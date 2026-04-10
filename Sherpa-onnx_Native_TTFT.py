import os
import sys
import time
import argparse
import csv
import soundfile as sf
import sherpa_onnx
from collections import deque

# ==========================================
# 常量配置
# ==========================================
MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"
CHUNK_DURATION = 0.1  # 流式切片大小 (100ms)


class WorkerSlot:
    """TTFT 并发流式工位状态机"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.is_active = False
        self.stream = None
        self.file_id = ""
        self.audio_samples = None
        self.cursor = 0
        self.start_time = 0.0
        self.ttft = None
        self.has_output = False
        self.input_finished = False


def create_recognizer(model_dir):
    tokens_path = os.path.join(model_dir, "tokens.txt")
    encoder_path = decoder_path = joiner_path = ""
    for f in os.listdir(model_dir):
        if f.startswith("encoder-") and f.endswith(".onnx"):
            encoder_path = os.path.join(model_dir, f)
        elif f.startswith("decoder-") and f.endswith(".onnx"):
            decoder_path = os.path.join(model_dir, f)
        elif f.startswith("joiner-") and f.endswith(".onnx"):
            joiner_path = os.path.join(model_dir, f)

    try:
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path, encoder=encoder_path, decoder=decoder_path, joiner=joiner_path,
            num_threads=1, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
            enable_endpoint_detection=False, provider="cpu"
        )
    except Exception:
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path, encoder=encoder_path, decoder=decoder_path, joiner=joiner_path,
            num_threads=1, sample_rate=16000, feature_dim=80, decoding_method="greedy_search", provider="cpu"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()

    # 初始化日志文件
    csv_filename = f"Results_Sherpa-onnx_{args.dataset_name}_Native_TTFT_C{args.concurrency}.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["FileID", "Chunks_Processed_At_TTFT", "TTFT(s)"])

    print(f"初始化 Sherpa-ONNX TTFT 并发压测 (并发数: {args.concurrency})...")
    recognizer = create_recognizer(MODEL_DIR)

    # 扫描音频
    all_files = []
    for root, _, files in os.walk(args.dataset_path):
        for f in files:
            if f.endswith(".wav") or f.endswith(".flac"):
                all_files.append(os.path.join(root, f))

    task_queue = deque(all_files)
    total_tasks = len(task_queue)
    if total_tasks == 0: return

    # 并发工位初始化
    slots = [WorkerSlot() for _ in range(args.concurrency)]
    processed_count = 0
    total_ttft_sum = 0.0
    global_start_time = time.time()

    sample_rate = 16000
    chunk_samples = int(sample_rate * CHUNK_DURATION)

    print("=" * 60)
    print(f"🚀 开始极限吞吐压测，实时详细数据正写入: {csv_filename}")
    print("=" * 60)

    while len(task_queue) > 0 or any(s.is_active for s in slots):
        # 1. 任务分配
        for s in slots:
            if not s.is_active and task_queue:
                file_path = task_queue.popleft()
                try:
                    audio, sr = sf.read(file_path, dtype="float32")
                    if sr != 16000: continue
                except Exception:
                    continue

                # 过滤极其短的录音（不够产生第一个词的）
                if len(audio) < sample_rate * 0.5: continue

                s.reset()
                s.is_active = True
                s.stream = recognizer.create_stream()
                s.file_id = os.path.basename(file_path)
                s.audio_samples = audio
                s.start_time = time.perf_counter()

        # 2. 模拟切片流式输入
        for s in slots:
            if s.is_active and not s.input_finished:
                start = s.cursor
                end = min(start + chunk_samples, len(s.audio_samples))
                chunk = s.audio_samples[start:end]

                import numpy as np
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')

                s.stream.accept_waveform(sample_rate, chunk)
                s.cursor += chunk_samples

                if s.cursor >= len(s.audio_samples):
                    s.stream.input_finished()
                    s.input_finished = True

        # 3. 批量推理
        ready_streams = [s.stream for s in slots if s.is_active and recognizer.is_ready(s.stream)]
        if ready_streams:
            try:
                recognizer.decode_streams(ready_streams)
            except AttributeError:
                for stream in ready_streams:
                    recognizer.decode_stream(stream)

        # 4. 重点：非阻塞实时捕捉首字延迟 (TTFT)
        current_time = time.perf_counter()
        for s in slots:
            if s.is_active:
                # 核心捕捉逻辑：一旦模型产生了非空文本，立刻记录 TTFT
                if not s.has_output:
                    res = recognizer.get_result(s.stream)
                    text = res if isinstance(res, str) else getattr(res, 'text', str(res))
                    if text.strip():
                        s.ttft = current_time - s.start_time
                        s.has_output = True

                        chunk_count = s.cursor // chunk_samples
                        total_ttft_sum += s.ttft
                        processed_count += 1

                        # 静默落盘
                        with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow([s.file_id, chunk_count, round(s.ttft, 4)])

                        # 控制台进度更新
                        if processed_count % 50 == 0 or processed_count == total_tasks:
                            print(
                                f"[{processed_count}/{total_tasks}] | 最近完成: {s.file_id[:20]:<20} | TTFT: {s.ttft:.4f}s")

                # 如果流结束了还没输出，或者已经记录过 TTFT，则可以回收工位
                if s.input_finished and not recognizer.is_ready(s.stream):
                    # 对于极短音频，如果跑到最后才出结果
                    if not s.has_output:
                        s.ttft = current_time - s.start_time
                        total_ttft_sum += s.ttft
                        processed_count += 1
                        with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow([s.file_id, "End_of_Audio", round(s.ttft, 4)])
                    s.reset()

    wall_time = time.time() - global_start_time
    avg_ttft = total_ttft_sum / processed_count if processed_count > 0 else 0

    print("=" * 40)
    print("TTFT 测试完成")
    print(f"并发线程数: {args.concurrency}")
    print(f"成功处理文件数: {processed_count}")
    print(f"评测总耗时 (Wall Time): {wall_time:.2f} 秒")
    print("=" * 40)
    print(f"平均首字延迟 (Average TTFT): {avg_ttft:.4f} 秒")
    print("=" * 40)


if __name__ == "__main__":
    main()