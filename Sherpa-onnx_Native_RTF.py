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
    """并发流式工位状态机"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.is_active = False
        self.stream = None
        self.file_id = ""
        self.audio_samples = None
        self.duration = 0.0
        self.cursor = 0
        self.start_time = 0.0
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
            enable_endpoint_detection=False, provider="cpu"  # 压测环境统一使用CPU或根据环境改cuda
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
    csv_filename = f"Results_Sherpa-onnx_{args.dataset_name}_Native_RTF_C{args.concurrency}.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["FileID", "Duration(s)", "Compute_Time(s)", "RTF"])

    print(f"初始化 Sherpa-ONNX RTF 并发压测 (并发数: {args.concurrency})...")
    recognizer = create_recognizer(MODEL_DIR)

    # 扫描音频
    all_files = []
    for root, _, files in os.walk(args.dataset_path):
        for f in files:
            if f.endswith(".wav") or f.endswith(".flac"):
                all_files.append(os.path.join(root, f))

    task_queue = deque(all_files)
    total_tasks = len(task_queue)
    if total_tasks == 0:
        print("未找到音频文件。")
        return

    # 并发工位初始化
    slots = [WorkerSlot() for _ in range(args.concurrency)]
    processed_count = 0
    total_rtf_sum = 0.0
    global_start_time = time.time()

    sample_rate = 16000
    chunk_samples = int(sample_rate * CHUNK_DURATION)

    print("=" * 60)
    print(f"🚀 开始极限吞吐压测，实时详细数据正写入: {csv_filename}")
    print("=" * 60)

    # 核心轮询调度
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

                s.reset()
                s.is_active = True
                s.stream = recognizer.create_stream()
                s.file_id = os.path.basename(file_path)
                s.audio_samples = audio
                s.duration = len(audio) / sr
                s.start_time = time.perf_counter()

        # 2. 模拟极限切片流式输入
        for s in slots:
            if s.is_active and not s.input_finished:
                start = s.cursor
                end = min(start + chunk_samples, len(s.audio_samples))
                chunk = s.audio_samples[start:end]

                # 尾部补齐
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
                # 兼容部分不支持批量解码的版本
                for stream in ready_streams:
                    recognizer.decode_stream(stream)

        # 4. 指标回收
        current_time = time.perf_counter()
        for s in slots:
            if s.is_active and s.input_finished and not recognizer.is_ready(s.stream):
                compute_time = current_time - s.start_time
                rtf = compute_time / s.duration if s.duration > 0 else 0

                total_rtf_sum += rtf
                processed_count += 1

                # 静默落盘
                with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([s.file_id, round(s.duration, 4), round(compute_time, 4), round(rtf, 4)])

                # 控制台进度更新 (每 50 个文件打印一次)
                if processed_count % 50 == 0 or processed_count == total_tasks:
                    print(f"[{processed_count}/{total_tasks}] | 最近完成: {s.file_id[:20]:<20} | RTF: {rtf:.4f}")

                s.reset()

    wall_time = time.time() - global_start_time
    avg_rtf = total_rtf_sum / processed_count if processed_count > 0 else 0

    print("=" * 40)
    print("RTF 测试完成")
    print(f"并发线程数: {args.concurrency}")
    print(f"成功处理文件数: {processed_count}")
    print(f"评测总耗时 (Wall Time): {wall_time:.2f} 秒")
    print("=" * 40)
    print(f"平均实时率 (Average RTF): {avg_rtf:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()