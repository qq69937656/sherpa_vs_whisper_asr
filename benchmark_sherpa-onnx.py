import sys
import os
import time
import wave
import numpy as np
import sherpa_onnx
from collections import deque

# ================= 配置区域 =================
# LibriSpeech test-clean 数据集根目录
LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"

# 模型目录
MODEL_DIR = "sherpa-onnx-streaming-zipformer-en-2023-06-21"

# 并发数 (模拟多少路会议同时进行)
CONCURRENT_STREAMS = 51

# 切片时长 (秒)
CHUNK_SECONDS = 0.1

# 设备: "cpu" 或 "cuda"
PROVIDER = "cuda"


# ===========================================

class WorkerSlot:
    """代表并发池中的一个工位"""

    def __init__(self):
        self.is_active = False
        self.stream = None
        self.filename = None
        self.samples = None
        self.duration = 0.0
        self.cursor = 0  # 当前喂到了第几个采样点
        self.start_time = 0.0  # 物理开始时间 (用于计算 RTF)
        self.ttft = None  # 首字延迟
        self.has_output = False
        self.input_finished = False

    def reset(self):
        """重置工位"""
        self.is_active = False
        self.stream = None
        self.filename = None
        self.samples = None
        self.duration = 0.0
        self.cursor = 0
        self.start_time = 0.0
        self.ttft = None
        self.has_output = False
        self.input_finished = False


def find_wav_files(root_dir):
    wav_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return sorted(wav_files)


def read_wav_file(filename):
    try:
        with wave.open(filename, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getframerate() != 16000:
                return None, 0
            num_frames = wf.getnframes()
            data = wf.readframes(num_frames)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            duration = num_frames / 16000.0
            return samples, duration
    except Exception as e:
        print(f"读取失败 {filename}: {e}")
        return None, 0


def main():
    # 0. 初始化
    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 目录不存在: {LIBRISPEECH_DIR}")
        return

    print(f"正在搜索文件...")
    all_files = find_wav_files(LIBRISPEECH_DIR)
    if not all_files:
        print("未找到 wav 文件")
        return

    # 任务队列
    task_queue = deque(all_files)
    total_files_count = len(all_files)
    print(f"共发现 {total_files_count} 个任务。")

    # 加载模型
    print(f"加载模型: {MODEL_DIR} ({PROVIDER})...")
    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=os.path.join(MODEL_DIR, "tokens.txt"),
            encoder=os.path.join(MODEL_DIR, "encoder-epoch-99-avg-1.onnx"),
            decoder=os.path.join(MODEL_DIR, "decoder-epoch-99-avg-1.onnx"),
            joiner=os.path.join(MODEL_DIR, "joiner-epoch-99-avg-1.onnx"),
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=PROVIDER,
            enable_endpoint_detection=False
        )
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return

    # 初始化并发池
    slots = [WorkerSlot() for _ in range(CONCURRENT_STREAMS)]
    finished_results = []
    sample_rate = 16000
    chunk_samples = int(sample_rate * CHUNK_SECONDS)

    print("\n" + "=" * 60)
    print(f"开始测试 (TTFT包含音频时长) | 并发: {CONCURRENT_STREAMS}")
    print("=" * 60)

    start_benchmark_time = time.time()
    files_processed = 0

    # === 主循环 ===
    while len(task_queue) > 0 or any(s.is_active for s in slots):

        # 1. 补货 (Replenish)
        for i in range(CONCURRENT_STREAMS):
            if not slots[i].is_active and len(task_queue) > 0:
                next_file = task_queue.popleft()
                samples, duration = read_wav_file(next_file)
                if samples is None: continue

                slots[i].reset()
                slots[i].is_active = True
                slots[i].stream = recognizer.create_stream()
                slots[i].filename = next_file
                slots[i].samples = samples
                slots[i].duration = duration
                slots[i].start_time = time.perf_counter()  # 记录物理开始时间
                slots[i].cursor = 0

        # 2. 喂数据 (Feed)
        for s in slots:
            if s.is_active and not s.input_finished:
                start = s.cursor
                end = min(start + chunk_samples, len(s.samples))
                chunk = s.samples[start:end]

                # 补零防止崩溃
                if len(chunk) < chunk_samples:
                    padding = np.zeros(chunk_samples - len(chunk), dtype=np.float32)
                    chunk = np.concatenate((chunk, padding))

                chunk = np.ascontiguousarray(chunk)
                s.stream.accept_waveform(sample_rate, chunk)
                s.cursor += chunk_samples  # 指针前移

                if start + chunk_samples >= len(s.samples):
                    s.stream.input_finished()
                    s.input_finished = True

        # 3. 推理 (Decode)
        ready_streams = [s.stream for s in slots if s.is_active and recognizer.is_ready(s.stream)]
        if ready_streams:
            recognizer.decode_streams(ready_streams)

        # 4. 检查结果 (Check)
        current_wall_time = time.perf_counter()

        for s in slots:
            if not s.is_active: continue

            # --- 计算 TTFT (关键修改) ---
            if not s.has_output:
                res = recognizer.get_result(s.stream)
                text = res if isinstance(res, str) else res.text
                if text.strip():
                    # Audio Lag: 模拟的音频时间 (例如 3 * 0.1 = 0.3s)
                    audio_lag = s.cursor / sample_rate
                    # Compute Lag: 机器实际计算耗时
                    compute_lag = current_wall_time - s.start_time

                    # 用户真实感知延迟
                    s.ttft = audio_lag + compute_lag
                    s.has_output = True

            # --- 检查完成 ---
            if s.input_finished and not recognizer.is_ready(s.stream):
                # 任务结束，计算 RTF
                compute_duration = current_wall_time - s.start_time
                rtf = compute_duration / s.duration

                # 如果一直没出字，TTFT = 总时长
                final_ttft = s.ttft if s.ttft else (s.duration + compute_duration)

                result = {
                    "filename": os.path.basename(s.filename),
                    "duration": s.duration,
                    "rtf": rtf,
                    "ttft": final_ttft
                }
                finished_results.append(result)
                files_processed += 1

                print(
                    f"[{files_processed}/{total_files_count}] {result['filename'][:25]}... | RTF: {rtf:.4f} | TTFT: {final_ttft:.4f}s")
                s.reset()  # 释放工位

    # === 统计报告 ===
    total_time = time.time() - start_benchmark_time
    print("\n" + "=" * 60)
    print("                测试报告")
    print("=" * 60)

    if finished_results:
        avg_rtf = sum(r['rtf'] for r in finished_results) / len(finished_results)
        avg_ttft = sum(r['ttft'] for r in finished_results) / len(finished_results)
        p90_rtf = np.percentile([r['rtf'] for r in finished_results], 90)

        print(f"总物理耗时:     {total_time:.2f} s")
        print(f"处理文件数:     {len(finished_results)}")
        print(f"并发流数:       {CONCURRENT_STREAMS}")
        print("-" * 60)
        print(f"平均 RTF:       {avg_rtf:.4f}")
        print(f"P90  RTF:       {p90_rtf:.4f}")
        print(f"平均 TTFT:      {avg_ttft:.4f} s (含音频播放时间)")
        print("-" * 60)

        # 保存 CSV
        with open("benchmark_librispeech_final.txt", "w") as f:
            f.write("Filename,Duration,RTF,TTFT\n")
            for r in finished_results:
                f.write(f"{r['filename']},{r['duration']:.4f},{r['rtf']:.4f},{r['ttft']:.4f}\n")
        print("详细结果已保存至 benchmark_librispeech_final.txt")


if __name__ == "__main__":
    main()
