import os
import sys
import time
import re
import warnings
import threading
import soundfile as sf
import sherpa_onnx
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================

# 数据集目录
LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"

# Sherpa-onnx 模型目录 (ASR)
ASR_MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"

# 并发数
PARALLEL_WORKERS = 49

# 分片大小
CHUNK_DURATION = 0.1

# 打印锁
print_lock = threading.Lock()

# 限制 PyTorch 线程数 (减少 CPU 竞争)
torch.set_num_threads(1)


# ================= 1. Sherpa-onnx 初始化 =================
def create_recognizer(model_dir):
    tokens = os.path.join(model_dir, "tokens.txt")
    encoder = ""
    decoder = ""
    joiner = ""

    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        sys.exit(1)

    for f in os.listdir(model_dir):
        if f.startswith("encoder-") and f.endswith(".onnx"):
            encoder = os.path.join(model_dir, f)
        elif f.startswith("decoder-") and f.endswith(".onnx"):
            decoder = os.path.join(model_dir, f)
        elif f.startswith("joiner-") and f.endswith(".onnx"):
            joiner = os.path.join(model_dir, f)

    try:
        # 优先使用 CUDA
        provider = "cuda" if torch.cuda.is_available() else "cpu"

        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,
            provider=provider
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        sys.exit(1)
    return recognizer


# ================= 2. 纯净版 TTFT 计算逻辑 =================

def process_file_ttft_pure(audio_path, recognizer):
    """
    计算 TTFT：
    严格定义为 [音频传输时间] + [ASR 解码得到第一个字符的时间]。
    不包含任何 NLP/Bert 后处理，确保反映真实的“首字上屏”速度。
    """
    filename = os.path.basename(audio_path)

    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
    except:
        return None

    if sample_rate != 16000: return None
    if len(audio) / sample_rate < 0.1: return None

    stream = recognizer.create_stream()
    chunk_samples = int(sample_rate * CHUNK_DURATION)

    # 虚拟时钟
    virtual_time = 0.0
    ttft = None
    first_raw_text = ""

    # 模拟流式分片
    for i in range(0, len(audio), chunk_samples):
        # 1. 计算理论上的音频到达时间
        chunk_index = i // chunk_samples
        audio_arrival_time = (chunk_index + 1) * CHUNK_DURATION

        chunk = audio[i: i + chunk_samples]

        # === 计时开始 (仅计算 ASR) ===
        t_start = time.perf_counter()

        stream.accept_waveform(sample_rate, chunk)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        t_end = time.perf_counter()
        # === 计时结束 ===

        # 2. 更新虚拟时钟
        # 系统时间 = max(当前系统时间, 音频到达时间) + ASR计算耗时
        compute_duration = t_end - t_start
        virtual_time = max(virtual_time, audio_arrival_time) + compute_duration

        # 3. 检查 ASR 原始输出
        raw_text = recognizer.get_result(stream)

        # 【关键修正】：一旦有原始文本，立刻结算，绝不运行 Bert
        if raw_text.strip():
            ttft = virtual_time
            first_raw_text = raw_text
            break  # 立即跳出，结束该文件的处理

    # 如果没出字，做最后一次检查
    if ttft is None:
        t_start = time.perf_counter()
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        raw_text = recognizer.get_result(stream)
        t_end = time.perf_counter()

        if raw_text.strip():
            virtual_time += (t_end - t_start)
            ttft = virtual_time
            first_raw_text = raw_text
        else:
            ttft = -1.0
            first_raw_text = "[NO SPEECH]"

    # 输出 (加锁)
    with print_lock:
        display_name = filename if len(filename) < 30 else filename[:27] + "..."
        if ttft > 0:
            print(f"{display_name:<30} | {ttft:<10.4f} | {first_raw_text[:40]}")
        else:
            print(f"{display_name:<30} | {'N/A':<10} | {first_raw_text}")

    return ttft


# ================= 3. 主程序 =================

def main():
    print("-" * 90)
    print(f"ASR 首字延迟(TTFT) 纯净测试")
    print(f"并发数: {PARALLEL_WORKERS} | 分片: {CHUNK_DURATION}s | 排除 BERT 干扰")

    try:
        # 只加载 ASR 模型，不需要 NLP
        recognizer = create_recognizer(ASR_MODEL_DIR)
        print("ASR 模型加载完成。")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"目录不存在 {LIBRISPEECH_DIR}")
        return

    file_list = []
    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        for f in files:
            if f.endswith(".wav") or f.endswith(".flac"):
                file_list.append(os.path.join(root, f))

    print(f"待处理文件: {len(file_list)}")
    print("-" * 90)
    print(f"{'文件名':<30} | {'TTFT(s)':<10} | {'ASR原始输出 (Raw)'}")
    print("-" * 90)

    ttft_results = []

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = [
            executor.submit(process_file_ttft_pure, f, recognizer)
            for f in file_list
        ]

        for future in as_completed(futures):
            res = future.result()
            if res is not None and res > 0:
                ttft_results.append(res)

    # ================= 报告 =================
    print("-" * 90)
    print("TTFT 测试完成 (ASR Only)")

    if ttft_results:
        count = len(ttft_results)
        avg_ttft = sum(ttft_results) / count
        p90 = np.percentile(ttft_results, 90)
        p99 = np.percentile(ttft_results, 99)

        print(f"有效样本数: {count}")
        print("=" * 40)
        print(f"平均 TTFT:      {avg_ttft:.4f} s")
        print(f"P90 延迟:       {p90:.4f} s")
        print(f"P99 延迟:       {p99:.4f} s")
        print("=" * 40)
        print("评价: 该指标反映了用户听到声音后，屏幕上弹出第一个字（无标点）的真实物理耗时。")
    else:
        print("无有效数据。")


if __name__ == "__main__":
    main()