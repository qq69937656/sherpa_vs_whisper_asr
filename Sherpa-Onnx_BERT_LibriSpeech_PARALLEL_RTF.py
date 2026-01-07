import os
import sys
import time
import re
import soundfile as sf
import sherpa_onnx
import numpy as np
import warnings
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================

# LibriSpeech 数据集根目录
LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"

# ASR 模型目录
ASR_MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"

# NLP 标点模型名称
PUNCT_MODEL_NAME = "oliverguhr/fullstop-punctuation-multilang-large"

# 并行线程数
PARALLEL_WORKERS = 29

# 模拟流式分片大小 (秒)
CHUNK_DURATION = 0.1

# 打印锁，防止控制台输出错乱
print_lock = threading.Lock()

# ================= 0. 环境调优 =================
# 关键优化：限制 PyTorch 内部线程数。
# 如果不限制，每个线程的 BERT 模型都会试图占用所有 CPU 核心，
# 导致 5 个线程互相打架，性能反而极其低下。
torch.set_num_threads(1)


# ================= 1. 文本后处理引擎 (NLP) =================
class TextPostProcessor:
    def __init__(self):
        print(f"Loading NLP Model: {PUNCT_MODEL_NAME} ...")
        try:
            from deepmultilingualpunctuation import PunctuationModel
            # 这里的 use_cuda=False 是为了保证在纯 CPU 压测下的公平性
            # 如果你有 GPU，库会自动检测，但多线程共用 GPU 模型需要小心显存
            self.model = PunctuationModel(model=PUNCT_MODEL_NAME)
        except Exception as e:
            print(f"NLP 模型加载失败: {e}")
            sys.exit(1)

    def process(self, text: str) -> str:
        if not text: return ""
        try:
            # 清洗
            clean_text = text.lower().strip()
            # 恢复标点 (BERT)
            punctuated = self.model.restore_punctuation(clean_text)
            # 正则修正
            punctuated = re.sub(r"\b(i)\b", "I", punctuated)
            punctuated = re.sub(r"\b(i')(?=m|ve|ll|d)\b", "I'", punctuated)

            def capitalize_match(match):
                return match.group(1) + match.group(2).upper()

            final_text = re.sub(r"(^|[.!?]\s+)([a-z])", capitalize_match, punctuated)
            return final_text
        except Exception:
            return text


# ================= 2. Sherpa-onnx 初始化 (ASR) =================
def create_recognizer(model_dir):
    tokens_path = os.path.join(model_dir, "tokens.txt")
    encoder_path = ""
    decoder_path = ""
    joiner_path = ""

    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        sys.exit(1)

    for f in os.listdir(model_dir):
        if f.startswith("encoder-") and f.endswith(".onnx"):
            encoder_path = os.path.join(model_dir, f)
        elif f.startswith("decoder-") and f.endswith(".onnx"):
            decoder_path = os.path.join(model_dir, f)
        elif f.startswith("joiner-") and f.endswith(".onnx"):
            joiner_path = os.path.join(model_dir, f)

    if not (encoder_path and decoder_path and joiner_path):
        print("模型文件缺失。")
        sys.exit(1)

    try:
        # num_threads=1: 在多线程环境下，必须限制单个解码器的线程数
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,
            provider="cuda"
        )
    except Exception as e:
        print(f"ASR 初始化失败: {e}")
        sys.exit(1)
    return recognizer


# ================= 3. 核心处理任务 =================

def process_single_file(audio_path, recognizer, post_processor):
    """
    单个文件的完整处理流程：读取 -> ASR -> NLP
    该函数将在线程池中运行。
    """
    audio_filename = os.path.basename(audio_path)

    try:
        # 读取音频
        audio, sample_rate = sf.read(audio_path, dtype="float32")
    except Exception as e:
        return None

    if sample_rate != 16000:
        return None

    audio_duration = len(audio) / sample_rate
    if audio_duration < 0.1:
        return None

    # === 计时开始 ===
    start_time = time.perf_counter()

    # 1. ASR
    # create_stream 是线程安全的，它会分配一个新的状态对象
    stream = recognizer.create_stream()
    chunk_size = int(sample_rate * CHUNK_DURATION)

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i: i + chunk_size]
        stream.accept_waveform(sample_rate, chunk)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    raw_text = recognizer.get_result(stream)

    # 2. NLP
    _ = post_processor.process(raw_text)

    # === 计时结束 ===
    end_time = time.perf_counter()

    process_time = end_time - start_time
    rtf = process_time / audio_duration

    # 格式化输出 (加锁)
    display_name = audio_filename if len(audio_filename) < 35 else audio_filename[:32] + "..."
    log_line = f"{display_name:<40} | {audio_duration:<12.2f} | {process_time:<15.4f} | {rtf:<10.4f}"

    with print_lock:
        print(log_line)

    return (audio_duration, process_time)


# ================= 4. 主程序 =================

def main():
    print("-" * 60)
    print(f"初始化并发系统 (Workers={PARALLEL_WORKERS})...")

    # 全局模型初始化 (主线程执行)
    post_processor = TextPostProcessor()
    recognizer = create_recognizer(ASR_MODEL_DIR)

    print("模型加载完成。")
    print("-" * 60)

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 数据集目录不存在 {LIBRISPEECH_DIR}")
        return

    # 收集文件
    all_audio_paths = []
    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        for f in files:
            if f.endswith(".wav") or f.endswith(".flac"):
                all_audio_paths.append(os.path.join(root, f))

    print(f"找到 {len(all_audio_paths)} 个音频文件。")
    print(f"开始并行处理...")
    print(f"{'文件名':<40} | {'音频时长(s)':<12} | {'完整生成耗时(s)':<15} | {'实时率(RTF)':<10}")
    print("-" * 90)

    # 统计数据
    total_audio_duration = 0.0
    total_process_time = 0.0
    processed_count = 0

    # 启动线程池
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        # 提交所有任务
        # future_to_file 字典用于追踪任务（可选，这里简化处理）
        futures = [
            executor.submit(process_single_file, path, recognizer, post_processor)
            for path in all_audio_paths
        ]

        # as_completed 会在任务完成时立即 yield，实现"完成一个处理一个"的效果
        for future in as_completed(futures):
            result = future.result()  # 获取 process_single_file 的返回值

            if result:
                duration, p_time = result
                total_audio_duration += duration
                total_process_time += p_time
                processed_count += 1

    # ================= 最终报告 =================
    print("-" * 90)
    print("并发性能测试完成")
    print(f"并发线程数: {PARALLEL_WORKERS}")
    print(f"处理文件总数: {processed_count}")

    if total_audio_duration > 0:
        avg_rtf = total_process_time / total_audio_duration
        print(f"总音频时长: {total_audio_duration:.2f} 秒")
        # 注意：在并发下，Total Process Time 是所有线程耗时的总和（CPU总时间），而不是墙上时钟时间
        print(f"总生成耗时(CPU Time): {total_process_time:.2f} 秒")
        print("=" * 40)
        print(f"平均实时率 (Average RTF): {avg_rtf:.5f}")
        print("=" * 40)

        # 性能评估逻辑
        if avg_rtf > 1.0:
            print("性能评价: 严重拥堵 (RTF > 1)")
            print("原因: CPU 核心数不足或 BERT 模型计算量过大导致资源争抢。")
        else:
            throughput = PARALLEL_WORKERS / avg_rtf
            print(f"性能评价: 系统通常能同时支持 {throughput:.1f} 路类似并发请求")

    else:
        print("未处理任何有效文件。")


if __name__ == "__main__":
    main()
