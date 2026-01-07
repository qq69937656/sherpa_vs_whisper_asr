import os
import sys
import time
import soundfile as sf
import sherpa_onnx
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================

# LibriSpeech 数据集根目录
LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"

# 模型目录
MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"

# 并行线程数
PARALLEL_JOBS = 49

# 分片大小 (秒)
CHUNK_DURATION = 0.1

# ================= 全局锁 =================

print_lock = threading.Lock()


# ================= Sherpa-onnx 初始化 =================

def create_recognizer(model_dir):
    tokens_path = os.path.join(model_dir, "tokens.txt")
    encoder_path = ""
    decoder_path = ""
    joiner_path = ""

    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在 {model_dir}")
        sys.exit(1)

    for f in os.listdir(model_dir):
        if f.startswith("encoder-") and f.endswith(".onnx"):
            encoder_path = os.path.join(model_dir, f)
        elif f.startswith("decoder-") and f.endswith(".onnx"):
            decoder_path = os.path.join(model_dir, f)
        elif f.startswith("joiner-") and f.endswith(".onnx"):
            joiner_path = os.path.join(model_dir, f)

    if not (os.path.exists(tokens_path) and encoder_path and decoder_path and joiner_path):
        print(f"错误: 模型文件缺失。")
        sys.exit(1)

    # print(f"加载模型: {os.path.basename(encoder_path)}") # 多线程下减少打印

    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=1,  # 单个 Stream 内部单线程，并发由外部 ThreadPool 控制
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,
            provider="cpu"
        )
    except Exception as e:
        print(f"初始化警告: {e}, 尝试默认参数...")
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu"
        )
    return recognizer


# ================= 单文件任务处理 =================

def process_file_ttft(audio_path, recognizer):
    """
    计算 TTFT = (分片数 * 0.1) + 当前分片计算耗时
    """
    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
    except Exception as e:
        return None

    if sample_rate != 16000:
        return None

    # 跳过极短音频 (不足一个分片的)
    chunk_samples = int(sample_rate * CHUNK_DURATION)
    if len(audio) < chunk_samples:
        return None

    stream = recognizer.create_stream()

    chunk_count = 0
    ttft = None
    got_token = False

    # 遍历音频分片
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i: i + chunk_samples]
        chunk_count += 1

        # === 计时开始 (只统计当前分片的处理时间) ===
        t_start = time.perf_counter()

        stream.accept_waveform(sample_rate, chunk)

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result_obj = recognizer.get_result(stream)

        # === 计时结束 ===
        t_end = time.perf_counter()
        process_time = t_end - t_start  # 当前分片的计算耗时

        # 检查是否有文本输出
        current_text = result_obj if isinstance(result_obj, str) else getattr(result_obj, 'text', str(result_obj))

        if len(current_text.strip()) > 0:
            # 【核心逻辑】TTFT = 已消耗的音频时长 + 最后的计算耗时
            audio_time_consumed = chunk_count * CHUNK_DURATION
            ttft = audio_time_consumed + process_time
            got_token = True
            break  # 只要拿到第一个字，任务结束

    # 如果循环结束还没有结果（例如音频前段静音很长），尝试结束输入并强制解码
    if not got_token:
        t_start = time.perf_counter()

        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result_obj = recognizer.get_result(stream)

        t_end = time.perf_counter()
        process_time = t_end - t_start

        current_text = result_obj if isinstance(result_obj, str) else getattr(result_obj, 'text', str(result_obj))

        if len(current_text.strip()) > 0:
            # 这种情况下，音频已经全部流完了
            audio_time_consumed = len(audio) / sample_rate
            ttft = audio_time_consumed + process_time
            got_token = True

    if not got_token:
        return None

    return {
        "filename": os.path.basename(audio_path),
        "ttft": ttft,
        "chunks": chunk_count,
        "calc_time": process_time  # 记录一下最后那次计算花了多久，用于调试观察
    }


# ================= 主程序 =================

def main():
    print("-" * 70)
    print(f"Sherpa-onnx 并行 TTFT 测试")
    print(f"并发数 (Threads): {PARALLEL_JOBS}")
    print(f"计算公式: TTFT = (Chunks * {CHUNK_DURATION}s) + Parse_Time")

    recognizer = create_recognizer(MODEL_DIR)
    print("模型加载完成。")
    print("-" * 70)

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 数据集目录不存在 {LIBRISPEECH_DIR}")
        return

    # 1. 扫描文件
    print(f"正在扫描文件列表: {LIBRISPEECH_DIR} ...")
    all_audio_paths = []
    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        for f in files:
            if f.endswith(".wav") or f.endswith(".flac"):
                all_audio_paths.append(os.path.join(root, f))

    print(f"共找到 {len(all_audio_paths)} 个音频文件。开始并行处理...")
    print(f"{'文件名':<40} | {'分片数':<6} | {'计算耗时(s)':<12} | {'TTFT(s)':<10}")
    print("-" * 75)

    # 2. 线程池并行处理
    total_ttft = 0.0
    success_count = 0

    # 记录总墙钟时间 (Wall Clock Time)
    start_wall_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
        future_to_file = {
            executor.submit(process_file_ttft, path, recognizer): path
            for path in all_audio_paths
        }

        for future in as_completed(future_to_file):
            result = future.result()

            if result:
                ttft = result['ttft']
                total_ttft += ttft
                success_count += 1

                with print_lock:
                    display_name = result['filename']
                    if len(display_name) > 35:
                        display_name = display_name[:32] + "..."

                    # 打印详细信息：用了几个分片，最后一次计算用了多久，总TTFT是多少
                    print(f"{display_name:<40} | {result['chunks']:<6} | {result['calc_time']:<12.4f} | {ttft:<10.4f}")

    end_wall_time = time.perf_counter()
    total_duration_sec = end_wall_time - start_wall_time

    # ================= 最终报告 =================
    print("-" * 75)
    print("TTFT 测试完成")
    print(f"并发线程数: {PARALLEL_JOBS}")
    print(f"成功处理文件数: {success_count}")

    if success_count > 0:
        avg_ttft = total_ttft / success_count
        print(f"评测总耗时 (Wall Time): {total_duration_sec:.2f} 秒")
        print("=" * 40)
        print(f"平均首字延迟 (Average TTFT): {avg_ttft:.4f} 秒")
        print("=" * 40)
        print("注：TTFT 包含了音频流逝的物理时间 (Audio Duration) 和模型计算延迟。")
    else:
        print("未检测到任何输出文本。")


if __name__ == "__main__":
    main()
