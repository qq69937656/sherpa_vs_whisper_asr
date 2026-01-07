import os
import sys
import time
import soundfile as sf
import sherpa_onnx
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 配置区域 =================

LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"
MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"

# 并行进程数 (建议设为 4-8，视显存大小而定)
# V100 32G 可以尝试 8 或 10，如果 OOM (显存不足) 就调小
PARALLEL_JOBS = 1

CHUNK_DURATION = 0.1

# ================= 全局变量 (仅在子进程中生效) =================
# 这个变量在每个子进程中是独立的，不会冲突
process_recognizer = None


# ================= Sherpa-onnx 初始化逻辑 =================

def init_worker(model_dir):
    """
    子进程初始化函数：每个进程启动时执行一次，加载模型到显存。
    """
    global process_recognizer

    tokens_path = os.path.join(model_dir, "tokens.txt")
    encoder_path = ""
    decoder_path = ""
    joiner_path = ""

    for f in os.listdir(model_dir):
        if f.startswith("encoder-") and f.endswith(".onnx"):
            encoder_path = os.path.join(model_dir, f)
        elif f.startswith("decoder-") and f.endswith(".onnx"):
            decoder_path = os.path.join(model_dir, f)
        elif f.startswith("joiner-") and f.endswith(".onnx"):
            joiner_path = os.path.join(model_dir, f)

    print(f"[PID {os.getpid()}] 正在加载模型到 GPU: {os.path.basename(encoder_path)} ...")

    try:
        # 每个进程拥有独立的 recognizer
        process_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,
            provider="cuda"  # 启用 GPU
        )
        print(f"[PID {os.getpid()}] ✅ 模型加载完成 (CUDA)")
    except Exception as e:
        print(f"[PID {os.getpid()}] ❌ 模型加载失败: {e}")
        sys.exit(1)


# ================= 单个文件处理任务 =================

def process_file_task(audio_path):
    """
    工作函数：直接使用全局的 process_recognizer
    """
    global process_recognizer
    if process_recognizer is None:
        return None

    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
    except Exception:
        return None

    if sample_rate != 16000:
        return None

    audio_duration = len(audio) / sample_rate
    if audio_duration < 0.1:
        return None

    start_time = time.perf_counter()

    # 创建流
    stream = process_recognizer.create_stream()
    chunk_size = int(sample_rate * CHUNK_DURATION)

    # 模拟流式
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i: i + chunk_size]
        stream.accept_waveform(sample_rate, chunk)
        while process_recognizer.is_ready(stream):
            process_recognizer.decode_stream(stream)

    stream.input_finished()
    while process_recognizer.is_ready(stream):
        process_recognizer.decode_stream(stream)

    _ = process_recognizer.get_result(stream)

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    return {
        "filename": os.path.basename(audio_path),
        "duration": audio_duration,
        "inference_time": inference_time,
        "rtf": inference_time / audio_duration
    }


# ================= 主程序 =================

def main():
    # 设置启动方式为 spawn (Linux/CUDA 必须)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    print("-" * 60)
    print(f"Sherpa-onnx 多进程 GPU 测试 (并发数: {PARALLEL_JOBS})")
    print("-" * 60)

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 数据集不存在 {LIBRISPEECH_DIR}")
        return

    # 1. 扫描文件
    all_audio_paths = []
    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        for f in files:
            if f.endswith(".wav") or f.endswith(".flac"):
                all_audio_paths.append(os.path.join(root, f))

    # 仅测试前 100 个文件用于验证稳定性
    # all_audio_paths = all_audio_paths[:100]

    print(f"共找到 {len(all_audio_paths)} 个文件。准备启动进程池...")

    total_audio_duration = 0.0
    total_inference_time = 0.0
    success_count = 0

    # 2. 启动进程池
    # initializer=init_worker 会确保每个子进程启动时先加载好模型
    with ProcessPoolExecutor(max_workers=PARALLEL_JOBS, initializer=init_worker, initargs=(MODEL_DIR,)) as executor:
        futures = {executor.submit(process_file_task, path): path for path in all_audio_paths}

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    total_audio_duration += result['duration']
                    total_inference_time += result['inference_time']
                    success_count += 1

                    print(
                        f"[{success_count}/{len(all_audio_paths)}] {result['filename'][:30]:<30} | RTF: {result['rtf']:.4f}")
            except Exception as e:
                print(f"任务异常: {e}")

    # 3. 结果
    print("-" * 60)
    if total_audio_duration > 0:
        avg_rtf = total_inference_time / total_audio_duration
        print(f"总音频时长: {total_audio_duration:.2f}s")
        print(f"总计算耗时: {total_inference_time:.2f}s")
        print(f"平均 RTF:  {avg_rtf:.5f}")
        print(f"加速比:    {1 / avg_rtf:.2f}x")
    else:
        print("无结果。")


if __name__ == "__main__":
    main()
