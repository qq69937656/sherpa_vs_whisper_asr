import os
import sys
import time
import soundfile as sf
import sherpa_onnx
import numpy as np

# ================= 配置区域 =================

# LibriSpeech 数据集根目录
# 程序会递归查找该目录下的所有 .wav (或 .flac) 文件
LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"

# 模型目录
MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"

# 模拟流式分片大小 (秒)
CHUNK_DURATION = 0.1


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
        print(f"错误: 模型文件缺失。\nTokens: {tokens_path}")
        sys.exit(1)

    print(
        f"加载模型:\n  Enc: {os.path.basename(encoder_path)}\n  Dec: {os.path.basename(decoder_path)}\n  Join: {os.path.basename(joiner_path)}")

    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            # 关闭端点检测以确保处理完整音频，且避免短音频报错
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
            provider="cuda"
        )
    return recognizer


# ================= 主程序 =================

def main():
    print("-" * 50)
    print("正在初始化 Sherpa-onnx (RTF 性能测试)...")
    recognizer = create_recognizer(MODEL_DIR)
    print("模型加载完成。")
    print("-" * 50)

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 数据集目录不存在 {LIBRISPEECH_DIR}")
        return

    # 全局统计变量
    total_audio_duration = 0.0  # 总音频时长 (秒)
    total_inference_time = 0.0  # 总推理耗时 (秒)
    file_count = 0

    print(f"开始遍历数据集: {LIBRISPEECH_DIR}")
    print(f"{'文件名':<40} | {'音频时长(s)':<10} | {'推理耗时(s)':<10} | {'实时率(RTF)':<10}")
    print("-" * 80)

    # 遍历目录
    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        # 查找音频文件 (支持 .wav 和 .flac)
        audio_files = [f for f in files if f.endswith(".wav") or f.endswith(".flac")]

        for audio_filename in audio_files:
            audio_path = os.path.join(root, audio_filename)

            # 1. 读取音频 (不计入推理时间)
            try:
                audio, sample_rate = sf.read(audio_path, dtype="float32")
            except Exception as e:
                print(f"读取失败 {audio_filename}: {e}")
                continue

            if sample_rate != 16000:
                # 简单跳过非16k音频
                continue

            # 获取音频时长
            audio_duration = len(audio) / sample_rate

            # 跳过过短音频 (避免极短片段导致的不稳定或报错)
            if audio_duration < 0.1:
                continue

            # ================= 计时开始 =================
            start_time = time.perf_counter()

            stream = recognizer.create_stream()
            chunk_size = int(sample_rate * CHUNK_DURATION)  # 1600 samples

            # 模拟流式输入 (全速运行，不 sleep)
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i: i + chunk_size]
                stream.accept_waveform(sample_rate, chunk)

                # 只要模型准备好，就立即解码
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            # 获取结果 (这一步通常很快，也可以算作流程一部分)
            _ = recognizer.get_result(stream)

            # ================= 计时结束 =================
            end_time = time.perf_counter()

            # 计算指标
            inference_time = end_time - start_time
            rtf = inference_time / audio_duration

            # 累加全局统计
            total_audio_duration += audio_duration
            total_inference_time += inference_time
            file_count += 1

            # 输出单文件结果
            # 为了版面整洁，截断过长的文件名
            display_name = audio_filename if len(audio_filename) < 35 else audio_filename[:32] + "..."
            print(f"{display_name:<40} | {audio_duration:<10.2f} | {inference_time:<10.4f} | {rtf:<10.4f}")

    # ================= 最终报告 =================
    print("-" * 80)
    print("性能测试完成")
    print(f"处理文件总数: {file_count}")

    if total_audio_duration > 0:
        avg_rtf = total_inference_time / total_audio_duration
        print(f"总音频时长: {total_audio_duration:.2f} 秒")
        print(f"总推理耗时: {total_inference_time:.2f} 秒")
        print("=" * 40)
        print(f"平均实时率 (Average RTF): {avg_rtf:.5f}")
        if avg_rtf < 1:
            speed_x = 1 / avg_rtf
            print(f"性能评价: 比实时快 {speed_x:.2f} 倍")
        else:
            print(f"性能评价: 比实时慢")
        print("=" * 40)
    else:
        print("未处理任何有效文件。")


if __name__ == "__main__":
    main()
