import os
import sys
import time
import re
import soundfile as sf
import sherpa_onnx
import numpy as np
import warnings

# 忽略 transformers 的版本警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================

# LibriSpeech 数据集根目录
LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"

# Sherpa-onnx 模型目录 (ASR)
ASR_MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"

# 标点模型名称 (NLP)
PUNCT_MODEL_NAME = "oliverguhr/fullstop-punctuation-multilang-large"

# 模拟流式分片大小 (秒)
CHUNK_DURATION = 0.1


# ================= 1. 文本后处理引擎 (NLP) =================

class TextPostProcessor:
    """
    负责对 ASR 的原始输出进行标点恢复和格式修正
    """

    def __init__(self):
        print(f"正在加载标点模型: {PUNCT_MODEL_NAME} ...")
        try:
            from deepmultilingualpunctuation import PunctuationModel
            # 自动加载到 GPU (如果可用) 或 CPU
            self.model = PunctuationModel(model=PUNCT_MODEL_NAME)
        except ImportError:
            print("错误: 缺少依赖库，请运行: pip install deepmultilingualpunctuation torch")
            sys.exit(1)
        except Exception as e:
            print(f"标点模型加载失败: {e}")
            sys.exit(1)

    def process(self, text: str) -> str:
        """
        执行流程: 原始文本 -> 清洗 -> BERT标点 -> 正则格式化 -> 最终文本
        """
        if not text:
            return ""

        try:
            # 1. 清洗 (转小写)
            clean_text = text.lower().strip()

            # 2. BERT 模型恢复标点 (这是耗时大户)
            punctuated = self.model.restore_punctuation(clean_text)

            # 3. 正则表达式规则修正
            # 修正单独的 i -> I
            punctuated = re.sub(r"\b(i)\b", "I", punctuated)
            # 修正 i'm, i'll 等
            punctuated = re.sub(r"\b(i')(?=m|ve|ll|d)\b", "I'", punctuated)

            # 强制句首大写
            def capitalize_match(match):
                return match.group(1) + match.group(2).upper()

            final_text = re.sub(r"(^|[.!?]\s+)([a-z])", capitalize_match, punctuated)

            return final_text

        except Exception as e:
            # 如果出错，为了不中断流程，返回原文本
            return text


# ================= 2. Sherpa-onnx 初始化 (ASR) =================

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
            enable_endpoint_detection=False,  # 关闭 VAD，确保测试完整音频
            provider="cuda"
        )
    except Exception as e:
        print(f"ASR 初始化失败: {e}")
        sys.exit(1)

    return recognizer


# ================= 3. 主程序 =================

def main():
    print("-" * 60)
    print("正在初始化 AI 引擎 (ASR + NLP)...")

    # 初始化两个模型
    post_processor = TextPostProcessor()
    recognizer = create_recognizer(ASR_MODEL_DIR)

    print("所有模型加载完成。")
    print("-" * 60)

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 数据集目录不存在 {LIBRISPEECH_DIR}")
        return

    # 全局统计
    total_audio_duration = 0.0
    total_process_time = 0.0  # 包含 ASR 和 NLP 的总时间
    file_count = 0

    print(f"开始遍历数据集: {LIBRISPEECH_DIR}")
    # 输出表头
    print(f"{'文件名':<40} | {'音频时长(s)':<12} | {'完整生成耗时(s)':<15} | {'实时率(RTF)':<10}")
    print("-" * 90)

    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        audio_files = [f for f in files if f.endswith(".wav") or f.endswith(".flac")]

        for audio_filename in audio_files:
            audio_path = os.path.join(root, audio_filename)

            # 读取音频
            try:
                audio, sample_rate = sf.read(audio_path, dtype="float32")
            except Exception:
                continue

            if sample_rate != 16000: continue

            audio_duration = len(audio) / sample_rate
            if audio_duration < 0.1: continue

            # ================= 计时开始 (ASR + NLP) =================
            start_time = time.perf_counter()

            # --- 步骤 1: 语音识别 (ASR) ---
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

            # --- 步骤 2: 文本后处理 (NLP) ---
            # 即使不输出文本，也必须执行这一步以统计耗时
            _ = post_processor.process(raw_text)

            # ================= 计时结束 =================
            end_time = time.perf_counter()

            # 计算
            process_time = end_time - start_time
            rtf = process_time / audio_duration

            # 累加
            total_audio_duration += audio_duration
            total_process_time += process_time
            file_count += 1

            # 输出单行
            display_name = audio_filename if len(audio_filename) < 35 else audio_filename[:32] + "..."
            print(f"{display_name:<40} | {audio_duration:<12.2f} | {process_time:<15.4f} | {rtf:<10.4f}")

    # ================= 最终报告 =================
    print("-" * 90)
    print("性能测试完成 (ASR + BERT Post-processing)")
    print(f"处理文件总数: {file_count}")

    if total_audio_duration > 0:
        avg_rtf = total_process_time / total_audio_duration
        print(f"总音频时长: {total_audio_duration:.2f} 秒")
        print(f"总生成耗时: {total_process_time:.2f} 秒")
        print("=" * 40)
        print(f"平均实时率 (Average RTF): {avg_rtf:.5f}")
        print("=" * 40)
        if avg_rtf > 1.0:
            print("注意: 当前 RTF > 1，说明处理速度慢于说话速度（非实时）。")
            print("原因可能是 BERT 标点模型计算量较大，建议在 GPU 环境运行或换用更小的 NLP 模型。")
    else:
        print("未处理任何有效文件。")


if __name__ == "__main__":
    main()
