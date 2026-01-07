import os
import sys
import re
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
import jiwer

# ================= 配置区域 =================

LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"
MODEL_SIZE = "medium.en"
CHUNK_DURATION = 1.0  # 1S 分片

# 【新增】静音阈值：RMS 能量低于此值视为静音，不送入模型
# LibriSpeech 通常较清晰，0.002 是一个经验保守值；嘈杂环境需调高
SILENCE_THRESHOLD = 0.01

# 【新增】最大词数限制：1秒内人类很难说超过 6 个单词
MAX_WORDS_PER_SEC = 6


# ================= 辅助函数 =================

def normalize_text(text):
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.strip().split()
    return words


def calculate_rms(audio_chunk):
    """计算音频切片的均方根能量 (Root Mean Square)"""
    if len(audio_chunk) == 0:
        return 0
    return np.sqrt(np.mean(audio_chunk ** 2))


# ================= Faster-Whisper 初始化 =================

def create_recognizer(model_size):
    try:
        # float16 + cuda
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)


# ================= 主程序 =================

def main():
    print("-" * 50)
    print(f"正在加载模型 {MODEL_SIZE} (包含能量VAD + 字数熔断机制)...")
    recognizer = create_recognizer(MODEL_SIZE)
    print("模型加载完成。")
    print("-" * 50)

    total_distance = 0
    total_ref_words = 0
    file_count = 0

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 数据集目录不存在 {LIBRISPEECH_DIR}")
        return

    print(f"开始遍历数据集: {LIBRISPEECH_DIR}")

    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        trans_files = [f for f in files if f.endswith(".trans.txt")]

        for trans_file in trans_files:
            trans_path = os.path.join(root, trans_file)
            ref_map = {}

            # 读取 trans.txt
            try:
                with open(trans_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            file_id, text = parts
                            ref_map[file_id] = text
            except Exception as e:
                print(f"读取文本文件失败 {trans_path}: {e}")
                continue

            for file_id, ref_text in ref_map.items():
                audio_filename = f"{file_id}.wav"
                audio_path = os.path.join(root, audio_filename)

                # 兼容 flac
                if not os.path.exists(audio_path):
                    flac_path = os.path.join(root, f"{file_id}.flac")
                    if os.path.exists(flac_path):
                        audio_path = flac_path
                    else:
                        continue

                try:
                    audio, sample_rate = sf.read(audio_path, dtype="float32")
                except Exception as e:
                    print(f"读取音频失败 {audio_path}: {e}")
                    continue

                if len(audio) < int(sample_rate * 0.05):
                    continue

                # ================= 核心推理逻辑 =================
                chunk_size = int(sample_rate * CHUNK_DURATION)
                hyp_segments_list = []
                last_text = ""  # 用于防重复

                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i: i + chunk_size]

                    if len(chunk) < 160:
                        continue

                    # --- 1. 硬规则：能量 VAD ---
                    # 如果这1秒全是静音，Whisper 极大概率会产生幻觉，必须在这里拦截
                    rms = calculate_rms(chunk)
                    if rms < SILENCE_THRESHOLD:
                        # 能量过低，直接视为静音，跳过推理
                        continue

                    # --- 2. 推理 ---
                    try:
                        # 移除 initial_prompt，因为在短切片下，Prompt 有时反而会诱导模型生成无关内容
                        segments, info = recognizer.transcribe(
                            chunk,
                            beam_size=1,
                            language="en",
                            vad_filter=True,  # 开启内置 VAD 过滤切片内部的静音
                            condition_on_previous_text=False,
                            temperature=0.0,  # 贪婪解码
                            # 提高压缩比阈值，防止循环重复
                            compression_ratio_threshold=2.0,
                            # 提高静音概率阈值
                            no_speech_threshold=0.4
                        )

                        current_chunk_text = ""
                        for segment in segments:
                            txt = segment.text.strip()

                            # --- 3. 硬规则：黑名单过滤 ---
                            hallucinations = ["see you", "watching", "subtitles", "amara", "org", "bye"]
                            if any(h in txt.lower() for h in hallucinations):
                                continue

                            current_chunk_text += " " + txt

                        current_chunk_text = current_chunk_text.strip()

                        # --- 4. 硬规则：字数熔断 (Length Filter) ---
                        # 1秒内如果输出了太多字，绝对是幻觉（例如输出了整句废话）
                        words_in_chunk = len(current_chunk_text.split())
                        if words_in_chunk > MAX_WORDS_PER_SEC:
                            # 丢弃该片段
                            continue

                        # --- 5. 硬规则：防重复 ---
                        if current_chunk_text and current_chunk_text != last_text:
                            hyp_segments_list.append(current_chunk_text)
                            last_text = current_chunk_text

                    except Exception as e:
                        continue

                hyp_text_raw = " ".join(hyp_segments_list)
                # ===============================================

                # ================= 评测逻辑 =================
                ref_words_list = normalize_text(ref_text)
                hyp_words_list = normalize_text(hyp_text_raw)

                ref_str_clean = " ".join(ref_words_list)
                hyp_str_clean = " ".join(hyp_words_list)

                try:
                    out = jiwer.process_words(ref_str_clean, hyp_str_clean)

                    curr_dist = out.substitutions + out.deletions + out.insertions
                    curr_wer = out.wer
                    curr_len = len(out.references[0]) if out.references else 0

                    total_distance += curr_dist
                    total_ref_words += curr_len
                    file_count += 1

                    print(
                        f"File: {os.path.basename(audio_path)} | WER: {curr_wer:.4f} | Dist: {curr_dist} | RefWords: {curr_len}")

                except Exception as e:
                    print(f"Jiwer 计算错误 {audio_filename}: {e}")

    # ================= 结果汇总 =================
    print("\n" + "=" * 50)
    print("评测完成")
    print(f"处理文件总数: {file_count}")

    if total_ref_words > 0:
        avg_wer = total_distance / total_ref_words
        print(f"总编辑距离: {total_distance}")
        print(f"总参考单词数: {total_ref_words}")
        print("-" * 25)
        print(f"平均错词率 (Average WER): {avg_wer:.2%}")
    else:
        print("未处理任何有效文件。")
    print("=" * 50)


if __name__ == "__main__":
    main()