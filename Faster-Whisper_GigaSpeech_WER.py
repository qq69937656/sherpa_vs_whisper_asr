import os
import sys
import re
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
import jiwer

# ================= 配置区域 =================

# 数据集根目录
GIGASPEECH_ROOT = "/opt/Audio_Datasets/TEDLIUM_WAV"
# 文本文件路径 (这是一个文件，不是目录)
TRANS_FILE = os.path.join(GIGASPEECH_ROOT, "text.txt")
# 音频存放目录
WAV_DIR = os.path.join(GIGASPEECH_ROOT, "wav")

MODEL_SIZE = "medium.en"
CHUNK_DURATION = 1.0  # 1S 分片

# 静音阈值：RMS 能量低于此值视为静音，不送入模型
SILENCE_THRESHOLD = 0.01

# 最大词数限制：1秒内人类很难说超过 6 个单词
MAX_WORDS_PER_SEC = 6

# 过滤条件：仅评测有效单词数 >= 3 的样本
MIN_WORD_COUNT = 3


# ================= 辅助函数 =================

def normalize_text(text):
    """
    基础文本标准化：
    1. 转小写
    2. 去除标点 (保留字母、数字、空格)
    3. 分词
    """
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # 去除标点
    words = text.strip().split()
    return words


def clean_gigaspeech_text(raw_text):
    """
    GigaSpeech 专用清洗：
    1. 去除 <COMMA>, <SIL>, <OTHER> 等所有尖括号标签
    2. 标准化处理
    3. 返回清洗后的单词列表
    """
    if not raw_text:
        return []

    # 1. 使用正则去除 <...> 内容
    # 替换为空格防止粘连，例如 "hello<SIL>world" -> "hello world"
    text_no_tags = re.sub(r'<[^>]+>', ' ', raw_text)

    # 2. 调用标准归一化进行分词
    words = normalize_text(text_no_tags)

    return words


def calculate_rms(audio_chunk):
    """计算音频切片的均方根能量"""
    if len(audio_chunk) == 0:
        return 0
    return np.sqrt(np.mean(audio_chunk ** 2))


# ================= Faster-Whisper 初始化 =================

def create_recognizer(model_size):
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)


# ================= 主程序 =================

def main():
    print("-" * 50)
    print(f"正在加载模型 {MODEL_SIZE} (包含能量VAD + GigaSpeech过滤规则)...")
    recognizer = create_recognizer(MODEL_SIZE)
    print("模型加载完成。")
    print("-" * 50)

    # 检查路径
    if not os.path.isfile(TRANS_FILE):
        print(f"错误: 找不到文本文件: {TRANS_FILE}")
        return
    if not os.path.isdir(WAV_DIR):
        print(f"错误: 找不到wav目录: {WAV_DIR}")
        return

    # 1. 预读取并过滤文本
    print(f"正在读取并过滤文本文件: {TRANS_FILE}")
    valid_tasks = {}  # 格式: {file_id: ref_word_list}
    skipped_count = 0

    try:
        with open(TRANS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 拆分 ID 和 文本 (通常以第一个空格分隔)
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    # 只有ID没有文本，视为无效
                    skipped_count += 1
                    continue

                file_id = parts[0]
                raw_content = parts[1]

                # 清洗文本
                ref_words = clean_gigaspeech_text(raw_content)

                # 过滤规则：单词数 >= 3
                if len(ref_words) >= MIN_WORD_COUNT:
                    valid_tasks[file_id] = ref_words
                else:
                    skipped_count += 1

    except Exception as e:
        print(f"读取文本文件时出错: {e}")
        return

    print(f"文本预处理完成。")
    print(f"符合评测条件的样本数: {len(valid_tasks)}")
    print(f"跳过样本数 (过短或仅含标签): {skipped_count}")
    print("-" * 50)

    # 2. 开始遍历评测
    total_distance = 0
    total_ref_words = 0
    processed_count = 0

    for file_id, ref_words_list in valid_tasks.items():
        audio_path = os.path.join(WAV_DIR, f"{file_id}.wav")

        # 检查音频是否存在
        if not os.path.exists(audio_path):
            # 也可以尝试兼容 flac 或 opus，如果数据集只有 wav 则不需要
            print(f"警告: 音频文件缺失 {audio_path}")
            continue

        try:
            # 读取音频
            audio, sample_rate = sf.read(audio_path, dtype="float32")
        except Exception as e:
            print(f"读取音频失败 {audio_path}: {e}")
            continue

        # 忽略极短音频
        if len(audio) < int(sample_rate * 0.05):
            continue

        # ================= 核心推理逻辑 (切片 + VAD + 熔断) =================
        chunk_size = int(sample_rate * CHUNK_DURATION)
        hyp_segments_list = []
        last_text = ""

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i: i + chunk_size]
            if len(chunk) < 160: continue

            # --- 1. 能量 VAD ---
            rms = calculate_rms(chunk)
            if rms < SILENCE_THRESHOLD:
                continue

            # --- 2. 推理 ---
            try:
                segments, info = recognizer.transcribe(
                    chunk,
                    beam_size=1,
                    language="en",
                    vad_filter=True,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    compression_ratio_threshold=2.0,
                    no_speech_threshold=0.4
                )

                current_chunk_text = ""
                for segment in segments:
                    txt = segment.text.strip()
                    # --- 3. 黑名单过滤 ---
                    hallucinations = ["see you", "watching", "subtitles", "amara", "org", "bye"]
                    if any(h in txt.lower() for h in hallucinations):
                        continue
                    current_chunk_text += " " + txt

                current_chunk_text = current_chunk_text.strip()

                # --- 4. 字数熔断 ---
                words_in_chunk = len(current_chunk_text.split())
                if words_in_chunk > MAX_WORDS_PER_SEC:
                    continue

                # --- 5. 防重复 ---
                if current_chunk_text and current_chunk_text != last_text:
                    hyp_segments_list.append(current_chunk_text)
                    last_text = current_chunk_text

            except Exception as e:
                continue

        hyp_text_raw = " ".join(hyp_segments_list)
        # ===============================================

        # ================= 评测计算 =================
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
            processed_count += 1

            print(f"File: {file_id} | WER: {curr_wer:.4f} | Dist: {curr_dist} | RefWords: {curr_len}")

        except Exception as e:
            print(f"Jiwer 计算错误 {file_id}: {e}")

    # ================= 结果汇总 =================
    print("\n" + "=" * 50)
    print("评测完成 (GigaSpeech Test)")
    print(f"实际处理文件数: {processed_count}")

    if total_ref_words > 0:
        avg_wer = total_distance / total_ref_words
        print(f"总编辑距离: {total_distance}")
        print(f"总参考单词数: {total_ref_words}")
        print("-" * 25)
        print(f"平均错词率 (Average WER): {avg_wer:.2%}")
    else:
        print("未生成任何有效统计数据。")
    print("=" * 50)


if __name__ == "__main__":
    main()