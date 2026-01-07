import os
import sys
import re
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
import jiwer

# ================= 配置区域 =================

# 数据集根目录
AMI_ROOT = "/opt/Audio_Datasets/AMICorpus"

# 限制处理的文件数量
MAX_FILES_TO_PROCESS = 10

MODEL_SIZE = "medium.en"
CHUNK_DURATION = 1.0  # 1S 分片

# 静音阈值
SILENCE_THRESHOLD = 0.01

# 字数熔断：1秒内超过6个词视为幻觉
MAX_WORDS_PER_SEC = 6


# ================= 辅助函数 =================

def normalize_text(text):
    """
    文本清洗：
    1. 去除换行符 (替换为空格)
    2. 转小写
    3. 去除标点
    4. 分词
    """
    if not text:
        return []

    # 将换行符替换为空格，防止单词粘连
    text = text.replace('\n', ' ').replace('\r', ' ')

    # 转小写
    text = text.lower()

    # 去除标点 (只保留单词、数字和空格)
    text = re.sub(r"[^\w\s]", "", text)

    # 去除多余空格并分词
    words = text.strip().split()
    return words


def calculate_rms(audio_chunk):
    """计算 RMS 能量"""
    if len(audio_chunk) == 0:
        return 0
    return np.sqrt(np.mean(audio_chunk ** 2))


def get_ami_file_pairs(root_dir, limit=10):
    """
    遍历目录，寻找 audio 文件夹下的 wav 和 txt 对
    返回 [(wav_path, txt_path), ...]
    """
    pairs = []
    print(f"正在搜索 {root_dir} 下的前 {limit} 个有效文件对...")

    for root, dirs, files in os.walk(root_dir):
        # 根据描述，音频位于 'audio' 目录下
        if os.path.basename(root) != 'audio':
            continue

        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                # 假设 txt 文件名与 wav 文件名一致（除了后缀）
                txt_name = os.path.splitext(file)[0] + ".txt"
                txt_path = os.path.join(root, txt_name)

                if os.path.exists(txt_path):
                    pairs.append((wav_path, txt_path))

                    # 达到限制数量，立即停止搜索
                    if len(pairs) >= limit:
                        return pairs

    return pairs


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
    print(f"正在加载模型 {MODEL_SIZE} (AMI Corpus 适配版)...")
    recognizer = create_recognizer(MODEL_SIZE)
    print("模型加载完成。")
    print("-" * 50)

    if not os.path.isdir(AMI_ROOT):
        print(f"错误: 数据集目录不存在: {AMI_ROOT}")
        return

    # 1. 获取文件列表 (限制 10 个)
    file_pairs = get_ami_file_pairs(AMI_ROOT, limit=MAX_FILES_TO_PROCESS)

    if not file_pairs:
        print("未找到任何成对的 wav/txt 文件。请检查目录结构。")
        return

    print(f"已找到 {len(file_pairs)} 个待测文件。")
    print("-" * 50)

    # 2. 遍历评测
    total_distance = 0
    total_ref_words = 0
    processed_count = 0

    for wav_path, txt_path in file_pairs:
        # --- A. 读取并处理文本 ---
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            ref_words_list = normalize_text(raw_text)

            if len(ref_words_list) == 0:
                print(f"警告: 文本为空，跳过 {os.path.basename(wav_path)}")
                continue

        except Exception as e:
            print(f"读取文本失败 {txt_path}: {e}")
            continue

        # --- B. 读取音频 ---
        try:
            audio, sample_rate = sf.read(wav_path, dtype="float32")
        except Exception as e:
            print(f"读取音频失败 {wav_path}: {e}")
            continue

        # 计算音频时长 (秒)
        duration_sec = len(audio) / sample_rate

        # 忽略极短音频
        if len(audio) < int(sample_rate * 0.05):
            continue

        # ================= 核心推理逻辑 (保持一致) =================
        chunk_size = int(sample_rate * CHUNK_DURATION)
        hyp_segments_list = []
        last_text = ""

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i: i + chunk_size]
            if len(chunk) < 160: continue

            # 1. 能量 VAD
            rms = calculate_rms(chunk)
            if rms < SILENCE_THRESHOLD:
                continue

            # 2. 推理
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
                    # 3. 黑名单过滤
                    hallucinations = ["see you", "watching", "subtitles", "amara", "org", "bye"]
                    if any(h in txt.lower() for h in hallucinations):
                        continue
                    current_chunk_text += " " + txt

                current_chunk_text = current_chunk_text.strip()

                # 4. 字数熔断
                words_in_chunk = len(current_chunk_text.split())
                if words_in_chunk > MAX_WORDS_PER_SEC:
                    continue

                # 5. 防重复
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

            # 按照要求输出：WER、编辑距离、官方文本单词个数、音频时长
            print(f"File: {os.path.basename(wav_path)} | "
                  f"WER: {curr_wer:.4f} | "
                  f"Dist: {curr_dist} | "
                  f"RefWords: {curr_len} | "
                  f"Duration: {duration_sec:.2f}s")

        except Exception as e:
            print(f"Jiwer 计算错误 {os.path.basename(wav_path)}: {e}")

    # ================= 结果汇总 =================
    print("\n" + "=" * 50)
    print("评测完成 (AMI Corpus Top 10)")
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