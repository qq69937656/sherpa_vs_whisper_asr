import os
import sys
import re
import jiwer
from faster_whisper import WhisperModel

# ================= 配置区域 =================

LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"
MODEL_SIZE = "medium.en"


# ================= 辅助函数 =================

def normalize_text(text):
    """
    文本归一化：转小写，去除标点符号，仅保留单词和空格
    """
    if not text:
        return []
    text = text.lower()
    # 去除标点符号
    text = re.sub(r"[^\w\s]", "", text)
    # 按空格分割
    words = text.strip().split()
    return words


# ================= Faster-Whisper 初始化 =================

def create_recognizer(model_size):
    try:
        # 使用 float16 + cuda 进行加速
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)


# ================= 主程序 =================

def main():
    print("-" * 50)
    print(f"正在加载模型 {MODEL_SIZE} (全量音频推理模式)...")
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
        # 寻找转写文件
        trans_files = [f for f in files if f.endswith(".trans.txt")]

        for trans_file in trans_files:
            trans_path = os.path.join(root, trans_file)
            ref_map = {}

            # 读取 trans.txt 构建参考文本映射
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

            # 遍历该目录下的所有音频对应关系
            for file_id, ref_text in ref_map.items():
                # 尝试寻找 .wav 或 .flac
                audio_filename = f"{file_id}.wav"
                audio_path = os.path.join(root, audio_filename)

                if not os.path.exists(audio_path):
                    flac_path = os.path.join(root, f"{file_id}.flac")
                    if os.path.exists(flac_path):
                        audio_path = flac_path
                    else:
                        # 找不到音频文件
                        continue

                # ================= 核心推理逻辑 (全量) =================
                try:
                    # 直接传入路径，Faster-Whisper 内部会处理解码
                    segments, info = recognizer.transcribe(
                        audio_path,
                        beam_size=5,  # 全量推理推荐使用 beam search 提高准确率
                        language="en",
                        vad_filter=True,  # 依然开启 VAD 过滤非人声片段
                        vad_parameters=dict(min_silence_duration_ms=500),
                        condition_on_previous_text=True  # 允许利用前文上下文，这对长句子识别很有帮助
                    )

                    # 拼接所有识别出的片段
                    hyp_segments_list = [segment.text for segment in segments]
                    hyp_text_raw = " ".join(hyp_segments_list)

                except Exception as e:
                    print(f"推理失败 {audio_path}: {e}")
                    continue
                # ====================================================

                # ================= 评测逻辑 =================
                # 使用相同的归一化规则
                ref_words_list = normalize_text(ref_text)
                hyp_words_list = normalize_text(hyp_text_raw)

                ref_str_clean = " ".join(ref_words_list)
                hyp_str_clean = " ".join(hyp_words_list)

                try:
                    # 计算 WER
                    out = jiwer.process_words(ref_str_clean, hyp_str_clean)

                    curr_dist = out.substitutions + out.deletions + out.insertions
                    curr_wer = out.wer
                    curr_len = len(out.references[0]) if out.references else 0

                    total_distance += curr_dist
                    total_ref_words += curr_len
                    file_count += 1

                    print(
                        f"File: {os.path.basename(audio_path)} | WER: {curr_wer:.4f} | Dist: {curr_dist} | RefWords: {curr_len}"
                    )

                except Exception as e:
                    print(f"Jiwer 计算错误 {file_id}: {e}")

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