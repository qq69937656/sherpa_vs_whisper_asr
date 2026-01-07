import os
import sys
import re
import time
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
import jiwer

# ================= 配置区域 =================

LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"
MODEL_SIZE = "medium.en"
CHUNK_DURATION = 1.0  # 1S 分片

# VAD & 过滤参数
SILENCE_THRESHOLD = 0.01
MAX_WORDS_PER_SEC = 6
MIN_WORD_COUNT = 3  # 仅影响是否计算WER，不影响RTF统计


# ================= 辅助函数 =================

def normalize_text(text):
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.strip().split()
    return words


def calculate_rms(audio_chunk):
    if len(audio_chunk) == 0:
        return 0
    return np.sqrt(np.mean(audio_chunk ** 2))


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
    print(f"正在加载模型 {MODEL_SIZE} (RTF 专项测试)...")
    recognizer = create_recognizer(MODEL_SIZE)
    print("模型加载完成。")
    print("-" * 50)

    # 打印表头
    print(f"{'File ID':<25} | {'ProcTime(s)':<12} | {'Audio(s)':<10} | {'RTF':<8}")
    print("-" * 65)

    # 全局统计变量
    total_processing_time = 0.0
    total_audio_duration = 0.0

    # WER 统计 (保留功能，但不作为输出重点)
    total_distance = 0
    total_ref_words = 0

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 目录不存在 {LIBRISPEECH_DIR}")
        return

    # 遍历 LibriSpeech
    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        trans_files = [f for f in files if f.endswith(".trans.txt")]

        for trans_file in trans_files:
            trans_path = os.path.join(root, trans_file)
            ref_map = {}

            # 1. 读取参考文本
            try:
                with open(trans_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            file_id, text = parts
                            ref_map[file_id] = text
            except Exception:
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

                # 2. 读取音频
                try:
                    audio, sample_rate = sf.read(audio_path, dtype="float32")
                except Exception:
                    continue

                if len(audio) < int(sample_rate * 0.05):
                    continue

                # ================= 核心计时循环 =================
                chunk_size = int(sample_rate * CHUNK_DURATION)
                hyp_segments_list = []
                last_text = ""

                # 单文件统计
                file_proc_time = 0.0
                file_audio_len = len(audio) / sample_rate

                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i: i + chunk_size]
                    if len(chunk) < 160: continue

                    # --- [计时开始] ---
                    t_start = time.time()

                    # A. VAD
                    rms = calculate_rms(chunk)

                    if rms >= SILENCE_THRESHOLD:
                        try:
                            # B. 推理
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

                            # C. 后处理
                            current_chunk_text = ""
                            for segment in segments:
                                txt = segment.text.strip()
                                # 黑名单
                                hallucinations = ["see you", "watching", "subtitles", "amara", "org", "bye"]
                                if any(h in txt.lower() for h in hallucinations):
                                    continue
                                current_chunk_text += " " + txt

                            current_chunk_text = current_chunk_text.strip()

                            # 字数熔断 & 防重复
                            if len(current_chunk_text.split()) <= MAX_WORDS_PER_SEC:
                                if current_chunk_text and current_chunk_text != last_text:
                                    hyp_segments_list.append(current_chunk_text)
                                    last_text = current_chunk_text

                        except Exception:
                            pass

                    # --- [计时结束] ---
                    t_end = time.time()
                    file_proc_time += (t_end - t_start)

                # ================= 统计与输出 =================
                total_processing_time += file_proc_time
                total_audio_duration += file_audio_len

                # 计算单文件 RTF
                curr_rtf = file_proc_time / file_audio_len if file_audio_len > 0 else 0

                # 格式化输出: 文件名 | 处理耗时 | 音频时长 | RTF
                print(f"{file_id:<25} | {file_proc_time:<12.4f} | {file_audio_len:<10.2f} | {curr_rtf:<8.4f}")

                # (后台计算 WER 用于验证准确性，不打印细节)
                hyp_text_raw = " ".join(hyp_segments_list)
                ref_words = normalize_text(ref_text)
                if len(ref_words) >= MIN_WORD_COUNT:
                    try:
                        hyp_words = normalize_text(hyp_text_raw)
                        out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
                        total_distance += (out.substitutions + out.deletions + out.insertions)
                        total_ref_words += len(out.references[0])
                    except:
                        pass

    # ================= 最终汇总 =================
    print("-" * 65)
    print("RTF 评测统计结果:")
    print(f"总音频时长: {total_audio_duration:.2f} 秒")
    print(f"总处理耗时: {total_processing_time:.2f} 秒")

    if total_audio_duration > 0:
        avg_rtf = total_processing_time / total_audio_duration
        print(f"★ 平均实时率 (Average RTF): {avg_rtf:.5f}")
        print(f"  (意味着处理 1小时音频需要 {avg_rtf * 60:.2f} 分钟)")

    if total_ref_words > 0:
        print(f"参考 WER: {total_distance / total_ref_words:.2%}")
    print("=" * 65)


if __name__ == "__main__":
    main()