import os
import sys
import re
import wave
import numpy as np
import jiwer
from faster_whisper import WhisperModel

# ================= 配置区域 =================

TEDLIUM_DIR = "/opt/Audio_Datasets/TEDLIUM_WAV"
TEXT_FILENAME = "text.txt"

# 建议保持 large-v3 以获得最佳识别率，但您可以改为 small.en 以复现您本地的结果
MODEL_SIZE = "medium.en"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

MIN_WORD_COUNT = 3


# ================= 文本处理函数 (关键修复) =================

def clean_tedlium_text(text):
    """
    TEDLIUM 专用清洗：
    1. 移除 <...> 标签
    2. 【关键】修复分词缩写，将 "you 're" 修复为 "you're"，以便与 Whisper 输出对齐
    """
    if not text:
        return ""

    # 移除 <SIL>, <COMA> 等
    text = re.sub(r"<[^>]+>", " ", text)

    # 【关键步骤】修复 TEDLIUM 的缩写空格问题
    # 例如: "don 't" -> "don't", "we 're" -> "we're", "s 's" -> "s's"
    # 逻辑：将 " '字母" 替换为 "'字母"
    text = re.sub(r"\s+'([a-z])", r"'\1", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text):
    """
    通用标准化：转小写，去标点，按空格分词
    """
    if not text:
        return []

    text = text.lower()
    # 保留撇号，因为我们在 clean_tedlium_text 里已经修正了缩写格式
    # 这样 "don't" (Ref) 和 "don't" (Hyp) 就能匹配上
    text = re.sub(r"[^a-z0-9\s']", "", text)
    words = text.strip().split()
    return words


# ================= 核心识别函数 (完全复用您的代码) =================

def transcribe_wav_file(model, file_path, verbose=False):
    """
    完全复用您提供的基于 wave 的代码逻辑
    """
    # 您的配置
    CHUNK_DURATION = 1.0
    SAMPLE_RATE = 16000
    FORCE_FLUSH_DURATION = 10.0
    HALLUCINATION_PHRASES = ["subtitle", "amara", "audio", "copyright", "subscribe"]

    def is_hallucination(segment):
        text = segment.text.strip().lower()
        duration = segment.end - segment.start
        if not text or duration <= 0: return True, "Empty"
        if (len(text) / duration) > 40: return True, "Too Fast"
        if segment.words and len(segment.words) > 1:
            if (duration / len(segment.words)) < 0.05: return True, "Word Density"
        if segment.no_speech_prob > 0.9: return True, "Silence Prob"
        if segment.avg_logprob < -1.5: return True, "Low Logprob"
        for bad in HALLUCINATION_PHRASES:
            if bad in text: return True, "Blacklist"
        return False, "OK"

    audio_buffer = np.array([], dtype=np.float32)
    final_transcript = []

    try:
        wf = wave.open(file_path, 'rb')
        if wf.getnchannels() != 1 or wf.getframerate() != 16000:
            return f"Error: Mono 16k required, got {wf.getnchannels()}ch {wf.getframerate()}Hz"
    except Exception as e:
        return f"Error: {e}"

    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    while True:
        data = wf.readframes(chunk_samples)
        is_eof = len(data) == 0

        if not is_eof:
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            new_chunk = audio_int16.astype(np.float32) / 32768.0
            audio_buffer = np.concatenate((audio_buffer, new_chunk))

        if is_eof and len(audio_buffer) == 0: break

        buffer_duration = len(audio_buffer) / SAMPLE_RATE
        if is_eof and buffer_duration < 0.2: break
        if buffer_duration < 0.5 and not is_eof: continue

        segments, _ = model.transcribe(
            audio_buffer,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            word_timestamps=True,
            no_speech_threshold=0.5
        )

        all_valid_words = []
        for seg in segments:
            is_bad, reason = is_hallucination(seg)
            if is_bad:
                if verbose: print(f"  [过滤] '{seg.text.strip()}' ({reason})")
                continue
            if not seg.words:
                class PseudoWord:
                    def __init__(self, w, start, end):
                        self.word = w
                        self.start = start
                        self.end = end

                all_valid_words.append(PseudoWord(seg.text, seg.start, seg.end))
            else:
                all_valid_words.extend(seg.words)

        word_count = len(all_valid_words)
        should_hold_back = (not is_eof) and (buffer_duration < FORCE_FLUSH_DURATION)

        text_to_commit = ""
        cut_time = 0.0

        if word_count > 0:
            if should_hold_back:
                num_to_hold = 1
                last_word_text = all_valid_words[-1].word.strip()
                if last_word_text in ['-', '—', '--']:
                    num_to_hold = 2

                if word_count > num_to_hold:
                    cut_index = -num_to_hold
                    cut_time = all_valid_words[cut_index - 1].end
                    committed_words = all_valid_words[:cut_index]
                    text_to_commit = "".join([w.word for w in committed_words])
                else:
                    cut_time = 0.0
                    text_to_commit = ""
            else:
                cut_time = all_valid_words[-1].end
                text_to_commit = "".join([w.word for w in all_valid_words])

        if text_to_commit:
            final_transcript.append(text_to_commit)

        if cut_time > 0:
            cut_sample_index = int(cut_time * SAMPLE_RATE)
            if cut_sample_index < len(audio_buffer):
                audio_buffer = audio_buffer[cut_sample_index:]
            else:
                audio_buffer = np.array([], dtype=np.float32)

        if is_eof: break

    wf.close()
    return "".join(final_transcript).strip()


# ================= 主程序 =================

def main():
    print("-" * 50)
    print(f"Loading Model: {MODEL_SIZE} on {DEVICE}...")
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    print("Model Loaded.")
    print("-" * 50)

    total_distance = 0
    total_ref_words = 0
    file_count = 0
    skipped_count = 0

    text_file_path = os.path.join(TEDLIUM_DIR, TEXT_FILENAME)

    if not os.path.exists(text_file_path):
        print(f"错误: 标注文件不存在 {text_file_path}")
        return

    # ================= 索引音频文件 =================
    print(f"Indexing audio files in {TEDLIUM_DIR} ...")
    audio_map = {}
    for root, dirs, files in os.walk(TEDLIUM_DIR):
        for f in files:
            if f.endswith(".wav"):
                # TEDLIUM 文件名ID通常不含扩展名
                file_id = os.path.splitext(f)[0]
                audio_map[file_id] = os.path.join(root, f)
    print(f"Indexed {len(audio_map)} audio files.")
    print("-" * 50)

    # ================= 处理流程 =================
    try:
        with open(text_file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line: continue

                parts = line.split(" ", 1)
                if len(parts) != 2: continue

                file_id, raw_ref_text = parts

                # 1. 清洗与预处理 (包含缩写修复)
                cleaned_ref_text = clean_tedlium_text(raw_ref_text)
                ref_words_list = normalize_text(cleaned_ref_text)

                if len(ref_words_list) < MIN_WORD_COUNT:
                    skipped_count += 1
                    continue

                if file_id not in audio_map:
                    continue
                audio_path = audio_map[file_id]

                try:
                    # 2. 识别
                    hyp_text_raw = transcribe_wav_file(model, audio_path)

                    if hyp_text_raw.startswith("Error"):
                        print(f"Skip {file_id}: {hyp_text_raw}")
                        continue

                    # 3. 计算 WER
                    hyp_words_list = normalize_text(hyp_text_raw)

                    ref_str_clean = " ".join(ref_words_list)
                    hyp_str_clean = " ".join(hyp_words_list)

                    out = jiwer.process_words(ref_str_clean, hyp_str_clean)

                    dist = out.substitutions + out.deletions + out.insertions
                    curr_len = len(out.references[0]) if out.references else 0

                    total_distance += dist
                    total_ref_words += curr_len
                    file_count += 1

                    print(f"File: {file_id} | WER: {out.wer:.4f} | Dist: {dist} | RefWords: {curr_len}")

                except Exception as e:
                    print(f"Error {file_id}: {e}")
                    continue

    except Exception as e:
        print(f"读取 text 文件失败: {e}")
        return

    # ================= 结果 =================
    print("\n" + "=" * 50)
    print("评测完成")
    print(f"有效处理文件数: {file_count}")
    print(f"跳过文件数: {skipped_count}")

    if total_ref_words > 0:
        avg_wer = total_distance / total_ref_words
        print(f"总编辑距离: {total_distance}")
        print(f"总参考单词数: {total_ref_words}")
        print("-" * 25)
        print(f"平均错词率 (Average WER): {avg_wer:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()