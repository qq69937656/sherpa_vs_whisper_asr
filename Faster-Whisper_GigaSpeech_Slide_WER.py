import os
import sys
import re
import wave
import numpy as np
import jiwer
from faster_whisper import WhisperModel

# ================= 配置区域 =================

# GigaSpeech 数据集根目录
GIGASPEECH_DIR = "/opt/Audio_Datasets/GigaSpeech_Test_WAV"
# 存放 wav 文件的子目录名
WAV_SUBDIR = "wav"
# 标注文件名
TEXT_FILENAME = "text"

MODEL_SIZE = "medium.en"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# 最小单词数限制 (含)，少于此数量的样本将被忽略
MIN_WORD_COUNT = 3


# ================= 文本处理函数 =================

def clean_gigaspeech_text(text):
    """
    GigaSpeech 专用清洗函数：
    1. 移除所有 <> 包含的标签 (如 <COMMA>, <SIL>, <OTHER>)
    2. 移除多余空格
    """
    if not text:
        return ""

    # 移除 <...> 标签
    text = re.sub(r"<[^>]+>", " ", text)

    # 移除连续空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text):
    """
    通用标准化：转小写，去标点，按空格分词
    """
    if not text:
        return []

    # 1. 转小写
    text = text.lower()

    # 2. 去除标点 (保留字母、数字、空格、撇号)
    # GigaSpeech 的文本清洗后可能还剩纯标点，这里统一再洗一次
    text = re.sub(r"[^a-z0-9\s']", "", text)

    # 3. 分词
    words = text.strip().split()
    return words


# ================= 核心识别函数 (保持不变) =================

def transcribe_wav_file(model, file_path, verbose=False):
    """
    使用智能截断/扣留策略进行识别
    """
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
            return "Error: Mono 16k required"
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

    text_file_path = os.path.join(GIGASPEECH_DIR, TEXT_FILENAME)
    wav_dir_path = os.path.join(GIGASPEECH_DIR, WAV_SUBDIR)

    if not os.path.exists(text_file_path):
        print(f"错误: 标注文件不存在 {text_file_path}")
        return
    if not os.path.exists(wav_dir_path):
        print(f"错误: 音频目录不存在 {wav_dir_path}")
        return

    print(f"Processing GigaSpeech Text: {text_file_path}")

    try:
        with open(text_file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # GigaSpeech text 格式通常为: AUDIO_ID TEXT_CONTENT
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue

                file_id, raw_ref_text = parts

                # ================= 1. 文本预处理与过滤 =================

                # 移除 <COMMA>, <SIL> 等标签
                cleaned_ref_text = clean_gigaspeech_text(raw_ref_text)

                # 标准化 (分词)
                ref_words_list = normalize_text(cleaned_ref_text)

                # 过滤规则: 单词数 < MIN_WORD_COUNT 则跳过
                if len(ref_words_list) < MIN_WORD_COUNT:
                    # print(f"Skip (Too Short): {file_id} | Words: {len(ref_words_list)}")
                    skipped_count += 1
                    continue

                # ================= 2. 查找音频 =================

                audio_path = os.path.join(wav_dir_path, f"{file_id}.wav")
                if not os.path.exists(audio_path):
                    # print(f"Warning: Audio not found {audio_path}")
                    continue

                # ================= 3. 执行识别 =================
                try:
                    hyp_text_raw = transcribe_wav_file(model, audio_path)

                    if hyp_text_raw.startswith("Error"):
                        print(f"Skip {file_id}: {hyp_text_raw}")
                        continue

                    hyp_words_list = normalize_text(hyp_text_raw)

                    # ================= 4. 计算 WER =================

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

    # ================= 结果汇总 =================
    print("\n" + "=" * 50)
    print("评测完成")
    print(f"有效处理文件数: {file_count}")
    print(f"因文本过短/纯标签跳过: {skipped_count}")

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