import os
import sys
import re
import wave
import numpy as np
import jiwer
from faster_whisper import WhisperModel

# ================= 配置区域 =================

AMICORPUS_DIR = "/opt/Audio_Datasets/AMICorpus"
MODEL_SIZE = "medium.en"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# 仅处理前 N 个文件
MAX_FILES_TO_PROCESS = 10


# ================= 辅助函数 =================

def get_audio_duration(file_path):
    """获取 wav 音频时长 (秒)"""
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0


def normalize_text(text):
    """
    标准化：
    1. 替换回车为对应空格
    2. 转小写
    3. 去除标点 (保留字母、数字、空格、撇号)
    """
    if not text:
        return []

    # 替换回车/换行 为空格，防止单词粘连
    text = text.replace('\n', ' ').replace('\r', ' ')

    text = text.lower()
    # 保留撇号以匹配 don't 等，其他标点去除
    text = re.sub(r"[^a-z0-9\s']", "", text)

    # 分词
    words = text.strip().split()
    return words


# ================= 核心识别函数 (基于智能截断逻辑) =================

def transcribe_wav_file(model, file_path, verbose=False):
    """
    复用经过验证的基于 wave 的流式模拟识别逻辑
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
        # AMICorpus 的音频可能是 16k 单声道，这里做个检查，兼容性处理
        if wf.getnchannels() != 1 or wf.getframerate() != 16000:
            return f"Error: Format mismatch ({wf.getnchannels()}ch {wf.getframerate()}Hz). Expect 1ch 16000Hz"
    except Exception as e:
        return f"Error opening wav: {e}"

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
            if is_bad: continue  # 过滤幻觉
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

    # 1. 扫描文件对 (WAV + TXT)
    # 只需要前10个，所以我们先遍历收集，满了就停
    print(f"Scanning {AMICORPUS_DIR} for first {MAX_FILES_TO_PROCESS} file pairs...")

    file_pairs = []  # List of (wav_path, txt_path)

    for root, dirs, files in os.walk(AMICORPUS_DIR):
        # 优化：题目说 "每个子目录下有audio目录"，我们可以只在 audio 结尾的目录里找
        # 或者直接全遍历
        for f in files:
            if f.endswith(".wav"):
                wav_path = os.path.join(root, f)

                # 假设 txt 文件名与 wav 文件名一致，且在同一目录 (根据题目描述)
                # 例如: ES2002a.wav -> ES2002a.txt
                txt_filename = os.path.splitext(f)[0] + ".txt"
                txt_path = os.path.join(root, txt_filename)

                if os.path.exists(txt_path):
                    file_pairs.append((wav_path, txt_path))

                    if len(file_pairs) >= MAX_FILES_TO_PROCESS:
                        break
        if len(file_pairs) >= MAX_FILES_TO_PROCESS:
            break

    if not file_pairs:
        print("未找到任何成对的 wav/txt 文件。")
        return

    print(f"Found {len(file_pairs)} pairs. Starting transcription...")
    print("=" * 100)
    print(f"{'Filename':<30} | {'Duration':<8} | {'RefWords':<8} | {'Dist':<6} | {'WER':<8}")
    print("-" * 100)

    total_distance = 0
    total_ref_words = 0
    processed_count = 0

    for wav_path, txt_path in file_pairs:
        file_id = os.path.basename(wav_path)

        # 1. 获取时长
        duration = get_audio_duration(wav_path)

        # 2. 读取官方文本 (去掉回车)
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                raw_ref_text = f.read().replace('\n', ' ').replace('\r', ' ')
        except Exception as e:
            print(f"{file_id:<30} | Error reading txt: {e}")
            continue

        # 3. 识别
        try:
            hyp_text_raw = transcribe_wav_file(model, wav_path)
            if hyp_text_raw.startswith("Error"):
                print(f"{file_id:<30} | {hyp_text_raw}")
                continue
        except Exception as e:
            print(f"{file_id:<30} | Transcribe Error: {e}")
            continue

        # 4. 归一化与评测
        ref_words = normalize_text(raw_ref_text)
        hyp_words = normalize_text(hyp_text_raw)

        if len(ref_words) == 0:
            print(f"{file_id:<30} | Skipped (Empty Ref)")
            continue

        ref_str = " ".join(ref_words)
        hyp_str = " ".join(hyp_words)

        try:
            out = jiwer.process_words(ref_str, hyp_str)
            dist = out.substitutions + out.deletions + out.insertions

            total_distance += dist
            total_ref_words += len(ref_words)
            processed_count += 1

            # 输出单行结果
            print(f"{file_id:<30} | {duration:<8.2f} | {len(ref_words):<8} | {dist:<6} | {out.wer:<8.4f}")

        except Exception as e:
            print(f"{file_id:<30} | WER Calc Error: {e}")

    # ================= 结果汇总 =================
    print("=" * 100)
    print(f"Processed Files: {processed_count}")

    if total_ref_words > 0:
        avg_wer = total_distance / total_ref_words
        print(f"Total Distance:  {total_distance}")
        print(f"Total Ref Words: {total_ref_words}")
        print("-" * 30)
        print(f"Average WER:     {avg_wer:.2%}")
    else:
        print("No valid data for WER calculation.")
    print("=" * 100)


if __name__ == "__main__":
    main()
