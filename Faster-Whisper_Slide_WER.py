import os
import sys
import re
import argparse
import soundfile as sf
import numpy as np
import jiwer
from faster_whisper import WhisperModel

# ============================================================
# 常量与配置
# ============================================================
MODEL_SIZE = "medium.en"
CHUNK_DURATION = 1.0  # 每次累加的音频分片长度
FORCE_FLUSH_DURATION = 10.0  # 强制刷新阈值（防止缓冲区无限增长）
MIN_WORD_COUNT = 3  # 过滤过短的参考文本

# 常见 Whisper 幻觉关键词（用于黑名单过滤）
HALLUCINATION_PHRASES = ["subtitle", "amara", "audio", "copyright", "subscribe", "see you", "watching", "org", "bye"]


def normalize_text(text, dataset_name):
    if not text: return ""
    text = text.replace("\n", " ").replace("\r", " ").lower()
    if dataset_name in ["TEDLIUM", "AMICorpus"]:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[{\[][^}\]]*[}\]]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.strip().split())


def is_hallucination(segment):
    text = segment.text.strip().lower()
    duration = segment.end - segment.start
    if not text or duration <= 0: return True
    if (len(text) / duration) > 40: return True
    if segment.words and len(segment.words) > 1 and (duration / len(segment.words)) < 0.05: return True
    if segment.no_speech_prob > 0.9 or segment.avg_logprob < -1.5: return True
    if any(bad in text for bad in HALLUCINATION_PHRASES): return True
    return False


def load_dataset_tasks(dataset_name, dataset_path):
    task_list = []
    print(f"正在扫描 {dataset_name} 数据集目录: {dataset_path} ...")
    if dataset_name == "LibriSpeech":
        for root, dirs, files in os.walk(dataset_path):
            for f in [x for x in files if x.endswith(".trans.txt")]:
                with open(os.path.join(root, f), "r", encoding="utf-8") as file:
                    for line in file:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            audio_path = os.path.join(root, f"{parts[0]}.wav")
                            if not os.path.exists(audio_path): audio_path = os.path.join(root, f"{parts[0]}.flac")
                            if os.path.exists(audio_path): task_list.append(
                                {"file_id": parts[0], "audio_path": audio_path,
                                 "ref_text": normalize_text(parts[1], dataset_name)})
    elif dataset_name == "AMICorpus":
        for root, dirs, files in os.walk(dataset_path):
            for wav_file in [x for x in files if x.endswith(".wav")]:
                file_id = os.path.splitext(wav_file)[0]
                txt_path = os.path.join(root, f"{file_id}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as file:
                        task_list.append({"file_id": file_id, "audio_path": os.path.join(root, wav_file),
                                          "ref_text": normalize_text(file.read(), dataset_name)})
    elif dataset_name == "TEDLIUM":
        audio_map = {os.path.splitext(f)[0]: os.path.join(root, f) for root, dirs, files in os.walk(dataset_path) for f
                     in files if f.endswith((".wav", ".flac"))}
        txt_path = os.path.join(dataset_path, "text.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2 and parts[0] in audio_map:
                        task_list.append({"file_id": parts[0], "audio_path": audio_map[parts[0]],
                                          "ref_text": normalize_text(parts[1], dataset_name)})

    valid_tasks = [t for t in task_list if len(t["ref_text"].split()) >= MIN_WORD_COUNT]
    print(f"扫描完毕！找到有效评测任务 {len(valid_tasks)} 个。")
    return valid_tasks


def transcribe_sliding_window(model, audio, sample_rate):
    """滑动窗口核心推理逻辑"""
    audio_buffer = np.array([], dtype=np.float32)
    final_transcript = []
    chunk_samples = int(sample_rate * CHUNK_DURATION)

    class PseudoWord:
        def __init__(self, w, start, end): self.word, self.start, self.end = w, start, end

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i: i + chunk_samples]
        audio_buffer = np.concatenate((audio_buffer, chunk))
        is_eof = (i + chunk_samples >= len(audio))
        buffer_duration = len(audio_buffer) / sample_rate

        if is_eof and buffer_duration < 0.2: break
        if buffer_duration < 0.5 and not is_eof: continue

        try:
            segments, _ = model.transcribe(
                audio_buffer, beam_size=1, language="en", condition_on_previous_text=False,
                vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300),
                word_timestamps=True, no_speech_threshold=0.5
            )

            all_valid_words = []
            for seg in segments:
                if is_hallucination(seg): continue
                if not seg.words:
                    all_valid_words.append(PseudoWord(seg.text, seg.start, seg.end))
                else:
                    all_valid_words.extend(seg.words)

            word_count = len(all_valid_words)
            should_hold_back = (not is_eof) and (buffer_duration < FORCE_FLUSH_DURATION)
            text_to_commit, cut_time = "", 0.0

            if word_count > 0:
                if should_hold_back:
                    num_to_hold = 2 if all_valid_words[-1].word.strip() in ['-', '—', '--'] else 1
                    if word_count > num_to_hold:
                        cut_index = -num_to_hold
                        cut_time = all_valid_words[cut_index - 1].end
                        committed_words = all_valid_words[:cut_index]
                        text_to_commit = "".join([w.word for w in committed_words])
                else:
                    cut_time = all_valid_words[-1].end
                    text_to_commit = "".join([w.word for w in all_valid_words])

            if text_to_commit:
                final_transcript.append(text_to_commit)

            if cut_time > 0:
                cut_sample_index = int(cut_time * sample_rate)
                audio_buffer = audio_buffer[cut_sample_index:] if cut_sample_index < len(audio_buffer) else np.array([],
                                                                                                                     dtype=np.float32)

        except Exception:
            break

    return "".join(final_transcript).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()

    print("-" * 50)
    print(f"初始化 Faster-Whisper [滑动窗口优化] 评测...")
    try:
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    except Exception as e:
        sys.exit(1)
    print("-" * 50)

    task_list = load_dataset_tasks(args.dataset_name, args.dataset_path)
    if not task_list: return

    total_distance, total_ref_words, processed_count = 0, 0, 0

    print("=" * 60)
    print(f"🎯 开始执行 {args.dataset_name} 滑动窗口 WER 评测")
    print("=" * 60)

    for task in task_list:
        try:
            audio, sr = sf.read(task["audio_path"], dtype="float32")
            if sr != 16000: continue

            hyp_text_raw = transcribe_sliding_window(model, audio, sr)
            hyp_text_clean = normalize_text(hyp_text_raw, args.dataset_name)

            out = jiwer.process_words(task["ref_text"], hyp_text_clean)
            curr_dist = out.substitutions + out.deletions + out.insertions
            curr_len = len(out.references[0]) if out.references else 0

            total_distance += curr_dist
            total_ref_words += curr_len
            processed_count += 1

            print(
                f"[{processed_count}/{len(task_list)}] ID: {task['file_id'][:25]:<25} | Words: {curr_len:<4} | Dist: {curr_dist:<3} | WER: {out.wer:.4f}")
        except Exception:
            pass

    print("\n" + "=" * 50)
    print(f"✅ {args.dataset_name} 数据集滑动窗口评测完成")
    if total_ref_words > 0: print(f"🚀 平均词错误率 (Average WER): {total_distance / total_ref_words:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()