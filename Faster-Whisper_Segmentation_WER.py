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
CHUNK_DURATION = 1.0  # 基础分片：固定 1.0 秒无重叠
MIN_WORD_COUNT = 3  # 过滤过短的参考文本
SILENCE_THRESHOLD = 0.01  # 能量型 VAD 阈值
MAX_WORDS_PER_SEC = 6  # 幻觉熔断阈值


def normalize_text(text, dataset_name):
    if not text: return ""
    text = text.replace("\n", " ").replace("\r", " ").lower()
    if dataset_name in ["TEDLIUM", "AMICorpus"]:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[{\[][^}\]]*[}\]]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.strip().split())


def calculate_rms(audio_chunk):
    if len(audio_chunk) == 0: return 0
    return np.sqrt(np.mean(audio_chunk ** 2))


def load_dataset_tasks(dataset_name, dataset_path):
    task_list = []
    print(f"正在扫描 {dataset_name} 数据集目录: {dataset_path} ...")

    if dataset_name == "LibriSpeech":
        for root, dirs, files in os.walk(dataset_path):
            trans_files = [f for f in files if f.endswith(".trans.txt")]
            for trans_file in trans_files:
                with open(os.path.join(root, trans_file), "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            file_id, raw_text = parts
                            audio_path = os.path.join(root, f"{file_id}.wav")
                            if not os.path.exists(audio_path):
                                audio_path = os.path.join(root, f"{file_id}.flac")
                            if os.path.exists(audio_path):
                                task_list.append({"file_id": file_id, "audio_path": audio_path,
                                                  "ref_text": normalize_text(raw_text, dataset_name)})

    elif dataset_name == "AMICorpus":
        for root, dirs, files in os.walk(dataset_path):
            wav_files = [f for f in files if f.endswith(".wav")]
            for wav_file in wav_files:
                file_id = os.path.splitext(wav_file)[0]
                txt_path = os.path.join(root, f"{file_id}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                    task_list.append({"file_id": file_id, "audio_path": os.path.join(root, wav_file),
                                      "ref_text": normalize_text(raw_text, dataset_name)})

    elif dataset_name == "TEDLIUM":
        audio_map = {os.path.splitext(f)[0]: os.path.join(root, f) for root, dirs, files in os.walk(dataset_path) for f
                     in files if f.endswith((".wav", ".flac"))}
        text_file_path = os.path.join(dataset_path, "text.txt")
        if os.path.exists(text_file_path):
            with open(text_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2 and parts[0] in audio_map:
                        task_list.append({"file_id": parts[0], "audio_path": audio_map[parts[0]],
                                          "ref_text": normalize_text(parts[1], dataset_name)})

    valid_tasks = [t for t in task_list if len(t["ref_text"].split()) >= MIN_WORD_COUNT]
    print(f"扫描完毕！找到有效评测任务 {len(valid_tasks)} 个。")
    return valid_tasks


def create_recognizer(model_size):
    try:
        return WhisperModel(model_size, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"❌ 错误: 模型加载失败 - {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()

    print("-" * 50)
    print(f"初始化 Faster-Whisper [基础分片(1s)准实时] 评测...")
    recognizer = create_recognizer(MODEL_SIZE)
    print("-" * 50)

    task_list = load_dataset_tasks(args.dataset_name, args.dataset_path)
    total_files = len(task_list)
    if total_files == 0: return

    total_distance, total_ref_words, processed_count = 0, 0, 0

    print("=" * 60)
    print(f"🎯 开始执行 {args.dataset_name} WER 评测")
    print("=" * 60)

    for task in task_list:
        try:
            audio, sample_rate = sf.read(task["audio_path"], dtype="float32")
            if sample_rate != 16000 or len(audio) < int(sample_rate * 0.05): continue
        except Exception:
            continue

        chunk_size = int(sample_rate * CHUNK_DURATION)
        hyp_segments_list = []
        last_text = ""

        # 核心分片推理
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i: i + chunk_size]
            if len(chunk) < 160 or calculate_rms(chunk) < SILENCE_THRESHOLD: continue

            try:
                segments, _ = recognizer.transcribe(
                    chunk, beam_size=1, language="en", vad_filter=True,
                    condition_on_previous_text=False, temperature=0.0,
                    compression_ratio_threshold=2.0, no_speech_threshold=0.4
                )

                current_chunk_text = " ".join(
                    seg.text.strip() for seg in segments
                    if
                    not any(h in seg.text.lower() for h in ["see you", "watching", "subtitles", "amara", "org", "bye"])
                ).strip()

                # 幻觉熔断与去重
                if current_chunk_text and len(current_chunk_text.split()) <= MAX_WORDS_PER_SEC:
                    if current_chunk_text != last_text:
                        hyp_segments_list.append(current_chunk_text)
                        last_text = current_chunk_text
            except Exception:
                pass

        hyp_text_clean = normalize_text(" ".join(hyp_segments_list), args.dataset_name)

        try:
            out = jiwer.process_words(task["ref_text"], hyp_text_clean)
            curr_dist = out.substitutions + out.deletions + out.insertions
            curr_len = len(out.references[0]) if out.references else 0

            total_distance += curr_dist
            total_ref_words += curr_len
            processed_count += 1

            print(
                f"[{processed_count}/{total_files}] ID: {task['file_id'][:25]:<25} | Words: {curr_len:<4} | Dist: {curr_dist:<3} | WER: {out.wer:.4f}")
        except Exception:
            pass

    print("\n" + "=" * 50)
    print(f"✅ {args.dataset_name} 数据集评测完成 (总计 {processed_count} 首)")
    if total_ref_words > 0:
        print(f"🚀 平均词错误率 (Average WER): {total_distance / total_ref_words:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()