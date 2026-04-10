import os
import sys
import time
import argparse
import csv
import threading
import queue
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

MODEL_SIZE = "medium.en"
CHUNK_DURATION = 1.0
FORCE_FLUSH_DURATION = 10.0
HALLUCINATION_PHRASES = ["subtitle", "amara", "audio", "copyright", "subscribe", "see you", "watching", "org", "bye"]

csv_lock = threading.Lock()
print_lock = threading.Lock()
stats_lock = threading.Lock()

g_ttft_sum = 0.0
g_processed_count = 0


def is_hallucination(segment):
    text = segment.text.strip().lower()
    duration = segment.end - segment.start
    if not text or duration <= 0: return True
    if (len(text) / duration) > 40: return True
    if segment.words and len(segment.words) > 1 and (duration / len(segment.words)) < 0.05: return True
    if segment.no_speech_prob > 0.9 or segment.avg_logprob < -1.5: return True
    if any(bad in text for bad in HALLUCINATION_PHRASES): return True
    return False


def create_recognizer():
    try:
        return WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    except Exception:
        sys.exit(1)


def worker_thread(task_queue, recognizer, total_tasks, csv_filename):
    global g_ttft_sum, g_processed_count

    class PseudoWord:
        def __init__(self, w, start, end): self.word, self.start, self.end = w, start, end

    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break

        file_id, audio_path = task
        try:
            audio, sample_rate = sf.read(audio_path, dtype="float32")
            if sample_rate != 16000: raise Exception
        except Exception:
            task_queue.task_done()
            continue

        if len(audio) < int(sample_rate * 0.05):
            task_queue.task_done()
            continue

        chunk_samples = int(sample_rate * CHUNK_DURATION)
        audio_buffer = np.array([], dtype=np.float32)
        cumulative_compute_time = 0.0
        ttft = None

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i: i + chunk_samples]
            audio_buffer = np.concatenate((audio_buffer, chunk))
            is_eof = (i + chunk_samples >= len(audio))
            current_audio_duration = len(audio_buffer) / sample_rate

            if is_eof and current_audio_duration < 0.2: break
            if current_audio_duration < 0.5 and not is_eof: continue

            t_start = time.perf_counter()
            text_to_commit = ""

            try:
                segments, _ = recognizer.transcribe(
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
                should_hold_back = (not is_eof and current_audio_duration < FORCE_FLUSH_DURATION)

                if word_count > 0:
                    if should_hold_back:
                        num_to_hold = 2 if all_valid_words[-1].word.strip() in ['-', '—', '--'] else 1
                        if word_count > num_to_hold:
                            text_to_commit = "".join([w.word for w in all_valid_words[:-num_to_hold]])
                    else:
                        text_to_commit = "".join([w.word for w in all_valid_words])

            except Exception:
                pass

            cumulative_compute_time += (time.perf_counter() - t_start)

            # TTFT = 当前累积投入的音频物理时长（等待时间） + 累积的GPU计算排队时间
            if text_to_commit:
                ttft = current_audio_duration + cumulative_compute_time
                chunk_idx = (i // chunk_samples) + 1

                with stats_lock:
                    g_ttft_sum += ttft
                    g_processed_count += 1
                    current_count = g_processed_count

                with csv_lock:
                    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerow([file_id, chunk_idx, round(ttft, 4)])

                with print_lock:
                    if current_count % 50 == 0 or current_count == total_tasks:
                        print(f"[{current_count}/{total_tasks}] | {file_id[:20]:<20} | TTFT: {ttft:.4f}s")

                break  # 拿到首字，早停

        if ttft is None:
            with csv_lock:
                with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([file_id, "End_of_Audio", "None"])

        task_queue.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()

    csv_filename = f"Results_Faster-Whisper_{args.dataset_name}_Slide_TTFT_C{args.concurrency}.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["FileID", "Chunks_Processed_At_TTFT", "TTFT(s)"])

    print(f"初始化 Faster-Whisper [滑动窗口] TTFT 并发压测 (并发数: {args.concurrency})...")
    recognizer = create_recognizer()

    all_files = [os.path.join(root, f) for root, _, files in os.walk(args.dataset_path) for f in files if
                 f.endswith((".wav", ".flac"))]
    task_queue = queue.Queue()
    for f in all_files: task_queue.put((os.path.splitext(os.path.basename(f))[0], f))
    total_tasks = task_queue.qsize()
    if total_tasks == 0: return

    print("=" * 60)
    print(f"🚀 开始极限吞吐压测，实时详细数据正写入: {csv_filename}")
    print("=" * 60)

    wall_start_time = time.time()
    threads = []
    for _ in range(args.concurrency):
        t = threading.Thread(target=worker_thread, args=(task_queue, recognizer, total_tasks, csv_filename))
        t.daemon = True
        t.start()
        threads.append(t)

    task_queue.join()
    wall_time = time.time() - wall_start_time

    avg_ttft = g_ttft_sum / g_processed_count if g_processed_count > 0 else 0

    print("=" * 40)
    print("滑动窗口 TTFT 并发测试完成")
    print(f"并发线程数: {args.concurrency}")
    print(f"成功出字文件数: {g_processed_count}")
    print(f"评测总耗时 (Wall Time): {wall_time:.2f} 秒")
    print("=" * 40)
    print(f"平均首字延迟 (Average TTFT): {avg_ttft:.4f} 秒")
    print("=" * 40)


if __name__ == "__main__":
    main()