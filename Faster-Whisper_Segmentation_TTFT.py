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
SILENCE_THRESHOLD = 0.01
MAX_WORDS_PER_SEC = 6

csv_lock = threading.Lock()
print_lock = threading.Lock()
stats_lock = threading.Lock()

g_ttft_sum = 0.0
g_processed_count = 0


def calculate_rms(audio_chunk):
    return np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) > 0 else 0


def create_recognizer():
    try:
        return WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    except Exception as e:
        sys.exit(1)


def worker_thread(task_queue, recognizer, total_tasks, csv_filename):
    global g_ttft_sum, g_processed_count

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

        chunk_size = int(sample_rate * CHUNK_DURATION)
        ttft = None

        # 核心分片推理（早停机制）
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i: i + chunk_size]
            if len(chunk) < 160: continue

            # 音频等待积累的时间 (1s 分片固有延迟)
            audio_wait_time = (i / sample_rate) + CHUNK_DURATION

            t_start = time.perf_counter()
            valid_text_found = False

            if calculate_rms(chunk) >= SILENCE_THRESHOLD:
                try:
                    segments, _ = recognizer.transcribe(
                        chunk, beam_size=1, language="en", vad_filter=True,
                        condition_on_previous_text=False, temperature=0.0, no_speech_threshold=0.4
                    )

                    current_chunk_text = " ".join(
                        seg.text.strip() for seg in segments
                        if not any(
                            h in seg.text.lower() for h in ["see you", "watching", "subtitles", "amara", "org", "bye"])
                    ).strip()

                    if current_chunk_text and len(current_chunk_text.split()) <= MAX_WORDS_PER_SEC:
                        valid_text_found = True
                except Exception:
                    pass

            calc_duration = time.perf_counter() - t_start

            if valid_text_found:
                ttft = audio_wait_time + calc_duration
                chunk_idx = (i // chunk_size) + 1

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

                # 早停：一旦抓到首字，退出当前文件的处理
                break

        # 如果跑完了音频都没有结果，也记录一次
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

    csv_filename = f"Results_Faster-Whisper_{args.dataset_name}_Segmentation_TTFT_C{args.concurrency}.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["FileID", "Chunks_Processed_At_TTFT", "TTFT(s)"])

    print(f"初始化 Faster-Whisper TTFT 并发压测 (并发数: {args.concurrency})...")
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
    print("TTFT 并发测试完成")
    print(f"并发线程数: {args.concurrency}")
    print(f"成功出字文件数: {g_processed_count}")
    print(f"评测总耗时 (Wall Time): {wall_time:.2f} 秒")
    print("=" * 40)
    print(f"平均首字延迟 (Average TTFT): {avg_ttft:.4f} 秒")
    print("=" * 40)


if __name__ == "__main__":
    main()