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

csv_lock = threading.Lock()
print_lock = threading.Lock()
stats_lock = threading.Lock()

g_total_proc_time, g_total_audio_dur, g_processed_count = 0.0, 0.0, 0


def calculate_rms(audio_chunk):
    return np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) > 0 else 0


def create_recognizer():
    try:
        return WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    except Exception as e:
        sys.exit(1)


def worker_thread(task_queue, recognizer, total_tasks, csv_filename):
    global g_total_proc_time, g_total_audio_dur, g_processed_count

    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break

        file_id, audio_path = task
        try:
            audio, sample_rate = sf.read(audio_path, dtype="float32")
            if sample_rate != 16000: continue
        except Exception:
            task_queue.task_done()
            continue

        file_audio_len = len(audio) / sample_rate
        if file_audio_len < 0.05:
            task_queue.task_done()
            continue

        chunk_size = int(sample_rate * CHUNK_DURATION)
        file_proc_time = 0.0

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i: i + chunk_size]
            if len(chunk) < 160: continue

            # 仅记录模型计算真实耗时（包含并发排队等待时间）
            t_start = time.perf_counter()
            if calculate_rms(chunk) >= SILENCE_THRESHOLD:
                try:
                    recognizer.transcribe(chunk, beam_size=1, vad_filter=True, condition_on_previous_text=False)
                except Exception:
                    pass

            file_proc_time += (time.perf_counter() - t_start)

        curr_rtf = file_proc_time / file_audio_len if file_audio_len > 0 else 0

        with stats_lock:
            g_total_proc_time += file_proc_time
            g_total_audio_dur += file_audio_len
            g_processed_count += 1
            current_count = g_processed_count

        with csv_lock:
            with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(
                    [file_id, round(file_audio_len, 4), round(file_proc_time, 4), round(curr_rtf, 4)])

        with print_lock:
            if current_count % 50 == 0 or current_count == total_tasks:
                print(f"[{current_count}/{total_tasks}] | 最近完成: {file_id[:20]:<20} | RTF: {curr_rtf:.4f}")

        task_queue.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()

    csv_filename = f"Results_Faster-Whisper_{args.dataset_name}_Segmentation_RTF_C{args.concurrency}.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["FileID", "Duration(s)", "Compute_Time(s)", "RTF"])

    print(f"初始化 Faster-Whisper RTF 并发压测 (并发数: {args.concurrency})...")
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

    avg_rtf = g_total_proc_time / g_total_audio_dur if g_total_audio_dur > 0 else 0

    print("=" * 40)
    print("RTF 并发测试完成")
    print(f"并发线程数: {args.concurrency}")
    print(f"成功处理文件数: {g_processed_count}")
    print(f"评测总耗时 (Wall Time): {wall_time:.2f} 秒")
    print("=" * 40)
    print(f"平均实时率 (Average RTF): {avg_rtf:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()