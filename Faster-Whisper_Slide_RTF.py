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

csv_lock = threading.Lock()
print_lock = threading.Lock()
stats_lock = threading.Lock()

g_total_proc_time, g_total_audio_dur, g_processed_count = 0.0, 0.0, 0


def create_recognizer():
    try:
        return WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    except Exception:
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
            if sample_rate != 16000: raise Exception
        except Exception:
            task_queue.task_done()
            continue

        file_audio_len = len(audio) / sample_rate
        if file_audio_len < 0.1:
            task_queue.task_done()
            continue

        chunk_samples = int(sample_rate * CHUNK_DURATION)
        audio_buffer = np.array([], dtype=np.float32)

        file_proc_time = 0.0

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i: i + chunk_samples]
            audio_buffer = np.concatenate((audio_buffer, chunk))
            is_eof = (i + chunk_samples >= len(audio))
            buffer_duration = len(audio_buffer) / sample_rate

            if is_eof and buffer_duration < 0.2: break
            if buffer_duration < 0.5 and not is_eof: continue

            # 仅记录模型处理耗时
            t_start = time.perf_counter()
            try:
                segments, _ = recognizer.transcribe(
                    audio_buffer, beam_size=1, language="en", condition_on_previous_text=False,
                    vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300),
                    word_timestamps=True, no_speech_threshold=0.5
                )

                all_valid_words = []
                for seg in segments:
                    if not seg.words:
                        class PseudoWord:
                            def __init__(self, w, start, end): self.word, self.start, self.end = w, start, end

                        all_valid_words.append(PseudoWord(seg.text, seg.start, seg.end))
                    else:
                        all_valid_words.extend(seg.words)

                cut_time = 0.0
                if len(all_valid_words) > 0:
                    if (not is_eof) and (buffer_duration < FORCE_FLUSH_DURATION):
                        num_to_hold = 2 if all_valid_words[-1].word.strip() in ['-', '—', '--'] else 1
                        if len(all_valid_words) > num_to_hold:
                            cut_time = all_valid_words[-num_to_hold - 1].end
                    else:
                        cut_time = all_valid_words[-1].end

                if cut_time > 0:
                    cut_sample_index = int(cut_time * sample_rate)
                    audio_buffer = audio_buffer[cut_sample_index:] if cut_sample_index < len(
                        audio_buffer) else np.array([], dtype=np.float32)

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
                print(f"[{current_count}/{total_tasks}] | {file_id[:20]:<20} | 滑窗RTF: {curr_rtf:.4f}")

        task_queue.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()

    csv_filename = f"Results_Faster-Whisper_{args.dataset_name}_Slide_RTF_C{args.concurrency}.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["FileID", "Duration(s)", "Compute_Time(s)", "RTF"])

    print(f"初始化 Faster-Whisper [滑动窗口] RTF 并发压测 (并发数: {args.concurrency})...")
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
    print("滑动窗口 RTF 测试完成")
    print(f"并发线程数: {args.concurrency}")
    print(f"成功处理文件数: {g_processed_count}")
    print(f"评测总耗时 (Wall Time): {wall_time:.2f} 秒")
    print("=" * 40)
    print(f"平均实时率 (Average RTF): {avg_rtf:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()