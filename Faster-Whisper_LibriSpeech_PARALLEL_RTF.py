import os
import sys
import re
import time
import soundfile as sf
import numpy as np
import threading
import queue
from faster_whisper import WhisperModel
import jiwer

# ================= 配置区域 =================

LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"
MODEL_SIZE = "medium.en"
CHUNK_DURATION = 1.0  # 1S 分片

# 并行数量
PARALLEL_NUM = 9

# VAD & 过滤参数
SILENCE_THRESHOLD = 0.01
MAX_WORDS_PER_SEC = 6
MIN_WORD_COUNT = 3

# ================= 全局共享资源 =================

# 线程锁
print_lock = threading.Lock()
stats_lock = threading.Lock()

# 全局统计变量
g_total_proc_time = 0.0
g_total_audio_dur = 0.0
g_total_ref_words = 0
g_total_distance = 0


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
        # CTranslate2 释放 GIL，允许 python 多线程并发调用底层 C++ 推理
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)


# ================= 工作线程逻辑 =================

def worker_thread(thread_id, task_queue, recognizer):
    """
    工作线程：不断从队列取文件进行处理
    """
    while True:
        try:
            # 非阻塞获取任务，如果队列空了就退出
            task = task_queue.get_nowait()
        except queue.Empty:
            break

        file_id, audio_path, ref_text = task

        # --- 开始处理单个文件 ---
        process_single_file(thread_id, file_id, audio_path, ref_text, recognizer)

        # 标记任务完成
        task_queue.task_done()


def process_single_file(thread_id, file_id, audio_path, ref_text, recognizer):
    global g_total_proc_time, g_total_audio_dur, g_total_ref_words, g_total_distance

    # 读取音频
    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
    except Exception:
        return

    file_audio_len = len(audio) / sample_rate
    if file_audio_len < 0.05:
        return

    # 局部统计
    chunk_size = int(sample_rate * CHUNK_DURATION)
    hyp_segments_list = []
    last_text = ""
    file_proc_time = 0.0

    # === 核心处理循环 ===
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i: i + chunk_size]
        if len(chunk) < 160: continue

        # 计时开始
        t_start = time.time()

        # 1. VAD
        rms = calculate_rms(chunk)
        if rms >= SILENCE_THRESHOLD:
            try:
                # 2. 推理 (并发竞争点)
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

                # 3. 后处理
                current_chunk_text = ""
                for segment in segments:
                    txt = segment.text.strip()
                    hallucinations = ["see you", "watching", "subtitles", "amara", "org", "bye"]
                    if any(h in txt.lower() for h in hallucinations):
                        continue
                    current_chunk_text += " " + txt

                current_chunk_text = current_chunk_text.strip()

                if len(current_chunk_text.split()) <= MAX_WORDS_PER_SEC:
                    if current_chunk_text and current_chunk_text != last_text:
                        hyp_segments_list.append(current_chunk_text)
                        last_text = current_chunk_text

            except Exception:
                pass

        # 计时结束
        t_end = time.time()
        file_proc_time += (t_end - t_start)

    # === 计算单文件结果 ===
    curr_rtf = file_proc_time / file_audio_len if file_audio_len > 0 else 0

    # 简单的 WER 计算 (仅用于汇总，不打印细节)
    dist = 0
    ref_len = 0
    hyp_text_raw = " ".join(hyp_segments_list)
    ref_words = normalize_text(ref_text)
    if len(ref_words) >= MIN_WORD_COUNT:
        try:
            hyp_words = normalize_text(hyp_text_raw)
            out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
            dist = out.substitutions + out.deletions + out.insertions
            ref_len = len(out.references[0])
        except:
            pass

    # === 更新全局统计 (加锁) ===
    with stats_lock:
        g_total_proc_time += file_proc_time
        g_total_audio_dur += file_audio_len
        g_total_distance += dist
        g_total_ref_words += ref_len

    # === 打印输出 (加锁) ===
    with print_lock:
        print(f"{file_id:<25} | {file_proc_time:<12.4f} | {file_audio_len:<10.2f} | {curr_rtf:<8.4f}")


# ================= 主程序 =================

def main():
    print("-" * 65)
    print(f"模式: 并行处理 (Threads={PARALLEL_NUM})")
    print(f"正在加载模型 {MODEL_SIZE} ...")
    recognizer = create_recognizer(MODEL_SIZE)
    print("模型加载完成。")
    print("-" * 65)

    # 打印表头
    print(f"{'File ID':<25} | {'ProcTime(s)':<12} | {'Audio(s)':<10} | {'RTF':<8}")
    print("-" * 65)

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 目录不存在 {LIBRISPEECH_DIR}")
        return

    # 1. 扫描文件并填充队列
    task_queue = queue.Queue()
    file_count = 0

    print("正在扫描数据集并填充任务队列...")
    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        trans_files = [f for f in files if f.endswith(".trans.txt")]
        for trans_file in trans_files:
            trans_path = os.path.join(root, trans_file)
            ref_map = {}
            try:
                with open(trans_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            ref_map[parts[0]] = parts[1]
            except:
                continue

            for fid, txt in ref_map.items():
                # 寻找音频路径
                audio_path = os.path.join(root, f"{fid}.wav")
                if not os.path.exists(audio_path):
                    flac_path = os.path.join(root, f"{fid}.flac")
                    if os.path.exists(flac_path):
                        audio_path = flac_path
                    else:
                        continue

                # 入队 (FileID, Path, RefText)
                task_queue.put((fid, audio_path, txt))
                file_count += 1

    print(f"任务队列填充完成，共 {file_count} 个文件。启动 {PARALLEL_NUM} 个线程开始并发处理...")
    print("-" * 65)

    # 2. 启动线程池
    threads = []
    wall_start_time = time.time()  # 记录墙钟时间，用于计算吞吐量

    for i in range(PARALLEL_NUM):
        t = threading.Thread(target=worker_thread, args=(i, task_queue, recognizer))
        t.daemon = True
        t.start()
        threads.append(t)

    # 3. 等待所有任务完成
    task_queue.join()
    wall_end_time = time.time()

    # ================= 最终汇总 =================
    total_wall_time = wall_end_time - wall_start_time

    print("-" * 65)
    print("并发评测完成统计:")
    print(f"并行线程数: {PARALLEL_NUM}")
    print(f"总音频时长: {g_total_audio_dur:.2f} 秒")
    print(f"累积处理耗时(Sum ProcTime): {g_total_proc_time:.2f} 秒 (所有线程耗时之和)")
    print(f"实际墙钟耗时(Wall Time):    {total_wall_time:.2f} 秒 (用户等待时间)")

    if g_total_audio_dur > 0:
        # 定义 A: 平均单个文件的延迟 RTF (反映资源竞争下的单任务速度)
        avg_latency_rtf = g_total_proc_time / g_total_audio_dur

        # 定义 B: 系统有效吞吐 RTF (反映并发带来的总加速比)
        effective_rtf = total_wall_time / g_total_audio_dur

        print("-" * 30)
        print(f"★ 平均单路 RTF (Average Latency RTF): {avg_latency_rtf:.5f}")
        print(f"   (注: 由于 {PARALLEL_NUM} 路并发争抢GPU，该数值通常比单线程模式高)")

        print(f"★ 系统吞吐 RTF (Effective System RTF): {effective_rtf:.5f}")
        print(f"   (注: 这是实际处理完所有数据与音频总时长的比值，数值越低越快)")

        # 计算加速比
        print("-" * 30)
        print(f"系统吞吐倍速: {1 / effective_rtf:.2f}x (每秒墙钟时间处理 {1 / effective_rtf:.2f}秒音频)")

    if g_total_ref_words > 0:
        print(f"平均 WER: {g_total_distance / g_total_ref_words:.2%}")
    print("=" * 65)


if __name__ == "__main__":
    main()
