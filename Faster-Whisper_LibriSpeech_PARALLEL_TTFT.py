import os
import sys
import re
import time
import soundfile as sf
import numpy as np
import threading
import queue
from faster_whisper import WhisperModel

# ================= 配置区域 =================

LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"
MODEL_SIZE = "medium.en"
CHUNK_DURATION = 1.0  # 1S 分片

# 并发数量 (GPU 模式下，并发数过高会导致单次推理排队延迟增加，建议 4-8)
PARALLEL_NUM = 9

# VAD & 过滤参数
SILENCE_THRESHOLD = 0.01
MAX_WORDS_PER_SEC = 6

# ================= 全局共享资源 =================

print_lock = threading.Lock()
stats_lock = threading.Lock()

# 收集结果
g_ttft_list = []  # 存储每个文件的 TTFT
g_processed_count = 0  # 处理文件总数


# ================= 辅助函数 =================

def calculate_rms(audio_chunk):
    if len(audio_chunk) == 0:
        return 0
    return np.sqrt(np.mean(audio_chunk ** 2))


def create_recognizer(model_size):
    try:
        print(f"正在加载 GPU 模型 (float16)...")
        # 【配置修改】使用 CUDA + float16
        model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16"
        )
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)


# ================= 工作线程逻辑 =================

def worker_thread(thread_id, task_queue, recognizer):
    while True:
        try:
            # 非阻塞获取任务
            task = task_queue.get_nowait()
        except queue.Empty:
            break

        file_id, audio_path = task

        # 执行核心逻辑：计算 TTFT
        ttft = process_until_first_token(file_id, audio_path, recognizer)

        # 记录数据
        if ttft is not None:
            with stats_lock:
                g_ttft_list.append(ttft)

        task_queue.task_done()


def process_until_first_token(file_id, audio_path, recognizer):
    """
    处理音频直到输出第一个有效文本，然后立即停止
    """
    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
    except Exception:
        return None

    # 忽略过短音频
    if len(audio) < int(sample_rate * 0.05):
        return None

    chunk_size = int(sample_rate * CHUNK_DURATION)

    # 逐秒处理
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i: i + chunk_size]
        if len(chunk) < 160: continue

        # === 关键指标 1: 音频流等待时间 ===
        # 第0片(0-1s) -> 等待1s; 第1片(1-2s) -> 等待2s
        audio_wait_time = (i / sample_rate) + CHUNK_DURATION

        # === 计时开始 (计算延迟) ===
        t_start = time.time()

        # 1. VAD 预判
        rms = calculate_rms(chunk)

        valid_text_found = False
        first_text = ""

        # 只有能量达标才送入 GPU
        if rms >= SILENCE_THRESHOLD:
            try:
                # 2. GPU 推理 (并发竞争点)
                segments, info = recognizer.transcribe(
                    chunk,
                    beam_size=1,
                    language="en",
                    vad_filter=True,  # 启用模型内部 VAD
                    condition_on_previous_text=False,
                    temperature=0.0,
                    no_speech_threshold=0.4
                )

                current_chunk_text = ""
                for segment in segments:
                    txt = segment.text.strip()
                    # 3. 过滤幻觉
                    hallucinations = ["see you", "watching", "subtitles", "amara", "org", "bye"]
                    if any(h in txt.lower() for h in hallucinations):
                        continue
                    current_chunk_text += " " + txt

                current_chunk_text = current_chunk_text.strip()

                # 4. 判断是否为有效首字
                if current_chunk_text and len(current_chunk_text.split()) <= MAX_WORDS_PER_SEC:
                    valid_text_found = True
                    first_text = current_chunk_text

            except Exception:
                pass

        # === 计时结束 ===
        t_end = time.time()

        # 纯计算耗时 (Computation Latency)
        calc_duration = t_end - t_start

        # 如果找到了字，计算 TTFT 并返回
        if valid_text_found:
            # === 关键指标 2: TTFT ===
            ttft = audio_wait_time + calc_duration

            with print_lock:
                print(
                    f"File: {file_id:<20} | TTFT: {ttft:.4f}s (Wait:{audio_wait_time:.0f}s + Calc:{calc_duration:.4f}s) | Text: {first_text[:20]}...")

            # 【早停】不用继续处理该文件的后续切片了
            return ttft

    # 如果遍历完整个文件都没字（全是静音或过滤掉了）
    with print_lock:
        print(f"File: {file_id:<20} | [No Speech Found]")
    return None


# ================= 主程序 =================

def main():
    print("-" * 65)
    print(f"模式: GPU 并发测试 (Threads={PARALLEL_NUM})")
    print(f"设备: CUDA (float16)")
    print(f"指标: 首字延迟 (TTFT)")
    print("-" * 65)

    recognizer = create_recognizer(MODEL_SIZE)
    print("模型加载完成。")
    print("-" * 65)

    task_queue = queue.Queue()
    file_count = 0

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"目录不存在: {LIBRISPEECH_DIR}")
        return

    print("正在扫描数据集...")
    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        for f in files:
            if f.endswith(".wav") or f.endswith(".flac"):
                file_id = os.path.splitext(f)[0]
                audio_path = os.path.join(root, f)
                task_queue.put((file_id, audio_path))
                file_count += 1

    print(f"共扫描到 {file_count} 个文件。启动 {PARALLEL_NUM} 个线程开始并发评测...")
    print(f"{'File ID':<20} | {'TTFT Result (Wait + Calc)':<40} | {'First Text'}")
    print("-" * 80)

    # 启动线程
    threads = []

    for i in range(PARALLEL_NUM):
        t = threading.Thread(target=worker_thread, args=(i, task_queue, recognizer))
        t.daemon = True
        t.start()
        threads.append(t)

    # 等待队列清空
    task_queue.join()

    # ================= 结果汇总 =================
    print("-" * 80)
    print("GPU 并发 TTFT 评测完成")
    print(f"文件总数: {file_count}")
    print(f"有输出的文件数: {len(g_ttft_list)}")

    if len(g_ttft_list) > 0:
        avg_ttft = sum(g_ttft_list) / len(g_ttft_list)
        min_ttft = min(g_ttft_list)
        max_ttft = max(g_ttft_list)

        # 计算标准差，评估抖动
        std_dev = np.std(g_ttft_list)

        print(f"★ 平均首字延迟 (Avg TTFT): {avg_ttft:.4f} 秒")
        print(f"  - 最小 TTFT: {min_ttft:.4f} 秒")
        print(f"  - 最大 TTFT: {max_ttft:.4f} 秒")
        print(f"  - 延迟波动 (Std Dev): {std_dev:.4f}")
        print("-" * 40)
        print("注意: 在 5 路并发下，'Calc' 时间包含了在 GPU 上的排队等待时间。")
        print("      单路 Calc 时间通常很短，但并发高时 Calc 会变长，导致 TTFT 增加。")
    else:
        print("未检测到有效语音。")
    print("=" * 80)


if __name__ == "__main__":
    main()
