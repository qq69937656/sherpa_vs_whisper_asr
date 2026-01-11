import os
import sys
import re
import wave
import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from faster_whisper import WhisperModel

# ================= 配置区域 =================

LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"
MODEL_SIZE = "medium.en"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# 并行数 (线程数)
PARALLEL_NUM = 6

# 算法参数
CHUNK_DURATION = 1.0
FORCE_FLUSH_DURATION = 10.0

# 打印锁，防止多线程打印混乱
print_lock = threading.Lock()


# ================= 辅助函数 =================

def get_audio_duration(file_path):
    """获取wav文件时长"""
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except:
        return 0.0


def is_hallucination(segment):
    """幻觉抑制过滤器"""
    HALLUCINATION_PHRASES = ["subtitle", "amara", "audio", "copyright", "subscribe"]
    text = segment.text.strip().lower()
    duration = segment.end - segment.start

    if not text or duration <= 0: return True
    if (len(text) / duration) > 40: return True
    if segment.words and len(segment.words) > 1:
        if (duration / len(segment.words)) < 0.05: return True
    if segment.no_speech_prob > 0.9: return True
    if segment.avg_logprob < -1.5: return True
    for bad in HALLUCINATION_PHRASES:
        if bad in text: return True
    return False


# ================= 核心识别逻辑 (Worker) =================

def process_single_file(model, file_path):
    """
    单个文件的完整处理流程 (带回滚机制的流式模拟)
    返回: (文件名, 音频时长, 处理耗时)
    """
    # 1. 获取音频时长
    duration = get_audio_duration(file_path)
    if duration < 0.1:
        return os.path.basename(file_path), 0.0, 0.0

    # 2. 开始计时
    start_time = time.perf_counter()

    audio_buffer = np.array([], dtype=np.float32)
    SAMPLE_RATE = 16000
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    try:
        wf = wave.open(file_path, 'rb')
        if wf.getnchannels() != 1 or wf.getframerate() != 16000:
            wf.close()
            return os.path.basename(file_path), duration, -1  # 错误标记
    except:
        return os.path.basename(file_path), duration, -1

    while True:
        # IO 读取
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

        # 模型推理 (此处会竞争 GPU 资源)
        try:
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
                if is_hallucination(seg): continue
                if not seg.words:
                    class PseudoWord:
                        def __init__(self, w, start, end):
                            self.word, self.start, self.end = w, start, end

                    all_valid_words.append(PseudoWord(seg.text, seg.start, seg.end))
                else:
                    all_valid_words.extend(seg.words)

            # 动态截断与回滚逻辑
            word_count = len(all_valid_words)
            should_hold_back = (not is_eof) and (buffer_duration < FORCE_FLUSH_DURATION)
            cut_time = 0.0

            if word_count > 0:
                if should_hold_back:
                    num_to_hold = 1
                    if all_valid_words[-1].word.strip() in ['-', '—', '--']:
                        num_to_hold = 2

                    if word_count > num_to_hold:
                        cut_index = -num_to_hold
                        cut_time = all_valid_words[cut_index - 1].end
                    else:
                        cut_time = 0.0
                else:
                    cut_time = all_valid_words[-1].end

            if cut_time > 0:
                cut_sample_index = int(cut_time * SAMPLE_RATE)
                if cut_sample_index < len(audio_buffer):
                    audio_buffer = audio_buffer[cut_sample_index:]
                else:
                    audio_buffer = np.array([], dtype=np.float32)

        except Exception as e:
            # 推理出错
            break

        if is_eof: break

    wf.close()

    end_time = time.perf_counter()
    cost_time = end_time - start_time

    return os.path.basename(file_path), duration, cost_time


# ================= 主程序 =================

def main():
    print("-" * 60)
    print(f"模式: 并行评测 (线程数: {PARALLEL_NUM})")
    print(f"模型: {MODEL_SIZE} | Device: {DEVICE}")

    # 1. 加载模型 (主线程加载，子线程共享)
    try:
        # 注意: 如果 GPU 显存较小，可以设置 device_index=[0] 或调整 intra_threads
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    print("模型加载完成。")

    # 2. 扫描文件
    wav_files = []
    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 目录不存在 {LIBRISPEECH_DIR}")
        return

    print("正在扫描数据集...")
    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    print(f"共找到 {len(wav_files)} 个 wav 文件。")
    print("-" * 60)
    print(f"{'Filename':<30} | {'Audio(s)':<8} | {'Cost(s)':<8} | {'RTF':<6}")
    print("-" * 60)

    total_audio_duration = 0.0
    total_process_time = 0.0  # 累积所有线程的处理时间
    processed_count = 0

    # 3. 线程池并行执行
    # max_workers=PARALLEL_NUM 保证了同时只有 5 个文件在处理
    with ThreadPoolExecutor(max_workers=PARALLEL_NUM) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_single_file, model, f): f for f in wav_files}

        # 随着任务完成，获取结果
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                fname, duration, cost = future.result()

                if cost < 0:  # 错误处理
                    continue

                rtf = cost / duration if duration > 0 else 0

                # 线程安全地累积数据
                with print_lock:
                    print(f"{fname:<30} | {duration:<8.2f} | {cost:<8.4f} | {rtf:<6.4f}")
                    total_audio_duration += duration
                    total_process_time += cost
                    processed_count += 1

            except Exception as e:
                print(f"Thread Error processing {file_path}: {e}")

    # ================= 结果汇总 =================
    print("-" * 60)
    print(f"并行任务结束。处理文件总数: {processed_count}")

    if total_audio_duration > 0:
        # 这里的 RTF 是 "平均单流 RTF" (Average Stream RTF)
        # 反映了在当前并发负载下，处理每一秒音频平均需要花费的计算时间
        avg_rtf = total_process_time / total_audio_duration

        # 计算吞吐量 (Throughput) 的参考指标
        # 理想情况下，总处理能力 = 总时长 / (总耗时 / 并发数)

        print(f"总音频时长: {total_audio_duration:.2f} s")
        print(f"总计算耗时: {total_process_time:.2f} s (各线程耗时累加)")
        print("=" * 30)
        print(f"平均实时率 (Average RTF): {avg_rtf:.4f}")
        print("=" * 30)
        print("注意: 由于 GPU 资源竞争，单流 RTF 通常高于单线程运行时的 RTF。")
    else:
        print("未处理任何有效文件。")


if __name__ == "__main__":
    main()
