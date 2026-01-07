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

# 并发数 (设置并行线程)
PARALLEL_NUM = 6

# 算法参数
CHUNK_DURATION = 1.0
FORCE_FLUSH_DURATION = 10.0

# 打印锁
print_lock = threading.Lock()


# ================= 辅助函数 =================

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


# ================= 核心 TTFT 测算逻辑 =================

def calculate_ttft_single_file(model, file_path):
    """
    处理单个文件，直到输出第一个文本为止。
    返回: (文件名, TTFT, 是否成功输出)
    """
    # 累计的纯计算耗时 (Wall clock time of inference)
    cumulative_compute_time = 0.0

    audio_buffer = np.array([], dtype=np.float32)
    SAMPLE_RATE = 16000
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    try:
        wf = wave.open(file_path, 'rb')
        if wf.getnchannels() != 1 or wf.getframerate() != 16000:
            wf.close()
            return os.path.basename(file_path), 0.0, False  # 格式错误
    except:
        return os.path.basename(file_path), 0.0, False

    ttft_result = 0.0
    has_output = False

    while True:
        # 1. 模拟流式读取
        data = wf.readframes(chunk_samples)
        is_eof = len(data) == 0

        if not is_eof:
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            new_chunk = audio_int16.astype(np.float32) / 32768.0
            audio_buffer = np.concatenate((audio_buffer, new_chunk))

        if is_eof and len(audio_buffer) == 0: break

        # 当前已送入的音频总时长 (用户等待录音的时间)
        current_audio_duration = len(audio_buffer) / SAMPLE_RATE

        if is_eof and current_audio_duration < 0.2: break
        if current_audio_duration < 0.5 and not is_eof: continue

        # ================= 推理计时开始 =================
        t0 = time.perf_counter()

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

            # 智能扣留逻辑 (Smart Holding)
            word_count = len(all_valid_words)
            should_hold_back = (not is_eof) and (current_audio_duration < FORCE_FLUSH_DURATION)
            text_to_commit = ""
            cut_time = 0.0

            if word_count > 0:
                if should_hold_back:
                    num_to_hold = 1
                    if all_valid_words[-1].word.strip() in ['-', '—', '--']:
                        num_to_hold = 2

                    if word_count > num_to_hold:
                        # 有足够的词，可以输出了！
                        cut_index = -num_to_hold
                        committed_words = all_valid_words[:cut_index]
                        text_to_commit = "".join([w.word for w in committed_words])

                        # 为了继续逻辑完整性，记录切割点 (虽然马上就return了)
                        cut_time = all_valid_words[cut_index - 1].end
                else:
                    # EOF 情况
                    text_to_commit = "".join([w.word for w in all_valid_words])

            # 执行 buffer 切割 (模拟状态更新)
            if cut_time > 0:
                cut_sample_index = int(cut_time * SAMPLE_RATE)
                if cut_sample_index < len(audio_buffer):
                    audio_buffer = audio_buffer[cut_sample_index:]
                else:
                    audio_buffer = np.array([], dtype=np.float32)

        except Exception:
            break

        t1 = time.perf_counter()
        # ================= 推理计时结束 =================

        # 累积计算时间
        cumulative_compute_time += (t1 - t0)

        # 【核心逻辑】如果产生了有效输出，计算 TTFT 并退出
        if text_to_commit:
            # TTFT = 等待音频录制的时间 + 累积的计算时间
            ttft_result = current_audio_duration + cumulative_compute_time
            has_output = True
            break  # <--- 得到第一个字，结束对该文件的处理

        if is_eof: break

    wf.close()
    return os.path.basename(file_path), ttft_result, has_output


# ================= 主程序 =================

def main():
    print("-" * 60)
    print(f"任务: 并发计算 TTFT (首字延迟)")
    print(f"并发数: {PARALLEL_NUM} | 模型: {MODEL_SIZE} | Device: {DEVICE}")
    print(f"计算公式: TTFT = 音频缓冲时长(1s步进) + 累积推理耗时")

    try:
        # 设置 inter_threads=1 避免 CPU 争抢，让 GPU 调度决定并行
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=1)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    print("模型加载完成。")

    # 扫描文件
    wav_files = []
    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 目录不存在 {LIBRISPEECH_DIR}")
        return

    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    print(f"共加载 {len(wav_files)} 个任务。开始并行处理...")
    print("-" * 60)
    print(f"{'Filename':<30} | {'TTFT(s)':<8} | {'Status':<10}")
    print("-" * 60)

    total_ttft = 0.0
    valid_count = 0
    no_output_count = 0

    with ThreadPoolExecutor(max_workers=PARALLEL_NUM) as executor:
        future_to_file = {executor.submit(calculate_ttft_single_file, model, f): f for f in wav_files}

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                fname, ttft, has_output = future.result()

                if has_output:
                    with print_lock:
                        print(f"{fname:<30} | {ttft:<8.4f} | {'OK':<10}")
                    total_ttft += ttft
                    valid_count += 1
                else:
                    # 整个文件读完都没有输出 (可能是纯静音或极短)
                    with print_lock:
                        print(f"{fname:<30} | {'N/A':<8} | {'No Output':<10}")
                    no_output_count += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # ================= 结果汇总 =================
    print("-" * 60)
    print(f"处理结束。")
    print(f"有效输出文件数: {valid_count}")
    print(f"无输出文件数:   {no_output_count} (不计入平均值)")

    if valid_count > 0:
        avg_ttft = total_ttft / valid_count
        print("=" * 30)
        print(f"平均首字延迟 (Avg TTFT): {avg_ttft:.4f} s")
        print("=" * 30)
        print("说明:")
        print("1. 最小理论 TTFT > 1.0s (因为必须等待第一个1s分片)。")
        print("2. 智能扣留策略可能会导致第一个1s不输出，从而延迟到2s+，这是为了保证准确率。")
        print("3. 并发争抢 GPU 会导致 '累积推理耗时' 增加，从而增加 TTFT。")
    else:
        print("未获得任何有效 TTFT 数据。")


if __name__ == "__main__":
    main()
