import os
import sys
import re
import wave
import time
import numpy as np
from faster_whisper import WhisperModel

# ================= 配置区域 =================

LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"
MODEL_SIZE = "medium.en"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# 算法参数
CHUNK_DURATION = 1.0  # 1秒分片
FORCE_FLUSH_DURATION = 10.0  # 强制刷新阈值


# ================= 辅助逻辑 =================

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
    """
    幻觉抑制过滤器 (物理约束策略)
    """
    HALLUCINATION_PHRASES = ["subtitle", "amara", "audio", "copyright", "subscribe"]
    text = segment.text.strip().lower()
    duration = segment.end - segment.start

    if not text or duration <= 0: return True

    # 1. 语速物理极限检测 (>40字符/秒 视为幻觉)
    if (len(text) / duration) > 40: return True

    # 2. 单词密度检测
    if segment.words and len(segment.words) > 1:
        if (duration / len(segment.words)) < 0.05: return True

    # 3. 置信度门控
    if segment.no_speech_prob > 0.9: return True
    if segment.avg_logprob < -1.5: return True

    # 4. 黑名单过滤
    for bad in HALLUCINATION_PHRASES:
        if bad in text: return True

    return False


# ================= 核心识别算法 (用于测速) =================

def transcribe_with_rtf_logic(model, file_path):
    """
    执行完整的优化算法逻辑，并返回处理该文件所消耗的“纯计算时间”
    """
    # 开始计时
    start_time = time.perf_counter()

    audio_buffer = np.array([], dtype=np.float32)
    SAMPLE_RATE = 16000
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    try:
        wf = wave.open(file_path, 'rb')
        if wf.getnchannels() != 1 or wf.getframerate() != 16000:
            wf.close()
            return -1  # 格式错误
    except:
        return -1

    while True:
        # 1. 模拟流式读取 (IO时间通常不计入算法RTF，但在端到端测试中包含也无妨，这里主要测处理耗时)
        data = wf.readframes(chunk_samples)
        is_eof = len(data) == 0

        if not is_eof:
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            new_chunk = audio_int16.astype(np.float32) / 32768.0
            # 策略一：声学上下文扩展 (拼接 Buffer)
            audio_buffer = np.concatenate((audio_buffer, new_chunk))

        if is_eof and len(audio_buffer) == 0: break

        buffer_duration = len(audio_buffer) / SAMPLE_RATE
        if is_eof and buffer_duration < 0.2: break
        if buffer_duration < 0.5 and not is_eof: continue

        # 2. 模型推理
        segments, _ = model.transcribe(
            audio_buffer,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            word_timestamps=True,  # 关键：开启字级时间戳
            no_speech_threshold=0.5
        )

        all_valid_words = []
        for seg in segments:
            # 策略四：幻觉抑制
            if is_hallucination(seg):
                continue
            if not seg.words:
                # 处理无字级时间戳的特殊情况
                class PseudoWord:
                    def __init__(self, w, start, end):
                        self.word, self.start, self.end = w, start, end

                all_valid_words.append(PseudoWord(seg.text, seg.start, seg.end))
            else:
                all_valid_words.extend(seg.words)

        # 策略二 & 三：动态截断与回滚
        word_count = len(all_valid_words)
        should_hold_back = (not is_eof) and (buffer_duration < FORCE_FLUSH_DURATION)
        cut_time = 0.0

        if word_count > 0:
            if should_hold_back:
                num_to_hold = 1
                last_word = all_valid_words[-1].word.strip()
                if last_word in ['-', '—', '--']:
                    num_to_hold = 2

                if word_count > num_to_hold:
                    # 截断点：倒数第 N 个词的前一个词的结束时间
                    cut_index = -num_to_hold
                    cut_time = all_valid_words[cut_index - 1].end
                else:
                    # 词数不够，全量扣留（回滚全部）
                    cut_time = 0.0
            else:
                # EOF 或 强制刷新，全部提交
                cut_time = all_valid_words[-1].end

        # 执行 Buffer 切割 (模拟回滚)
        if cut_time > 0:
            cut_sample_index = int(cut_time * SAMPLE_RATE)
            if cut_sample_index < len(audio_buffer):
                audio_buffer = audio_buffer[cut_sample_index:]
            else:
                audio_buffer = np.array([], dtype=np.float32)

        if is_eof: break

    wf.close()

    # 结束计时
    end_time = time.perf_counter()
    return end_time - start_time


# ================= 主程序 =================

def main():
    print("-" * 60)
    print(f"正在计算 RTF (模型: {MODEL_SIZE}, Device: {DEVICE})...")

    # 预热/加载模型 (不计入 RTF)
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    print("模型加载完成，开始遍历数据集...")
    print("-" * 60)
    print(f"{'Filename':<30} | {'Audio(s)':<8} | {'Cost(s)':<8} | {'RTF':<6}")
    print("-" * 60)

    total_audio_duration = 0.0
    total_process_time = 0.0
    file_count = 0

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 目录不存在 {LIBRISPEECH_DIR}")
        return

    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        for f in files:
            if f.endswith(".wav"):
                wav_path = os.path.join(root, f)

                # 1. 获取音频时长
                duration = get_audio_duration(wav_path)
                if duration < 0.1: continue

                # 2. 执行识别并测速
                process_time = transcribe_with_rtf_logic(model, wav_path)

                if process_time < 0:  # 发生错误
                    continue

                # 3. 计算 RTF
                rtf = process_time / duration

                # 4. 打印单行结果
                print(f"{f:<30} | {duration:<8.2f} | {process_time:<8.4f} | {rtf:<6.4f}")

                # 5. 累积
                total_audio_duration += duration
                total_process_time += process_time
                file_count += 1

    # ================= 结果汇总 =================
    print("-" * 60)
    print(f"处理文件总数: {file_count}")

    if total_audio_duration > 0:
        avg_rtf = total_process_time / total_audio_duration
        print(f"总音频时长: {total_audio_duration:.2f} s")
        print(f"总推理耗时: {total_process_time:.2f} s")
        print("=" * 30)
        print(f"平均实时率 (Average RTF): {avg_rtf:.4f}")
        print("=" * 30)
    else:
        print("未处理任何有效文件。")


if __name__ == "__main__":
    main()
