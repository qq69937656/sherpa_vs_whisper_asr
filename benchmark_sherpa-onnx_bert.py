import sys
import os
import time
import wave
import numpy as np
import sherpa_onnx
from collections import deque
import torch
from transformers import pipeline

# ================= 配置区域 =================
LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"
ASR_MODEL_DIR = "sherpa-onnx-streaming-zipformer-en-2023-06-21"
PUNCT_MODEL_NAME = "oliverguhr/fullstop-punctuation-multilang-large"

# 并发数
CONCURRENT_STREAMS = 6
# 切片时长 (秒)
CHUNK_SECONDS = 0.1
# Bert 修正间隔 (秒)
BERT_INTERVAL = 0.5

# 【关键修改】分别设置设备
# 1. ASR 设备: 你想测 CPU 就写 "cpu"，想测 GPU 就写 "cuda"
ASR_PROVIDER = "cuda"

# 2. Bert 设备: 强制使用 GPU (ID 0)
# 如果没显卡，代码会自动回退到 CPU (-1) 并打印警告
BERT_DEVICE_ID = 0 if torch.cuda.is_available() else -1


# ===========================================

class PunctuationRestorer:
    def __init__(self, model_name, device_id):
        device_str = "GPU (cuda:0)" if device_id >= 0 else "CPU"
        print(f"正在加载 Bert 模型: {model_name} 到 {device_str}...")

        try:
            # device 参数接收 int: -1 代表 CPU, 0+ 代表 GPU ID
            self.pipe = pipeline(
                "token-classification",
                model=model_name,
                aggregation_strategy="none",
                device=device_id
            )
            print("Bert 模型加载完成。")
        except Exception as e:
            print(f"Bert 模型加载失败: {e}")
            sys.exit(1)

    def restore(self, text):
        if not text or not text.strip(): return text
        try:
            _ = self.pipe(text)
            return text
        except Exception:
            return text


class WorkerSlot:
    def __init__(self):
        self.is_active = False
        self.stream = None
        self.filename = None
        self.samples = None
        self.duration = 0.0
        self.cursor = 0
        self.start_time = 0.0

        self.raw_text = ""
        self.ttft = None
        self.has_output = False
        self.input_finished = False
        self.last_punct_time = 0.0

    def reset(self):
        self.is_active = False
        self.stream = None
        self.filename = None
        self.samples = None
        self.duration = 0.0
        self.cursor = 0
        self.start_time = 0.0
        self.raw_text = ""
        self.ttft = None
        self.has_output = False
        self.input_finished = False
        self.last_punct_time = 0.0


def find_wav_files(root_dir):
    wav_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return sorted(wav_files)


def read_wav_file(filename):
    try:
        with wave.open(filename, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getframerate() != 16000: return None, 0
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, wf.getnframes() / 16000.0
    except:
        return None, 0


def main():
    if not os.path.exists(LIBRISPEECH_DIR):
        print("数据集不存在");
        return

    all_files = find_wav_files(LIBRISPEECH_DIR)
    if not all_files: print("未找到文件"); return

    task_queue = deque(all_files)
    print(f"任务总数: {len(all_files)}")

    # 1. 初始化 ASR (使用 ASR_PROVIDER)
    print(f"加载 ASR 模型到: {ASR_PROVIDER}")
    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=os.path.join(ASR_MODEL_DIR, "tokens.txt"),
            encoder=os.path.join(ASR_MODEL_DIR, "encoder-epoch-99-avg-1.onnx"),
            decoder=os.path.join(ASR_MODEL_DIR, "decoder-epoch-99-avg-1.onnx"),
            joiner=os.path.join(ASR_MODEL_DIR, "joiner-epoch-99-avg-1.onnx"),
            num_threads=1, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search",
            provider=ASR_PROVIDER,  # <--- 这里的改动
            enable_endpoint_detection=False
        )
    except Exception as e:
        print(e); return

    # 2. 初始化 Bert (使用 BERT_DEVICE_ID)
    bert_restorer = PunctuationRestorer(PUNCT_MODEL_NAME, BERT_DEVICE_ID)

    slots = [WorkerSlot() for _ in range(CONCURRENT_STREAMS)]
    finished_results = []
    chunk_samples = int(16000 * CHUNK_SECONDS)

    print(f"\n开始测试 | ASR: {ASR_PROVIDER} | Bert: {'GPU' if BERT_DEVICE_ID >= 0 else 'CPU'}")
    print(f"并发数: {CONCURRENT_STREAMS} | ASR切片: {CHUNK_SECONDS}s")

    start_benchmark_time = time.time()
    files_processed = 0

    while len(task_queue) > 0 or any(s.is_active for s in slots):
        current_wall_time = time.perf_counter()

        # 填充 Slot
        for i in range(CONCURRENT_STREAMS):
            if not slots[i].is_active and len(task_queue) > 0:
                fname = task_queue.popleft()
                samples, duration = read_wav_file(fname)
                if samples is None: continue
                s = slots[i];
                s.reset();
                s.is_active = True
                s.stream = recognizer.create_stream()
                s.filename = fname;
                s.samples = samples;
                s.duration = duration
                s.start_time = current_wall_time;
                s.last_punct_time = current_wall_time

        # 喂数据
        for s in slots:
            if s.is_active and not s.input_finished:
                start = s.cursor
                end = min(start + chunk_samples, len(s.samples))
                chunk = s.samples[start:end]
                if len(chunk) < chunk_samples:
                    chunk = np.concatenate((chunk, np.zeros(chunk_samples - len(chunk), dtype=np.float32)))
                s.stream.accept_waveform(16000, np.ascontiguousarray(chunk))
                s.cursor += chunk_samples
                if start + chunk_samples >= len(s.samples):
                    s.stream.input_finished();
                    s.input_finished = True

        # ASR 解码
        ready = [s.stream for s in slots if s.is_active and recognizer.is_ready(s.stream)]
        if ready: recognizer.decode_streams(ready)

        loop_time = time.perf_counter()

        # 结果检查
        for s in slots:
            if not s.is_active: continue

            res = recognizer.get_result(s.stream)
            text = res if isinstance(res, str) else res.text
            text = text.strip()

            if text != s.raw_text:
                s.raw_text = text
                # 计算 TTFT (包含说话时间)
                if not s.has_output and len(text) > 0:
                    audio_elapsed = s.cursor / 16000.0
                    latency = loop_time - s.start_time
                    s.ttft = audio_elapsed + latency
                    s.has_output = True

            # Bert 修正
            if (loop_time - s.last_punct_time >= BERT_INTERVAL) and len(s.raw_text) > 0:
                _ = bert_restorer.restore(s.raw_text)
                s.last_punct_time = time.perf_counter()

            # 任务结束
            if s.input_finished and not recognizer.is_ready(s.stream):
                if len(s.raw_text) > 0: _ = bert_restorer.restore(s.raw_text)

                # RTF (机器处理时间 / 音频时间)
                process_duration = time.perf_counter() - s.start_time
                rtf = process_duration / s.duration
                ttft_val = s.ttft if s.ttft else (s.duration + process_duration)

                finished_results.append({"filename": os.path.basename(s.filename), "rtf": rtf, "ttft": ttft_val})
                files_processed += 1
                print(f"[{files_processed}] RTF: {rtf:.4f} | TTFT: {ttft_val:.4f}s")
                s.reset()

    if finished_results:
        avg_rtf = sum(r['rtf'] for r in finished_results) / len(finished_results)
        avg_ttft = sum(r['ttft'] for r in finished_results) / len(finished_results)
        print("\n" + "=" * 60)
        print(f"平均 RTF:  {avg_rtf:.4f}")
        print(f"平均 TTFT: {avg_ttft:.4f} s")
        print("=" * 60)


if __name__ == "__main__":
    main()
