import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# =========================
# 配置
# =========================
OUTPUT_ROOT = "/opt/Audio_Datasets/GigaSpeech_XS_WAV"
WAV_ROOT = os.path.join(OUTPUT_ROOT, "wav")
TEXT_FILE = os.path.join(OUTPUT_ROOT, "transcripts.txt")

os.makedirs(WAV_ROOT, exist_ok=True)

# =========================
# 加载 XS-test（只加载 test）
# =========================
dataset = load_dataset(
    "speechcolab/gigaspeech",
    "xs",
    split="test",
    use_auth_token=True
)

print(f"XS-test segments: {len(dataset)}")

# =========================
# 导出
# =========================
with open(TEXT_FILE, "w", encoding="utf-8") as txt_f:
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # 000000 这种格式
        utt_id = f"{idx:06d}"

        # 每 100 个 wav 一个目录
        subdir = utt_id[:4]
        out_dir = os.path.join(WAV_ROOT, subdir)
        os.makedirs(out_dir, exist_ok=True)

        wav_path = os.path.join(out_dir, f"{utt_id}.wav")

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        text = sample["text"].strip()

        # 保存 wav
        sf.write(wav_path, audio, sr)

        # 保存文本
        txt_f.write(f"{utt_id}\t{text}\n")

print("✅ XS-test 导出完成")
print(f"wav 目录: {WAV_ROOT}")
print(f"文本文件: {TEXT_FILE}")
