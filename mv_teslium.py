import os
import subprocess

SPH_DIR = "/opt/Audio_Datasets/TEDLIUM/sph"
STM_DIR = "/opt/Audio_Datasets/TEDLIUM/stm"
OUT_WAV_DIR = "/opt/Audio_Datasets/TEDLIUM_WAV"
TEXT_OUT = "/opt/Audio_Datasets/TEDLIUM_WAV/text.txt"

os.makedirs(OUT_WAV_DIR, exist_ok=True)

wav_index = 0
text_lines = []

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

for stm_file in sorted(os.listdir(STM_DIR)):
    if not stm_file.endswith(".stm"):
        continue

    stm_path = os.path.join(STM_DIR, stm_file)

    with open(stm_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(";;"):
                continue

            parts = line.strip().split()
            if len(parts) < 6:
                continue

            rec_id = parts[0]
            start = float(parts[3])
            end = float(parts[4])
            text = " ".join(parts[6:])

            sph_path = os.path.join(SPH_DIR, rec_id + ".sph")
            if not os.path.exists(sph_path):
                continue

            subdir = f"{wav_index // 100:04d}"
            out_dir = os.path.join(OUT_WAV_DIR, subdir)
            os.makedirs(out_dir, exist_ok=True)

            wav_name = f"{wav_index:06d}.wav"
            wav_path = os.path.join(out_dir, wav_name)

            cmd = (
                f"sox {sph_path} {wav_path} "
                f"trim {start} ={end}"
            )
            run(cmd)

            text_lines.append(f"{wav_index:06d} {text}\n")
            wav_index += 1

with open(TEXT_OUT, "w", encoding="utf-8") as f:
    f.writelines(text_lines)

print(f"Done. Generated {wav_index} wav files.")
