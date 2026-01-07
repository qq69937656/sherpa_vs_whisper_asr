import os
import sys
import re
import soundfile as sf
import sherpa_onnx
import jiwer

# ================= 配置区域 =================

# TEDLIUM 数据集根目录
# 结构:
#   /root_dir/text.txt
#   /root_dir/legacy/train/sph/... (或其他深层子目录结构)
TEDLIUM_DIR = "/opt/Audio_Datasets/TEDLIUM_WAV"

# 模型目录
MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"
CHUNK_DURATION = 0.1


# ================= 1. TEDLIUM 文本清洗函数 =================

def normalize_tedlium_text(text):
    """
    针对 TEDLIUM 的标准化处理：
    1. 去除 <COMMA>, <SIL>, <OTHER> 等标签
    2. 转小写
    3. 去除标点和特殊符号
    4. 返回纯净的单词串
    """
    if not text:
        return ""

    # 1. 去除尖括号标签 <...>
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. 转小写
    text = text.lower()

    # 3. 去除所有非字母数字字符 (只保留 a-z, 0-9 和空格)
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # 4. 规范化空格
    words = text.strip().split()
    return " ".join(words)


# ================= 2. 建立音频文件索引 =================

def index_audio_files(root_dir):
    """
    递归遍历目录，建立 {file_id: full_path} 的映射
    """
    audio_map = {}
    print(f"正在建立音频文件索引: {root_dir} ...")

    count = 0
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            # 支持 wav 和 flac
            if f.endswith(".wav") or f.endswith(".flac"):
                # 获取不带后缀的文件名作为 ID
                file_id = os.path.splitext(f)[0]
                full_path = os.path.join(root, f)
                audio_map[file_id] = full_path
                count += 1

    print(f"索引建立完成，共找到 {count} 个音频文件。")
    return audio_map


# ================= 3. Sherpa-onnx 初始化 =================

def create_recognizer(model_dir):
    tokens_path = os.path.join(model_dir, "tokens.txt")
    encoder_path = ""
    decoder_path = ""
    joiner_path = ""

    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在 {model_dir}")
        sys.exit(1)

    for f in os.listdir(model_dir):
        if f.startswith("encoder-") and f.endswith(".onnx"):
            encoder_path = os.path.join(model_dir, f)
        elif f.startswith("decoder-") and f.endswith(".onnx"):
            decoder_path = os.path.join(model_dir, f)
        elif f.startswith("joiner-") and f.endswith(".onnx"):
            joiner_path = os.path.join(model_dir, f)

    if not (os.path.exists(tokens_path) and encoder_path and decoder_path and joiner_path):
        print(f"错误: 模型文件缺失。\nTokens: {tokens_path}")
        sys.exit(1)

    print(
        f"加载模型:\n  Enc: {os.path.basename(encoder_path)}\n  Dec: {os.path.basename(decoder_path)}\n  Join: {os.path.basename(joiner_path)}")

    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,  # 必须关闭
            provider="cpu"
        )
    except Exception as e:
        print(f"初始化警告: {e}, 尝试默认参数...")
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu"
        )
    return recognizer


# ================= 4. 主程序 =================

def main():
    print("-" * 50)
    print("初始化 Sherpa-onnx (TEDLIUM 评测)...")
    recognizer = create_recognizer(MODEL_DIR)
    print("模型加载完成。")
    print("-" * 50)

    # 1. 建立音频索引 (递归查找所有子目录)
    if not os.path.exists(TEDLIUM_DIR):
        print(f"错误: 数据集目录不存在 {TEDLIUM_DIR}")
        return

    audio_map = index_audio_files(TEDLIUM_DIR)

    # 2. 读取文本索引文件
    text_file_path = os.path.join(TEDLIUM_DIR, "text.txt")
    if not os.path.exists(text_file_path):
        print(f"错误: 找不到文本索引文件 {text_file_path}")
        return

    # 统计变量
    total_distance = 0
    total_ref_words = 0
    file_count = 0
    ignored_short_count = 0
    missing_audio_count = 0

    print(f"开始评测: {text_file_path}")

    with open(text_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 分割 ID 和 文本
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue

            file_id = parts[0]
            raw_ref_text = parts[1]

            # --- 文本清洗与过滤 ---
            ref_clean = normalize_tedlium_text(raw_ref_text)

            # 单词数计算
            ref_words_list = ref_clean.split()
            ref_word_count = len(ref_words_list)

            # 过滤小于3个单词的样本
            if ref_word_count < 3:
                ignored_short_count += 1
                continue

            # --- 查找音频文件 ---
            # 直接从建立好的索引中获取路径
            if file_id not in audio_map:
                # print(f"警告: 未找到音频文件 ID: {file_id}")
                missing_audio_count += 1
                continue

            audio_path = audio_map[file_id]

            # --- 读取音频 ---
            try:
                audio, sample_rate = sf.read(audio_path, dtype="float32")
            except Exception as e:
                print(f"读取失败 {audio_path}: {e}")
                continue

            if sample_rate != 16000:
                print(f"跳过非16k音频: {file_id} ({sample_rate}Hz)")
                continue

            # 极短音频保护
            if len(audio) < int(sample_rate * 0.05):
                continue

            # --- 核心推理逻辑 ---
            stream = recognizer.create_stream()
            chunk_size = int(sample_rate * CHUNK_DURATION)  # 1600 samples

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i: i + chunk_size]
                stream.accept_waveform(sample_rate, chunk)

                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            # 获取结果
            result_obj = recognizer.get_result(stream)
            hyp_text_raw = result_obj if isinstance(result_obj, str) else getattr(result_obj, 'text', str(result_obj))

            # --- 结果清洗与 WER 计算 ---
            hyp_clean = normalize_tedlium_text(hyp_text_raw)

            try:
                out = jiwer.process_words(ref_clean, hyp_clean)

                curr_dist = out.substitutions + out.deletions + out.insertions
                curr_wer = out.wer
                curr_len = len(out.references[0]) if out.references else 0

                total_distance += curr_dist
                total_ref_words += curr_len
                file_count += 1

                print(f"ID: {file_id} | WER: {curr_wer:.4f} | Dist: {curr_dist} | RefWords: {curr_len}")

            except Exception as e:
                print(f"评测计算错误 {file_id}: {e}")

    # ================= 结果汇总 =================
    print("\n" + "=" * 50)
    print("TEDLIUM 评测完成")
    print(f"有效处理文件数: {file_count}")
    print(f"忽略的过短文件数(Words < 3): {ignored_short_count}")
    print(f"未找到音频的文件数: {missing_audio_count}")

    if total_ref_words > 0:
        avg_wer = total_distance / total_ref_words
        print(f"总编辑距离: {total_distance}")
        print(f"总参考单词数: {total_ref_words}")
        print("-" * 25)
        print(f"平均错词率 (Average WER): {avg_wer:.2%}")
    else:
        print("未生成任何有效统计数据。")
    print("=" * 50)


if __name__ == "__main__":
    main()