import os
import sys
import re
import soundfile as sf
import sherpa_onnx
import jiwer

# ================= 配置区域 =================

# GigaSpeech 数据集根目录
# 结构:
#   /root_dir/text  (索引文件)
#   /root_dir/wav/  (音频目录)
GIGASPEECH_DIR = "/opt/Audio_Datasets/GigaSpeech_Test_WAV"

# 模型目录
MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"
CHUNK_DURATION = 0.1


# ================= 1. GigaSpeech 文本清洗函数 =================

def normalize_gigaspeech_text(text):
    """
    针对 GigaSpeech 的标准化处理：
    1. 去除 <COMMA>, <SIL>, <OTHER> 等标签
    2. 转小写
    3. 去除标点和特殊符号
    4. 返回纯净的单词串
    """
    if not text:
        return ""

    # 1. 去除尖括号标签 <...>
    # 例如: "HELLO <COMMA> WORLD" -> "HELLO  WORLD"
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. 转小写
    text = text.lower()

    # 3. 去除所有非字母数字字符 (只保留 a-z, 0-9 和空格)
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # 4. 规范化空格 (去除首尾空格，中间多个空格变一个)
    words = text.strip().split()
    return " ".join(words)


# ================= 2. Sherpa-onnx 初始化 =================

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


# ================= 3. 主程序 =================

def main():
    print("-" * 50)
    print("初始化 Sherpa-onnx (GigaSpeech 评测)...")
    recognizer = create_recognizer(MODEL_DIR)
    print("模型加载完成。")
    print("-" * 50)

    # 路径定义
    text_file_path = os.path.join(GIGASPEECH_DIR, "text")
    wav_dir_path = os.path.join(GIGASPEECH_DIR, "wav")

    if not os.path.exists(text_file_path):
        print(f"错误: 找不到索引文件 {text_file_path}")
        return
    if not os.path.exists(wav_dir_path):
        print(f"错误: 找不到 wav 目录 {wav_dir_path}")
        return

    # 统计变量
    total_distance = 0
    total_ref_words = 0
    file_count = 0
    ignored_short_count = 0  # 统计因过短被忽略的数量

    print(f"开始处理数据集: {GIGASPEECH_DIR}")

    # 读取并解析 text 文件
    # 格式通常为: FILE_ID  TEXT CONTENT...
    with open(text_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 分割 ID 和 文本
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                # 只有ID没有文本，跳过
                continue

            file_id = parts[0]
            raw_ref_text = parts[1]

            # 1. 预处理参考文本
            ref_clean = normalize_gigaspeech_text(raw_ref_text)

            # 将文本拆分为单词列表以计算长度
            ref_words_list = ref_clean.split()
            ref_word_count = len(ref_words_list)

            # 2. 【核心修改】过滤逻辑
            # 如果去标签后文本为空，或者单词数小于 3，则忽略
            if ref_word_count < 3:
                # print(f"忽略短文本 (Words={ref_word_count}): {file_id}")
                ignored_short_count += 1
                continue

            # 3. 定位音频文件
            audio_path = os.path.join(wav_dir_path, f"{file_id}.wav")

            if not os.path.exists(audio_path):
                print(f"警告: 音频文件丢失 {audio_path}")
                continue

            # 4. 读取音频
            try:
                audio, sample_rate = sf.read(audio_path, dtype="float32")
            except Exception as e:
                print(f"读取失败 {audio_path}: {e}")
                continue

            if sample_rate != 16000:
                print(f"跳过非16k音频: {file_id} ({sample_rate}Hz)")
                continue

            # 这里的音频长度过滤可以稍微放宽，因为我们已经按单词数过滤了
            # 但为了保护模型不崩溃，还是保留极短音频过滤
            if len(audio) < int(sample_rate * 0.05):
                continue

            # 5. 核心推理逻辑
            stream = recognizer.create_stream()
            chunk_size = int(sample_rate * CHUNK_DURATION)  # 1600 samples

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i: i + chunk_size]
                stream.accept_waveform(sample_rate, chunk)

                # 检查 is_ready 防止越界
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            # 获取结果
            result_obj = recognizer.get_result(stream)
            hyp_text_raw = result_obj if isinstance(result_obj, str) else getattr(result_obj, 'text', str(result_obj))

            # 6. 标准化预测文本
            hyp_clean = normalize_gigaspeech_text(hyp_text_raw)

            # 7. 计算 WER
            try:
                # 传入 pure string，jiwer 自动按空格分词计算
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
    print("GigaSpeech 评测完成")
    print(f"有效处理文件数: {file_count}")
    print(f"忽略的过短文件数(Words < 3): {ignored_short_count}")

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