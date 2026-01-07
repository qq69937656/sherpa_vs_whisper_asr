import os
import sys
import re
import soundfile as sf
import sherpa_onnx
import jiwer

# ================= 配置区域 =================

# AMICorpus 数据集根目录
AMI_DIR = "/opt/Audio_Datasets/AMICorpus"

# 模型目录
MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"
CHUNK_DURATION = 0.1

# 最大处理文件数 (只测试前10个)
MAX_FILES = 10


# ================= 1. 文本清洗函数 =================

def normalize_ami_text(text):
    """
    AMICorpus 标准化：
    1. 替换回车/换行
    2. 转小写
    3. 去除标点
    4. 返回空格分隔的单词串
    """
    if not text:
        return ""

    # 替换换行符
    text = text.replace("\n", " ").replace("\r", " ")

    # 转小写
    text = text.lower()

    # 去除标点 (只留字母数字和空格)
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # 规范化空格
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
            enable_endpoint_detection=False,
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
    print(f"初始化 Sherpa-onnx (AMICorpus 快速评测, Limit={MAX_FILES})...")
    recognizer = create_recognizer(MODEL_DIR)
    print("模型加载完成。")
    print("-" * 50)

    if not os.path.exists(AMI_DIR):
        print(f"错误: 数据集目录不存在 {AMI_DIR}")
        return

    # 统计变量
    total_distance = 0
    total_ref_words = 0
    processed_count = 0  # 已成功处理的文件数

    stop_processing = False  # 停止标志

    print(f"开始遍历: {AMI_DIR}")

    # 使用 os.walk 递归
    for root, dirs, files in os.walk(AMI_DIR):
        if stop_processing:
            break

        wav_files = [f for f in files if f.endswith(".wav")]

        for wav_filename in wav_files:
            if stop_processing:
                break

            wav_path = os.path.join(root, wav_filename)
            txt_filename = os.path.splitext(wav_filename)[0] + ".txt"
            txt_path = os.path.join(root, txt_filename)

            # 检查 txt
            if not os.path.exists(txt_path):
                continue

            # 1. 读取文本
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except Exception as e:
                print(f"读取文本失败 {txt_path}: {e}")
                continue

            ref_clean = normalize_ami_text(raw_text)

            # 单词数过滤
            ref_words_list = ref_clean.split()
            ref_word_count = len(ref_words_list)

            if ref_word_count < 3:
                # 忽略过短文件，不计入 processed_count
                continue

            # 2. 读取音频
            try:
                audio, sample_rate = sf.read(wav_path, dtype="float32")
            except Exception as e:
                print(f"读取音频失败 {wav_path}: {e}")
                continue

            if sample_rate != 16000:
                print(f"跳过非16k音频: {wav_filename}")
                continue

            if len(audio) < int(sample_rate * 0.05):
                continue

            # 计算时长 (秒)
            duration_sec = len(audio) / sample_rate

            # 3. 推理
            stream = recognizer.create_stream()
            chunk_size = int(sample_rate * CHUNK_DURATION)

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i: i + chunk_size]
                stream.accept_waveform(sample_rate, chunk)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            result_obj = recognizer.get_result(stream)
            hyp_text_raw = result_obj if isinstance(result_obj, str) else getattr(result_obj, 'text', str(result_obj))

            # 4. 评测
            hyp_clean = normalize_ami_text(hyp_text_raw)

            try:
                out = jiwer.process_words(ref_clean, hyp_clean)

                curr_dist = out.substitutions + out.deletions + out.insertions
                curr_wer = out.wer
                curr_len = len(out.references[0]) if out.references else 0

                total_distance += curr_dist
                total_ref_words += curr_len
                processed_count += 1

                # 输出详细信息：文件名、WER、编辑距离、单词数、时长
                print(f"[{processed_count}/{MAX_FILES}] File: {wav_filename}")
                print(f"  Duration: {duration_sec:.2f}s | Words: {curr_len} | Dist: {curr_dist} | WER: {curr_wer:.4f}")

                # 达到数量限制，设置停止标志
                if processed_count >= MAX_FILES:
                    stop_processing = True

            except Exception as e:
                print(f"评测计算错误 {wav_filename}: {e}")

    # ================= 结果汇总 =================
    print("\n" + "=" * 50)
    print("测试结束 (Limit Reached)")
    print(f"处理文件总数: {processed_count}")

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