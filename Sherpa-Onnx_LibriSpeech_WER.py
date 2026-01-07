import os
import sys
import re
import soundfile as sf
import sherpa_onnx
import jiwer

# ================= 配置区域 =================

LIBRISPEECH_DIR = "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean"
MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"
CHUNK_DURATION = 0.1


# ================= 文本标准化函数 =================

def normalize_text(text):
    """
    手动实现文本标准化：
    1. 转小写
    2. 去除标点符号 (只保留字母、数字、空格)
    3. 按空格分词，返回单词列表
    """
    if not text:
        return []

    # 1. 转小写
    text = text.lower()

    # 2. 正则去标点：替换所有 [非单词字符 且 非空格] 为空
    # \w 匹配 [a-zA-Z0-9_]
    text = re.sub(r"[^\w\s]", "", text)

    # 3. 去除首尾空格并分词
    words = text.strip().split()
    return words


# ================= Sherpa-onnx 初始化 =================

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
            enable_endpoint_detection=False,  # 必须关闭，否则短音频会报错
            provider="cuda"
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
            provider="cuda"
        )
    return recognizer


# ================= 主程序 =================

def main():
    print("-" * 50)
    print("正在初始化 Sherpa-onnx...")
    recognizer = create_recognizer(MODEL_DIR)
    print("模型加载完成。")
    print("-" * 50)

    total_distance = 0
    total_ref_words = 0
    file_count = 0

    if not os.path.exists(LIBRISPEECH_DIR):
        print(f"错误: 数据集目录不存在 {LIBRISPEECH_DIR}")
        return

    print(f"开始遍历数据集: {LIBRISPEECH_DIR}")

    for root, dirs, files in os.walk(LIBRISPEECH_DIR):
        trans_files = [f for f in files if f.endswith(".trans.txt")]

        for trans_file in trans_files:
            trans_path = os.path.join(root, trans_file)
            ref_map = {}

            # 读取 trans.txt
            try:
                with open(trans_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            file_id, text = parts
                            ref_map[file_id] = text
            except Exception as e:
                print(f"读取文本文件失败 {trans_path}: {e}")
                continue

            for file_id, ref_text in ref_map.items():
                audio_filename = f"{file_id}.wav"
                audio_path = os.path.join(root, audio_filename)

                # 兼容性: 找 flac
                if not os.path.exists(audio_path):
                    flac_path = os.path.join(root, f"{file_id}.flac")
                    if os.path.exists(flac_path):
                        audio_path = flac_path
                    else:
                        continue

                try:
                    audio, sample_rate = sf.read(audio_path, dtype="float32")
                except Exception as e:
                    print(f"读取音频失败 {audio_path}: {e}")
                    continue

                if sample_rate != 16000:
                    print(f"警告: 采样率非16k ({sample_rate}Hz) - 跳过 {audio_filename}")
                    continue

                # 跳过极短音频
                if len(audio) < int(sample_rate * 0.05):
                    continue

                # ================= 核心推理逻辑 =================
                stream = recognizer.create_stream()
                chunk_size = int(sample_rate * CHUNK_DURATION)  # 1600 samples

                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i: i + chunk_size]
                    stream.accept_waveform(sample_rate, chunk)

                    # 检查是否积攒了足够的帧数再解码
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)

                stream.input_finished()

                # 处理剩余尾部
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                # 获取结果 (兼容 str 和 object 返回值)
                result_obj = recognizer.get_result(stream)
                hyp_text_raw = result_obj if isinstance(result_obj, str) else getattr(result_obj, 'text',
                                                                                      str(result_obj))
                # ===============================================

                # ================= 评测逻辑 (修复版) =================

                # 1. 预处理为单词列表
                ref_words_list = normalize_text(ref_text)
                hyp_words_list = normalize_text(hyp_text_raw)

                # 2. 【关键修复】将列表拼回字符串
                # jiwer 如果接收到 list of strings，会把它当做 batch (多句话) 处理。
                # 拼接回字符串后，jiwer 才会把它当做 single sentence 处理。
                ref_str_clean = " ".join(ref_words_list)
                hyp_str_clean = " ".join(hyp_words_list)

                # 3. 计算指标
                # 由于我们已经手动 normalize 过了，这里直接计算即可
                try:
                    out = jiwer.process_words(ref_str_clean, hyp_str_clean)

                    curr_dist = out.substitutions + out.deletions + out.insertions
                    curr_wer = out.wer

                    # 这里的 references 是 [[word1, word2...]]
                    curr_len = len(out.references[0]) if out.references else 0

                    total_distance += curr_dist
                    total_ref_words += curr_len
                    file_count += 1

                    print(
                        f"File: {os.path.basename(audio_path)} | WER: {curr_wer:.4f} | Dist: {curr_dist} | RefWords: {curr_len}")

                except Exception as e:
                    print(f"Jiwer 计算错误 {audio_filename}: {e}")
                    # 如果空串导致报错，做简单处理
                    if len(ref_words_list) == 0:
                        print(f"  Ref 为空，Hyp 长度: {len(hyp_words_list)}")

    # ================= 结果汇总 =================
    print("\n" + "=" * 50)
    print("评测完成")
    print(f"处理文件总数: {file_count}")

    if total_ref_words > 0:
        avg_wer = total_distance / total_ref_words
        print(f"总编辑距离: {total_distance}")
        print(f"总参考单词数: {total_ref_words}")
        print("-" * 25)
        print(f"平均错词率 (Average WER): {avg_wer:.2%}")
    else:
        print("未处理任何有效文件。")
    print("=" * 50)


if __name__ == "__main__":
    main()
