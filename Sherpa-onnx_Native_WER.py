import os
import sys
import re
import argparse
import soundfile as sf
import sherpa_onnx
import jiwer

# ============================================================
# 常量与配置
# ============================================================
# Sherpa-ONNX 流式 Zipformer 模型目录 (假定与脚本同级或绝对路径)
MODEL_DIR = "./sherpa-onnx-streaming-zipformer-en-2023-06-21"
CHUNK_DURATION = 0.1  # 流式输入的音频分片长度（秒）


# ============================================================
# 1. 文本清洗函数集
# ============================================================
def normalize_standard_text(text):
    """通用的文本标准化（适用于 LibriSpeech 和 AMICorpus）"""
    if not text: return ""
    text = text.replace("\n", " ").replace("\r", " ").lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.strip().split())


def normalize_tedlium_text(text):
    """TEDLIUM 专用的文本标准化（需要去除尖括号标签）"""
    if not text: return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.strip().split())


# ============================================================
# 2. 数据集加载与路由工厂
# ============================================================
def load_dataset_tasks(dataset_name, dataset_path):
    """
    根据数据集类型，扫描目录并返回统一的任务列表。
    格式: [{"file_id": id, "audio_path": path, "ref_text": text}, ...]
    """
    task_list = []
    print(f"正在扫描 {dataset_name} 数据集目录: {dataset_path} ...")

    if dataset_name == "LibriSpeech":
        for root, dirs, files in os.walk(dataset_path):
            trans_files = [f for f in files if f.endswith(".trans.txt")]
            for trans_file in trans_files:
                with open(os.path.join(root, trans_file), "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            file_id, text = parts
                            audio_path = os.path.join(root, f"{file_id}.wav")
                            if not os.path.exists(audio_path):
                                audio_path = os.path.join(root, f"{file_id}.flac")
                            if os.path.exists(audio_path):
                                task_list.append({
                                    "file_id": file_id,
                                    "audio_path": audio_path,
                                    "ref_text": normalize_standard_text(text)
                                })

    elif dataset_name == "AMICorpus":
        for root, dirs, files in os.walk(dataset_path):
            wav_files = [f for f in files if f.endswith(".wav")]
            for wav_file in wav_files:
                file_id = os.path.splitext(wav_file)[0]
                txt_path = os.path.join(root, f"{file_id}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    task_list.append({
                        "file_id": file_id,
                        "audio_path": os.path.join(root, wav_file),
                        "ref_text": normalize_standard_text(text)
                    })

    elif dataset_name == "TEDLIUM":
        # 建立音频索引
        audio_map = {}
        for root, dirs, files in os.walk(dataset_path):
            for f in files:
                if f.endswith(".wav") or f.endswith(".flac"):
                    audio_map[os.path.splitext(f)[0]] = os.path.join(root, f)

        text_file_path = os.path.join(dataset_path, "text.txt")
        if os.path.exists(text_file_path):
            with open(text_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        file_id, raw_text = parts
                        if file_id in audio_map:
                            task_list.append({
                                "file_id": file_id,
                                "audio_path": audio_map[file_id],
                                "ref_text": normalize_tedlium_text(raw_text)
                            })

    # 过滤掉参考文本过短的任务（少于3个词），避免影响整体WER
    valid_tasks = [t for t in task_list if len(t["ref_text"].split()) >= 3]
    print(f"扫描完毕！找到有效测试音频 {len(valid_tasks)} 条。")
    return valid_tasks


# ============================================================
# 3. Sherpa-ONNX 在线识别器初始化
# ============================================================
def create_recognizer(model_dir):
    tokens_path = os.path.join(model_dir, "tokens.txt")
    encoder_path = decoder_path = joiner_path = ""

    if not os.path.exists(model_dir):
        print(f"❌ 错误: 模型目录不存在 {model_dir}")
        sys.exit(1)

    for f in os.listdir(model_dir):
        if f.startswith("encoder-") and f.endswith(".onnx"):
            encoder_path = os.path.join(model_dir, f)
        elif f.startswith("decoder-") and f.endswith(".onnx"):
            decoder_path = os.path.join(model_dir, f)
        elif f.startswith("joiner-") and f.endswith(".onnx"):
            joiner_path = os.path.join(model_dir, f)

    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path, encoder=encoder_path, decoder=decoder_path, joiner=joiner_path,
            num_threads=1, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
            enable_endpoint_detection=False, provider="cpu"
        )
    except Exception:
        # 兼容旧版本
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens_path, encoder=encoder_path, decoder=decoder_path, joiner=joiner_path,
            num_threads=1, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
            provider="cpu"
        )
    return recognizer


# ============================================================
# 4. 主执行流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()

    print("-" * 50)
    print(f"初始化 Sherpa-ONNX 模型 (当前数据集: {args.dataset_name})...")
    recognizer = create_recognizer(MODEL_DIR)
    print("模型加载完成。")
    print("-" * 50)

    # 加载统一格式的测试任务
    task_list = load_dataset_tasks(args.dataset_name, args.dataset_path)
    total_files = len(task_list)

    if total_files == 0:
        print("未找到有效测试数据，程序退出。")
        return

    total_distance = 0
    total_ref_words = 0
    processed_count = 0

    print("=" * 60)
    print(f"开始 {args.dataset_name} WER 指标评测...")
    print("=" * 60)

    for task in task_list:
        audio_path = task["audio_path"]
        ref_text = task["ref_text"]
        file_id = task["file_id"]

        try:
            audio, sample_rate = sf.read(audio_path, dtype="float32")
        except Exception as e:
            print(f"读取音频失败 {audio_path}: {e}")
            continue

        if sample_rate != 16000 or len(audio) < int(sample_rate * 0.05):
            continue

        # --- 流式推理核心 ---
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

        # 将预测结果标准化（直接借用 LibriSpeech 的标准化逻辑即可）
        hyp_clean = normalize_standard_text(hyp_text_raw)

        # --- WER 计算 ---
        try:
            out = jiwer.process_words(ref_text, hyp_clean)
            curr_dist = out.substitutions + out.deletions + out.insertions
            curr_wer = out.wer
            curr_len = len(out.references[0]) if out.references else 0

            total_distance += curr_dist
            total_ref_words += curr_len
            processed_count += 1
            duration_sec = len(audio) / sample_rate

            # 漂亮的控制台输出
            print(f"[{processed_count}/{total_files}] ID: {file_id}")
            print(f"  --> Dur: {duration_sec:.1f}s | Words: {curr_len} | Dist: {curr_dist} | WER: {curr_wer:.4f}")

        except Exception as e:
            print(f"评测计算错误 {file_id}: {e}")

    # ============================================================
    # 结果汇总输出
    # ============================================================
    print("\n" + "=" * 50)
    print(f"🎯 {args.dataset_name} 数据集 WER 测试完成")
    print(f"成功处理文件数: {processed_count}")

    if total_ref_words > 0:
        avg_wer = total_distance / total_ref_words
        print(f"总编辑距离 (Distance): {total_distance}")
        print(f"总参考单词数 (Ref Words): {total_ref_words}")
        print("-" * 25)
        print(f"🚀 平均词错误率 (Average WER): {avg_wer:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()