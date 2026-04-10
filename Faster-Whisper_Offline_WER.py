import os
import sys
import re
import argparse
import jiwer
from faster_whisper import WhisperModel

# ============================================================
# 常量与配置
# ============================================================
MODEL_SIZE = "medium.en"
MIN_WORD_COUNT = 3  # 过滤有效单词数过少的音频（如仅包含<SIL>）


# ============================================================
# 1. 文本清洗函数集
# ============================================================
def normalize_text(text, dataset_name):
    """
    统一的文本标准化函数，根据数据集类型动态适配清洗规则
    """
    if not text:
        return ""

    # 统一换行符替换与转小写
    text = text.replace("\n", " ").replace("\r", " ").lower()

    # 针对 TEDLIUM 和 AMICorpus 移除特定的标记（如 <SIL>, <NOISE>, {breath}）
    if dataset_name in ["TEDLIUM", "AMICorpus"]:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[{\[][^}\]]*[}\]]", " ", text)

    # 移除标点符号（仅保留字母、数字和空格）
    text = re.sub(r"[^\w\s]", "", text)

    # 规范空格
    return " ".join(text.strip().split())


# ============================================================
# 2. 数据集加载路由工厂
# ============================================================
def load_dataset_tasks(dataset_name, dataset_path):
    """
    根据数据集类型扫描目录，返回统一格式的任务列表
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
                            file_id, raw_text = parts
                            audio_path = os.path.join(root, f"{file_id}.wav")
                            if not os.path.exists(audio_path):
                                audio_path = os.path.join(root, f"{file_id}.flac")
                            if os.path.exists(audio_path):
                                ref_clean = normalize_text(raw_text, dataset_name)
                                task_list.append({
                                    "file_id": file_id,
                                    "audio_path": audio_path,
                                    "ref_text": ref_clean
                                })

    elif dataset_name == "AMICorpus":
        for root, dirs, files in os.walk(dataset_path):
            wav_files = [f for f in files if f.endswith(".wav")]
            for wav_file in wav_files:
                file_id = os.path.splitext(wav_file)[0]
                txt_path = os.path.join(root, f"{file_id}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                    ref_clean = normalize_text(raw_text, dataset_name)
                    task_list.append({
                        "file_id": file_id,
                        "audio_path": os.path.join(root, wav_file),
                        "ref_text": ref_clean
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
                            ref_clean = normalize_text(raw_text, dataset_name)
                            task_list.append({
                                "file_id": file_id,
                                "audio_path": audio_map[file_id],
                                "ref_text": ref_clean
                            })

    # 过滤掉清洗后文本过短的任务（如仅包含噪声标签的片段）
    initial_count = len(task_list)
    valid_tasks = [t for t in task_list if len(t["ref_text"].split()) >= MIN_WORD_COUNT]
    skipped_count = initial_count - len(valid_tasks)

    print(
        f"扫描完毕！找到原始文件 {initial_count} 个，过滤无效短文本 {skipped_count} 个，最终有效评测任务 {len(valid_tasks)} 个。")
    return valid_tasks


# ============================================================
# 3. Faster-Whisper 初始化
# ============================================================
def create_recognizer(model_size):
    try:
        # 使用 GPU 及 float16 精度加速离线推理
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        return model
    except Exception as e:
        print(f"❌ 错误: 模型加载失败 - {e}")
        sys.exit(1)


# ============================================================
# 4. 主执行流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, help="数据集类型")
    parser.add_argument("--dataset_path", required=True, help="数据集物理路径")
    args = parser.parse_args()

    print("-" * 50)
    print(f"初始化 Faster-Whisper 离线评测 (模型: {MODEL_SIZE})...")
    recognizer = create_recognizer(MODEL_SIZE)
    print("模型加载完成。")
    print("-" * 50)

    # 加载任务列表
    task_list = load_dataset_tasks(args.dataset_name, args.dataset_path)
    total_files = len(task_list)

    if total_files == 0:
        print("未找到有效测试数据，程序退出。")
        return

    # 评测统计量
    total_distance = 0
    total_ref_words = 0
    processed_count = 0

    print("=" * 60)
    print(f"🎯 开始执行 {args.dataset_name} 全量音频离线 WER 评测")
    print("=" * 60)

    for task in task_list:
        audio_path = task["audio_path"]
        ref_text_clean = task["ref_text"]
        file_id = task["file_id"]

        # ================= 核心离线推理逻辑 =================
        try:
            # 采用 Faster-Whisper 内置的高效 VAD 与长音频处理策略
            segments, info = recognizer.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=True
            )

            # 拼接识别文本
            hyp_segments_list = [segment.text for segment in segments]
            hyp_text_raw = " ".join(hyp_segments_list)

        except Exception as e:
            print(f"推理失败 {audio_path}: {e}")
            continue

        # ================= WER 对齐计算 =================
        hyp_text_clean = normalize_text(hyp_text_raw, args.dataset_name)

        try:
            out = jiwer.process_words(ref_text_clean, hyp_text_clean)

            curr_dist = out.substitutions + out.deletions + out.insertions
            curr_wer = out.wer
            curr_len = len(out.references[0]) if out.references else 0

            total_distance += curr_dist
            total_ref_words += curr_len
            processed_count += 1

            # 进度打印
            print(f"[{processed_count}/{total_files}] ID: {file_id[:25]:<25} | "
                  f"Words: {curr_len:<4} | Dist: {curr_dist:<3} | WER: {curr_wer:.4f}")

        except Exception as e:
            print(f"Jiwer 计算错误 {file_id}: {e}")

    # ============================================================
    # 结果汇总输出
    # ============================================================
    print("\n" + "=" * 50)
    print(f"✅ {args.dataset_name} 数据集评测完成")
    print(f"成功处理文件数: {processed_count}")

    if total_ref_words > 0:
        avg_wer = total_distance / total_ref_words
        print(f"总编辑距离 (Distance): {total_distance}")
        print(f"总参考单词数 (Ref Words): {total_ref_words}")
        print("-" * 25)
        print(f"🚀 平均词错误率 (Average WER): {avg_wer:.2%}")
    else:
        print("未生成任何有效统计数据。")
    print("=" * 50)


if __name__ == "__main__":
    main()