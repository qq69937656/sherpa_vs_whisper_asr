import os
import sys
import re
import jiwer
from faster_whisper import WhisperModel

# ================= 配置区域 =================

# 数据集根目录
TEDLIUM_DIR = "/opt/Audio_Datasets/TEDLIUM_WAV"
MODEL_SIZE = "medium.en"


# ================= 辅助函数 =================

def normalize_text(text):
    """
    文本归一化：
    1. 转小写
    2. 去除 TEDLIUM 常见的特殊标记，如 <unk>, {breath}, [noise] 等
    3. 去除标点符号
    4. 仅保留单词和空格
    """
    if not text:
        return []

    text = text.lower()

    # 【新增】去除 <...>, {...}, [...] 格式的非语音标记
    # TEDLIUM 数据集中常包含 <unk>, {breath}, [noise] 等
    text = re.sub(r"[<{\[][^>}\]]*[>}\]]", "", text)

    # 去除标点符号 (保留字母、数字、空格)
    text = re.sub(r"[^\w\s]", "", text)

    # 替换多个空格为一个空格，并分割
    words = text.strip().split()
    return words


def load_reference_map(dataset_dir):
    """
    在根目录下寻找 .txt 文件并加载参考文本
    格式假设：Filename_ID Transcript_Content
    """
    ref_map = {}
    txt_files = [f for f in os.listdir(dataset_dir) if f.endswith(".txt")]

    if not txt_files:
        print(f"错误: 在 {dataset_dir} 下未找到 .txt 参考文本文件")
        sys.exit(1)

    # 默认取第一个 txt 文件（通常只有一个）
    trans_file_path = os.path.join(dataset_dir, txt_files[0])
    print(f"正在加载参考文本: {trans_file_path}")

    try:
        with open(trans_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 按照第一个空格分割：第一部分是ID，剩余部分是文本
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    file_id, text = parts
                    ref_map[file_id] = text
    except Exception as e:
        print(f"读取参考文本文件失败: {e}")
        sys.exit(1)

    print(f"已加载 {len(ref_map)} 条参考文本。")
    return ref_map


# ================= Faster-Whisper 初始化 =================

def create_recognizer(model_size):
    try:
        # 使用 float16 + cuda 进行加速
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)


# ================= 主程序 =================

def main():
    print("-" * 50)
    print(f"正在加载模型 {MODEL_SIZE} (TEDLIUM 全量推理模式)...")
    recognizer = create_recognizer(MODEL_SIZE)
    print("模型加载完成。")
    print("-" * 50)

    if not os.path.exists(TEDLIUM_DIR):
        print(f"错误: 数据集目录不存在 {TEDLIUM_DIR}")
        return

    # 1. 加载参考文本字典
    ref_map = load_reference_map(TEDLIUM_DIR)

    total_distance = 0
    total_ref_words = 0
    file_count = 0

    print(f"开始遍历音频文件: {TEDLIUM_DIR}")

    # 2. 遍历目录寻找 wav 文件
    for root, dirs, files in os.walk(TEDLIUM_DIR):
        for filename in files:
            # 兼容 wav 和 flac
            if filename.lower().endswith(('.wav', '.flac')):
                audio_path = os.path.join(root, filename)

                # 提取文件名主体 (不含扩展名) 作为 ID
                # 假设：文件名 "JaneDoe_2015.wav" -> ID "JaneDoe_2015"
                file_id = os.path.splitext(filename)[0]

                # 检查是否存在于参考文本中
                if file_id not in ref_map:
                    # 如果找不到，有些数据集可能是通过文件名前缀匹配的
                    # 如果需要更复杂的匹配逻辑（如 split('_')[0]），请在此处修改
                    # print(f"警告: 未找到对应的参考文本 ID: {file_id}，跳过。")
                    continue

                ref_text = ref_map[file_id]

                # ================= 核心推理逻辑 =================
                try:
                    segments, info = recognizer.transcribe(
                        audio_path,
                        beam_size=5,  # 提高准确率
                        language="en",
                        vad_filter=True,  # 过滤非人声
                        condition_on_previous_text=True
                    )

                    hyp_segments_list = [segment.text for segment in segments]
                    hyp_text_raw = " ".join(hyp_segments_list)

                except Exception as e:
                    print(f"推理失败 {filename}: {e}")
                    continue
                # ===============================================

                # ================= 评测逻辑 =================
                ref_words_list = normalize_text(ref_text)
                hyp_words_list = normalize_text(hyp_text_raw)

                ref_str_clean = " ".join(ref_words_list)
                hyp_str_clean = " ".join(hyp_words_list)

                # 如果参考文本清洗后为空（例如只有噪声标记），则跳过计算 WER
                if not ref_str_clean:
                    continue

                try:
                    out = jiwer.process_words(ref_str_clean, hyp_str_clean)

                    curr_dist = out.substitutions + out.deletions + out.insertions
                    curr_wer = out.wer
                    curr_len = len(out.references[0]) if out.references else 0

                    total_distance += curr_dist
                    total_ref_words += curr_len
                    file_count += 1

                    print(
                        f"File: {filename} | WER: {curr_wer:.4f} | Dist: {curr_dist} | RefWords: {curr_len}"
                    )

                except Exception as e:
                    print(f"Jiwer 计算错误 {filename}: {e}")

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
        print("未处理任何有效文件 (或参考文本匹配失败)。")
    print("=" * 50)


if __name__ == "__main__":
    main()