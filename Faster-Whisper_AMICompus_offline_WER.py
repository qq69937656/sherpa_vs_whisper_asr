import os
import sys
import re
import jiwer
from faster_whisper import WhisperModel

# ================= 配置区域 =================

# 数据集根目录
AMI_DIR = "/opt/Audio_Datasets/AMICorpus"
MODEL_SIZE = "medium.en"

# 最小有效单词数限制 (少于3个单词的样本不参与评测)
MIN_WORD_COUNT = 3

# 【新增】最大处理文件数限制
MAX_FILES = 10


# ================= 辅助函数 =================

def normalize_text(text):
    """
    文本归一化处理：
    1. 【AMI特定】将换行符替换为空格
    2. 转小写
    3. 移除 <...> 格式的特殊标签
    4. 移除常规标点符号
    5. 拆分为单词列表
    """
    if not text:
        return []

    # 1. 替换回车为空格 (防止换行导致的单词粘连)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # 2. 转小写
    text = text.lower()

    # 3. 去除 <...> 格式的标签 (如 <SIL>, <vocal>, 等)
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. 去除除了字母、数字、空格以外的标点符号
    text = re.sub(r"[^\w\s]", "", text)

    # 5. 分割单词
    words = text.strip().split()
    return words


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
    print(f"正在加载模型 {MODEL_SIZE} (AMI Corpus)...")
    recognizer = create_recognizer(MODEL_SIZE)
    print("模型加载完成。")
    print(f"注意：程序将在处理完 {MAX_FILES} 个有效文件后停止。")
    print("-" * 50)

    if not os.path.exists(AMI_DIR):
        print(f"错误: 数据集目录不存在 {AMI_DIR}")
        return

    total_distance = 0
    total_ref_words = 0
    file_count = 0  # 记录已处理的有效文件数
    skipped_count = 0  # 记录跳过的文件数
    stop_processing = False  # 控制双层循环退出的标志

    print(f"开始遍历数据集: {AMI_DIR}")

    # 遍历目录结构
    for root, dirs, files in os.walk(AMI_DIR):
        if stop_processing:
            break

        # 筛选出 wav 文件
        wav_files = [f for f in files if f.lower().endswith('.wav')]

        for wav_file in wav_files:
            # 如果达到数量限制，退出循环
            if file_count >= MAX_FILES:
                stop_processing = True
                break

            wav_path = os.path.join(root, wav_file)

            # 构建对应的 txt 文件路径
            # 假设文件名一致: recording.wav -> recording.txt
            file_id = os.path.splitext(wav_file)[0]
            txt_file = file_id + ".txt"
            txt_path = os.path.join(root, txt_file)

            # 1. 检查参考文本是否存在
            if not os.path.exists(txt_path):
                continue

            # 2. 读取参考文本
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    raw_ref_text = f.read()
            except Exception as e:
                print(f"无法读取文本文件 {txt_path}: {e}")
                continue

            # 3. 预处理参考文本
            ref_words_list = normalize_text(raw_ref_text)

            # 4. 长度过滤规则 (忽略少于3个单词的样本)
            if len(ref_words_list) < MIN_WORD_COUNT:
                skipped_count += 1
                continue

            # ================= 核心推理逻辑 =================
            try:
                segments, info = recognizer.transcribe(
                    wav_path,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    condition_on_previous_text=True
                )

                hyp_segments_list = [segment.text for segment in segments]
                hyp_text_raw = " ".join(hyp_segments_list)

            except Exception as e:
                print(f"推理异常 {wav_file}: {e}")
                continue
            # ===============================================

            # ================= 评测逻辑 =================
            hyp_words_list = normalize_text(hyp_text_raw)

            ref_str_clean = " ".join(ref_words_list)
            hyp_str_clean = " ".join(hyp_words_list)

            try:
                out = jiwer.process_words(ref_str_clean, hyp_str_clean)

                curr_dist = out.substitutions + out.deletions + out.insertions
                curr_wer = out.wer
                curr_len = len(out.references[0]) if out.references else 0

                total_distance += curr_dist
                total_ref_words += curr_len
                file_count += 1

                print(
                    f"[{file_count}/{MAX_FILES}] File: {wav_file} | WER: {curr_wer:.4f} | Dist: {curr_dist} | RefWords: {curr_len}"
                )

            except Exception as e:
                print(f"Jiwer 计算错误 {wav_file}: {e}")

    # ================= 结果汇总 =================
    print("\n" + "=" * 50)
    if stop_processing:
        print(f"已达到最大处理数量限制 ({MAX_FILES}个)，停止运行。")
    print("AMI Corpus 评测完成")
    print(f"有效处理文件数: {file_count}")
    print(f"跳过短/无效文件数: {skipped_count}")

    if total_ref_words > 0:
        avg_wer = total_distance / total_ref_words
        print("-" * 25)
        print(f"总编辑距离: {total_distance}")
        print(f"总参考单词数: {total_ref_words}")
        print(f"平均错词率 (Average WER): {avg_wer:.2%}")
    else:
        print("未生成有效统计数据。")
    print("=" * 50)


if __name__ == "__main__":
    main()