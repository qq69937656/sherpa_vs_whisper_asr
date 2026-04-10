import argparse
import subprocess
import sys
import os

# =========================
# 模块与文件映射配置
# =========================
# 优化点：去除了文件名中的 {dataset}，实现单脚本通吃多数据集
MODULES = {
    "fw_offline": "Faster-Whisper_Offline_{metric}.py",
    "fw_segment": "Faster-Whisper_Segmentation_{metric}.py",
    "fw_slide": "Faster-Whisper_Slide_{metric}.py",
    "so_native": "Sherpa-onnx_Native_{metric}.py",
    "so_bert": "Sherpa-onnx_BERT_{metric}.py"
}

# 统一管理数据集的物理存储位置
DATASET_PATHS = {
    "LibriSpeech": "/opt/Audio_Datasets/LibriSpeech_WAV/test-clean",
    "TEDLIUM": "/opt/Audio_Datasets/TEDLIUM_WAV",
    "AMICorpus": "/opt/Audio_Datasets/AMICorpus"
}

DATASETS = list(DATASET_PATHS.keys())
METRICS = ["WER", "RTF", "TTFT"]

# RTF / TTFT 仅支持 LibriSpeech
REALTIME_ONLY_DATASET = "LibriSpeech"


def get_description():
    return """
=================================================================
  流式语音识别模型性能仿真与并发评测系统 (Streaming-ASR-SimBench)
=================================================================
本程序用于统一调度不同架构（Faster-Whisper, Sherpa-onnx）在多种
配置下的性能测试。支持跨数据集的准确率（WER）评估，以及针对实时
场景的延迟（TTFT）与并发性能（RTF）压测。

[1. 支持的模型模块 (--module)]
  fw_offline  : Faster-Whisper 离线测试 (Topline, 仅测 WER)
  fw_segment  : Faster-Whisper 基础分片准实时测试 (配置 A1)
  fw_slide    : Faster-Whisper 滑动窗口重叠优化测试 (配置 A2)
  so_native   : Sherpa-onnx 原生流式测试 (配置 B1)
  so_bert     : Sherpa-onnx 结合 BERT 语义优化测试 (配置 B2)

[2. 支持的评测数据集 (--dataset)]
  LibriSpeech : 理想朗读语音 (支持测试 WER, RTF, TTFT)
  TEDLIUM     : 即兴演讲场景 (支持测试 WER)
  AMICorpus   : 多人会议场景 (支持测试 WER)

[3. 支持的评估指标 (--metric)]
  WER  : 词错误率 (串行测试准确度，无需指定并发数)
  RTF  : 实时率 (测试并发吞吐量，需指定并发数)
  TTFT : 首字延迟 (测试初始响应速度，需指定并发数)
-----------------------------------------------------------------
"""


def validate_args(parser, args):
    if args.module not in MODULES:
        parser.error(f"不支持的模块: '{args.module}'\n请在 {list(MODULES.keys())} 中选择。")
    if args.dataset not in DATASETS:
        parser.error(f"不支持的数据集: '{args.dataset}'\n请在 {DATASETS} 中选择。")
    if args.metric not in METRICS:
        parser.error(f"不支持的指标: '{args.metric}'\n请在 {METRICS} 中选择。")
    if args.module == "fw_offline" and args.metric != "WER":
        parser.error("逻辑冲突：'fw_offline' (离线测试) 仅支持测试 'WER' 指标。")
    if args.metric in ["RTF", "TTFT"] and args.dataset != REALTIME_ONLY_DATASET:
        parser.error(f"数据集限制：'{args.metric}' 指标目前设定仅支持 '{REALTIME_ONLY_DATASET}'。")


def build_script_name(module, metric):
    # 针对 Sherpa-onnx 的 WER 指标合并优化
    if module == "so_bert" and metric == "WER":
        print("💡 提示：Sherpa-onnx 的 BERT 语义后处理不改变 WER 指标，已自动路由至 Native 版本的测试程序。")
        module = "so_native"

    return MODULES[module].format(metric=metric)


def run_script(script_name, dataset_name, metric, concurrency):
    if not os.path.exists(script_name):
        print(f"❌ 错误：未找到测试脚本文件 '{script_name}'")
        sys.exit(1)

    dataset_path = DATASET_PATHS[dataset_name]

    # 基础命令：传入数据集名称和路径
    cmd = [
        sys.executable, script_name,
        "--dataset_name", dataset_name,
        "--dataset_path", dataset_path
    ]

    # 根据指标决定是否传递并发参数
    if metric in ["RTF", "TTFT"]:
        cmd.extend(["--concurrency", str(concurrency)])
    else:
        if concurrency != 1:
            print(f"💡 提示：当前指标为 '{metric}'，测试与并发数无关，已自动忽略并发参数。")

    print(f"\n🚀 正在启动评测任务: {' '.join(cmd)}\n" + "=" * 55)

    try:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠️ 评测被用户手动中断")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=get_description(),
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s --module MODULE --dataset DATASET --metric METRIC [--concurrency N]"
    )

    parser.add_argument("--module", help="选择要评测的模型模块")
    parser.add_argument("--dataset", help="选择评测数据集")
    parser.add_argument("--metric", help="选择要统计的评估指标")
    parser.add_argument("--concurrency", type=int, default=1, help="设置并发路数 (默认: 1)")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if not all([args.module, args.dataset, args.metric]):
        parser.error("参数不完整。必须同时指定 --module, --dataset 和 --metric。")

    validate_args(parser, args)
    script_name = build_script_name(args.module, args.metric)
    run_script(script_name, args.dataset, args.metric, args.concurrency)
    print("\n✅ 评测调度执行完毕")


if __name__ == "__main__":
    main()