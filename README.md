# Streaming-ASR-SimBench (V1.0)

**流式语音识别模型性能仿真与并发评测系统** 面向实时会议场景的高保真性能仿真与量化评测工具。

---

## 📖 项目概述

[cite_start]**Streaming-ASR-SimBench** 是一套面向实时会议语音转写场景的流式语音识别模型性能仿真与并发评测系统 [cite: 37][cite_start]。该系统针对流式（Streaming）与非流式（Non-Streaming）模型在实时处理与系统吞吐能力方面的差异，构建了离线流式仿真引擎 [cite: 37]。

[cite_start]通过对标准音频文件进行分帧切片并以真实语速节流发送，系统可在无需物理麦克风的条件下，精确复现真实会议场景的连续语音流输入特征 [cite: 37][cite_start]。这为工业级低延迟、高并发会议字幕系统的架构设计与部署规划提供了定量化的决策支持 [cite: 39]。

---

## ✨ 核心特性

* [cite_start]**离线流式仿真技术**：通过精确控制分片粒度与帧间时间间隔，复现连续语音流输入特征 [cite: 250]。
* [cite_start]**跨架构统一评测框架**：兼容流式 Transducer 架构（如 Sherpa-ONNX）与分片式 AED 架构（如 Faster-Whisper）的统一调用与对比 [cite: 38, 252]。
* [cite_start]**语义增强量化机制**：内置基于 BERT 的语义后处理模块，可实时统计标点恢复和大小写规范化带来的性能损耗（如延迟和显存带宽压力） [cite: 143, 254]。
* [cite_start]**高并发负载仿真**：利用多线程并发或状态机轮询机制，在单进程内模拟 GPU 高负载运行状态，评估模型在特定环境下的最大并发处理能力 [cite: 38, 256]。
* [cite_start]**标准化数据工厂**：内置对 LibriSpeech、TED-LIUM 及 AMI Corpus 等数据集的动态路由与标准化清洗规则 [cite: 61, 258]。

---

## 🏗️ 系统架构

[cite_start]系统采用模块化分层架构设计，实现了模型架构解耦、输入机制仿真化、指标统计实时化以及评测流程自动化 [cite: 54, 57]。

1.  [cite_start]**评测调度控制模块**：负责模型选择、数据集路径解析、指标配置及并发数设置 [cite: 60, 62]。
2.  [cite_start]**流式语音仿真模块**：执行音频读取、分帧切片与时间节流控制 [cite: 115]。
3.  [cite_start]**模型与算法执行模块**：核心运算引擎，支持离线基准、基础分片、滑动窗口优化等 5 种执行模式 [cite: 131, 134]。
4.  [cite_start]**指标计算与统计模块**：并行监测机制，实时采集 WER、RTF 与 TTFT 。
5.  [cite_start]**结果输出模块**：输出单文件指标及批量平均统计数据，支持日志与结构化数据记录 [cite: 163]。

---

## 🛠️ 运行环境

* [cite_start]**操作系统**：Ubuntu 20.04.6 LTS [cite: 41]
* [cite_start]**运行环境**：Python 3.12 [cite: 42]
* [cite_start]**核心库**：PyTorch 2.1+, Sherpa-onnx, Faster-Whisper, Transformers [cite: 44, 45, 46]
* [cite_start]**硬件要求**：支持 CUDA 的 NVIDIA 显卡 (显存 ≥ 8GB)，多核 CPU [cite: 50, 51]

---

## 🚀 快速上手

### 1. 克隆仓库
```bash
git clone [https://github.com/qq69937656/sherpa_vs_whisper_asr.git](https://github.com/qq69937656/sherpa_vs_whisper_asr.git)
cd sherpa_vs_whisper_asr
```

### 2. 执行评测任务

主程序 main_controller.py 采用命令行接口设计。

示例：使用 Sherpa-ONNX 原生流式模型，在 LibriSpeech 数据集上运行 50 路并发的首字延迟（TTFT）测试：

```bash
python main_controller.py \
  --module so_native \
  --dataset LibriSpeech \
  --metric TTFT \
  --concurrency 50
```

### 3. 核心参数详细规范 [cite: 66, 67, 68, 74, 75, 76]

| 参数 | 属性 | 说明 | 可选值 (Options) |
| :--- | :--- | :--- | :--- |
| `--module` | **必填** | 决定系统拉起的底层算法模型与配置策略 [cite: 68] | `fw_offline`, `fw_segment`, `fw_slide`, `so_native`, `so_bert` [cite: 69, 70, 71, 72, 73] |
| `--dataset` | **必填** | 系统内置路径映射映射字典，自动下发物理路径与清洗规则 [cite: 74] | `LibriSpeech`, `TEDLIUM`, `AMICorpus` [cite: 74] |
| `--metric` | **必填** | 选择评测的性能指标维度 [cite: 75] | `WER` (串行准确率), `RTF` (并发吞吐量), `TTFT` (并发首字延迟) [cite: 75] |
| `--concurrency` | 选填 | 并发任务路数（默认为 1），仅在测 RTF 或 TTFT 时生效 [cite: 76] | 整数 (Integer) [cite: 76] |

---

## 💡 执行模式深度说明 [cite: 134]

### Faster-Whisper 分支（非流式架构适配） [cite: 135]
* **fw_offline** (全量离线基准模式): 性能基准上界（Topline），不进行流式分片，直接将长音频全量送入模型，仅用于提取理想状态下的 WER [cite: 137]。
* **fw_segment** (基础分片准实时模式): 将音频在内存中严格切分为固定时长（1.0s）的无重叠分片，量化强制干预引发的单词截断错误 [cite: 138]。
* **fw_slide** (滑动窗口重叠优化模式): 引入动态音频缓存与基于字级时间戳的“时间回退与裁剪”机制，验证上下文拼接对识别准确率的修复效果 [cite: 139]。

### Sherpa-ONNX 分支（原生流式架构） [cite: 140]
* **so_native** (原生流式实时模式): 依靠 Transducer 内部状态机实现即时推理，音频切分为 0.1s 极小粒度数据包，评估极限吞吐下的延迟与并发能力 [cite: 142]。
* **so_bert** (挂载 BERT 语义增强模式): 在流式输出末端挂载轻量级 BERT 模型进行异步节流调用，模拟真实工业应用中语义修饰带来的边际延迟负担 [cite: 143]。

---

## 📊 性能指标定义 [cite: 145]

系统采用并行监测机制，实时采集以下关键性能指标：

* **WER (Word Error Rate)**：词错误率。将模型输出文本与标准文本对齐，计算插入、删除与替换错误比例，衡量识别准确度 [cite: 145]。
* **TTFT (Time To First Token)**：首字上屏延迟。统计起点为首个音频分片生成时刻，终点为首个识别文本输出时刻，反映系统响应速度 [cite: 145]。
* **RTF (Real Time Factor)**：实时率。模型总推理耗时与音频总时长的比值，反映模型在特定硬件环境下的处理效率 [cite: 145]。

---

## 🛡️ 系统关键技术

* **多进程路由调度**：主控调度层与执行层实现进程级物理隔离，任务结束后彻底回收 GPU 显存，杜绝显存泄漏 [cite: 85, 111]。
* **并行指标采集与锁控制**：在高并发场景下利用线程安全锁（threading.Lock）实现安全静默落盘，避免 I/O 阻塞导致测量失真 [cite: 196, 201, 203]。
* **双轨数据输出策略**：终端仅抽样显示进度，全量实验原始数据实时追加写入本地 CSV，便于后续性能分析与绘图 [cite: 201, 202, 203]。

---

**编写人**：李鲲程 [cite: 3]
**编写时间**：2026.1.9 [cite: 4]
