# sherpa_vs_whisper_asr
实时会议场景下流式（Sherpa-onnx）与分片式（Faster-Whisper）ASR 对比测评。包括四种场景，三个指标。
四种场景：1）Faster-Whisper基础分片（1秒钟）
2）Faster-Whisper + 滑动窗口：通过滑动窗口算法补充语音数据上下文信息，提高识别准确率。
3）Sherpa-onnx原生流式
4）Sherpa-onnx流式+语义后处理

三个指标：WER、RTF和TTFT，WER评测采用LibriSpeech（标准朗读）、TEDLIUM（个人演讲）及 AMI Meeting Corpus（真实会议）数据集，RTF和TTFT采用LibriSpeech数据集。

测试环境：硬件：NVIDIA Tesla V100-PCIE-32GB；Intel(R) Xeon(R) Silver 4316 CPU（32 CPU）；64GB RAM。
        软件：Ubuntu 20.04.6 LTS；Python 3.12.9；PyTorch 2.2.2 (CUDA 12.1)；Sherpa-onnx 1.12.20；Faster-Whisper 1.2.0。
        其中Sherpa-onnx为官网下载源码编译以支持GPU，官方源仅支持CPU
