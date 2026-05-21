# vLLM-MLX 文档

**Apple Silicon 的 MLX 推理后端** - 在 Mac 上对文本、图像、视频和音频进行 GPU 加速推理

## 什么是 vLLM-MLX？

vllm-mlx 通过集成以下组件，为 vLLM 带来原生 Apple Silicon GPU 加速：

- **[MLX](https://github.com/ml-explore/mlx)**：Apple 的机器学习框架，具有统一内存和 Metal 内核
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)**：经过优化的 LLM 推理，支持 KV cache 和量化
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)**：用于多模态推理的视觉语言模型 VLM
- **[mlx-audio](https://github.com/Blaizzy/mlx-audio)**：基于原生语音的 TTS 和 STT
- **[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)**：用于语义搜索和 RAG 的文本 embeddings

## 主要特性

- **多模态** - 在同一平台上处理文本、图像、视频和音频
- **原生 GPU 加速**，支持 Apple Silicon（M1、M2、M3、M4、M5）
- **原生 TTS 语音** - 支持西班牙语、法语、中文、日语及其他 5 种语言
- **OpenAI API 兼容** - 可直接替换 OpenAI 客户端
- **Embeddings** - 兼容 OpenAI 的 `/v1/embeddings` 端点
- **MCP Tool Calling** - 通过 Model Context Protocol 集成外部工具
- **Paged KV Cache** - 支持前缀共享的高效内存缓存
- **Continuous Batching** - 为多并发用户提供高吞吐量

## 快速链接

### 入门指南
- [安装](getting-started/installation.md)
- [快速开始](getting-started/quickstart.md)

### 用户指南
- [兼容 OpenAI 的服务器](guides/server.md)
- [Python API](guides/python-api.md)
- [多模态（图像与视频）](guides/multimodal.md)
- [音频（STT/TTS）](guides/audio.md)
- [Embeddings](guides/embeddings.md)
- [Reasoning 模型](guides/reasoning.md)
- [Tool Calling](guides/tool-calling.md)
- [MCP 与 Tool Calling](guides/mcp-tools.md)
- [Continuous Batching](guides/continuous-batching.md)

### 参考文档
- [CLI 命令](reference/cli.md)
- [支持的模型](reference/models.md)
- [配置说明](reference/configuration.md)

### 基准测试
- [LLM 基准测试](benchmarks/llm.md)
- [图像基准测试](benchmarks/image.md)
- [视频基准测试](benchmarks/video.md)
- [音频基准测试](benchmarks/audio.md)

### 开发者文档
- [架构设计](../development/architecture.md)
- [贡献指南](../development/contributing.md)

## 环境要求

- 搭载 Apple Silicon 的 macOS（M1/M2/M3/M4/M5）
- Python 3.10 及以上
- 推荐 8GB 及以上内存

## 许可证

Apache 2.0，详情请参阅 [LICENSE](../../LICENSE)。
