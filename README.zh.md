# vllm-mlx

**其他语言版本：** [English](README.md) · [Español](README.es.md) · [Français](README.fr.md) · [中文](README.zh.md)

**连续批处理 + OpenAI 和 Anthropic API 集成于一个服务。Apple Silicon 原生推理。**

[![PyPI version](https://img.shields.io/pypi/v/vllm-mlx.svg)](https://pypi.org/project/vllm-mlx/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/vllm-mlx.svg)](https://pypi.org/project/vllm-mlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)
[![GitHub stars](https://img.shields.io/github/stars/waybarrios/vllm-mlx.svg?style=social)](https://github.com/waybarrios/vllm-mlx)

---

## vllm-mlx 是什么？

面向 Apple Silicon Mac 的 vLLM 风格推理服务器。与直接使用 `Ollama` 或 `mlx-lm` 不同，vllm-mlx 内置了**连续批处理（continuous batching）、分页 KV cache、prefix caching 以及 SSD 分层 KV cache**，并在同一个进程中同时暴露 **OpenAI `/v1/*` 和 Anthropic `/v1/messages`** 接口。可在 Metal 上通过统一内存运行 LLM、视觉模型、音频模型和嵌入模型，无需任何格式转换步骤。

## 快速开始（30 秒）

```bash
pip install vllm-mlx
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

**OpenAI SDK：**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(model="default", messages=[{"role": "user", "content": "你好！"}])
print(r.choices[0].message.content)
```

**Anthropic SDK / Claude Code：**

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

## 功能特性

### API
- **兼容 OpenAI**：`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、`/v1/rerank`、`/v1/responses`
- **兼容 Anthropic**：`/v1/messages`（流式、工具调用、system prompts）
- **MCP 工具调用**：12 种解析器（OpenAI、Anthropic、Gemini、Qwen、DeepSeek、Gemma 等）
- **结构化输出**：通过 `response_format` 的 JSON Schema（基于 lm-format-enforcer）

### 吞吐与内存
- **连续批处理**：高并发下的高吞吐
- **分页 KV cache**：内存高效，支持 prefix 共享
- **SSD 分层 KV cache**：为长上下文 agent 场景将 prefix cache 溢出到磁盘（`--ssd-cache-dir`）
- **Warm prompts**：启动时预加载热门 prefix（`--warm-prompts`），TTFT 提升 1.3-2.25 倍
- **Prefix cache**：基于 trie，跨请求共享

### 多模态
- **文本 + 图像 + 视频 + 音频** 集成于一个服务
- 视觉模型：Gemma 3、Gemma 4、Qwen3-VL、Pixtral、Llama vision
- **聊天中的音频输入**（`audio_url` 内容块）
- **原生 TTS**：11 种声音，15+ 种语言（Kokoro、Chatterbox、VibeVoice、VoxCPM）
- **STT**：Whisper 系列，M4 Max 上 RTF 最高可达 197 倍

### 推理与高级功能
- **思维链提取**：Qwen3、DeepSeek-R1（`--reasoning-parser`）
- **MoE 专家裁剪**：`--moe-top-k`，Qwen3-30B-A3B 上 +7-16%
- **投机解码**：`--mtp`，用于 Qwen3-Next
- **稀疏 prefill**：基于注意力的 `--spec-prefill`，降低 TTFT

### 可观测性
- **Prometheus 指标**：使用 `--metrics` 开启 `/metrics` 端点
- **内置基准测试**：`vllm-mlx bench-serve`，支持 prompt 扫描及 CSV/JSON 输出

### 原生 GPU 加速
- 仅支持 Apple Silicon（M1、M2、M3、M4、M5），通过 MLX 使用 Metal kernel
- 统一内存，无需模型转换

## 性能

**LLM 解码（M4 Max，128 GB，greedy，单流）：**

| 模型 | Tok/s | 内存 |
|------|------:|-----:|
| Qwen3-0.6B-8bit | 417.9 | 0.7 GB |
| Llama-3.2-3B-Instruct-4bit | 205.6 | 1.8 GB |
| Qwen3-30B-A3B-4bit | 127.7 | ~18 GB |

**音频 speech-to-text（M4 Max，RTF = real-time factor）：**

| 模型 | RTF | 适用场景 |
|------|----:|---------|
| whisper-tiny | 197x | 实时 / 低延迟 |
| whisper-large-v3-turbo | 55x | 质量与速度兼顾 |
| whisper-large-v3 | 24x | 最高精度 |

完整结果（包括连续批处理、KV cache 量化 4-bit / 8-bit / fp16、MoE top-k 扫描）见 [docs/benchmarks/](docs/benchmarks/)。

## 示例

### Anthropic API（Claude Code、OpenCode）

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --port 8000
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

### 推理模型（Qwen3、DeepSeek-R1）

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

```python
r = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "17 乘以 23 等于多少？"}],
)
print("思考过程：", r.choices[0].message.reasoning)
print("答案：",     r.choices[0].message.content)
```

### 多模态（图像 + 文本）

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

```python
r = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "这张图里有什么？"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    ]}],
)
```

### 结构化输出（JSON Schema）

```python
r = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "列出 3 种颜色。"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "schema": {"type": "object", "properties": {"colors": {"type": "array", "items": {"type": "string"}}}}
        },
    },
)
```

### 重排序（`/v1/rerank`）

```bash
curl http://localhost:8000/v1/rerank -H 'Content-Type: application/json' -d '{
  "model": "default",
  "query": "apple silicon 推理",
  "documents": ["MLX 是苹果的框架", "M 系列芯片上的 Metal kernel", "NVIDIA 上的 CUDA"]
}'
```

### 嵌入（Embeddings）

```bash
vllm-mlx serve <llm-model> --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

```python
emb = client.embeddings.create(model="mlx-community/all-MiniLM-L6-v2-4bit", input=["你好", "世界"])
```

### 音频（TTS / STT）

```bash
pip install vllm-mlx[audio]
brew install espeak-ng        # macOS，非英语 TTS 需要

python examples/tts_example.py "Hello, how are you?" --play
python examples/tts_multilingual.py "你好，世界" --lang zh --play
```

### 内置基准测试

```bash
vllm-mlx bench-serve --url http://localhost:8000 --concurrency 5 --prompts prompts.txt --output results.csv
```

### Prometheus 指标

```bash
vllm-mlx serve <model> --metrics
curl http://localhost:8000/metrics
```

## 安装

**使用 uv（推荐）：**

```bash
uv tool install vllm-mlx                 # 作为系统级 CLI
# 或在项目中
uv pip install vllm-mlx
```

**使用 pip：**

```bash
pip install vllm-mlx

# 音频扩展
pip install vllm-mlx[audio]
brew install espeak-ng
python -m spacy download en_core_web_sm
```

**从源码安装：**

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

更多选项见 [安装指南](docs/getting-started/installation.md)。

## 文档

- **入门**：[安装](docs/getting-started/installation.md) · [快速开始](docs/getting-started/quickstart.md)
- **服务器与 API**：[OpenAI 服务器](docs/guides/server.md) · [Anthropic Messages API](docs/guides/server.md#anthropic-messages-api) · [Python API](docs/guides/python-api.md)
- **功能**：[多模态](docs/guides/multimodal.md) · [音频](docs/guides/audio.md) · [嵌入](docs/guides/embeddings.md) · [推理模型](docs/guides/reasoning.md) · [MCP 与工具调用](docs/guides/mcp-tools.md) · [工具解析器](docs/guides/tool-calling.md)
- **性能**：[连续批处理](docs/guides/continuous-batching.md) · [Warm Prompts](docs/guides/warm-prompts.md) · [MoE Top-K](docs/guides/moe-top-k.md)
- **参考**：[CLI](docs/reference/cli.md) · [模型](docs/reference/models.md) · [配置](docs/reference/configuration.md)
- **基准测试**：[LLM](docs/benchmarks/llm.md) · [图像](docs/benchmarks/image.md) · [视频](docs/benchmarks/video.md) · [音频](docs/benchmarks/audio.md)

## 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          vllm-mlx 服务器                                │
│   OpenAI /v1/*  ·  Anthropic /v1/messages  ·  /v1/rerank  ·  /metrics   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  连续批处理 · 分页 KV cache · Prefix cache · SSD 分层                   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
        ┌─────────────┬────────────┴────────────┬─────────────┐
        ▼             ▼                         ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    mlx-lm     │ │   mlx-vlm     │ │   mlx-audio   │ │mlx-embeddings │
│    (LLMs)     │ │   (视觉)      │ │  (TTS + STT)  │ │  (嵌入向量)   │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  MLX · Metal kernel · 统一内存                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 贡献

欢迎提交 bug 修复、性能优化、文档改进以及在不同 Apple Silicon 芯片上的 benchmark。详见 [贡献指南](docs/development/contributing.md)。

## 许可证

Apache 2.0。详见 [LICENSE](LICENSE)。

## 引用

```bibtex
@software{vllm_mlx2025,
  author = {Barrios, Wayner},
  title  = {vllm-mlx: Apple Silicon MLX Backend for vLLM},
  year   = {2025},
  url    = {https://github.com/waybarrios/vllm-mlx},
  note   = {Native GPU-accelerated LLM and vision-language model inference on Apple Silicon}
}
```

## 致谢

- [MLX](https://github.com/ml-explore/mlx)。Apple 的 ML 框架。
- [mlx-lm](https://github.com/ml-explore/mlx-lm)。LLM 推理库。
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)。视觉语言模型。
- [mlx-audio](https://github.com/Blaizzy/mlx-audio)。Text-to-Speech 与 Speech-to-Text。
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)。文本嵌入。
- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX)。vllm-mlx 的社区分支。
- [vLLM](https://github.com/vllm-project/vllm)。高吞吐 LLM 推理服务。vllm-mlx 受 vLLM 启发，借鉴了其连续批处理和分页 KV cache 的设计，通过 MLX 适配到 Apple Silicon。

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=waybarrios/vllm-mlx&type=Date)](https://star-history.com/#waybarrios/vllm-mlx&Date)

---

**如果 vllm-mlx 对你有帮助，请给仓库点一个 star。这能帮助更多 Apple Silicon 开发者发现它。**
