# vLLM-MLX

**vLLM-like inference for Apple Silicon** - GPU-accelerated Text, Image, Video & Audio on Mac

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)
[![GitHub](https://img.shields.io/badge/GitHub-waybarrios%2Fvllm--mlx-blue?logo=github)](https://github.com/waybarrios/vllm-mlx)

## Overview

vllm-mlx brings native Apple Silicon GPU acceleration to vLLM by integrating:

- **[MLX](https://github.com/ml-explore/mlx)**: Apple's ML framework with unified memory and Metal kernels
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)**: Optimized LLM inference with KV cache and quantization
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)**: Vision-language models for multimodal inference
- **[mlx-audio](https://github.com/Blaizzy/mlx-audio)**: Speech-to-Text and Text-to-Speech with native voices
- **[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)**: Text embeddings for semantic search and RAG

## Features

- **Multimodal** - Text, Image, Video & Audio in one platform
- **Native GPU acceleration** on Apple Silicon (M1, M2, M3, M4)
- **Native TTS voices** - Spanish, French, Chinese, Japanese + 5 more languages
- **OpenAI API compatible** - drop-in replacement for OpenAI client
- **Anthropic Messages API** - native `/v1/messages` endpoint for Claude Code and OpenCode
- **Embeddings** - OpenAI-compatible `/v1/embeddings` endpoint with mlx-embeddings
- **Reasoning Models** - extract thinking process from Qwen3, DeepSeek-R1
- **MCP Tool Calling** - integrate external tools via Model Context Protocol
- **Paged KV Cache** - memory-efficient caching with prefix sharing
- **Continuous Batching** - high throughput for multiple concurrent users

## Quick Start

### Installation

**Using uv (recommended):**

```bash
# Install as CLI tool (system-wide)
uv tool install git+https://github.com/waybarrios/vllm-mlx.git

# Or install in a project/virtual environment
uv pip install git+https://github.com/waybarrios/vllm-mlx.git
```

**Using pip:**

```bash
# Install from GitHub
pip install git+https://github.com/waybarrios/vllm-mlx.git

# Or clone and install in development mode
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

### Start Server

```bash
# Simple mode (single user, max throughput)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# Continuous batching (multiple users)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching

# With API key authentication
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --api-key your-secret-key
```

### Use with OpenAI SDK

```python
from openai import OpenAI

# Without API key (local development)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# With API key (production)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-secret-key")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Use with Anthropic SDK

vllm-mlx exposes an Anthropic-compatible `/v1/messages` endpoint, so tools like Claude Code and OpenCode can connect directly.

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key="not-needed")

response = client.messages.create(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

To use with Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

See [Anthropic Messages API docs](docs/guides/server.md#anthropic-messages-api) for streaming, tool calling, system messages, and token counting.

### Multimodal (Images & Video)

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)
```

### Audio (TTS/STT)

```bash
# Install audio dependencies
pip install vllm-mlx[audio]
python -m spacy download en_core_web_sm
brew install espeak-ng  # macOS, for non-English languages
```

```bash
# Text-to-Speech (English)
python examples/tts_example.py "Hello, how are you?" --play

# Text-to-Speech (Spanish)
python examples/tts_multilingual.py "Hola mundo" --lang es --play

# List available models and languages
python examples/tts_multilingual.py --list-models
python examples/tts_multilingual.py --list-languages
```

**Supported TTS Models:**
| Model | Languages | Description |
|-------|-----------|-------------|
| Kokoro | EN, ES, FR, JA, ZH, IT, PT, HI | Fast, 82M params, 11 voices |
| Chatterbox | 15+ languages | Expressive, voice cloning |
| VibeVoice | EN | Realtime, low latency |
| VoxCPM | ZH, EN | High quality Chinese/English |

### Reasoning Models

Extract the thinking process from reasoning models like Qwen3 and DeepSeek-R1:

```bash
# Start server with reasoning parser
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)

# Access reasoning separately from the answer
print("Thinking:", response.choices[0].message.reasoning)
print("Answer:", response.choices[0].message.content)
```

**Supported Parsers:**
| Parser | Models | Description |
|--------|--------|-------------|
| `qwen3` | Qwen3 series | Requires both `<think>` and `</think>` tags |
| `deepseek_r1` | DeepSeek-R1 | Handles implicit `<think>` tag |

### Embeddings

Generate text embeddings for semantic search, RAG, and similarity:

```bash
# Start server with an embedding model pre-loaded
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

```python
# Generate embeddings using the OpenAI SDK
embeddings = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input=["Hello world", "How are you?"]
)
print(f"Dimensions: {len(embeddings.data[0].embedding)}")
```

See [Embeddings Guide](docs/guides/embeddings.md) for details on supported models and lazy loading.

## Documentation

For full documentation, see the [docs](docs/) directory:

- **Getting Started**
  - [Installation](docs/getting-started/installation.md)
  - [Quick Start](docs/getting-started/quickstart.md)

- **User Guides**
  - [OpenAI-Compatible Server](docs/guides/server.md)
  - [Anthropic Messages API](docs/guides/server.md#anthropic-messages-api)
  - [Python API](docs/guides/python-api.md)
  - [Multimodal (Images & Video)](docs/guides/multimodal.md)
  - [Audio (STT/TTS)](docs/guides/audio.md)
  - [Embeddings](docs/guides/embeddings.md)
  - [Reasoning Models](docs/guides/reasoning.md)
  - [MCP & Tool Calling](docs/guides/mcp-tools.md)
  - [Continuous Batching](docs/guides/continuous-batching.md)

- **Reference**
  - [CLI Commands](docs/reference/cli.md)
  - [Supported Models](docs/reference/models.md)
  - [Configuration](docs/reference/configuration.md)

- **Benchmarks**
  - [LLM Benchmarks](docs/benchmarks/llm.md)
  - [Image Benchmarks](docs/benchmarks/image.md)
  - [Video Benchmarks](docs/benchmarks/video.md)
  - [Audio Benchmarks](docs/benchmarks/audio.md)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           vLLM API Layer                                │
│                    (OpenAI-compatible interface)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            MLXPlatform                                  │
│               (vLLM platform plugin for Apple Silicon)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
        ┌─────────────┬────────────┴────────────┬─────────────┐
        ▼             ▼                         ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    mlx-lm     │ │   mlx-vlm     │ │   mlx-audio   │ │mlx-embeddings │
│(LLM inference)│ │ (Vision+LLM)  │ │  (TTS + STT)  │ │ (Embeddings)  │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
        │             │                         │             │
        └─────────────┴─────────────────────────┴─────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              MLX                                        │
│                (Apple ML Framework - Metal kernels)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Performance

**LLM Performance (M4 Max, 128GB):**

| Model | Speed | Memory |
|-------|-------|--------|
| Qwen3-0.6B-8bit | 402 tok/s | 0.7 GB |
| Llama-3.2-1B-4bit | 464 tok/s | 0.7 GB |
| Llama-3.2-3B-4bit | 200 tok/s | 1.8 GB |

**Continuous Batching (5 concurrent requests):**

| Model | Single | Batched | Speedup |
|-------|--------|---------|---------|
| Qwen3-0.6B-8bit | 328 tok/s | 1112 tok/s | **3.4x** |
| Llama-3.2-1B-4bit | 299 tok/s | 613 tok/s | **2.0x** |

**Audio - Speech-to-Text (M4 Max, 128GB):**

| Model | RTF* | Use Case |
|-------|------|----------|
| whisper-tiny | **197x** | Real-time, low latency |
| whisper-large-v3-turbo | **55x** | Best quality/speed balance |
| whisper-large-v3 | **24x** | Highest accuracy |

*RTF = Real-Time Factor. RTF of 100x means 1 minute transcribes in ~0.6 seconds.

See [benchmarks](docs/benchmarks/) for detailed results.

## Gemma 3 Support

vllm-mlx includes native support for Gemma 3 vision models. Gemma 3 is automatically detected as MLLM.

### Usage

```bash
# Start server with Gemma 3
vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Verify it loaded as MLLM (not LLM)
curl http://localhost:8000/health
# Should show: "model_type": "mllm"
```

### Long Context Patch (mlx-vlm)

Gemma 3's default `sliding_window=1024` limits context to ~10K tokens on Apple Silicon (Metal GPU timeout at higher context). To enable longer context (up to ~50K tokens), patch mlx-vlm:

**Location:** `~/.../site-packages/mlx_vlm/models/gemma3/language.py`

Find the `make_cache` method and replace with:

```python
def make_cache(self):
    import os
    # Set GEMMA3_SLIDING_WINDOW=8192 for ~40K context
    # Set GEMMA3_SLIDING_WINDOW=0 for ~50K context (full KVCache)
    sliding_window = int(os.environ.get('GEMMA3_SLIDING_WINDOW', self.config.sliding_window))

    caches = []
    for i in range(self.config.num_hidden_layers):
        if (
            i % self.config.sliding_window_pattern
            == self.config.sliding_window_pattern - 1
        ):
            caches.append(KVCache())
        elif sliding_window == 0:
            caches.append(KVCache())  # Full context for all layers
        else:
            caches.append(RotatingKVCache(max_size=sliding_window, keep=0))
    return caches
```

**Usage:**

```bash
# Default (~10K max context)
vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Extended context (~40K max)
GEMMA3_SLIDING_WINDOW=8192 vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Maximum context (~50K max)
GEMMA3_SLIDING_WINDOW=0 vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000
```

**Benchmark Results (M4 Max 128GB):**

| Setting | Max Context | Memory |
|---------|-------------|--------|
| Default (1024) | ~10K tokens | ~16GB |
| `GEMMA3_SLIDING_WINDOW=8192` | ~40K tokens | ~25GB |
| `GEMMA3_SLIDING_WINDOW=0` | ~50K tokens | ~35GB |

## Contributing

We welcome contributions! See [Contributing Guide](docs/development/contributing.md) for details.

- Bug fixes and improvements
- Performance optimizations
- Documentation improvements
- Benchmarks on different Apple Silicon chips

Submit PRs to: [https://github.com/waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx)

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Baseline Benchmark — MiniMax-M2.5 on M3 Ultra

Benchmark measured on **Mac Studio M3 Ultra (256GB)** running **MiniMax-M2.5-MLX-4bit (229B MoE, ~120GB)** with `--reasoning-parser deepseek_r1 --tool-call-parser minimax --enable-auto-tool-choice --enable-prefix-cache`.

Date: 2026-02-24 | Branch: `feat-minimax-parser` | Server mode: Simple (no continuous batching)

### TTFT (Time To First Token)

| Prompt Size | TTFT | Decode Speed |
|-------------|------|-------------|
| Short (~50 tok) | 0.331s | 54.1 tok/s |
| Medium (~500 tok) | 0.753s | 53.5 tok/s |
| Long (~2K tok) | 1.396s | 52.1 tok/s |

### Decode Throughput

| Output Length | Speed | Total Time |
|--------------|-------|------------|
| 128 tokens | 52.8 tok/s | 2.8s |
| 512 tokens | 52.0 tok/s | 10.2s |
| 2048 tokens | 49.6 tok/s | 41.7s |

### Prefix Cache (Multi-Turn Conversation)

| Turn | TTFT | Prompt Growth |
|------|------|--------------|
| Turn 1 | 0.497s | baseline |
| Turn 2 | 0.585s | +assistant reply |
| Turn 3 | 0.871s | +2 turns |
| Turn 4 | 1.040s | +3 turns |

Turn 4 / Turn 1 TTFT ratio: **2.09x** (ideally should stay close to 1.0x with perfect cache hits)

### Tool Calling

| Test | Result | Latency | Tools Called |
|------|--------|---------|-------------|
| Simple (weather) | OK | 2.00s | get_weather |
| Multi-arg (search) | OK | 2.51s | search_web |
| Code execution | OK | 3.99s | run_python |
| Multi-tool selection | OK | 3.04s | get_weather, search_web |

Accuracy: **4/4 (100%)**  |  Avg latency: **2.89s**

### Reasoning Separation (deepseek_r1 parser)

| Test | Reasoning | Content | Separated |
|------|-----------|---------|-----------|
| Math (17*23) | 1013 chars | 259 chars | OK |
| Logic | 2154 chars | 0 chars | PARTIAL |
| Code | 2011 chars | 0 chars | PARTIAL |

Properly separated: **1/3** (reasoning parser needs improvement for non-math tasks)

### Long Generation Stability

| Metric | Value |
|--------|-------|
| Max tokens | 8192 |
| Completed | Yes (finish_reason=length) |
| Decode speed | 33.0 tok/s |
| Total time | 249.0s |
| Output chars | 36,609 |

### Run the Benchmark

```bash
# Start server
python -m vllm_mlx.cli serve <model> \
  --tool-call-parser minimax --enable-auto-tool-choice \
  --reasoning-parser deepseek_r1 --enable-prefix-cache

# Run all tests
python benchmark_minmax.py --output results.json

# Run specific test
python benchmark_minmax.py --test ttft
python benchmark_minmax.py --test tool_call
```

---

## Optimization Roadmap

Prioritized optimization plan based on analysis of upstream vLLM features, adapted for MLX on Apple Silicon. Focused on large-model serving (MiniMax-M2.5 229B MoE, ~120GB on M3 Ultra 256GB).

See [docs/plans/](docs/plans/) for detailed implementation plans.

### Tier 0: Quick Wins (Merged)

Low-risk, high-impact changes. [Full plan](docs/plans/tier0-pinned-prefix-cache.md).

| Optimization | Impact | Status |
|-------------|--------|--------|
| **GC Control** -- disable Python GC during generation | Eliminates latency spikes | Merged |
| **Detokenizer Caching** -- incremental token counting | O(n^2) -> O(1) per chunk | Merged |
| **Schema Error Hardening** -- graceful fallback on bad schemas | Prevents server crashes | Merged |
| **Pinned Prefix Cache** -- prevent system prompt eviction | Turn4/Turn1 TTFT 2.09x -> ~1.0x | Merged |

### Tier 1: High Impact Optimizations (Partially Merged)

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| **Fix Streaming Tool Truncation** -- parser-aware fallback for incomplete tool calls | Eliminates silently lost MiniMax/Llama/Mistral tool calls | Merged |
| **Frequency-Aware Cache Eviction** -- LRU-LFU hybrid for multi-turn | System prompt blocks survive 100x longer under pressure | Merged |
| **KV Block Reuse Ordering** -- prefer high-frequency blocks on hash collision | Higher cache hit ratio for hybrid models | Merged |
| **Jump-Forward Decoding** -- skip constrained tokens in guided gen | 2-5x faster JSON generation | Planned |
| **Structural Tags for Tool Calling** -- guarantee valid JSON args | <1% tool call failure rate | Planned |
| **LogitsProcessor Interface** -- clean extensibility for custom logits | Plugin architecture for constrained decoding | Planned |

### Tier 2: Evaluate Later (High Potential, High Complexity)

| Optimization | Expected Impact | Upstream Ref |
|-------------|----------------|-------------|
| **Beam Search + Structured Output** -- best-quality tool arguments | Highest quality constrained generation | [vLLM #35022](https://github.com/vllm-project/vllm/pull/35022) |
| **Draft Model Speculative Decoding** -- faster generation overall | 1.5-2x decode speedup | [vLLM #32662](https://github.com/vllm-project/vllm/pull/32662) |
| **Inter-Prefill-Budget** -- better TTFT under concurrent load | 37% TTFT reduction for concurrent requests | [vLLM #33743](https://github.com/vllm-project/vllm/pull/33743) |
| **Tool Calling Redesign** -- better architecture for tool call parsing | Cleaner, more maintainable tool call path | [vLLM #22977](https://github.com/vllm-project/vllm/pull/22977) |

### Estimated Combined Impact (Tier 0 + Tier 1)

For 10-turn agentic conversations with tool calling (e.g., OpenClaw on MiniMax-M2.5):

- **TTFT**: 2x improvement (pinned prefix cache + GC control)
- **Tool call reliability**: 95% -> 99%+ (structural tags + streaming fix)
- **Throughput consistency**: Eliminate GC-induced spikes
- **Long generation**: O(n^2) -> O(1) per-chunk overhead

---

## Citation

If you use vLLM-MLX in your research or project, please cite:

```bibtex
@software{vllm_mlx2025,
  author = {Barrios, Wayner},
  title = {vLLM-MLX: Apple Silicon MLX Backend for vLLM},
  year = {2025},
  url = {https://github.com/waybarrios/vllm-mlx},
  note = {Native GPU-accelerated LLM and vision-language model inference on Apple Silicon}
}
```

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - LLM inference library
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language models
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - Text-to-Speech and Speech-to-Text
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) - Text embeddings
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
