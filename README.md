# Rapid-MLX

**Run AI on your Mac. Faster than anything else.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1500%2B-brightgreen.svg)](tests/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple)](https://support.apple.com/en-us/HT211814)

Rapid-MLX turns your Mac into a local AI server. It runs the same models as ChatGPT — Qwen, Llama, Mistral, Gemma, DeepSeek — directly on Apple Silicon, with no cloud, no subscription, and no data leaving your machine.

**Why not Ollama?** Rapid-MLX is **2-4x faster** on the same models. It uses Apple's [MLX framework](https://github.com/ml-explore/mlx), built specifically for the M-series GPU, instead of the generic C++ backend that Ollama and llama.cpp use. More speed from the same hardware.

| | Your Mac runs AI | How fast | What works |
|:---|:---:|:---:|:---:|
| **16 GB MacBook Air** | Qwen3.5-4B | 158 tok/s | Chat, coding, tools |
| **64 GB Mac Mini / Studio** | Qwen3.5-35B | 80 tok/s | Best balance of smart + fast |
| **128+ GB Mac Studio / Pro** | Qwen3.5-122B | 57 tok/s | Frontier-level intelligence |

---

## Quick Start

```bash
# 1. Install (one command, checks Apple Silicon + Python automatically)
curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/install.sh | bash

# 2. Pick a model and start serving
rapid-mlx serve mlx-community/Qwen3.5-9B-4bit --tool-call-parser hermes --port 8000

# 3. Use with any OpenAI-compatible app
OPENAI_BASE_URL=http://localhost:8000/v1 claude
```

That's it — you now have an AI server on `localhost:8000`. Works with Claude Code, Cursor, Aider, Open WebUI, or any app that speaks the OpenAI API.

<details>
<summary>Alternative install methods</summary>

**pip** (if you manage your own venv):
```bash
pip install git+https://github.com/raullenchai/Rapid-MLX.git
```

**From source** (for development):
```bash
git clone https://github.com/raullenchai/Rapid-MLX.git
cd rapid-mlx && pip install -e .
```
</details>

**Try it with Python:**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## Choose Your Model

### Decode Speed — 14 models, 5 engines

All benchmarks on Mac Studio M3 Ultra (256GB). Sorted by model size.

```
Qwen3.5-4B                  Rapid-MLX  ███████████████████████████████████████████████       158
                            vllm-mlx   ██████████████████████████████████████████████        155
                            mlx-lm     ██████████████████████████████████████████████████    168
                            llama.cpp                                           not supported
                            Ollama                                              not supported

Hermes-3-Llama 8B           Rapid-MLX  █████████████████████████████████████                 123
                            vllm-mlx   ████████████████████████████████████                  122
                            mlx-lm     ██████████████████████████████████████                127
                            llama.cpp                                           not supported
                            Ollama                                              not supported

Qwen3.5-9B                ⚡ Rapid-MLX  █████████████████████████████████                     109
                            vllm-mlx   ███████████████████████████████                       104
                            mlx-lm     ██████████████████                                    61
                            llama.cpp                                           not supported
                            Ollama     ████████                                              26

GLM-4.7-Flash 9B          ⚡ Rapid-MLX  ██████████████████                                    60
                            vllm-mlx   █████████████████                                     56
                            mlx-lm                                              not supported
                            llama.cpp                                           not supported
                            Ollama                                              not supported

Gemma 3 12B                 Rapid-MLX  ███████████████                                       49
                            vllm-mlx                                            not supported
                            mlx-lm     ██████████████████████                                73
                            llama.cpp                                           not supported
                            Ollama     ████████████████                                      54

Phi-4 Mini 14B            ⚡ Rapid-MLX  ████████████████████████████████████████████████████  174
                            vllm-mlx   ███████████████████████████████████████████████████   170
                            mlx-lm     ███████████████████████                               77
                            llama.cpp  ████████████████                                      55
                            Ollama     ███████████████                                       51

GPT-OSS 20B               ⚡ Rapid-MLX  █████████████████████████████████████                 123
                            vllm-mlx   ████████████████████████                              79
                            mlx-lm     ████████████████████████                              79
                            llama.cpp                                           not supported
                            Ollama                                              not supported

Mistral Small 24B         ⚡ Rapid-MLX  ██████████████                                        48
                            vllm-mlx   ██████████████                                        47
                            mlx-lm     ████████████                                          41
                            llama.cpp                                           not supported
                            Ollama                                              not supported

Devstral-Small-2 24B        Rapid-MLX  ██████████████                                        48
                            vllm-mlx   ██████████████                                        48
                            mlx-lm     ███████████████                                       49
                            llama.cpp                                           not supported
                            Ollama                                              not supported

Qwen3.5-27B               ⚡ Rapid-MLX  ████████████                                          39
                            vllm-mlx   ███████████                                           38
                            mlx-lm     ████████████                                          39
                            llama.cpp                                           not supported
                            Ollama                                              not supported

Qwen3.5-35B-A3B             Rapid-MLX  █████████████████████████                             82
                            vllm-mlx   ████████████████████████                              80
                            mlx-lm     █████████████████████████                             85
                            llama.cpp                                           not supported
                            Ollama                                              not supported

Qwen3-Coder-Next 80B        Rapid-MLX  ██████████████████████                                74
                            vllm-mlx   █████████████████████                                 69
                            mlx-lm     ███████████████████████                               76
                            llama.cpp                                           not supported
                            Ollama                                              not supported

GLM-4.5-Air                 Rapid-MLX  ██████████████                                        46
                            vllm-mlx   ████████████████                                      54
                            mlx-lm     █████████████████                                     56
                            llama.cpp                                           not supported
                            Ollama                                              not supported

Qwen3.5-122B-A10B           Rapid-MLX  █████████████                                         44
                            vllm-mlx   █████████████                                         43
                            mlx-lm     █████████████                                         45
                            llama.cpp                                           not supported
                            Ollama                                              not supported
```

*tok/s decode speed. ⚡ = fastest across all engines. Engines: [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX), [vllm-mlx](https://github.com/waybarrios/vllm-mlx) (upstream), [mlx-lm](https://github.com/ml-explore/mlx-examples), [Ollama](https://ollama.com), [llama.cpp](https://github.com/ggml-org/llama.cpp).*

### Time to First Token (TTFT) — multi-turn

Prompt caching makes multi-turn conversations feel instant. Rapid-MLX is **1.3-4.1x faster** than upstream vllm-mlx on first-token latency across most models.

| Model | Rapid-MLX | vllm-mlx | Ollama | llama.cpp | vs upstream |
|:---|:---:|:---:|:---:|:---:|:---:|
| Hermes-3-Llama 8B | **0.08s** | 0.18s | — | — | **2.3x** |
| Qwen3.5-9B | **0.18s** | 0.26s | 0.27s | — | **1.4x** |
| GLM-4.7-Flash 9B | **0.15s** | 0.23s | — | — | **1.5x** |
| Phi-4 Mini 14B | 0.10s | 0.15s | 0.06s | **0.03s** | 1.5x |
| GPT-OSS 20B | **0.12s** | 0.27s | — | — | **2.3x** |
| Mistral Small 24B | **0.10s** | 0.38s | — | — | **3.8x** |
| Devstral-Small-2 24B | **0.10s** | 0.38s | — | — | **3.7x** |
| Qwen3.5-35B-A3B | **0.19s** | 0.24s | — | — | **1.3x** |
| Qwen3-Coder-Next 80B | **0.10s** | 0.27s | — | — | **2.6x** |
| GLM-4.5-Air | **0.12s** | 0.47s | — | — | **4.1x** |

*Seconds to first visible token in a multi-turn conversation. Lower is better. Bold = fastest engine for that model. Ollama has no prompt cache — every turn re-prefills the full context.*

### What fits my Mac?

Model weights must fit in unified memory. If Activity Monitor shows red memory pressure, the model is too big — switch to a smaller one or a lower quantization.

| Your Mac | Best Model | RAM Used | Speed | Quality |
|----------|-----------|---------|-------|---------|
| **16 GB** MacBook Air/Pro | [Qwen3.5-4B 4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) | 2.4 GB | 158 tok/s | Good for chat and simple tasks |
| **24 GB** MacBook Pro | [Qwen3.5-9B 4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | 5.1 GB | 106 tok/s | Great all-rounder |
| **32 GB** Mac Mini / Studio | [Qwen3.5-27B 4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | 15.3 GB | 38 tok/s | Solid coding model |
| **64 GB** Mac Mini / Studio | [Qwen3.5-35B-A3B 8bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-8bit) | 37 GB | 80 tok/s | **Sweet spot** — smart + fast |
| **96 GB** Mac Studio / Pro | [Qwen3.5-122B mxfp4](https://huggingface.co/nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx) | 65 GB | 57 tok/s | Best model, fits comfortably |
| **128+ GB** Mac Studio / Pro | [Qwen3.5-122B 8bit](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-8bit) | 130 GB | 43 tok/s | Maximum quality |

### Copy-paste commands

Pick the one that matches your Mac:

```bash
# 16 GB — lightweight, fast
rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit --tool-call-parser hermes --port 8000

# 24 GB — best small model
rapid-mlx serve mlx-community/Qwen3.5-9B-4bit --tool-call-parser hermes --port 8000

# 64 GB — sweet spot
rapid-mlx serve mlx-community/Qwen3.5-35B-A3B-8bit --tool-call-parser hermes --prefill-step-size 8192 --port 8000

# 96+ GB — best model
rapid-mlx serve nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx --tool-call-parser hermes --kv-bits 8 --prefill-step-size 8192 --port 8000

# Coding agent — fast MoE, great for Claude Code / Cursor
rapid-mlx serve lmstudio-community/Qwen3-Coder-Next-MLX-4bit --tool-call-parser hermes --prefill-step-size 8192 --port 8000

# Vision — image understanding
rapid-mlx serve mlx-community/Qwen3-VL-4B-Instruct-MLX-4bit --mllm --port 8000
```

<details>
<summary><strong>Other model families (DeepSeek, Mistral, GLM, Llama)</strong></summary>

| Model Family | `--tool-call-parser` | `--reasoning-parser` | Notes |
|-------------|---------------------|---------------------|-------|
| Qwen3.5 (all sizes) | `hermes` | `qwen3` | **Recommended** — 100% tool calling |
| Qwen3-Coder-Next | `hermes` | *(none)* | Fast coding, non-thinking mode |
| GLM-4.7 | `glm47` | *(none)* | 100% tool calling |
| MiniMax-M2.5 | `minimax` | `minimax` | XML tool format |
| GPT-OSS | `seed_oss` | `gpt_oss` | Native format |
| DeepSeek-R1 | `deepseek` | `deepseek_r1` | With reasoning |
| Llama 3.x | `llama` | *(none)* | JSON tool format |
| Mistral | `hermes` | *(none)* | Hermes-compatible |

All 17 parsers include automatic recovery — if a quantized model outputs broken tool calls as text, they're auto-converted back to structured format.

</details>

---

## Works With

| Client | Status | Notes |
|--------|--------|-------|
| [Claude Code](https://claude.ai/claude-code) | Verified | Env var config, streaming tools |
| [Cursor](https://cursor.com) | Verified | Settings UI config |
| [Aider](https://aider.chat) | Verified | Code editing agent |
| [Open WebUI](https://github.com/open-webui/open-webui) | Verified | Self-hosted ChatGPT UI, Docker one-liner |
| [Continue.dev](https://continue.dev) | Verified | YAML config, VS Code + JetBrains |
| [OpenClaw](https://github.com/nicepkg/openclaw) | Verified | 14 tools, multi-round, streaming |
| [OpenCode](https://github.com/opencode-ai/opencode) | Verified | JSON config |
| [LangChain](https://langchain.com) | Compatible | Standard OpenAI client |
| Any OpenAI SDK client | Compatible | Drop-in `base_url` swap |

<details>
<summary><strong>Client setup instructions</strong></summary>

**Claude Code:**
```bash
OPENAI_BASE_URL=http://localhost:8000/v1 claude
# Or add to ~/.claude/settings.json:
# { "env": { "OPENAI_BASE_URL": "http://localhost:8000/v1" } }
```

**Cursor:** Settings > Models > OpenAI API Base → `http://localhost:8000/v1`

**Continue.dev** (`~/.continue/config.yaml`):
```yaml
models:
  - name: rapid-mlx
    provider: openai
    model: default
    apiBase: http://localhost:8000/v1
    apiKey: not-needed
```

**Aider:**
```bash
aider --openai-api-base http://localhost:8000/v1 --openai-api-key not-needed
```

**Open WebUI** (Docker one-liner):
```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e ENABLE_OLLAMA_API=False \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

**OpenCode** (`~/.config/opencode/opencode.json`):
```json
{
  "provider": {
    "openai-compatible": {
      "apiKey": "not-needed",
      "models": {
        "default": {
          "id": "default",
          "name": "rapid-mlx local",
          "api_base": "http://localhost:8000/v1"
        }
      }
    }
  }
}
```

</details>

> **Community-maintained tables.** Tested on M3 Ultra so far — if you verify a client or model on your hardware, [open an issue](https://github.com/raullenchai/Rapid-MLX/issues) or PR to update.

---

## Benchmarks

*This section is for ML engineers who want the numbers. If you just want to run models, see [Quick Start](#quick-start) and [Choose Your Model](#choose-your-model) above.*

15 models tested across 5 engines on **Mac Studio M3 Ultra (256GB)**. Same model, same hardware, head-to-head.

**Rapid-MLX is the fastest or tied on 11 of 13 models** vs upstream vllm-mlx, mlx-lm, Ollama, and llama.cpp.

| Model | Rapid-MLX | Best Alternative | Speedup |
|-------|----------|-----------------|---------|
| **Phi-4 Mini 14B** | **174** tok/s | 77 (mlx-lm) / 51 (Ollama) | **2.3x** / **3.4x** |
| **Qwen3.5-4B** | **~158** tok/s | 168 (mlx-lm) | ~1.0x |
| **GPT-OSS 20B** | **123** tok/s | 79 (mlx-lm / upstream) | **1.6x** |
| **Hermes-3-Llama 8B** | **123** tok/s | 127 (mlx-lm) | ~1.0x |
| **Qwen3.5-9B** | **109** tok/s | 61 (mlx-lm) / 26 (Ollama) | **1.8x** / **4.2x** |
| **Qwen3.5-35B-A3B** | **82** tok/s | 85 (mlx-lm) | ~1.0x |
| **Qwen3-Coder 80B** | **74** tok/s | 76 (mlx-lm) | ~1.0x |
| **GLM-4.7-Flash 9B** | **60** tok/s | 56 (upstream) | 1.07x |
| **Devstral-Small-2 24B** | **48** tok/s | 49 (mlx-lm) | ~1.0x |
| **Mistral Small 24B** | **48** tok/s | 41 (mlx-lm) | **1.2x** |
| **Qwen3.5-122B-A10B** | **44** tok/s | 45 (mlx-lm) | ~1.0x |
| **Qwen3.5-27B** | 39 tok/s | 39 (mlx-lm) | ~1.0x |
| Gemma 3 12B | 49 tok/s | 73 (mlx-lm) / 54 (Ollama) | 0.7x |

> **Why faster than Ollama/llama.cpp?** They use C++ with generic Metal shaders. Rapid-MLX uses Apple's [MLX framework](https://github.com/ml-explore/mlx) — purpose-built for Apple Silicon unified memory with native Metal compute kernels and zero-copy GPU access.
>
> **Where it's slower:** Gemma 3 doesn't benefit from MLX optimizations. We report honestly.
>
> **Same speed = still wins.** On models where decode is tied with mlx-lm, Rapid-MLX adds prompt cache (10-30x faster TTFT), 17 tool parsers, reasoning separation, vision, audio, and cloud routing — with zero speed overhead.

### TTFT — Prompt Cache Advantage

Prompt cache keeps multi-turn conversations fast. Ollama and mlx-lm re-prefill the full context every turn.

| Model | Rapid-MLX (cached) | vllm-mlx (upstream) | Speedup |
|-------|-------------------|-------------------|---------|
| Hermes-3-Llama 8B | **0.08s** | 0.18s | 2.3x |
| Phi-4 Mini 14B | **0.10s** | 0.15s | 1.5x |
| Qwen3-Coder-Next 80B | **0.10s** | 0.27s | 2.7x |
| Devstral-Small-2 24B | **0.10s** | 0.38s | 3.8x |
| Mistral Small 24B | **0.10s** | 0.38s | 3.8x |
| GLM-4.5-Air | **0.12s** | 0.47s | 3.9x |
| Qwen3.5-9B | **0.18s** | 0.26s | 1.4x |

### Capability Comparison

| Feature | Rapid-MLX | Ollama | llama.cpp | mlx-lm |
|---------|-----------|--------|-----------|--------|
| **Tool calling** | 100% (Qwen) | 100% (Qwen) | 80% (Phi-4) | N/A |
| **Tool call recovery** | 100% | 100% | 100% | N/A |
| **Think-tag leak** | 0% | 0% | 0% | N/A |
| **Prompt cache** | ✓ (0.1s TTFT) | ✗ (0.27s) | ✗ | ✗ |
| **Vision** | ✓ | ✓ | ✗ | ✗ |
| **Audio (STT/TTS)** | ✓ | ✗ | ✗ | ✗ |
| **17 tool parsers** | ✓ | ✗ | ✗ | ✗ |
| **Cloud routing** | ✓ | ✗ | ✗ | ✗ |
| **Streaming** | ✓ | ✓ | ✓ | ✗ |
| **OpenAI API** | ✓ | ✓ | ✓ | ✗ |

<details>
<summary><strong>Eval benchmarks (17 models, 4 suites)</strong></summary>

17 models across tool calling (30 scenarios), coding (HumanEval+), reasoning (MATH-500), and general knowledge (MMLU-Pro). All with `enable_thinking: false` on M3 Ultra.

| Model | Quant | RAM | Decode | Tools | Code | Reason | General | Avg |
|-------|-------|-----|--------|-------|------|--------|---------|-----|
| Qwen3.5-122B-A10B | 8bit | 129.8 GB | 43 t/s | 87% | **90%** | **90%** | **90%** | **89%** |
| Qwen3.5-122B-A10B | mxfp4 | 65.0 GB | 57 t/s | **90%** | **90%** | 80% | **90%** | 88% |
| Qwen3.5-35B-A3B | 8bit | 36.9 GB | 80 t/s | **90%** | **90%** | 80% | 80% | 85% |
| Qwen3-Coder-Next | 6bit | 64.8 GB | 66 t/s | 87% | **90%** | 80% | 70% | 82% |
| Qwen3-Coder-Next | 4bit | 44.9 GB | 74 t/s | **90%** | **90%** | 70% | 70% | 80% |
| GLM-4.5-Air | 4bit | 60.3 GB | 54 t/s | 73% | **90%** | 70% | 80% | 78% |
| GLM-4.7-Flash | 8bit | 31.9 GB | 57 t/s | 73% | **100%** | **90%** | 50% | 78% |
| Qwen3.5-27B | 4bit | 15.3 GB | 38 t/s | 83% | **90%** | 50% | 80% | 76% |
| Qwen3.5-35B-A3B | 4bit | 19.6 GB | 95 t/s | 87% | **90%** | 50% | 70% | 74% |
| Qwen3.5-9B | 4bit | 5.1 GB | 106 t/s | 83% | 70% | 60% | 70% | 71% |
| MiniMax-M2.5 | 4bit | 128.9 GB | 50 t/s | 87% | 10%\* | 80% | **90%** | 67% |
| Devstral-Small-2 | 4bit | 13.4 GB | 47 t/s | 17% | **90%** | 70% | 70% | 62% |
| GPT-OSS-20B | mxfp4-q8 | 12.1 GB | 124 t/s | 80% | 20% | 60% | **90%** | 62% |
| Qwen3.5-4B | 4bit | 2.4 GB | 158 t/s | 73% | 50% | 50% | 50% | 56% |
| Mistral-Small-3.2 | 4bit | 13.4 GB | 47 t/s | 17% | 80% | 60% | 60% | 54% |
| Hermes-3-Llama-8B | 4bit | 4.6 GB | 123 t/s | 17% | 20% | 30% | 40% | 27% |
| Qwen3-0.6B | 4bit | 0.4 GB | 365 t/s | 30% | 20% | 20% | 30% | 25% |

\* *MiniMax coding score likely affected by a code extraction parser issue, not model capability.*

</details>

*Benchmark script: [`scripts/benchmark_engines.py`](scripts/benchmark_engines.py). Run your own: `python scripts/benchmark_engines.py --engine rapid-mlx ollama --runs 3`. Eval suites: [evals/](evals/)*

---

## Features

### Tool Calling

Full OpenAI-compatible tool calling with 17 parser formats and **automatic recovery when quantized models break**. Models at 4-bit degrade after multiple tool rounds — Rapid-MLX auto-detects broken output and converts it back to structured `tool_calls`.

### Reasoning Separation

Models with chain-of-thought (Qwen3, DeepSeek-R1) output reasoning in a separate `reasoning_content` field — never mixed into `content`. 0% leak rate.

### Prompt Cache

Persistent KV cache across requests — only new tokens are prefilled on each turn. 10-30x faster multi-turn conversations vs cold prefill. Always on, no flags needed.

### Smart Cloud Routing

Large-context requests auto-route to a cloud LLM (GPT-5, Claude, etc.) when local prefill would be slow. Routing based on new tokens after cache hit. `--cloud-model openai/gpt-5 --cloud-threshold 20000`

### Multimodal

Vision, audio (STT/TTS), video understanding, and text embeddings — all through the same OpenAI-compatible API.

<details>
<summary><strong>All features (32 total)</strong></summary>

**Tool Calling (13):** Text-format recovery, 17 parsers, streaming, tool logits bias (2-5x faster structured output), disconnect guard, think-tag filter, chunk-boundary leak fix, developer role normalization, logprobs API, system prompt auto-injection, end-to-end agent simulation tests.

**Reasoning (3):** MiniMax/Qwen3/DeepSeek parsers, Chinese reasoning pattern recognition, clean `reasoning_content` field.

**Performance (6):** Prompt cache, configurable prefill step size, KV cache quantization (4/8 bit), speculative decoding, cloud routing, frequency-aware cache eviction.

**Reliability (6):** Accurate `prompt_tokens` reporting, EOS cache fix, crash prevention on malformed `response_format`, GC control during generation, system prompt pinning, 1500+ tests.

**Multimodal (4):** Vision (Qwen-VL), audio STT (Whisper), audio TTS (Kokoro), text embeddings.

</details>

---

<details>
<summary><strong>Server Flags Reference</strong></summary>

### Core

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | HuggingFace model name or local path | *(required)* |
| `--host` | Host to bind to | `0.0.0.0` |
| `--port` | Port to bind to | `8000` |
| `--max-tokens` | Default max tokens for generation | `32768` |
| `--continuous-batching` | Multi-user mode with scheduler | off |

### Tool Calling & Reasoning

| Flag | Description | Default |
|------|-------------|---------|
| `--tool-call-parser` | Parser: `hermes`, `minimax`, `qwen`, `llama`, `deepseek`, etc. | *(none)* |
| `--reasoning-parser` | Parser: `qwen3`, `deepseek_r1`, `minimax`, `gpt_oss` | *(none)* |
| `--enable-tool-logits-bias` | Jump-forward decoding for faster tool calls | off |

### Performance

| Flag | Description | Default |
|------|-------------|---------|
| `--prefill-step-size` | Tokens per prefill chunk | `2048` |
| `--kv-bits` | KV cache quantization: `4` or `8` bit | *(full precision)* |
| `--draft-model` | Draft model for speculative decoding | *(none)* |
| `--num-draft-tokens` | Speculative tokens per step | `4` |

### Cloud Routing

| Flag | Description | Default |
|------|-------------|---------|
| `--cloud-model` | litellm model string (e.g. `openai/gpt-5`) | *(disabled)* |
| `--cloud-threshold` | New token threshold to trigger cloud routing | `20000` |

### Security & Other

| Flag | Description | Default |
|------|-------------|---------|
| `--api-key` | API key for authentication | *(no auth)* |
| `--rate-limit` | Requests per minute per client | *(unlimited)* |
| `--timeout` | Request timeout in seconds | `300` |
| `--mllm` | Force multimodal (vision) mode | auto-detect |
| `--mcp-config` | MCP configuration file for tool integration | *(none)* |
| `--embedding-model` | Pre-load embedding model at startup | *(none)* |

</details>

<details>
<summary><strong>Architecture</strong></summary>

```
                    ┌──────────────────────────────────────┐
                    │     OpenAI-compatible API (port 8000) │
                    │    /v1/chat/completions, /v1/models   │
                    └──────────────────┬───────────────────┘
                                       │
                              ┌────────┴────────┐
                              │  Cloud Router   │ (optional)
                              │  new_tokens >   │
                              │  threshold?     │
                              └───┬─────────┬───┘
                            yes   │         │  no
                     ┌────────────┘         └──────────────┐
                     ▼                                     ▼
          ┌─────────────────┐               ┌──────────────────────┐
          │  Cloud LLM      │               │   Local MLX Engine   │
          │  (via litellm)  │               │                      │
          │  GPT-5, Claude, │               │  ┌────────────────┐  │
          │  Gemini, etc.   │               │  │ SimpleEngine   │  │
          └─────────────────┘               │  │ + prompt cache │  │
                                            │  └───────┬────────┘  │
                                            │          │           │
                                            │  ┌───────┴────────┐  │
                                            │  │  mlx-lm/mlx-vlm│  │
                                            │  │  MLX + Metal   │  │
                                            │  └────────────────┘  │
                                            └──────────────────────┘
```

**SimpleEngine** (default) — Single-user, persistent prompt cache, maximum throughput.

**BatchedEngine** (`--continuous-batching`) — Multi-user, paged KV cache with prefix sharing.

**Cloud Router** (`--cloud-model`) — Routes large-context cold requests to cloud. Routing based on new tokens after cache hit, not total input.

</details>

<details>
<summary><strong>Troubleshooting</strong></summary>

**"parameters not found in model" warnings at startup** — Normal for VLMs. Vision weights are auto-skipped.

**Out of memory / very slow (<5 tok/s)** — Model too big. Check [What fits my Mac?](#what-fits-my-mac) Use `--kv-bits 4` for long contexts. Close other apps.

**Empty responses** — Remove `--reasoning-parser` for non-thinking models. Only use it with Qwen3 (thinking), MiniMax, DeepSeek-R1.

**Tool calls as plain text** — Set the correct `--tool-call-parser` for your model. Even without it, Rapid-MLX auto-recovers most cases.

**Slow first response** — Cold start is normal. Subsequent turns hit prompt cache (10-30x faster). Use `--prefill-step-size 8192` to speed up cold starts.

**Server hangs after client disconnect** — Fixed in this fork. Upgrade to latest.

</details>

---

## Roadmap

| Technique | Expected Gain | Status |
|-----------|---------------|--------|
| [MTP in SimpleEngine](https://arxiv.org/abs/2404.19737) — multi-token prediction | 1.4x decode | In progress |
| [Standard Speculative Decode](https://arxiv.org/abs/2302.01318) — draft model acceleration | 1.5-2.3x decode | Not started |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) — feature-level draft on Metal | 3-6.5x decode | Not started |
| [ReDrafter](https://arxiv.org/abs/2403.09919) — Apple's RNN draft head | 1.4-1.5x decode | Not started |
| Auto-optimization per model — zero-config best settings | N/A | Not started |

---

## Contributing

Issues and PRs welcome at [github.com/raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX).

We need community data — hardware benchmarks, client verifications, model reports. If you test a model on your Mac, [open an issue](https://github.com/raullenchai/Rapid-MLX/issues/new) with your hardware, model, decode speed, and what worked.

## License

Apache 2.0 — see [LICENSE](LICENSE).
