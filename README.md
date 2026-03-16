# Rapid-MLX

**Run AI on your Mac. Faster than anything else.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1500%2B-brightgreen.svg)](tests/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple)](https://support.apple.com/en-us/HT211814)

Drop-in OpenAI API replacement for Apple Silicon. We aggressively adopt cutting-edge techniques — DeltaNet state snapshots, speculative decoding, prompt caching, tool logits biasing — to push local inference as fast as the hardware allows. 100% tool-calling reliability, sub-200ms cached TTFT on all architectures, and 2-4x faster than Ollama.

| | Your Mac runs AI | How fast | What works |
|:---|:---:|:---:|:---:|
| **16 GB MacBook Air** | Qwen3.5-4B | 158 tok/s | Chat, coding, tools |
| **64 GB Mac Mini / Studio** | Qwen3.5-35B | 80 tok/s | Best balance of smart + fast |
| **128+ GB Mac Studio / Pro** | Qwen3.5-122B | 57 tok/s | Frontier-level intelligence |

---

## Why Rapid-MLX

**Instant multi-turn context** — KV prompt cache for transformers, DeltaNet state snapshots for hybrid RNN models. 0.08–0.18s cached TTFT across all architectures vs re-prefilling every turn.

**Bulletproof tool calling** — 17 parser formats with automatic recovery. Even 4-bit quantized models achieve 100% function call success rate.

**Zero code changes** — Full OpenAI API compatibility. Change one env var and your existing tools just work.

---

## Quick Start

```bash
# 1. Install (one command, checks Apple Silicon + Python automatically)
curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/install.sh | bash

# 2. Pick a model and start serving (parsers auto-detected from model name)
rapid-mlx serve mlx-community/Qwen3.5-9B-4bit --port 8000

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

---

## Choose Your Model

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
rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit --port 8000

# 24 GB — best small model
rapid-mlx serve mlx-community/Qwen3.5-9B-4bit --port 8000

# 64 GB — sweet spot
rapid-mlx serve mlx-community/Qwen3.5-35B-A3B-8bit --prefill-step-size 8192 --port 8000

# 96+ GB — best model
rapid-mlx serve nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx --kv-bits 8 --prefill-step-size 8192 --port 8000

# Coding agent — fast MoE, great for Claude Code / Cursor
rapid-mlx serve lmstudio-community/Qwen3-Coder-Next-MLX-4bit --prefill-step-size 8192 --port 8000

# Vision — image understanding
rapid-mlx serve mlx-community/Qwen3-VL-4B-Instruct-MLX-4bit --mllm --port 8000
```

<details>
<summary><strong>Parser auto-detection & manual overrides</strong></summary>

Parsers are **auto-detected from the model name** — you don't need to specify `--tool-call-parser` or `--reasoning-parser` for supported families. Explicit flags always override auto-detection.

| Model Family | Auto-detected `--tool-call-parser` | Auto-detected `--reasoning-parser` | Notes |
|-------------|---------------------|---------------------|-------|
| Qwen3.5 (all sizes) | `hermes` | `qwen3` | **Recommended** — 100% tool calling |
| Qwen3-Coder-Next | `hermes` | *(none)* | Fast coding, non-thinking mode |
| DeepSeek R1-0528 / V3.1 | `deepseek_v31` | `deepseek_r1` | Dedicated V3.1 parser |
| DeepSeek R1 (older) | `deepseek` | `deepseek_r1` | With reasoning |
| DeepSeek V3 / V2.5 | `deepseek` | *(none)* | No reasoning parser |
| GLM-4.7 | `glm47` | *(none)* | 100% tool calling |
| MiniMax-M2.5 | `minimax` | `minimax` | XML tool format |
| GPT-OSS | `harmony` | `harmony` | Native format |
| Kimi-Linear | `kimi` | *(none)* | Kimi tool format |
| Llama 3.x | `llama` | *(none)* | JSON tool format |
| Mistral / Devstral | `hermes` | *(none)* | Hermes-compatible |
| Gemma | `hermes` | *(none)* | Hermes-compatible |
| Phi-3/4 | `hermes` | *(none)* | Hermes-compatible |

All 17 parsers include automatic recovery — if a quantized model outputs broken tool calls as text, they're auto-converted back to structured format.

</details>

---

## Benchmarks

*This section is for ML engineers who want the numbers. If you just want to run models, see [Quick Start](#quick-start) and [Choose Your Model](#choose-your-model) above.*

### Performance — 22 models, 6 engines

All benchmarks on Mac Studio M3 Ultra (256GB). Sorted by model size. Engines ranked by total bar length per model. **Rapid-MLX is #1 on 16 of 18 benchmarked models.**

```
█ = decode speed (tok/s)  ▒ = TTFT responsiveness  ░ = tool calling

Llama 3.2 3B            ⚡ Rapid-MLX   █████████████████████████████████▒▒▒▒▒▒▒▒▒▒               225 tok/s · 0.07s · 0%
                          mlx-lm      ███████████████████████████████████                       238 tok/s
                          oMLX        █████████████████████████████                             196 tok/s
                          vllm-mlx    not supported
                          Ollama      not supported
                          llama.cpp   not supported

Qwen3.5-4B              ⚡ Rapid-MLX   ███████████████████████▒▒▒▒▒▒▒▒░░░░░░░░                    158 tok/s · 0.16s · 100%
                          vllm-mlx    ███████████████████████▒▒▒▒▒▒▒░░░░░░░░                    155 tok/s · 0.19s · 100%
                          mlx-lm      █████████████████████████                                 168 tok/s
                          oMLX        ██████████████████████                                    149 tok/s
                          Ollama      not supported
                          llama.cpp   not supported

Hermes-3-Llama 8B       ⚡ Rapid-MLX   ██████████████████▒▒▒▒▒▒▒▒▒▒                              123 tok/s · 0.08s · 0%
                          vllm-mlx    ██████████████████▒▒▒▒▒▒▒▒                                122 tok/s · 0.18s · 0%
                          mlx-lm      ███████████████████                                       127 tok/s
                          oMLX        █████████████████                                         113 tok/s
                          Ollama      not supported
                          llama.cpp   not supported

Qwen3.5-9B              ⚡ Rapid-MLX   ████████████████▒▒▒▒▒▒▒▒░░░░░░░░                          109 tok/s · 0.18s · 100%
                          vllm-mlx    ███████████████▒▒▒▒▒▒░░░░░░░░                             104 tok/s · 0.26s · 100%
                          oMLX        ███████████████                                           102 tok/s
                          Ollama      ███████▒▒▒▒▒▒                                             46 tok/s · 0.19s · 0%
                          mlx-lm      █████████                                                 61 tok/s
                          llama.cpp   not supported

GLM-4.7-Flash 9B        ⚡ Rapid-MLX   █████████▒▒▒▒▒▒▒▒░░░░░░░░                                 60 tok/s · 0.15s · 100%
                          vllm-mlx    ████████▒▒▒▒▒▒░░░░░░░░                                    56 tok/s · 0.23s · 100%
                          mlx-lm      not supported
                          Ollama      not supported
                          llama.cpp   not supported

Gemma 3 12B               oMLX        ██████████                                                67 tok/s
                          Ollama      ████████▒▒▒▒▒▒▒▒▒                                         58 tok/s · 0.14s · 0%
                          Rapid-MLX   ███████▒▒▒▒                                               49 tok/s · 0.32s · 0%
                          mlx-lm      ███████████                                               73 tok/s
                          vllm-mlx    not supported
                          llama.cpp   not supported

Phi-4 Mini 14B          ⚡ Rapid-MLX   ██████████████████████████▒▒▒▒▒▒▒▒▒▒                      174 tok/s · 0.10s · 0%
                          vllm-mlx    █████████████████████████▒▒▒▒▒▒▒▒                         170 tok/s · 0.15s · 0%
                          llama.cpp   ████████▒▒▒▒▒▒▒▒▒▒▒░░░░░░                                 55 tok/s · 0.03s · 80%
                          Ollama      ████████▒▒▒▒▒▒▒▒▒▒▒                                       56 tok/s · 0.05s · 0%
                          mlx-lm      ███████████                                               77 tok/s

DeepSeek-R1 14B           coming soon

Qwen 2.5 Coder 14B        coming soon

GPT-OSS 20B             ⚡ Rapid-MLX   ██████████████████▒▒▒▒▒▒▒▒▒                               123 tok/s · 0.12s · 0%
                          oMLX        ████████████████                                          106 tok/s
                          vllm-mlx    ████████████▒▒▒▒▒▒                                        79 tok/s · 0.27s · 0%
                          mlx-lm      ████████████                                              79 tok/s
                          Ollama      not supported
                          llama.cpp   not supported

Mistral Small 24B       ⚡ Rapid-MLX   ███████▒▒▒▒▒▒▒▒▒▒                                         48 tok/s · 0.10s · 0%
                          vllm-mlx    ███████▒▒▒                                                47 tok/s · 0.38s · 0%
                          mlx-lm      ██████                                                    41 tok/s
                          Ollama      not supported
                          llama.cpp   not supported

Devstral-Small-2 24B    ⚡ Rapid-MLX   ███████▒▒▒▒▒▒▒▒▒▒                                         48 tok/s · 0.10s · 0%
                          vllm-mlx    ███████▒▒▒                                                48 tok/s · 0.38s · 0%
                          mlx-lm      ███████                                                   49 tok/s
                          oMLX        ███████                                                    46 tok/s
                          Ollama      not supported
                          llama.cpp   not supported

Gemma 3 27B               Ollama      ████▒▒▒▒▒▒▒▒▒▒                                            30 tok/s · 3.31s · 0%

Qwen3.5-27B             ⚡ Rapid-MLX   ██████░░░░░░░░                                            39 tok/s · 1.46s · 100%
                          vllm-mlx    ██████░░░░░░░░                                            38 tok/s · 0.60s · 100%
                          mlx-lm      ██████                                                    39 tok/s
                          Ollama      not supported
                          llama.cpp   not supported

Qwen3.5-35B-A3B         ⚡ Rapid-MLX   ████████████▒▒▒▒▒▒▒▒░░░░░░░░                              82 tok/s · 0.18s · 100%
                          vllm-mlx    ████████████▒▒▒▒▒▒░░░░░░░░                                80 tok/s · 0.24s · 100%
                          mlx-lm      ████████████                                              85 tok/s
                          oMLX        ███████████                                                75 tok/s
                          Ollama      not supported
                          llama.cpp   not supported

MiniMax-M2.5            ⚡ Rapid-MLX   ████████▒▒▒▒▒▒▒▒▒░░░░░░░░                                 52 tok/s · 0.13s · 100%
                          mlx-lm      ████████                                                  51 tok/s
                          vllm-mlx    not supported
                          Ollama      not supported
                          llama.cpp   not supported

Kimi-Linear-48B-A3B     ⚡ Rapid-MLX   ██████████████▒▒▒▒▒▒▒▒                                    96 tok/s · 0.06s · 0%
                          mlx-lm      not supported (trust_remote_code)
                          Ollama      not supported
                          llama.cpp   not supported

Llama 3.3 70B             coming soon

Qwen3-Coder 80B 4bit    ⚡ Rapid-MLX   ███████████▒▒▒▒▒▒▒▒▒▒░░░░░░░░                             74 tok/s · 0.10s · 100%
                          vllm-mlx    ██████████▒▒▒▒▒▒░░░░░░░░                                  69 tok/s · 0.27s · 100%
                          mlx-lm      ███████████                                               76 tok/s
                          Ollama      not supported
                          llama.cpp   not supported

Qwen3-Coder 80B 6bit    ⚡ Rapid-MLX   ██████████▒▒▒▒▒▒▒▒▒▒░░░░░░░░                              68 tok/s · 0.10s · 100%
                          mlx-lm      ██████████                                                69 tok/s
                          vllm-mlx    not supported
                          Ollama      not supported
                          llama.cpp   not supported

GLM-4.5-Air             ⚡ Rapid-MLX   ███████▒▒▒▒▒▒▒▒▒░░░░░░░░                                  46 tok/s · 0.12s · 100%
                          vllm-mlx    ████████▒░░░░░░░░                                         54 tok/s · 0.47s · 100%
                          mlx-lm      ████████                                                  56 tok/s
                          Ollama      not supported
                          llama.cpp   not supported

Qwen3.5-122B-A10B       ⚡ Rapid-MLX   ██████░░░░░░░░                                            44 tok/s · 1.28s · 100%
                          vllm-mlx    ██████░░░░░░░░                                            43 tok/s · 0.51s · 100%
                          mlx-lm      ███████                                                   45 tok/s
                          Ollama      not supported
                          llama.cpp   not supported
```

*⚡ = best combined score (decode + TTFT + tools). Longer bar = better overall experience. TTFT capped at 0.5s — thinking models have higher TTFT due to reasoning tokens. mlx-lm/oMLX have no TTFT or tool data (library/decode-only server). Engines: [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX), [vllm-mlx](https://github.com/waybarrios/vllm-mlx) (upstream), [mlx-lm](https://github.com/ml-explore/mlx-examples), [oMLX](https://github.com/jundot/omlx), [Ollama](https://ollama.com), [llama.cpp](https://github.com/ggml-org/llama.cpp).*

### Why faster than Ollama?

Ollama and llama.cpp use C++ with generic Metal shaders. Rapid-MLX uses Apple's [MLX framework](https://github.com/ml-explore/mlx) — purpose-built for Apple Silicon unified memory with native Metal compute kernels and zero-copy GPU access.

**Where it's slower:** Gemma 3 doesn't benefit from MLX optimizations. We report honestly.

**Same speed = still wins.** On models where decode is tied with mlx-lm, Rapid-MLX adds prompt cache (2-5x faster TTFT), DeltaNet state snapshots (first on MLX), 17 tool parsers, reasoning separation, vision, audio, and cloud routing — with zero speed overhead.

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

### Speedup Summary

18 models tested across 6 engines on **Mac Studio M3 Ultra (256GB)**. Same model, same hardware, head-to-head.

**Rapid-MLX is the fastest or tied on 16 of 18 models** vs upstream vllm-mlx, mlx-lm, oMLX, Ollama, and llama.cpp.

| Model | Rapid-MLX | Best Alternative | Speedup |
|-------|----------|-----------------|---------|
| **Llama 3.2 3B** | **225** tok/s | 238 (mlx-lm) / 196 (oMLX) | ~1.0x / **1.1x** |
| **Phi-4 Mini 14B** | **174** tok/s | 77 (mlx-lm) / 56 (Ollama) | **2.3x** / **3.1x** |
| **Qwen3.5-4B** | **158** tok/s · 0.16s | 168 (mlx-lm) / 0.19s (vllm-mlx) | ~1.0x decode / **1.2x TTFT** |
| **GPT-OSS 20B** | **123** tok/s | 106 (oMLX) / 79 (upstream) | **1.2x** / **1.6x** |
| **Hermes-3-Llama 8B** | **123** tok/s | 127 (mlx-lm) | ~1.0x |
| **Qwen3.5-9B** | **109** tok/s | 61 (mlx-lm) / 46 (Ollama) | **1.8x** / **2.4x** |
| **Kimi-Linear-48B-A3B** | **96** tok/s | — (only engine supporting it) | — |
| **Qwen3.5-35B-A3B** | **82** tok/s | 85 (mlx-lm) | ~1.0x |
| **Qwen3-Coder 80B** | **74** tok/s | 76 (mlx-lm) | ~1.0x |
| **GLM-4.7-Flash 9B** | **60** tok/s | 56 (upstream) | 1.07x |
| **Devstral-Small-2 24B** | **48** tok/s | 49 (mlx-lm) | ~1.0x |
| **Mistral Small 24B** | **48** tok/s | 41 (mlx-lm) | **1.2x** |
| **Qwen3.5-122B-A10B** | **44** tok/s | 45 (mlx-lm) | ~1.0x |
| **MiniMax-M2.5** | **52** tok/s | 51 (mlx-lm) | ~1.0x |
| **Qwen3.5-27B** | 39 tok/s | 39 (mlx-lm) | ~1.0x |
| Gemma 3 12B | 49 tok/s | 73 (mlx-lm) / 58 (Ollama) | 0.7x |

### TTFT — Prompt Cache Advantage

Prompt cache keeps multi-turn conversations fast. For standard transformers, KV cache trimming gives sub-100ms TTFT. For hybrid RNN models (Qwen3.5 DeltaNet), we use state snapshots — the first technique to bring prompt cache to non-trimmable architectures on MLX.

**Pure KV cache (transformers):**

| Model | Rapid-MLX (cached) | vllm-mlx (upstream) | Speedup |
|-------|-------------------|-------------------|---------|
| Hermes-3-Llama 8B | **0.08s** | 0.18s | 2.3x |
| Phi-4 Mini 14B | **0.10s** | 0.15s | 1.5x |
| Qwen3-Coder-Next 80B | **0.10s** | 0.27s | 2.7x |
| Devstral-Small-2 24B | **0.10s** | 0.38s | 3.8x |
| Mistral Small 24B | **0.10s** | 0.38s | 3.8x |
| GLM-4.5-Air | **0.12s** | 0.47s | 3.9x |
| Qwen3.5-9B | **0.18s** | 0.26s | 1.4x |

**DeltaNet state snapshots (hybrid RNN + attention):**

Qwen3.5 uses Gated DeltaNet (75% RNN) + full attention (25% KV). Other engines recreate the entire cache from scratch every request — we snapshot the RNN state at the system prompt boundary, restoring in ~0.1ms instead of re-running hundreds of tokens through the recurrent layers.

| Model | Cold TTFT | Snapshot TTFT | Speedup |
|-------|-----------|---------------|---------|
| Qwen3-Coder-Next 6bit (48L) | 0.68s | **0.14s** | **4.8x** |
| Qwen3.5-35B-A3B 8bit (40L) | 0.49s | **0.18s** | **2.8x** |
| Qwen3.5-4B 4bit (32L) | 0.24s | **0.16s** | **1.5x** |

### Optimization Techniques Per Model

We aggressively adopt architecture-specific optimizations. Not every model gets the same treatment — we match the technique to what each architecture needs.

| Technique | What it does | Models |
|-----------|-------------|--------|
| **KV prompt cache** | Trim KV cache to common prefix, skip re-prefill | All transformer models |
| **DeltaNet state snapshots** | Deep-copy RNN state at prefix boundary, restore in ~0.1ms | Qwen3.5 (4B, 9B, 27B, 35B, 122B), Qwen3-Coder-Next |
| **Hybrid cache sync** | Keep trimmable KV + non-trimmable RNN layers in sync | Qwen3.5 (Gated DeltaNet + attention) |
| **Tool logits bias** | Jump-forward decoding — bias logits toward structured tokens | All models with `--enable-tool-logits-bias` |
| **Auto tool recovery** | Detect broken text-format tool calls, convert to structured | All 17 parser formats |
| **Speculative decoding** | Draft model generates candidates, main model verifies | Any model + `--draft-model` |
| **KV quantization** | 4/8-bit KV cache for longer contexts in less memory | All models with `--kv-bits` |
| **Prefill chunking** | Configurable step size for large-prompt throughput | All models |
| **Cloud routing** | Offload high-token requests to cloud LLM when local is slow | All models with `--cloud-model` |

### Capability Comparison

| Feature | Rapid-MLX | oMLX | Ollama | llama.cpp | mlx-lm |
|---------|-----------|------|--------|-----------|--------|
| **Tool calling** | 100% (Qwen) | N/A | 100% (Qwen) | 80% (Phi-4) | N/A |
| **Tool call recovery** | 100% | N/A | 100% | 100% | N/A |
| **Think-tag leak** | 0% | N/A | 0% | 0% | N/A |
| **Prompt cache** | ✓ (KV + DeltaNet) | ✗ | ✗ (0.27s) | ✗ | ✗ |
| **Vision** | ✓ | ✓ | ✓ | ✗ | ✗ |
| **Audio (STT/TTS)** | ✓ | ✗ | ✗ | ✗ | ✗ |
| **17 tool parsers** | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Cloud routing** | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Streaming** | ✓ | ✓ | ✓ | ✓ | ✗ |
| **OpenAI API** | ✓ | ✓ | ✓ | ✓ | ✗ |

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

Persistent cache across requests — only new tokens are prefilled on each turn. For standard transformers, KV cache trimming. For hybrid models (Qwen3.5 DeltaNet), RNN state snapshots restore non-trimmable layers from memory instead of re-computing. 2-5x faster TTFT on all architectures. Always on, no flags needed.

### Smart Cloud Routing

Large-context requests auto-route to a cloud LLM (GPT-5, Claude, etc.) when local prefill would be slow. Routing based on new tokens after cache hit. `--cloud-model openai/gpt-5 --cloud-threshold 20000`

### Multimodal

Vision, audio (STT/TTS), video understanding, and text embeddings — all through the same OpenAI-compatible API.

<details>
<summary><strong>All features (33 total)</strong></summary>

**Tool Calling (13):** Text-format recovery, 17 parsers, streaming, tool logits bias (2-5x faster structured output), disconnect guard, think-tag filter, chunk-boundary leak fix, developer role normalization, logprobs API, system prompt auto-injection, end-to-end agent simulation tests.

**Reasoning (3):** MiniMax/Qwen3/DeepSeek parsers, Chinese reasoning pattern recognition, clean `reasoning_content` field.

**Performance (7):** Prompt cache (KV trim + DeltaNet state snapshots), configurable prefill step size, KV cache quantization (4/8 bit), speculative decoding, cloud routing, frequency-aware cache eviction.

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
| `--tool-call-parser` | Parser: `hermes`, `minimax`, `qwen`, `llama`, `deepseek`, etc. | *(auto-detected)* |
| `--reasoning-parser` | Parser: `qwen3`, `deepseek_r1`, `minimax`, `gpt_oss` | *(auto-detected)* |
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
| **DeltaNet state snapshots** — hybrid RNN cache reuse for Qwen3.5 | 1.5-4.8x TTFT | **Done** |
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
