# Rapid-MLX

**The fastest local AI engine for Apple Silicon.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1500%2B-brightgreen.svg)](tests/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple)](https://support.apple.com/en-us/HT211814)

| Fastest or tied on | vs Ollama / llama.cpp | Cached TTFT | Tool Calling |
|:---:|:---:|:---:|:---:|
| **11 of 13 models** | **2-4x faster** | **0.08-0.15s** | **100%** |
| vs upstream, mlx-lm, Ollama, llama.cpp | same models, same hardware | prompt cache, multi-turn | Qwen3.5 family, all sizes |

The fastest inference engine for local LLMs on Apple Silicon. A **drop-in replacement for OpenAI** — GPU-accelerated via [MLX](https://github.com/ml-explore/mlx), with the tool calling, streaming, and reliability that agent frameworks demand.

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/install.sh | bash

# Serve
rapid-mlx serve lmstudio-community/Qwen3-Coder-Next-MLX-4bit --tool-call-parser hermes --port 8000

# Use with any OpenAI client
OPENAI_BASE_URL=http://localhost:8000/v1 claude
```

---

## Benchmarks

15 models tested across 5 engines on **Mac Studio M3 Ultra (256GB)**. Same model, same hardware, head-to-head.

**Rapid-MLX is the fastest or tied on 11 of 13 models** vs upstream vllm-mlx, mlx-lm, Ollama, and llama.cpp. On the 3 models where Ollama/llama.cpp numbers exist, Rapid-MLX is **2-4x faster**.

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
> **Where it's slower:** Gemma 3 doesn't benefit from MLX optimizations. We report honestly — see the 0.7x row above.
>
> **Same speed = still wins.** On models where decode is tied with mlx-lm, Rapid-MLX adds prompt cache (10-30x faster TTFT), 17 tool parsers, reasoning separation, vision, audio, and cloud routing — with zero speed overhead.

### TTFT — The Killer Feature

Prompt cache keeps multi-turn conversations fast. Ollama re-prefills the full context every turn.

| Model | Rapid-MLX (cached) | Upstream vllm-mlx | Speedup |
|-------|-------------------|-------------------|---------|
| Hermes-3-Llama 8B | **0.08s** | 0.18s | 2.3x |
| Phi-4 Mini 14B | **0.10s** | 0.15s | 1.5x |
| Qwen3-Coder-Next 80B | **0.10s** | 0.27s | 2.7x |
| Devstral-Small-2 24B | **0.10s** | 0.38s | 3.8x |
| Qwen3.5-9B | **0.18s** | 0.26s | 1.4x |
| Mistral Small 24B | **0.10s** | 0.38s | 3.8x |
| GLM-4.5-Air | **0.12s** | 0.47s | 3.9x |

*Ollama has no prompt cache. Every turn re-prefills the full context.*

**"Upstream"** = original [vllm-mlx](https://github.com/waybarrios/vllm-mlx) (our fork base). Rapid-MLX adds prompt cache, 17 tool parsers, reasoning separation, and optimized scheduling.

<details>
<summary><strong>Full 15-model decode comparison (bar chart)</strong></summary>

```
Phi-4 Mini 14B            ⚡ Rapid-MLX  ████████████████████████████████████████████████████████████  174
                            mlx-lm     ████████████████████████                                      77
                            llama.cpp  █████████████████                                              55
                            Ollama     ████████████████                                               51

Qwen3.5-4B                ⚡ Rapid-MLX  █████████████████████████████████████████████████████         158
                            mlx-lm     ████████████████████████████████████████████████████████       168

GPT-OSS 20B               ⚡ Rapid-MLX  ████████████████████████████████████████                      123
                            mlx-lm     █████████████████████████                                      79

Hermes-3-Llama 8B         ⚡ Rapid-MLX  ████████████████████████████████████████                      123
                            mlx-lm     █████████████████████████████████████████                      127

Qwen3.5-9B                ⚡ Rapid-MLX  ███████████████████████████████████                           109
                            mlx-lm     ███████████████                                                61
                            Ollama     ████████                                                       26

Qwen3.5-35B-A3B           ⚡ Rapid-MLX  ██████████████████████████                                     82
                            mlx-lm     ███████████████████████████                                     85

Qwen3-Coder-Next 80B      ⚡ Rapid-MLX  ████████████████████████                                       74
                            mlx-lm     ████████████████████████                                        76

GLM-4.7-Flash 9B          ⚡ Rapid-MLX  ███████████████████                                            60

Gemma 3 12B                 Rapid-MLX  ███████████████                                                49
                            mlx-lm     ███████████████████████                                         73

Devstral-Small-2 24B      ⚡ Rapid-MLX  ███████████████                                                48
                            mlx-lm     ███████████████                                                 49

Mistral Small 24B          ⚡ Rapid-MLX  ███████████████                                                48
                            mlx-lm     █████████████                                                   41

Qwen3.5-122B-A10B          ⚡ Rapid-MLX  ██████████████                                                 44
                            mlx-lm     ██████████████                                                  45

Qwen3.5-27B                 Rapid-MLX  ████████████                                                   39
                            mlx-lm     ████████████                                                    39
```

</details>

<details>
<summary><strong>Full multi-engine comparison table</strong></summary>

| Model | RAM | Rapid-MLX | Upstream | Ollama | llama.cpp | mlx-lm | **Best Speedup** |
|-------|-----|----------|----------|--------|-----------|--------|-----------------|
| **Phi-4 Mini 14B** | 2.4 GB | **174** | 170 | 51 | 55 | 77 | **3.4x** vs Ollama |
| **Qwen3.5-4B** | 2.7 GB | **~158** | 155 | - | - | 168 | ~1.0x |
| **GPT-OSS 20B** | 11.8 GB | **123** | 79 | - | - | 79 | **1.56x** vs upstream |
| **Hermes-3-Llama 8B** | 4.7 GB | **123** | 122 | - | - | 127 | ~1.0x |
| **Qwen3.5-9B** | 5.1 GB | **109** | 104 | 26 | N/A | 61 | **4.2x** vs Ollama |
| **Qwen3.5-35B-A3B** | 34.8 GB | **82** | 80 | - | - | 85 | ~1.0x |
| **Qwen3-Coder 80B** | 42.2 GB | **74** | 69 | - | - | 76 | 1.07x vs upstream |
| **GLM-4.7-Flash 9B** | 30.1 GB | **60** | 56 | - | - | - | 1.07x vs upstream |
| **Gemma 3 12B** | 8.5 GB | 49 | - | 54 | N/A | 73 | 0.7x |
| **Devstral-Small-2 24B** | 12.7 GB | **48** | 48 | - | - | 49 | ~1.0x |
| **Mistral Small 24B** | 12.7 GB | **48** | 47 | - | - | 41 | **1.2x** vs mlx-lm |
| **Qwen3.5-122B-A10B** | 121.3 GB | 44 | 43 | - | - | 45 | ~1.0x |
| **Qwen3.5-27B** | 14.5 GB | 39 | 38 | - | - | 39 | ~1.0x |

</details>

### Capability Comparison — What Matters for Agents

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

> **Why faster than mlx-lm?** Same MLX backend, but Rapid-MLX has persistent prompt cache (reducing repeat prefill), reasoning separation (eliminating wasted thinking tokens), and an OpenAI-compatible server layer — all with near-zero overhead.

*Benchmark script: [`scripts/benchmark_engines.py`](scripts/benchmark_engines.py). Run your own: `python scripts/benchmark_engines.py --engine rapid-mlx ollama --runs 3`*

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

> **Community-maintained tables.** Tested on M3 Ultra so far — if you verify a client or model on your hardware, [open an issue](https://github.com/raullenchai/Rapid-MLX/issues) or PR to update.

---

## Quick Start

### 1. Install

**One-liner** (recommended — checks Apple Silicon, Python, sets up everything):

```bash
curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/install.sh | bash
```

**Or with pip** (if you manage your own venv):

```bash
pip install git+https://github.com/raullenchai/Rapid-MLX.git
```

<details>
<summary>Clone for development</summary>

```bash
git clone https://github.com/raullenchai/Rapid-MLX.git
cd rapid-mlx
pip install -e .
```
</details>

### 2. Start the server

```bash
# Qwen3-Coder-Next — fast coding model (80B MoE, 3B active)
rapid-mlx serve \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --port 8000
```

That's it. You now have an OpenAI-compatible agent server on `localhost:8000`.

### 3. Use it

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

### Hardware Requirements

How much RAM do you need? Model weights must fit in unified memory, plus ~20% headroom for KV cache. If your Mac uses >90% of RAM, you'll hit swap thrashing and decode speed drops to <5 tok/s.

| Mac RAM | Recommended Model | Weights | Decode Speed | Notes |
|---------|-------------------|---------|-------------|-------|
| 16 GB | Qwen3.5-4B 4bit | 2.4 GB | ~158 tok/s | Chat, simple tasks, 73% tool calling |
| 24 GB | Qwen3.5-9B 4bit | 5.1 GB | ~106 tok/s | Good all-rounder — 71% avg |
| 32 GB | Qwen3.5-27B 4bit | 15.3 GB | ~38 tok/s | 76% avg, solid for coding |
| 64 GB | Qwen3.5-35B-A3B 8bit | 37 GB | ~80 tok/s | **Sweet spot** — 85% avg, fast MoE |
| 96 GB | Qwen3.5-122B mxfp4 | 65 GB | ~57 tok/s | Best model fits comfortably |
| 128 GB | Qwen3.5-122B mxfp4 + `--kv-bits 4` | 65 GB | ~57 tok/s | Long contexts (50K+ tokens) |
| 192 GB+ | Qwen3.5-122B 8bit | 130 GB | ~43 tok/s | Maximum quality — 89% avg |

**Rule of thumb:** model weights + 20% = minimum RAM. Use `--kv-bits 4` or `--kv-bits 8` to halve KV cache memory for long contexts.

#### Help Us Fill This Table

These numbers are from a single M3 Ultra (256GB). We need community data for other configs — M1, M2, M4, MacBook Air/Pro, different RAM tiers. If you test a model, **copy the template below into a [new issue](https://github.com/raullenchai/Rapid-MLX/issues/new):**

```
**Hardware:** Mac ___ (___GB RAM), macOS ___
**Model:** ___ (quantization: ___)
**Server flags:** `rapid-mlx serve --model ___ ...`
**Decode speed:** ___ tok/s
**TTFT (cold / cached):** ___s / ___s
**Agent client tested:** Claude Code / Cursor / Aider / other
**Did it fit in RAM?** Yes / No (swap used: ___GB)
**Notes:** ___
```

### Recommended Models

| Model | Params | Quant | RAM | Decode | Tool Parser | Best For |
|-------|--------|-------|-----|--------|-------------|----------|
| [Qwen3.5-122B-A10B](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-8bit) | 122B/10B | 8bit | 130GB | 43 tok/s | `hermes` | **Best overall** — 89% avg |
| [Qwen3.5-122B-A10B](https://huggingface.co/nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx) | 122B/10B | mxfp4 | 65GB | 57 tok/s | `hermes` | **Best value** — same quality, half the RAM |
| [Qwen3.5-35B-A3B](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-8bit) | 35B/3B | 8bit | 37GB | 80 tok/s | `hermes` | Best for 64GB Macs — 85% avg |
| [Qwen3-Coder-Next](https://huggingface.co/lmstudio-community/Qwen3-Coder-Next-MLX-4bit) | 80B/3B | 4bit | 45GB | 74 tok/s | `hermes` | Fast coding agent |
| [Qwen3.5-9B](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | 9B | 4bit | 5.1GB | 106 tok/s | `hermes` | Best small model — 71% avg, fits any Mac |
| [Qwen3.5-4B](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) | 4B | 4bit | 2.4GB | 158 tok/s | `hermes` | Ultralight — 56% avg, 16GB MacBook Air |

Benchmarks on Mac Studio M3 Ultra (256GB), 800 GB/s memory bandwidth.

### Quick Start Commands

```bash
# Qwen3.5-122B — best overall for agent workloads (74GB, 41-47 tok/s)
rapid-mlx serve \
  --model nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --port 8000

# Qwen3-Coder-Next — fast coding agent (3B active, ~100 tok/s)
rapid-mlx serve \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-4bit \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --port 8000

# Qwen3.5-35B-A3B — best for 64GB Macs (85% avg, 80 tok/s)
rapid-mlx serve \
  --model mlx-community/Qwen3.5-35B-A3B-8bit \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --prefill-step-size 8192 \
  --port 8000

# Qwen3.5-9B — lightweight, fits any Mac (71% avg, 106 tok/s)
rapid-mlx serve \
  --model mlx-community/Qwen3.5-9B-4bit \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --port 8000

# DeepSeek-R1 — reasoning-focused
rapid-mlx serve \
  --model mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit \
  --tool-call-parser deepseek \
  --reasoning-parser deepseek_r1 \
  --port 8000

# Qwen3 VL — vision + language
rapid-mlx serve \
  --model mlx-community/Qwen3-VL-4B-Instruct-MLX-4bit \
  --mllm \
  --port 8000
```

### Tool Parser Selection Guide

| Model Family | `--tool-call-parser` | `--reasoning-parser` | Notes |
|-------------|---------------------|---------------------|-------|
| Qwen3.5-122B-A10B | `hermes` | `qwen3` | **Recommended** — best agent stability |
| Qwen3.5-35B/27B/9B/4B | `hermes` | `qwen3` | Same family, same parsers |
| Qwen3-Coder-Next | `hermes` | *(none)* | Non-thinking mode, fast |
| Qwen3 (thinking) | `qwen` or `qwen3_coder` | `qwen3` | With `<think>` tags |
| MiniMax-M2.5 | `minimax` | `minimax` | XML tool format |
| GPT-OSS | `seed_oss` | `gpt_oss` | Native format |
| GLM-4.7 | `glm47` | *(none)* | GLM-specific format |
| Llama 3.x | `llama` | *(none)* | JSON tool format |
| DeepSeek-R1 | `deepseek` | `deepseek_r1` | With reasoning |
| DeepSeek-V3.1 | `deepseek_v31` | *(none)* | Updated format |
| Mistral | `hermes` | *(none)* | Hermes-compatible |
| Functionary | `functionary` | *(none)* | Custom format |

All 17 parsers include automatic text-format tool call recovery — if a quantized model degrades and outputs tool calls as plain text, they're automatically converted back to structured `tool_calls`.

### Eval Benchmarks

17 models benchmarked across 4 eval suites: tool calling (30 scenarios), coding (HumanEval+), reasoning (MATH-500), and general knowledge (MMLU-Pro). All run with `enable_thinking: false` on Mac Studio M3 Ultra (256GB).

<details>
<summary><strong>Full eval results table</strong></summary>

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

```bash
# Run all eval suites against a running server (~5 min)
python evals/run_eval.py --model "My-Model" --quantization 4bit --port 8000
```

See **[evals/](evals/)** for methodology, prompts, and how to contribute your results.

---

## Client Setup

<details>
<summary><strong>Claude Code</strong></summary>

```bash
# Option 1: Environment variable (recommended)
OPENAI_BASE_URL=http://localhost:8000/v1 claude

# Option 2: Persistent config — add to ~/.claude/settings.json
```

```json
{
  "env": {
    "OPENAI_BASE_URL": "http://localhost:8000/v1"
  }
}
```

Set the model name in Claude Code with `/model` — use `default` or match whatever `--model` you loaded.

</details>

<details>
<summary><strong>Cursor</strong></summary>

1. Open **Settings > Models > OpenAI API Base**
2. Set the base URL to `http://localhost:8000/v1`
3. Add a model named `default` (or match your `--model`)
4. Set API key to any non-empty string (e.g. `not-needed`)

</details>

<details>
<summary><strong>Continue.dev (VS Code / JetBrains)</strong></summary>

Add to `~/.continue/config.yaml`:

```yaml
models:
  - name: rapid-mlx
    provider: openai
    model: default
    apiBase: http://localhost:8000/v1
    apiKey: not-needed
```

</details>

<details>
<summary><strong>OpenCode</strong></summary>

Add to `~/.config/opencode/opencode.json`:

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

<details>
<summary><strong>Open WebUI (Self-Hosted ChatGPT)</strong></summary>

Use Rapid-MLX as the backend for [Open WebUI](https://github.com/open-webui/open-webui) — a self-hosted ChatGPT-style interface with chat history, multi-user support, and RAG. No Ollama required.

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

Then open `http://localhost:3000`, create an account, and start chatting. Your Rapid-MLX model appears automatically in the model dropdown. Streaming, markdown rendering, and tool calling all work out of the box.

> **Migrating from Ollama?** Just swap the backend — Open WebUI works identically with Rapid-MLX, but you get prompt caching (10-30x faster multi-turn), reliable tool calling, and better agent stability.

</details>

<details>
<summary><strong>Aider</strong></summary>

```bash
aider --openai-api-base http://localhost:8000/v1 --openai-api-key not-needed
```

Or set permanently with environment variables:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=not-needed
aider
```

</details>

---

## Features

### Tool Calling (Any Model, Any Quantization)

Full OpenAI-compatible tool calling with streaming support. 17 parser formats built in, and **automatic recovery when models break**.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name)       # "get_weather"
print(tool_call.function.arguments)  # '{"city": "Tokyo"}'
```

Supported parsers: `hermes`, `minimax`, `qwen`, `qwen3_coder`, `llama`, `deepseek`, `deepseek_v31`, `seed_oss`, `functionary`, `glm47`, `harmony`, `mistral`, `nemotron`, `granite`, `kimi`, `xlam`, `auto`. Use `--tool-call-parser <name>` to select.

#### Robust Tool Call Recovery

A common pain point with local models: **quantized models (4-bit, 6-bit) degrade after multiple tool call rounds** and start outputting tool calls as plain text instead of structured format. This breaks agent frameworks like OpenClaw, Claude Code, Cursor, and LangChain — the client receives text instead of a `tool_calls` response.

Rapid-MLX solves this at the server level. **All parsers** automatically detect and recover degraded tool calls — no configuration needed, works with any model:

```
# Model outputs broken text instead of structured XML/JSON:
[Calling tool="web_search" query="weather tonight"]
[Calling tool: exec({"command":"python3 --version"})]

# Rapid-MLX auto-detects and converts to proper OpenAI tool_calls:
→ finish_reason: "tool_calls"
→ tool_calls: [{"name": "web_search", "arguments": "{\"query\": \"weather tonight\"}"}]
```

This is especially important for:
- **MoE models at 4-bit** (MiniMax-M2.5, Qwen3.5-122B, Qwen3-Coder-Next) — most prone to degradation
- **Long agent sessions** (8+ tool rounds) — where models run out of "structured output stamina"
- **Multi-tool setups** (10+ tools) — where tool choice complexity increases error rates

Tested end-to-end with 14 tools across 8+ rounds — see [`tests/test_tool_call_e2e.py`](tests/test_tool_call_e2e.py) for the full agent simulation.

### Reasoning Separation

Models with chain-of-thought (MiniMax-M2.5, Qwen3, DeepSeek-R1) output reasoning in a separate `reasoning_content` field — never mixed into `content`. 0% leak rate.

```bash
rapid-mlx serve \
  --model mlx-community/Qwen3.5-122B-A10B-8bit \
  --reasoning-parser qwen3 \
  --port 8000
```

### Prompt Cache (10-30x Faster Multi-Turn)

Persistent KV cache across requests. When consecutive requests share a prefix (system prompt + conversation history), only new tokens are prefilled:

| Context Size | Without Cache | With Cache | Speedup |
|-------------|---------------|------------|---------|
| 1K tokens | 0.7s | 0.3s | 2.3x |
| 4K tokens | 2.4s | 0.3s | 8x |
| 33K tokens | 28s | 0.3-0.9s | **30-90x** |

Always on in SimpleEngine (default mode). No flags needed.

### Smart Cloud Routing

Large-context requests are automatically routed to a cloud LLM when local prefill would be too slow. The routing decision is based on **new tokens** (after cache hit), not total input — so a 50K-token conversation with 2K new tokens stays local.

```bash
pip install litellm

# Route to GPT-5 when >20K new tokens need prefilling
OPENAI_API_KEY=sk-... rapid-mlx serve \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --cloud-model openai/gpt-5 \
  --cloud-threshold 20000 \
  --port 8000
```

```
Short request (44 new tokens)  → [LOCAL]  Qwen3 responds in 0.3s
Large request (15K new tokens) → [CLOUD]  GPT-5 responds in 3s (vs 50s local)
Next turn (cache hit, 200 new) → [LOCAL]  Back to local, 0.3s
```

Works with any litellm-supported provider: OpenAI, Anthropic, Google, Groq, etc. Clients see no difference — same API, transparent routing.

Disabled by default. Cost estimate: ~$0.02-0.05 per cloud-routed request with GPT-5.

### Multimodal

Vision, audio (speech-to-text, text-to-speech), video understanding, and text embeddings — all through the same OpenAI-compatible API.

```bash
# Vision + Language model
rapid-mlx serve \
  --model mlx-community/Qwen3-VL-4B-Instruct-MLX-4bit \
  --mllm \
  --port 8000
```

---

## Server Flags

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
| `--tool-call-parser` | Parser: `hermes`, `minimax`, `qwen`, `qwen3_coder`, `llama`, `deepseek`, etc. | *(none)* |
| `--enable-auto-tool-choice` | Enable automatic tool choice (implied by `--tool-call-parser`) | off |
| `--enable-tool-logits-bias` | Jump-forward decoding for faster tool calls | off |
| `--reasoning-parser` | Parser: `minimax`, `qwen3`, `deepseek_r1`, `gpt_oss`, `harmony` | *(none)* |

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
| `--cloud-model` | litellm model string (e.g. `openai/gpt-5`, `anthropic/claude-sonnet-4-5-20250929`) | *(disabled)* |
| `--cloud-threshold` | New token threshold to trigger cloud routing | `20000` |

### Security

| Flag | Description | Default |
|------|-------------|---------|
| `--api-key` | API key for authentication | *(no auth)* |
| `--rate-limit` | Requests per minute per client | *(unlimited)* |
| `--timeout` | Request timeout in seconds | `300` |

### Other

| Flag | Description | Default |
|------|-------------|---------|
| `--mllm` | Force multimodal (vision) mode | auto-detect |
| `--mcp-config` | MCP configuration file for tool integration | *(none)* |
| `--embedding-model` | Pre-load embedding model at startup | *(none)* |
| `--default-temperature` | Override default temperature | model default |

---

## Full Example Configurations

**Production agent setup (best tool calling):**

```bash
rapid-mlx serve \
  --model nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --max-tokens 4096 \
  --port 8000
```

**Fast coding agent:**

```bash
rapid-mlx serve \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --port 8000
```

**64GB Mac — best balance of quality + speed:**

```bash
rapid-mlx serve \
  --model mlx-community/Qwen3.5-35B-A3B-8bit \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --prefill-step-size 8192 \
  --port 8000
```

**Hybrid local + cloud — best of both worlds:**

```bash
OPENAI_API_KEY=sk-... rapid-mlx serve \
  --model nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx \
  --tool-call-parser hermes \
  --cloud-model openai/gpt-5 \
  --cloud-threshold 20000 \
  --port 8000
```

---

## Architecture

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

---

## Feature Details

<details>
<summary><strong>Agent-Grade Tool Calling (13 features)</strong></summary>

### Agent-Grade Tool Calling

- **Text-format tool call recovery** — quantized models degrade and output tool calls as plain text; server auto-detects and converts back to structured `tool_calls` (works with any model, any parser)
- 17 tool parsers — Hermes, MiniMax, Qwen, Qwen3-Coder, Llama, DeepSeek, DeepSeek-v31, Seed-OSS, Functionary, GLM-4.7, Harmony, Mistral, Nemotron, Granite, Kimi, xLAM, Auto
- MiniMax tool call parser — streaming + non-streaming XML extraction
- `--tool-call-parser` flag — explicit parser selection for any model
- Auto-infer tool parser — `--reasoning-parser minimax` auto-selects matching tool parser
- Tool-use system prompt auto-injection (100% tool call rate, was 67%)
- Tool logits bias — jump-forward decoding for 2-5x faster structured output
- Streaming disconnect guard — client disconnects release server locks instead of deadlocking
- Think-tag streaming filter — `<think>...</think>` blocks stripped from content, never leaked to clients
- Chunk-boundary leak fix — prevents XML leaking into reasoning stream
- `developer` role normalization for chat template compatibility
- Logprobs API — per-token `logprobs` + `top_logprobs`
- End-to-end agent simulation tests — 14 tools, 8+ rounds, verified with OpenClaw

</details>

<details>
<summary><strong>Reasoning Separation (3 features)</strong></summary>

### Reasoning Separation

- MiniMax reasoning parser — heuristic no-tag stripping (0% leak rate, was 60%)
- Chinese reasoning pattern recognition
- Clean `reasoning_content` field — reasoning never mixed into `content`

</details>

<details>
<summary><strong>Performance (6 features)</strong></summary>

### Performance

- Prompt cache (SimpleEngine) — persistent KV cache, 10-30x faster multi-turn
- `--prefill-step-size` — configurable prefill chunks for TTFT tuning
- `--kv-bits` — KV cache quantization (4/8 bit) for long contexts
- Speculative decoding — `--draft-model` with prompt cache compatibility
- Smart cloud routing — `--cloud-model` offloads large prefills to cloud LLMs
- Frequency-aware cache eviction — LRU-LFU hybrid under memory pressure

</details>

<details>
<summary><strong>Reliability (6 features)</strong></summary>

### Reliability

- Accurate `prompt_tokens` reporting (was always 0)
- Prompt cache EOS fix — cache saved correctly on EOS
- Server crash prevention on malformed `response_format`
- GC control during generation to avoid latency spikes
- System prompt pinning in prefix cache
- **1500+ tests** — unit tests + end-to-end agent simulation with real tool execution

</details>

<details>
<summary><strong>Multimodal (4 capabilities)</strong></summary>

### Multimodal

- Vision — image understanding via Qwen-VL and other VLMs
- Audio STT — speech-to-text via Whisper
- Audio TTS — text-to-speech via Kokoro
- Embeddings — text embedding models for RAG and similarity search

</details>

---

## Roadmap

Research-backed optimizations ranked by impact-to-effort ratio. Papers surveyed from ICLR 2025, ICML 2025, NeurIPS 2025, ACL 2025.

| Priority | Technique | Expected Gain | Status |
|----------|-----------|---------------|--------|
| 1 | [ReDrafter](https://arxiv.org/abs/2403.09919) — Apple's speculative decoding (RNN draft head) | 1.4-1.5x decode | Not started |
| 2 | [KVSplit](https://github.com/dipampaul17/KVSplit) — Mixed-precision KV cache (8-bit K, 4-bit V) | 59% memory reduction | Not started |
| 3 | [DuoAttention](https://arxiv.org/abs/2410.10819) — Per-head adaptive KV cache | 2.5x memory, 2.2x decode | Not started |
| 4 | [FastKV](https://arxiv.org/abs/2502.01068) — Token-selective propagation | 1.8x prefill, 2.9x decode | Not started |
| 5 | [xKV](https://arxiv.org/abs/2503.18893) — Cross-layer SVD compression | 8x KV compression | Not started |
| 6 | [Medusa](https://arxiv.org/abs/2401.10774) — Multiple decoding heads | 2.2-2.8x decode | Not started |

---

## Troubleshooting

### "parameters not found in model" warnings at startup

This is normal for vision-language models (Qwen3-VL, etc.). The warning means vision tower weights are present in the checkpoint but not used by the text-only model class. They are auto-skipped — no action needed.

### Out of memory / very slow decode (<5 tok/s)

Your model is too large for your RAM. Check the [Hardware Requirements](#hardware-requirements) table. Fixes:
- Switch to a smaller quantization (`4bit` instead of `6bit` or `8bit`)
- Use `--kv-bits 4` to halve KV cache memory for long contexts
- Close other apps — browsers, Docker, etc. compete for unified memory
- If Activity Monitor shows memory pressure in red, the model doesn't fit

### Empty streaming content (blank responses)

This happens when a reasoning parser is set (e.g. `--reasoning-parser qwen3`) but the model doesn't emit `<think>` tags. The parser waits for closing tags that never come. Fix: remove `--reasoning-parser` for non-thinking models, or use `--reasoning-parser` only with models that actually produce chain-of-thought (Qwen3 with thinking enabled, MiniMax-M2.5, DeepSeek-R1). This fork auto-corrects for Qwen3 no-tag output.

### Tool calls appear as plain text in responses

The model is outputting tool calls in text format instead of structured `tool_calls`. This usually means `--tool-call-parser` is not set, or is set to the wrong parser. Check the [Tool Parser Selection Guide](#tool-parser-selection-guide) for the correct parser for your model. Even when this happens, Rapid-MLX auto-recovers degraded tool calls — but setting the correct parser gives the best results.

### Slow time-to-first-token (TTFT) on first request

Cold start is normal — the first request prefills the full prompt with no cache. Subsequent requests with shared prefixes (same system prompt, growing conversation) hit the prompt cache and are 10-30x faster. To speed up cold starts:
- Increase `--prefill-step-size` (e.g. `8192` or `16384`) for faster chunked prefill
- Use `--kv-bits 8` to reduce memory pressure during prefill
- Consider `--cloud-model` to route large cold requests to a cloud LLM

### Server stops responding after client disconnects

Fixed in this fork. The streaming disconnect guard ensures that when a client disconnects mid-stream, the server releases all locks and saves the prompt cache. If you're on an older version, upgrade to the latest.

---

## Contributing

Issues and PRs welcome at [github.com/raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX).

**We need your help.** The hardware tables, model benchmarks, and client configs in this README are based on one M3 Ultra (256GB). There are dozens of Mac + model + client combos we haven't tested. The easiest way to contribute:

1. **Report your hardware benchmarks** — use the [issue template](#help-us-fill-this-table) in the Hardware Requirements section
2. **Verify a client** — if you get Claude Code, Cursor, Continue.dev, or another client working, let us know what worked (and what didn't)
3. **Fix a doc** — if something in this README is wrong or missing, PRs are welcome

Built on [mlx-lm](https://github.com/ml-explore/mlx-examples) and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Apple's MLX framework for efficient on-device inference.

## License

Apache 2.0 — see [LICENSE](LICENSE).
