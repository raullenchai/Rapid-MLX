<p align="center">
  <img src="https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/docs/assets/logo.png" alt="Rapid-MLX" width="200">
</p>

<h1 align="center">Rapid-MLX</h1>

<p align="center">
  <strong>Run AI on your Mac. Faster than anything else.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/tests-1900%2B-brightgreen.svg" alt="Tests"></a>
  <a href="https://support.apple.com/en-us/HT211814"><img src="https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple" alt="Apple Silicon"></a>
</p>

<p align="center">
  Drop-in OpenAI API replacement for Apple Silicon. 2-4x faster than Ollama, with full tool-calling support.
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/docs/assets/demo.gif" alt="Rapid-MLX vs Ollama — 2.4x faster on Qwen3.5-9B" width="800">
  <br>
  <em>Same model (Qwen3.5-9B), same Mac, head-to-head. Rapid-MLX: 79 tok/s vs Ollama: 33 tok/s.</em>
</p>

| | Your Mac runs AI | How fast | What works |
|:---|:---:|:---:|:---:|
| **16 GB MacBook Air** | Qwen3.5-4B | 168 tok/s | Chat, coding, tools |
| **64 GB Mac Mini / Studio** | Qwen3.5-35B | 83 tok/s | Best balance of smart + fast |
| **96+ GB Mac Studio / Pro** | Qwen3.5-122B | 57 tok/s | Frontier-level intelligence |

---

## Quick Start

**Step 1 — Install:**
```bash
curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/install.sh | bash
```
Then close and reopen your terminal (or run `source ~/.zshrc`).

**Step 2 — Start the server:**
```bash
rapid-mlx serve qwen3.5-9b
```
First run downloads the model (~5 GB) — you'll see a progress bar. Wait for `Ready: http://localhost:8000/v1`.

**Step 3 — Test it** (open a **second** terminal tab):
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Say hello"}]}'
```
You should get a JSON response with the AI's reply. To stop the server: Ctrl+C in the first terminal.

That's it — you now have an AI server on `localhost:8000`. Next: scroll down to [**Choose Your Model**](#choose-your-model) to pick the best model for your Mac, or see [**Works With**](#works-with) to connect Claude Code, Cursor, or other apps.

> **Tip:** If you get "Connection refused", the server is still loading. Wait for the "Ready" message.
>
> **Tip:** Qwen3.5 models "think" before responding (chain-of-thought). First responses may take 10-30s — this is normal. For faster responses without thinking, add `--no-thinking`.

<details>
<summary>Other install methods</summary>

**Homebrew:**
```bash
brew install raullenchai/rapid-mlx/rapid-mlx
```

**pip** (use a virtual environment on macOS Sonoma+):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install rapid-mlx
```

**From source** (for development):
```bash
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX && python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Vision models** (adds torch + torchvision, ~2.5 GB extra):
```bash
pip install 'rapid-mlx[vision]'
```
</details>

**Try it with Python** (make sure the server is running, then `pip install openai`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say hello"}],
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
| [Anthropic SDK](https://docs.anthropic.com/en/docs/sdks) | Compatible | Via `/v1/messages` endpoint |
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

**OpenCode** (`opencode.json` in your project root):
```json
{
  "provider": {
    "openai": {
      "api": "http://localhost:8000/v1",
      "models": {
        "default": {
          "name": "rapid-mlx local",
          "limit": { "context": 32768, "output": 8192 }
        }
      },
      "options": { "apiKey": "not-needed" }
    }
  }
}
```

**Anthropic SDK** (`pip install anthropic`):
```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8000", api_key="not-needed")

message = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Say hello"}],
)
print(message.content[0].text)
```

</details>

---

## Choose Your Model

### What fits my Mac?

Model weights must fit in unified memory. If Activity Monitor shows red memory pressure, the model is too big — switch to a smaller one or a lower quantization.

| Your Mac | Best Model | RAM Used | Speed | Quality |
|----------|-----------|---------|-------|---------|
| **16 GB** MacBook Air/Pro | [Qwen3.5-4B 4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) | 2.4 GB | 168 tok/s | Good for chat and simple tasks |
| **24 GB** MacBook Pro | [Qwen3.5-9B 4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | 5.1 GB | 108 tok/s | Great all-rounder |
| **32 GB** Mac Mini / Studio | [Qwen3.5-27B 4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | 15.3 GB | 39 tok/s | Solid coding model |
| **36 GB** MacBook Pro M3/M4 Pro | [Qwen3.5-27B 4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | 15.3 GB | 39 tok/s | Same as 32 GB — extra headroom for long contexts |
| **48 GB** Mac Mini / Studio | [Qwen3.5-35B-A3B 8bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-8bit) | 37 GB | 83 tok/s | **Sweet spot** — smart + fast |
| **64 GB** Mac Mini / Studio | [Qwen3.5-35B-A3B 8bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-8bit) | 37 GB | 83 tok/s | Same model, more room for KV cache |
| **96 GB** Mac Studio / Pro | [Qwen3.5-122B mxfp4](https://huggingface.co/nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx) | 65 GB | 57 tok/s | Best model, fits comfortably |
| **192 GB** Mac Studio / Pro | [Qwen3.5-122B 8bit](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-8bit) | 130 GB | 44 tok/s | Maximum quality |

### Copy-paste commands

Pick the one that matches your Mac. Short aliases work — run `rapid-mlx models` to see all 20.

```bash
# 16 GB — lightweight, fast
rapid-mlx serve qwen3.5-4b --port 8000

# 24 GB — best small model
rapid-mlx serve qwen3.5-9b --port 8000

# 32 GB — solid coding model
rapid-mlx serve qwen3.5-27b --port 8000

# 64 GB — sweet spot
rapid-mlx serve qwen3.5-35b --prefill-step-size 8192 --port 8000

# 96+ GB — best model
rapid-mlx serve qwen3.5-122b --kv-bits 8 --prefill-step-size 8192 --port 8000

# Coding agent — fast MoE, great for Claude Code / Cursor
rapid-mlx serve qwen3-coder --prefill-step-size 8192 --port 8000

# Vision — image understanding (see note below)
rapid-mlx serve qwen3-vl-4b --mllm --port 8000
```

> **Vision deps:** Install into the same environment where rapid-mlx lives:
> - `install.sh` users: `~/.rapid-mlx/bin/pip install 'rapid-mlx[vision]'`
> - `pip` users: `pip install 'rapid-mlx[vision]'` (in the same venv)
> - `brew` users: `$(brew --prefix)/opt/rapid-mlx/libexec/bin/pip install 'rapid-mlx[vision]'`

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

22 models tested across 6 engines on **Mac Studio M3 Ultra (256GB)**. Rapid-MLX uses Apple's [MLX framework](https://github.com/ml-explore/mlx) — purpose-built for unified memory with native Metal compute kernels — which is why it beats C++-based engines (Ollama, llama.cpp) on most models. **#1 on 16 of 18 benchmarked models.**

| Model | Rapid-MLX | Best Alternative | Speedup |
|-------|----------|-----------------|---------|
| **Phi-4 Mini 14B** | **180** tok/s | 77 (mlx-lm) / 56 (Ollama) | **2.3x** / **3.2x** |
| **Qwen3.5-4B** | **168** tok/s | 155 (mlx-lm serve) | **1.1x** |
| **GPT-OSS 20B** | **127** tok/s · 100% tools | 79 (mlx-lm serve) | **1.6x** |
| **Qwen3.5-9B** | **108** tok/s | 46 (Ollama) | **2.3x** |
| **Kimi-Linear-48B** | **94** tok/s · 100% tools | — (only engine) | — |
| **Qwen3.5-35B-A3B** | **83** tok/s · 100% tools | 75 (oMLX) | **1.1x** |
| **Qwen3-Coder 80B** | **74** tok/s · 100% tools | 69 (mlx-lm serve) | **1.1x** |
| **Qwen3.5-122B** | **44** tok/s · 100% tools | 43 (mlx-lm serve) | ~1.0x |

*Full benchmark data with all 18 models, TTFT tables, DeltaNet snapshots, and engine comparison below.*

<details>
<summary><strong>TTFT — Prompt Cache Advantage</strong></summary>

Prompt cache keeps multi-turn conversations fast. For standard transformers, KV cache trimming gives sub-100ms TTFT. For hybrid RNN models (Qwen3.5 DeltaNet), we use state snapshots — the first technique to bring prompt cache to non-trimmable architectures on MLX.

**Pure KV cache (transformers):**

| Model | Rapid-MLX (cached) | mlx-lm serve | Speedup |
|-------|-------------------|-------------------|---------|
| Kimi-Linear-48B | **0.08s** | — | — |
| Llama 3.2 3B | **0.10s** | — | — |
| Hermes-3-Llama 8B | **0.10s** | 0.18s | 1.8x |
| Phi-4 Mini 14B | **0.13s** | 0.15s | 1.2x |
| Devstral-Small-2 24B | **0.13s** | 0.38s | 2.9x |
| Mistral Small 24B | **0.13s** | 0.38s | 2.9x |
| GLM-4.7-Flash 9B | **0.13s** | 0.23s | 1.8x |
| GLM-4.5-Air | **0.14s** | 0.47s | 3.4x |
| Qwen3-Coder-Next 80B | **0.16s** | 0.27s | 1.7x |
| GPT-OSS 20B | **0.16s** | 0.27s | 1.7x |
| Qwen3.5-9B | **0.22s** | 0.26s | 1.2x |

**DeltaNet state snapshots (hybrid RNN + attention):**

Qwen3.5 uses Gated DeltaNet (75% RNN) + full attention (25% KV). Other engines recreate the entire cache from scratch every request — we snapshot the RNN state at the system prompt boundary, restoring in ~0.1ms instead of re-running hundreds of tokens through the recurrent layers.

| Model | Cold TTFT | Snapshot TTFT | Speedup |
|-------|-----------|---------------|---------|
| Qwen3-Coder-Next 6bit (48L) | 0.66s | **0.16s** | **4.3x** |
| Qwen3.5-35B-A3B 8bit (40L) | 0.49s | **0.19s** | **2.6x** |
| Qwen3.5-27B 4bit (40L) | 0.58s | **0.27s** | **2.1x** |
| Qwen3.5-9B 4bit (40L) | 0.27s | **0.22s** | **1.2x** |
| Qwen3.5-4B 4bit (32L) | 0.24s | **0.16s** | **1.5x** |

</details>

<details>
<summary><strong>Capability Comparison</strong></summary>

| Feature | Rapid-MLX | oMLX | Ollama | llama.cpp | mlx-lm serve |
|---------|-----------|------|--------|-----------|-------------|
| **Tool calling** | 100% (Qwen/GLM/GPT-OSS/Kimi) | N/A | 100% (Qwen) | 80% (Phi-4) | N/A |
| **Tool call recovery** | 100% | N/A | 100% | 100% | N/A |
| **Tool injection fallback** | Yes | No | No | No | No |
| **Think-tag leak** | 0% | N/A | 0% | 0% | N/A |
| **Prompt cache** | KV + DeltaNet | No | No | No | No |
| **Vision** | Yes | Yes | Yes | No | No |
| **Audio (STT/TTS)** | Yes | No | No | No | No |
| **17 tool parsers** | Yes | No | No | No | No |
| **Cloud routing** | Yes | No | No | No | No |
| **Streaming** | Yes | Yes | Yes | Yes | Yes |
| **OpenAI API** | Yes | Yes | Yes | Yes | Yes |

</details>

<details>
<summary><strong>Optimization Techniques Per Model</strong></summary>

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

</details>

<details>
<summary><strong>Eval benchmarks (17 models, 4 suites)</strong></summary>

17 models across tool calling (30 scenarios), coding (HumanEval+), reasoning (MATH-500), and general knowledge (MMLU-Pro). All with `enable_thinking: false` on M3 Ultra.

| Model | Quant | RAM | Decode | Tools | Code | Reason | General | Avg |
|-------|-------|-----|--------|-------|------|--------|---------|-----|
| Qwen3.5-122B-A10B | 8bit | 129.8 GB | 44 t/s | 87% | **90%** | **90%** | **90%** | **89%** |
| Qwen3.5-122B-A10B | mxfp4 | 65.0 GB | 57 t/s | **90%** | **90%** | 80% | **90%** | 88% |
| Qwen3.5-35B-A3B | 8bit | 36.9 GB | 83 t/s | **90%** | **90%** | 80% | 80% | 85% |
| Qwen3-Coder-Next | 6bit | 64.8 GB | 66 t/s | 87% | **90%** | 80% | 70% | 82% |
| Qwen3-Coder-Next | 4bit | 44.9 GB | 74 t/s | **90%** | **90%** | 70% | 70% | 80% |
| GLM-4.5-Air | 4bit | 60.3 GB | 46 t/s | 73% | **90%** | 70% | 80% | 78% |
| GLM-4.7-Flash | 8bit | 31.9 GB | 58 t/s | 73% | **100%** | **90%** | 50% | 78% |
| Qwen3.5-27B | 4bit | 15.3 GB | 39 t/s | 83% | **90%** | 50% | 80% | 76% |
| Qwen3.5-35B-A3B | 4bit | 19.6 GB | 95 t/s | 87% | **90%** | 50% | 70% | 74% |
| Qwen3.5-9B | 4bit | 5.1 GB | 108 t/s | 83% | 70% | 60% | 70% | 71% |
| MiniMax-M2.5 | 4bit | 128.9 GB | 52 t/s | 87% | 10%\* | 80% | **90%** | 67% |
| Devstral-Small-2 | 4bit | 13.4 GB | 49 t/s | 17% | **90%** | 70% | 70% | 62% |
| GPT-OSS-20B | mxfp4-q8 | 12.1 GB | 127 t/s | 80% | 20% | 60% | **90%** | 62% |
| Qwen3.5-4B | 4bit | 2.4 GB | 168 t/s | 73% | 50% | 50% | 50% | 56% |
| Mistral-Small-3.2 | 4bit | 13.4 GB | 49 t/s | 17% | 80% | 60% | 60% | 54% |
| Hermes-3-Llama-8B | 4bit | 4.6 GB | 127 t/s | 17% | 20% | 30% | 40% | 27% |
| Qwen3-0.6B | 4bit | 0.4 GB | 365 t/s | 30% | 20% | 20% | 30% | 25% |

\* *MiniMax coding score likely affected by a code extraction parser issue, not model capability.*

</details>

*Benchmark script: [`scripts/benchmark_engines.py`](scripts/benchmark_engines.py). Run your own: `python scripts/benchmark_engines.py --engine rapid-mlx ollama --runs 3`. Eval suites: [evals/](evals/)*

---

## Features

### Tool Calling

Full OpenAI-compatible tool calling with 17 parser formats and **automatic recovery when quantized models break**. Models at 4-bit degrade after multiple tool rounds — Rapid-MLX auto-detects broken output and converts it back to structured `tool_calls`.

### Reasoning Separation

Models with chain-of-thought (Qwen3, DeepSeek-R1) output reasoning in a separate `reasoning_content` field — cleanly separated from `content` in streaming mode. Works with Qwen3, DeepSeek-R1, MiniMax, and GPT-OSS reasoning formats.

### Prompt Cache

Persistent cache across requests — only new tokens are prefilled on each turn. For standard transformers, KV cache trimming. For hybrid models (Qwen3.5 DeltaNet), RNN state snapshots restore non-trimmable layers from memory instead of re-computing. 2-5x faster TTFT on all architectures. Always on, no flags needed.

### Smart Cloud Routing

Large-context requests auto-route to a cloud LLM (GPT-5, Claude, etc.) when local prefill would be slow. Routing based on new tokens after cache hit. `--cloud-model openai/gpt-5 --cloud-threshold 20000`

### Multimodal

Vision, audio (STT/TTS), video understanding, and text embeddings — all through the same OpenAI-compatible API.

<details>
<summary><strong>All features (37 total)</strong></summary>

**Tool Calling (15):** Text-format recovery, 17 parsers, streaming, tool logits bias (2-5x faster structured output), disconnect guard, think-tag filter, chunk-boundary leak fix, developer role normalization, logprobs API, system prompt tool injection fallback for incompatible chat templates, end-to-end agent simulation tests.

**Reasoning (3):** MiniMax/Qwen3/DeepSeek parsers, Chinese reasoning pattern recognition, clean `reasoning_content` field.

**Performance (9):** Prompt cache (KV trim + DeltaNet state snapshots), SSE template pre-computation, MTP (multi-token prediction), configurable prefill step size, KV cache quantization (4/8 bit), speculative decoding, cloud routing, frequency-aware cache eviction.

**Reliability (6):** Accurate `prompt_tokens` reporting, EOS cache fix, crash prevention on malformed `response_format`, GC control during generation, system prompt pinning, 1900+ tests.

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

**Server hangs after client disconnect** — Fixed in v0.3.0+. Upgrade to latest.

</details>

---

## Roadmap

| Technique | Expected Gain | Status |
|-----------|---------------|--------|
| **DeltaNet state snapshots** — hybrid RNN cache reuse for Qwen3.5 | 1.5-4.3x TTFT | **Done** |
| **SSE streaming optimization** — pre-computed templates, micro-opts | +10.5% composite | **Done** |
| **Tool injection fallback** — system prompt injection for broken templates | 0→100% tools | **Done** |
| [MTP in SimpleEngine](https://arxiv.org/abs/2404.19737) — multi-token prediction | 1.4x decode | **Done** |
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
