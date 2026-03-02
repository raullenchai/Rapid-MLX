# vLLM-MLX

**Production-grade agent inference server for Apple Silicon**

[![Fork](https://img.shields.io/badge/Fork-raullenchai%2Fvllm--mlx-orange?logo=github)](https://github.com/raullenchai/vllm-mlx)
[![Upstream](https://img.shields.io/badge/Upstream-waybarrios%2Fvllm--mlx-blue?logo=github)](https://github.com/waybarrios/vllm-mlx)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1500%2B-brightgreen.svg)](tests/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple)](https://support.apple.com/en-us/HT211814)

Run local LLMs as a **drop-in replacement for OpenAI** — with the reliability that agent frameworks demand. GPU-accelerated on Mac via [MLX](https://github.com/ml-explore/mlx), built for tools like **Claude Code, Cursor, OpenClaw, Aider, LangChain, and any OpenAI-compatible client**.

The upstream [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) is a solid MLX inference wrapper. This fork transforms it into a **production agent server** — the kind of infrastructure you need when AI agents call tools, reason through problems, and run multi-turn sessions that last hours.

### Why This Fork?

Local models are fast and private, but they break in agent workflows. Quantized models forget their tool format after a few rounds. Streaming responses leak XML into content. Disconnected clients lock the server. Multi-turn conversations re-prefill the entire context every time.

**We fix all of that at the server level**, so every client — Cursor, Claude Code, your custom agent — just works.

---

## Highlights vs. Upstream

| Capability | Upstream | This Fork |
|-----------|----------|-----------|
| **Tool calling** | Not supported | 7 parser formats, streaming, **auto-recovery of degraded outputs** |
| **Agent reliability** | N/A | Disconnect guard, think-tag filtering, text-format tool call recovery |
| **Reasoning separation** | Not supported | Clean `reasoning_content` field (0% leak rate) |
| **Multi-turn TTFT** | Full prefill every turn | **10-30x faster** — persistent prompt cache |
| **Long-context prefill** | 50s for 52K tokens | **<1s** — smart cloud routing offloads to GPT-5/Claude |
| **Decode speed** | Baseline | 65-100 tok/s on M3 Ultra |
| **KV cache quantization** | Not available | 4-bit and 8-bit, halves memory for long contexts |
| **Speculative decoding** | Not available | `--draft-model` with prompt cache compatibility |
| **Logprobs API** | Not available | Per-token `logprobs` + `top_logprobs` |
| **Test coverage** | Minimal | **1500+ tests** (unit + end-to-end agent simulation) |

### Tested With

| Client | Status | Notes |
|--------|--------|-------|
| [OpenClaw](https://github.com/nicepkg/openclaw) | Verified | 14 tools, multi-round, streaming |
| [Aider](https://aider.chat) | Verified | Code editing agent |
| [LangChain](https://langchain.com) | Compatible | Standard OpenAI client |
| [Claude Code](https://claude.ai/claude-code) | Verified | Env var config, streaming tools |
| [Cursor](https://cursor.com) | Verified | Settings UI config |
| [Continue.dev](https://continue.dev) | Verified | YAML config, VS Code + JetBrains |
| [OpenCode](https://github.com/opencode-ai/opencode) | Verified | JSON config |
| Any OpenAI SDK client | Compatible | Drop-in `base_url` swap |

> **These tables are community-maintained.** Tested on M3 Ultra so far — if you verify a client or model on your hardware, [open an issue](https://github.com/raullenchai/vllm-mlx/issues) or PR to update the tables. We can't test every Mac + model + client combo alone.

---

## Quick Start

### 1. Install

```bash
pip install git+https://github.com/raullenchai/vllm-mlx.git
```

Or clone for development:

```bash
git clone https://github.com/raullenchai/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

### 2. Start the server

```bash
# Qwen3-Coder-Next — fast coding model (80B MoE, 3B active)
python -m vllm_mlx.server \
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

Works as a drop-in backend for any OpenAI-compatible client:

```bash
# Claude Code
OPENAI_BASE_URL=http://localhost:8000/v1 claude

# Cursor — set in Settings > Models > OpenAI API Base
# http://localhost:8000/v1

# Aider
aider --openai-api-base http://localhost:8000/v1

# Any OpenAI SDK — just set base_url
```

---

## Client Setup

### Claude Code

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

### Cursor

1. Open **Settings > Models > OpenAI API Base**
2. Set the base URL to `http://localhost:8000/v1`
3. Add a model named `default` (or match your `--model`)
4. Set API key to any non-empty string (e.g. `not-needed`)

### Continue.dev (VS Code / JetBrains)

Add to `~/.continue/config.yaml`:

```yaml
models:
  - name: vllm-mlx
    provider: openai
    model: default
    apiBase: http://localhost:8000/v1
    apiKey: not-needed
```

### OpenCode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "openai-compatible": {
      "apiKey": "not-needed",
      "models": {
        "default": {
          "id": "default",
          "name": "vllm-mlx local",
          "api_base": "http://localhost:8000/v1"
        }
      }
    }
  }
}
```

### Aider

```bash
aider --openai-api-base http://localhost:8000/v1 --openai-api-key not-needed
```

Or set permanently with environment variables:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=not-needed
aider
```

---

## Features

### Tool Calling (Any Model, Any Quantization)

Full OpenAI-compatible tool calling with streaming support. 7 parser formats built in, and **automatic recovery when models break**.

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

Supported parsers: `hermes`, `minimax`, `qwen`, `qwen3_coder`, `llama`, `deepseek`, `functionary`, and more. Use `--tool-call-parser <name>` to select.

#### Robust Tool Call Recovery

A common pain point with local models: **quantized models (4-bit, 6-bit) degrade after multiple tool call rounds** and start outputting tool calls as plain text instead of structured format. This breaks agent frameworks like OpenClaw, Claude Code, Cursor, and LangChain — the client receives text instead of a `tool_calls` response.

vllm-mlx solves this at the server level. **All parsers** automatically detect and recover degraded tool calls — no configuration needed, works with any model:

```
# Model outputs broken text instead of structured XML/JSON:
[Calling tool="web_search" query="weather tonight"]
[Calling tool: exec({"command":"python3 --version"})]

# vllm-mlx auto-detects and converts to proper OpenAI tool_calls:
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
python -m vllm_mlx.server \
  --model lmstudio-community/MiniMax-M2.5-MLX-4bit \
  --reasoning-parser minimax \
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

**New.** Large-context requests are automatically routed to a cloud LLM when local prefill would be too slow. The routing decision is based on **new tokens** (after cache hit), not total input — so a 50K-token conversation with 2K new tokens stays local.

```bash
pip install litellm

# Route to GPT-5 when >20K new tokens need prefilling
OPENAI_API_KEY=sk-... python -m vllm_mlx.server \
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

---

## Supported Models

### Recommended

| Model | Params | Quant | RAM | Decode | Tool Parser | Best For |
|-------|--------|-------|-----|--------|-------------|----------|
| [Qwen3.5-122B-A10B](https://huggingface.co/nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx) | 122B/10B | mxfp4 | 74GB | **41-47 tok/s** | `hermes` | **Best overall** — tool calling + reasoning |
| [Qwen3-Coder-Next](https://huggingface.co/lmstudio-community/Qwen3-Coder-Next-MLX-4bit) | 80B/3B | 4bit | 42GB | **~80-100 tok/s** | `hermes` | Speed + coding |
| [Qwen3-Coder-Next](https://huggingface.co/lmstudio-community/Qwen3-Coder-Next-MLX-6bit) | 80B/3B | 6bit | 60GB | ~65 tok/s | `hermes` | **Best balance** |
| [MiniMax-M2.5](https://huggingface.co/lmstudio-community/MiniMax-M2.5-MLX-4bit) | 229B/10B | 4bit | 130GB | 33-38 tok/s | `minimax` | Deep reasoning (192GB+) |

Benchmarks on Mac Studio M3 Ultra (256GB), 800 GB/s memory bandwidth.

### Hardware Requirements

How much RAM do you need? Model weights must fit in unified memory, plus ~20% headroom for KV cache. If your Mac uses >90% of RAM, you'll hit swap thrashing and decode speed drops to <5 tok/s.

| Mac RAM | Recommended Model | Weights | Decode Speed | Notes |
|---------|-------------------|---------|-------------|-------|
| 16 GB | Llama 3.2 3B 4bit | ~2 GB | ~100 tok/s | Small tasks only, limited tool calling |
| 32 GB | Qwen3-Coder-Next 4bit | ~42 GB | ~80 tok/s | Fits with swap, ok for short sessions |
| 64 GB | Qwen3-Coder-Next 6bit | ~60 GB | ~65 tok/s | Sweet spot for coding agents |
| 96 GB | Qwen3.5-122B mxfp4 | ~74 GB | ~45 tok/s | Best model fits comfortably |
| 128 GB | Qwen3.5-122B mxfp4 + `--kv-bits 4` | ~74 GB | ~45 tok/s | Long contexts (50K+ tokens) |
| 192 GB | Qwen3.5-122B + large KV headroom | ~74 GB | ~45 tok/s | 100K+ context, multi-turn agent sessions |
| 256 GB | MiniMax-M2.5 4bit or Qwen3.5 8bit | 130 GB+ | 33-38 tok/s | Maximum model quality |

**Rule of thumb:** model weights + 20% = minimum RAM. Use `--kv-bits 4` or `--kv-bits 8` to halve KV cache memory for long contexts.

#### Help Us Fill This Table

These numbers are from a single M3 Ultra (256GB). We need community data for other configs — M1, M2, M4, MacBook Air/Pro, different RAM tiers. If you test a model, **copy the template below into a [new issue](https://github.com/raullenchai/vllm-mlx/issues/new):**

```
**Hardware:** Mac ___ (___GB RAM), macOS ___
**Model:** ___ (quantization: ___)
**Server flags:** `python -m vllm_mlx.server --model ___ ...`
**Decode speed:** ___ tok/s
**TTFT (cold / cached):** ___s / ___s
**Agent client tested:** Claude Code / Cursor / Aider / other
**Did it fit in RAM?** Yes / No (swap used: ___GB)
**Notes:** ___
```

### Model Comparison: Qwen3.5-122B vs MiniMax-M2.5

Tested on Mac Studio M3 Ultra (256GB) with OpenClaw (14 tools, multi-turn agent sessions):

**Speed (with prompt cache hit):**

| Metric | Qwen3.5-122B mxfp4 | MiniMax-M2.5 4bit |
|--------|---------------------|-------------------|
| Decode (short, <100 tok) | 25-37 tok/s | 33-38 tok/s |
| Decode (long, 300+ tok) | **41-47 tok/s** | ~35 tok/s |
| Decode peak (800 tok) | **46.3 tok/s** | 38 tok/s |
| TTFT (cache hit) | **0.4-1.3s** | ~1-2s |
| TTFT (cache miss, 8K tok) | 12s | ~12s |
| Prefill throughput | ~670 tok/s | ~700 tok/s |

**Agent reliability (OpenClaw, 14 tools):**

| Metric | Qwen3.5-122B | MiniMax-M2.5 |
|--------|--------------|--------------|
| Tool-call loops | **None (78+ rounds)** | Confirmed — gets stuck in edit→read cycles |
| Instruction following (IFEval) | **93.4%** | ~70% (estimated) |
| Long conversation stability | **Stable at 23K+ tokens** | Degrades after ~90K tokens |
| Max single output | **2196 tokens** (code review) | ~800 tokens |
| Multi-step agent tasks | Completes reliably | Loops on complex tasks |

**Resource usage:**

| Metric | Qwen3.5-122B | MiniMax-M2.5 |
|--------|--------------|--------------|
| Model weight RAM | **74 GB** | 130 GB |
| KV cache headroom (256GB) | **~180 GB** | ~120 GB |
| Active params per token | 10B | 10B |

**Verdict:** Qwen3.5-122B is the recommended model for agent workloads — faster decode on long outputs, half the memory, and no tool-call loops. MiniMax-M2.5 is slightly faster on short outputs but prone to agentic instability.

### Quick Start Commands

```bash
# Qwen3.5-122B — best overall for agent workloads (74GB, 41-47 tok/s)
python -m vllm_mlx.server \
  --model nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --port 8000

# Qwen3-Coder-Next — fast coding agent (3B active, ~100 tok/s)
python -m vllm_mlx.server \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-4bit \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --port 8000

# MiniMax-M2.5 — deep reasoning with tool calling
python -m vllm_mlx.server \
  --model lmstudio-community/MiniMax-M2.5-MLX-4bit \
  --tool-call-parser minimax \
  --reasoning-parser minimax \
  --max-tokens 2048 \
  --port 8000

# Llama 3.2 — lightweight, fast
python -m vllm_mlx.server \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --tool-call-parser llama \
  --port 8000

# Mistral — general purpose
python -m vllm_mlx.server \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --tool-call-parser hermes \
  --port 8000

# DeepSeek-R1 — reasoning-focused
python -m vllm_mlx.server \
  --model mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit \
  --tool-call-parser deepseek \
  --reasoning-parser deepseek_r1 \
  --port 8000

# Qwen3 VL — vision + language
python -m vllm_mlx.server \
  --model mlx-community/Qwen3-VL-4B-Instruct-MLX-4bit \
  --mllm \
  --port 8000
```

### Tool Parser Selection Guide

| Model Family | `--tool-call-parser` | `--reasoning-parser` | Notes |
|-------------|---------------------|---------------------|-------|
| Qwen3.5-122B-A10B | `hermes` | `qwen3` | **Recommended** — best agent stability |
| Qwen3-Coder-Next | `hermes` | *(none)* | Non-thinking mode, fast |
| Qwen3 (thinking) | `qwen` or `qwen3_coder` | `qwen3` | With `<think>` tags |
| MiniMax-M2.5 | `minimax` | `minimax` | XML tool format |
| Llama 3.x | `llama` | *(none)* | JSON tool format |
| DeepSeek-R1 | `deepseek` | `deepseek_r1` | With reasoning |
| Mistral | `hermes` | *(none)* | Hermes-compatible |
| Functionary | `functionary` | *(none)* | Custom format |

All parsers include automatic text-format tool call recovery — if a quantized model degrades and outputs tool calls as plain text, they're automatically converted back to structured `tool_calls`.

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
python -m vllm_mlx.server \
  --model nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --max-tokens 4096 \
  --port 8000
```

**Fast coding agent:**

```bash
python -m vllm_mlx.server \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --port 8000
```

**Deep reasoning + tool calling:**

```bash
python -m vllm_mlx.server \
  --model lmstudio-community/MiniMax-M2.5-MLX-4bit \
  --tool-call-parser minimax \
  --reasoning-parser minimax \
  --prefill-step-size 4096 \
  --kv-bits 4 \
  --port 8000
```

**Hybrid local + cloud — best of both worlds:**

```bash
OPENAI_API_KEY=sk-... python -m vllm_mlx.server \
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

## What This Fork Adds (vs. Upstream)

### Agent-Grade Tool Calling (13 features)

- **Text-format tool call recovery** — quantized models degrade and output tool calls as plain text; server auto-detects and converts back to structured `tool_calls` (works with any model, any parser)
- 7 tool parsers — Hermes, MiniMax, Qwen, Qwen3-Coder, Llama, DeepSeek, Functionary
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

### Reasoning Separation (3 features)

- MiniMax reasoning parser — heuristic no-tag stripping (0% leak rate, was 60%)
- Chinese reasoning pattern recognition
- Clean `reasoning_content` field — reasoning never mixed into `content`

### Performance (6 features)

- Prompt cache (SimpleEngine) — persistent KV cache, 10-30x faster multi-turn
- `--prefill-step-size` — configurable prefill chunks for TTFT tuning
- `--kv-bits` — KV cache quantization (4/8 bit) for long contexts
- Speculative decoding — `--draft-model` with prompt cache compatibility
- Smart cloud routing — `--cloud-model` offloads large prefills to cloud LLMs
- Frequency-aware cache eviction — LRU-LFU hybrid under memory pressure

### Reliability (6 features)

- Accurate `prompt_tokens` reporting (was always 0)
- Prompt cache EOS fix — cache saved correctly on EOS
- Server crash prevention on malformed `response_format`
- GC control during generation to avoid latency spikes
- System prompt pinning in prefix cache
- **1500+ tests** — unit tests + end-to-end agent simulation with real tool execution

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

The model is outputting tool calls in text format instead of structured `tool_calls`. This usually means `--tool-call-parser` is not set, or is set to the wrong parser. Check the [Tool Parser Selection Guide](#tool-parser-selection-guide) for the correct parser for your model. Even when this happens, vllm-mlx auto-recovers degraded tool calls — but setting the correct parser gives the best results.

### Slow time-to-first-token (TTFT) on first request

Cold start is normal — the first request prefills the full prompt with no cache. Subsequent requests with shared prefixes (same system prompt, growing conversation) hit the prompt cache and are 10-30x faster. To speed up cold starts:
- Increase `--prefill-step-size` (e.g. `8192` or `16384`) for faster chunked prefill
- Use `--kv-bits 8` to reduce memory pressure during prefill
- Consider `--cloud-model` to route large cold requests to a cloud LLM

### Server stops responding after client disconnects

Fixed in this fork. The streaming disconnect guard ensures that when a client disconnects mid-stream, the server releases all locks and saves the prompt cache. If you're on an older version, upgrade to the latest.

---

## Contributing

Issues and PRs welcome at [github.com/raullenchai/vllm-mlx](https://github.com/raullenchai/vllm-mlx).

**We need your help.** The hardware tables, model benchmarks, and client configs in this README are based on one M3 Ultra (256GB). There are dozens of Mac + model + client combos we haven't tested. The easiest way to contribute:

1. **Report your hardware benchmarks** — use the [issue template](#help-us-fill-this-table) in the Hardware Requirements section
2. **Verify a client** — if you get Claude Code, Cursor, Continue.dev, or another client working, let us know what worked (and what didn't)
3. **Fix a doc** — if something in this README is wrong or missing, PRs are welcome

Built on [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) — all upstream features (multimodal, audio, embeddings, Anthropic API, MCP) are available.

## License

Apache 2.0 — see [LICENSE](LICENSE).
