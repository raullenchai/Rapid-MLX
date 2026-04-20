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
  <a href="tests/"><img src="https://img.shields.io/badge/tests-2100%2B-brightgreen.svg" alt="Tests"></a>
  <a href="https://support.apple.com/en-us/HT211814"><img src="https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple" alt="Apple Silicon"></a>
</p>

<p align="center">
  Run local AI models on your Mac — no cloud, no API costs. Works with Cursor, Claude Code, and any OpenAI-compatible app.
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/docs/assets/demo.gif" alt="Rapid-MLX demo — install, serve Gemma 4, chat, tool calling" width="700">
  <br>
  <em>pip install → serve Gemma 4 26B → chat + tool calling → works with PydanticAI, LangChain, Aider, and more.</em>
</p>

| | Your Mac | Model | Speed (tok/s = words/sec) | What works |
|:---|:---:|:---:|:---:|:---:|
| **16 GB** MacBook Air | Qwen3.5-4B | 160 tok/s | Chat, coding, tools |
| **32+ GB** Mac Mini / Studio | Nemotron-Nano 30B | 141 tok/s | 🆕 Fastest 30B, 100% tools |
| **32+ GB** Mac Mini / Studio | Qwen3.6-35B | 95 tok/s | 256 experts, 262K context |
| **64 GB** Mac Mini / Studio | Qwen3.5-35B | 83 tok/s | Best balance of smart + fast |
| **96+ GB** Mac Studio / Pro | Qwen3.5-122B | 57 tok/s | Frontier-level intelligence |

<details>
<summary><b>New to local AI? Quick glossary</b></summary>

- **tok/s** (tokens per second) — roughly how many words the AI generates per second. Higher = faster.
- **4bit / 8bit** — compression levels for models. 4bit uses less memory (recommended); 8bit is higher quality.
- **TTFT** (Time To First Token) — how long before the AI starts responding.
- **Tool calling** — the AI can call functions in your code. Used by Cursor, Claude Code, and coding assistants.
- **OpenAI API compatible** — Rapid-MLX speaks the same language as ChatGPT's API, so any app that works with ChatGPT can work with Rapid-MLX by just changing the server address.
- **Ollama / llama.cpp** — other popular tools for running local AI. Rapid-MLX is 2-4x faster on Apple Silicon.

</details>

---

## Quick Start

**Step 1 — Install** (pick one):

```bash
# Homebrew (recommended — just works, no Python version issues)
brew install raullenchai/rapid-mlx/rapid-mlx

# pip (requires Python 3.10+ — macOS ships 3.9, so install Python first if needed)
pip install rapid-mlx

# Or one-liner with auto-setup (installs Python if needed)
curl -fsSL https://raullenchai.github.io/Rapid-MLX/install.sh | bash
```

> **"No matching distribution" error?** Your Python is too old. Run `python3 --version` — if it says 3.9, install a newer Python: `brew install python@3.12` then `python3.12 -m pip install rapid-mlx`

**Step 2 — Serve a model:**
```bash
rapid-mlx serve gemma-4-26b
```
First run downloads the model (~14 GB) — you'll see a progress bar. Wait for `Ready: http://localhost:8000/v1`.

**Step 3 — Chat** (open a **second** terminal tab):
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

That's it — you now have an OpenAI-compatible AI server on `localhost:8000`. Point any app at `http://localhost:8000/v1` and it just works.

> **Tip:** Run `rapid-mlx models` to see all available model aliases. For a smaller/faster model, try `rapid-mlx serve qwen3.5-9b` (~5 GB).

<details>
<summary>More install options</summary>

**From source** (for development):
```bash
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX && pip install -e .
```

**Vision models** (adds torch + torchvision, ~2.5 GB extra):
```bash
pip install 'rapid-mlx[vision]'
```

**Audio** (TTS/STT via mlx-audio):
```bash
pip install 'rapid-mlx[audio]'
```
</details>

**Try it with Python** (make sure the server is running, then `pip install openai`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")  # any value works, no real key needed

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(response.choices[0].message.content)
```

---

## Works With

### Agent Harnesses (MHI-tested)

| Harness | Type | Notes |
|---------|------|-------|
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | Agent | 62 tools, multi-turn ([test](tests/integrations/test_hermes.py)) |
| [PydanticAI](https://ai.pydantic.dev) | Framework | Typed agents, structured output ([test](tests/integrations/test_pydantic_ai_full.py)) |
| [LangChain](https://langchain.com) | Framework | `ChatOpenAI`, tools, streaming ([test](tests/integrations/test_langchain.py)) |
| [smolagents](https://github.com/huggingface/smolagents) | Framework | CodeAgent + ToolCallingAgent ([test](tests/integrations/test_smolagents_full.py)) |
| [OpenClaude](https://github.com/Gitlawb/openclaude) (Anthropic SDK) | Agent | `CLAUDE_CODE_USE_OPENAI=1` ([test](tests/integrations/test_anthropic_sdk.py)) |
| [Aider](https://aider.chat) | Agent | CLI edit-and-commit, architect mode ([test](tests/integrations/test_aider.sh)) |
| [Goose](https://github.com/block/goose) | Agent | Ollama provider via `OLLAMA_HOST` |
| [Claw Code](https://github.com/ultraworkers/claw-code) | Agent | OpenAI & Anthropic endpoints |

### UI / IDE Clients

| Client | Status | Setup |
|--------|--------|-------|
| [Cursor](https://cursor.com) | Compatible | Settings → OpenAI Base URL |
| [Continue.dev](https://continue.dev) | Compatible | VS Code / JetBrains extension |
| [LibreChat](https://librechat.ai) | Tested | Docker ([test](tests/integrations/test_librechat_docker.py)) |
| [Open WebUI](https://github.com/open-webui/open-webui) | Tested | Docker ([test](tests/integrations/test_openwebui.py)) |
| Any OpenAI-compatible app | Compatible | Point at `http://localhost:8000/v1` |

### Model-Harness Index (MHI)

MHI measures how well a model works with a specific agent harness. It combines three dimensions:

| Dimension | Weight | What it measures | Source |
|---|---|---|---|
| **Tool Calling** | 50% | Can the model+harness execute function calls correctly? | `rapid-mlx agents --test` |
| **HumanEval** | 30% | Can the model generate correct code? | [HumanEval](https://github.com/openai/human-eval) (10 tasks) |
| **MMLU** | 20% | Does the harness degrade base knowledge? | [tinyMMLU](https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU) (10 tasks) |

**MHI = 0.50 × ToolCalling + 0.30 × HumanEval + 0.20 × MMLU** (scale 0-100)

| Model | Best MHI | Best Harness | Tool Calling |
|---|---|---|---|
| **Qwopus 27B** | **92** | All (Hermes, PydanticAI, LangChain, smolagents) | 100% |
| **Qwen3.5 27B** | **82** | Hermes / PydanticAI / LangChain | 100% |
| **Llama 3.3 70B** | **83** | smolagents (text-based) | 100% |
| **Nemotron Nano 30B** | **59** | PydanticAI / LangChain | 91-93% |
| **Gemma 4 26B** | **62** | Hermes / smolagents | 100% |

<details>
<summary>Full MHI table (25 model-harness combinations) + methodology</summary>

**MHI = 0.50 × ToolCalling + 0.30 × HumanEval + 0.20 × MMLU** (scale 0-100)

Run `rapid-mlx agents` to see all supported agents and `python3 scripts/mhi_eval.py` to compute MHI on your own setup.

| Model + Harness | Tool Calling | HumanEval | MMLU | **MHI** |
|---|---|---|---|---|
| **Qwopus 27B** + Hermes | 100% | 80% | 90% | **92** |
| **Qwopus 27B** + PydanticAI | 100% | 80% | 90% | **92** |
| **Qwen3.5 27B** + Hermes | 100% | 40% | 100% | **82** |
| **Llama 3.3 70B** + smolagents | 100% | 50% | 90% | **83** |
| **DeepSeek-R1 32B** + smolagents | 100% | 30% | 100% | **79** |
| **Gemma 4 26B** + Hermes | 100% | 0% | 60% | **62** |
| **Nemotron Nano 30B** + PydanticAI | 93% | 0% | 60% | **59** |

</details>

**Quick setup for popular apps:**

**Cursor:** Settings → Models → Add Model:
```
OpenAI API Base:  http://localhost:8000/v1
API Key:          not-needed
Model name:       default          (or qwen3.5-9b — either works)
```
Cursor's agent/composer mode uses tool calls automatically — Rapid-MLX handles them natively with Qwen3.5 models, no extra flags needed.

**Claw Code:**
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=not-needed
claw --model "openai/default" prompt "summarize this repo"
```

**OpenClaude:**
```bash
CLAUDE_CODE_USE_OPENAI=1 OPENAI_BASE_URL=http://localhost:8000/v1 \
OPENAI_API_KEY=not-needed OPENAI_MODEL=default openclaude -p "hello"
```

**Hermes Agent** (`~/.hermes/config.yaml`):
```yaml
model:
  provider: "custom"
  default: "default"
  base_url: "http://localhost:8000/v1"
  context_length: 32768
```

**Goose:**
```bash
GOOSE_PROVIDER=ollama OLLAMA_HOST=http://localhost:8000 \
GOOSE_MODEL=default goose run --text "hello"
```

**Claude Code:**
```bash
OPENAI_BASE_URL=http://localhost:8000/v1 claude
```

<details>
<summary><strong>More client setup instructions</strong></summary>

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

**PydanticAI** (`pip install pydantic-ai`):
```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    model_name="default",
    provider=OpenAIProvider(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    ),
)
agent = Agent(model)
print(agent.run_sync("What is 2+2?").output)
```

**smolagents** (`pip install smolagents`):
```python
from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(
    model_id="default",
    api_base="http://localhost:8000/v1",
    api_key="not-needed",
)
agent = CodeAgent(tools=[], model=model)
agent.run("What is 5 multiplied by 7?")
```

**LibreChat** (`librechat.yaml`, under `endpoints.custom`):
```yaml
- name: "Rapid-MLX"
  apiKey: "rapid-mlx"
  baseURL: "http://localhost:8000/v1/"
  models:
    default: ["default"]
    fetch: true
  titleConvo: true
  titleModel: "current_model"
  modelDisplayLabel: "Rapid-MLX"
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

The model has to fit in your Mac's RAM. If your Mac slows down or Activity Monitor shows red memory pressure, pick a smaller model from the table below.

| Your Mac | Best Model | RAM Used | Speed | Quality |
|----------|-----------|---------|-------|---------|
| **16 GB** MacBook Air/Pro | [Qwen3.5-4B 4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) | 2.4 GB | 160 tok/s | Good for chat and simple tasks |
| **24 GB** MacBook Pro | [Qwen3.5-9B 4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | 5.1 GB | 108 tok/s | Great all-rounder |
| **32 GB** Mac Mini / Studio | [Qwen3.5-27B 4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | 15.3 GB | 39 tok/s | Solid coding model |
| **32 GB** Mac Mini / Studio | 🆕 [Nemotron-Nano 30B 4bit](https://huggingface.co/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit) | 18 GB | 141 tok/s | Fastest 30B, 100% tool calling |
| **32 GB** Mac Mini / Studio | [Qwen3.6-35B-A3B 4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) | 20 GB | 95 tok/s | 256 MoE experts, 262K context |
| **36 GB** MacBook Pro M3/M4 Pro | [Qwen3.5-27B 4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | 15.3 GB | 39 tok/s | Same as 32 GB — extra headroom for long contexts |
| **48 GB** Mac Mini / Studio | [Qwen3.5-35B-A3B 8bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-8bit) | 37 GB | 83 tok/s | **Sweet spot** — smart + fast |
| **64 GB** Mac Mini / Studio | [Qwen3.5-35B-A3B 8bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-8bit) | 37 GB | 83 tok/s | Same model, more room for KV cache |
| **96 GB** Mac Studio / Pro | [Qwen3.5-122B mxfp4](https://huggingface.co/nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx) | 65 GB | 57 tok/s | Best model, fits comfortably |
| **192 GB** Mac Studio / Pro | [Qwen3.5-122B 8bit](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-8bit) | 130 GB | 44 tok/s | Maximum quality |

> **4bit vs 8bit:** 4bit models are compressed to use less memory (recommended for most users). 8bit models are higher quality but need more RAM. "mxfp4" is a high-quality 4bit format.

### Copy-paste commands

Pick the one that matches your Mac. Short aliases work — run `rapid-mlx models` to see all available models.

```bash
# 16 GB — lightweight, fast
rapid-mlx serve qwen3.5-4b --port 8000

# 24 GB — best small model
rapid-mlx serve qwen3.5-9b --port 8000

# 32 GB — solid coding model
rapid-mlx serve qwen3.5-27b --port 8000

# 32 GB — Nemotron Nano (fastest 30B, 141 tok/s, NVIDIA MoE)
rapid-mlx serve nemotron-30b --port 8000

# 32+ GB — Qwen 3.6 (256 experts, 262K context)
rapid-mlx serve qwen3.6-35b --port 8000

# 64 GB — sweet spot
rapid-mlx serve qwen3.5-35b --prefill-step-size 8192 --port 8000  # faster first response

# 96+ GB — best model
rapid-mlx serve qwen3.5-122b --prefill-step-size 8192 --port 8000

# Coding agent — fast MoE, great for Claude Code / Cursor
rapid-mlx serve qwen3-coder --prefill-step-size 8192 --port 8000  # MoE = only uses part of the model, so it's fast

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
| 🆕 Qwen3.6 | `qwen3_coder_xml` | `qwen3` | XML tool format, 262K context |
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

Tested on **Mac Studio M3 Ultra (256GB)**. Rapid-MLX uses Apple's [MLX framework](https://github.com/ml-explore/mlx) — purpose-built for unified memory with native Metal compute kernels — which is why it beats C++-based engines (Ollama, llama.cpp) on most models. Ollama numbers tested with **v0.20.4** (latest, with MLX backend).

| Model | Rapid-MLX | Best Alternative | Speedup |
|-------|----------|-----------------|---------|
| **Phi-4 Mini 14B** | **180** tok/s | 77 (mlx-lm) / 56 (Ollama) | **2.3x** / **3.2x** |
| **Qwen3.5-4B** | **160** tok/s | 155 (mlx-lm serve) | **1.0x** |
| **Nemotron-Nano 30B** | **141** tok/s · 100% tools | — | — |
| **GPT-OSS 20B** | **127** tok/s · 100% tools | 79 (mlx-lm serve) | **1.6x** |
| **Qwen3.5-9B** | **108** tok/s | 41 (Ollama) | **2.6x** |
| **Qwen3.6-35B-A3B** | **95** tok/s · 100% tools | — | — |
| **Kimi-Linear-48B** | **94** tok/s · 100% tools | — (only engine) | — |
| **Gemma 4 26B-A4B** | **85** tok/s | 68 (Ollama) | **1.3x** |
| **Gemma 4 E4B** | **83** tok/s | — | — |
| **Qwen3.5-35B-A3B** | **83** tok/s · 100% tools | 75 (oMLX) | **1.1x** |
| **Qwen3-Coder 80B** | **74** tok/s · 100% tools | 69 (mlx-lm serve) | **1.1x** |
| **Qwen3.5-122B** | **44** tok/s · 100% tools | 43 (mlx-lm serve) | ~1.0x |
| **Gemma 4 31B** | **31** tok/s | — | — |

*Full benchmark data with all models, TTFT tables, DeltaNet snapshots, and engine comparison below.*

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
| Gemma 4 E4B | **0.25s** | — (day-0) | — |
| Gemma 4 26B-A4B | **0.25s** | — (day-0) | — |
| Gemma 4 31B | **0.34s** | 0.57s (mlx-vlm bf16) | **1.7x** |

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
| **Auto tool recovery** | Detect broken text-format tool calls, convert to structured | All 18 parser formats (incl. Gemma 4) |
| **KV cache quantization** | Quantize prefix cache entries to reduce memory | All models with `--kv-cache-quantization` |
| **Prefill chunking** | Configurable step size for large-prompt throughput | All models |
| **Cloud routing** | Offload high-token requests to cloud LLM when local is slow | All models with `--cloud-model` |

</details>

<details>
<summary><strong>Eval benchmarks (20 models, 4 suites)</strong></summary>

Tool calling (30 scenarios), coding (HumanEval+), reasoning (MATH-500), general knowledge (MMLU-Pro). Top models:

| Model | Decode | Tools | Code | Reason | General | Avg |
|-------|--------|-------|------|--------|---------|-----|
| Qwen3.5-122B 8bit | 44 t/s | 87% | 90% | 90% | 90% | **89%** |
| Qwen3.5-35B 8bit | 83 t/s | 90% | 90% | 80% | 80% | **85%** |
| Qwen3-Coder-Next 4bit | 74 t/s | 90% | 90% | 70% | 70% | **80%** |
| Qwen3.5-27B 4bit | 39 t/s | 83% | 90% | 50% | 80% | **76%** |
| Qwen3.5-9B 4bit | 108 t/s | 83% | 70% | 60% | 70% | **71%** |

Run your own: `python scripts/benchmark_engines.py --engine rapid-mlx ollama --runs 3`

</details>

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

Also: logprobs API, structured JSON output (`response_format`), continuous batching, KV cache quantization (`--kv-cache-quantization`), and [2100+ tests](tests/).

---

<details>
<summary><strong>Server Flags Reference</strong></summary>

> You don't need any flags to get started — the defaults work for most setups. These are for advanced tuning.

### Core

| Flag | Description | Default |
|------|-------------|---------|
| `<model>` | HuggingFace model name, local path, or alias (positional arg) | *(required)* |
| `--host` | Host to bind to | `0.0.0.0` |
| `--port` | Port to bind to | `8000` |
| `--max-tokens` | Default max tokens for generation | `32768` |

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
| `--kv-cache-quantization` | Quantize prefix cache entries for memory savings | off |
| `--enable-prefix-cache` | Cache common prefixes across requests | off |
| `--gpu-memory-utilization` | Fraction of device memory to use (0.0-1.0) | `0.90` |

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
<summary><strong>Common Issues</strong></summary>

**"parameters not found in model" warnings at startup** — Normal for VLMs. Vision weights are auto-skipped.

**Out of memory / very slow (<5 tok/s)** — Model too big. Check [What fits my Mac?](#what-fits-my-mac) Try a smaller quantization (4bit) or smaller model.

**Empty responses** — Remove `--reasoning-parser` for non-thinking models.

**Tool calls as plain text** — Set the correct `--tool-call-parser` for your model. Even without it, Rapid-MLX auto-recovers most cases.

**Other issues?** Run `rapid-mlx doctor` for self-diagnostics.

**Slow first response** — Two different causes: (1) Qwen3.5 models reason before answering — add `--no-thinking` to skip reasoning for faster responses, or (2) cold start on long prompts — add `--prefill-step-size 8192` to speed up processing. Subsequent turns hit prompt cache and are 10-30x faster.

**Server hangs after client disconnect** — Fixed in v0.3.0+. Upgrade to latest.

</details>

---

## Troubleshooting

Run the built-in self-diagnostic (works from `pip install`, no dev tools needed):

```bash
rapid-mlx doctor
```

```
Rapid-MLX Doctor
============================================================
  [metal] OK        # Apple Silicon Metal GPU available
  [imports] OK      # Core modules import cleanly
  [cli] OK          # CLI commands respond
  [model_load] OK   # Inference pipeline works
Result: PASS
```

---

## Development

### Quick start

```bash
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX
pip install -e ".[dev]"
```

### Testing

Two layers: **user-facing doctor** (ships with pip) and **dev test suite** (source checkout only).

#### Dev test commands

| Command | What | Time | Needs server? |
|---------|------|------|---------------|
| `make lint` | ruff lint | ~10s | No |
| `make test` | pytest unit suite (2000+ tests) | ~30s | No |
| `make smoke` | lint + unit | ~1 min | No |
| `make stress` | 8-scenario stress test | ~5 min | Yes |
| `make soak` | 10-min agent soak test | 10 min | Yes |

For stress/soak, start a server first:
```bash
rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit --enable-auto-tool-choice --tool-call-parser hermes
# In another terminal:
make stress
```

Or use the script directly for more options:
```bash
python scripts/dev_test.py smoke              # lint + unit
python scripts/dev_test.py stress --port 8000 # custom port
python scripts/dev_test.py full               # everything
```

#### Regression harness (multi-model)

```bash
make check              # 1 model (~10 min, auto starts server)
make full               # 3 models + 11 agent profiles (~1 hr)
make benchmark          # all local models (overnight)
```

### Architecture

```
vllm_mlx/
  server.py              # App factory + model loading + CLI (1047 lines)
  config/                # ServerConfig singleton
  service/
    helpers.py           # Shared request helpers
    postprocessor.py     # Streaming pipeline (100% test coverage)
  routes/
    chat.py              # /v1/chat/completions
    completions.py       # /v1/completions
    anthropic.py         # /v1/messages (Anthropic API)
    health.py, models.py, embeddings.py, audio.py, mcp_routes.py
  engine/                # BatchedEngine (continuous batching)
  reasoning/             # 7 reasoning parsers (Qwen3, DeepSeek, MiniMax, ...)
  tool_parsers/          # 20+ tool call parsers
  agents/                # 11 agent profiles (YAML)
  runtime/               # Model registry, cache persistence
  doctor/                # User self-diagnostic
scripts/                 # Dev-only (NOT shipped with pip)
  dev_test.py            # Unified test entry point
  stress_test.py         # 8-scenario stress test
  agent_soak_test.py     # 10-min agent soak test
  cross_model_stress.py  # Multi-model validation
tests/                   # pytest unit tests (2000+)
harness/                 # Regression baselines + thresholds
```

---

## Roadmap

| Technique | Expected Gain | Status |
|-----------|---------------|--------|
| [Standard Speculative Decode](https://arxiv.org/abs/2302.01318) — draft model acceleration | 1.5-2.3x decode | Not started |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) — feature-level draft on Metal | 3-6.5x decode | Not started |
| [ReDrafter](https://arxiv.org/abs/2403.09919) — Apple's RNN draft head | 1.4-1.5x decode | Not started |

---

## Contributing

We welcome contributions of all sizes! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

**Easy first contributions** (no model download needed):
- [Add a model alias](https://github.com/raullenchai/Rapid-MLX/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) — map a short name to a HuggingFace model ID
- [Request model support](https://github.com/raullenchai/Rapid-MLX/issues/new?template=model_support.yml) — tell us which model you want

**Testing contributions** (needs a Mac with Apple Silicon):
- Benchmark a model and share results
- Test with your favorite AI client (Cursor, Aider, LangChain, etc.)
- [Report a bug](https://github.com/raullenchai/Rapid-MLX/issues/new?template=bug_report.yml)

### Contributors

<a href="https://github.com/raullenchai/Rapid-MLX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=raullenchai/Rapid-MLX" />
</a>

## License

Apache 2.0 — see [LICENSE](LICENSE).
