<img width="1600" height="800" alt="banner" src="https://github.com/user-attachments/assets/f3743bb7-7287-4b24-ac97-a7037974396f" />
<p align="center">

<h1 align="center">Rapid-MLX</h1>

<p align="center">
  <strong>The fastest way to run AI models on Apple Silicon.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/tests-3300%2B-brightgreen.svg" alt="Tests"></a>
  <a href="https://support.apple.com/en-us/HT211814"><img src="https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple" alt="Apple Silicon"></a>
  <a href="https://github.com/raullenchai/Rapid-MLX/stargazers"><img src="https://img.shields.io/github/stars/raullenchai/Rapid-MLX?style=social" alt="GitHub stars"></a>
</p>

<p align="center">
  Rapid-MLX is a local LLM inference engine for Macs. It speaks the OpenAI Chat / Responses / Embeddings APIs, so anything that talks to ChatGPT — Cursor, Claude Code, Aider, Codex CLI, LangChain, PydanticAI, your own scripts — just works against <code>http://localhost:8000</code>. No cloud, no API key, no per-token cost.
</p>

<p align="center">
  <sub>
    <a href="https://rapidmlx.com"><b>rapidmlx.com</b></a> ·
    <a href="https://rapidmlx.com/desktop">Desktop app</a> ·
    <a href="https://rapidmlx.com/performance/">Community benchmarks</a> ·
    <a href="https://models.rapidmlx.com/">Model mirror</a>
  </sub>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/docs/assets/demo.gif" alt="Rapid-MLX demo — install, serve Gemma 4, chat, tool calling" width="700">
  <br>
  <em>pip install → serve Gemma 4 → chat + tool calling → works with PydanticAI, LangChain, Aider, and more.</em>
</p>

---

## What runs on your Mac

| | Your Mac | Model | Speed (tok/s) | What works |
|:---|:---:|:---:|:---:|:---:|
| **16 GB** MacBook Air | Qwen3.5-4B | 147 tok/s | Chat, coding, tools |
| **24 GB** MacBook Pro | Qwen3.5-9B | 101 tok/s | Great all-rounder |
| **32+ GB** Mac Mini / Studio | Gemma 4 12B | 64 tok/s | Vision + tools |
| **32+ GB** Mac Mini / Studio | GPT-OSS 20B | 119 tok/s | Harmony-native, 100% tools |
| **32+ GB** Mac Mini / Studio | Qwen3.6-35B-A3B | 93 tok/s | 256 MoE experts, 262K context |
| **48+ GB** Mac Mini / Studio | Qwen3.5-35B-A3B 8bit | 80 tok/s | Best balance of smart + fast |
| **96+ GB** Mac Studio / Pro | Qwen3.5-122B | 57 tok/s¹ | Frontier-level intelligence |
| **128+ GB** Mac Studio Ultra | DeepSeek V4 Flash 158B-A13B | 31–56 tok/s¹ | Day-0 frontier MoE, 1M context |

<sub>Single-user end-to-end throughput (B=1: one request at a time, 256 max output tokens, `output_tokens / wall-clock` incl. first-token latency), median of 3 rounds. `chat_template_kwargs.enable_thinking=False` passed where the engine honours it. Tested on M3 Ultra 256 GB / rapid-mlx v0.6.83 (fused top-p sampler). ¹ carried over from 2026-04 bench — disk-constrained on this refresh. The full sizing table with HuggingFace links is in [Choose Your Model](#choose-your-model).</sub>

<details>
<summary><b>New to local AI? Quick glossary</b></summary>

- **tok/s** (tokens per second) — roughly how many words the AI generates per second. Higher = faster.
- **4bit / 8bit** — compression levels for models. 4bit uses less memory (recommended); 8bit is higher quality.
- **TTFT** (Time To First Token) — how long before the AI starts responding.
- **Tool calling** — the AI can call functions in your code. Used by Cursor, Claude Code, and coding assistants.
- **OpenAI-compatible** — Rapid-MLX speaks the same HTTP API as ChatGPT, so any app that works with ChatGPT can work with Rapid-MLX by pointing at `http://localhost:8000/v1`.
- **Ollama / llama.cpp** — other popular local-AI runtimes. The only **apples-to-apples** row in our benchmark is GPT-OSS 20B (identical weights both sides) — Rapid-MLX runs it **2.3× faster than Ollama** under B=4 concurrent load. On the **Qwen3 closest-tag** rows (Qwen3.5/3.6 DeltaNet isn't on llama.cpp yet) Rapid-MLX leads 1.7–2.4×. The **Gemma 4** row ties Ollama's Gemma 3 (different architectures, 1.0×). Against `mlx-lm serve` (same MLX weights) Rapid-MLX is **1.2–1.5× faster**. Full caveats in [Benchmarks](#benchmarks).

</details>

---

## Quick Start

**1. Install** (pick one):

```bash
# uv (recommended — one command, isolated env, auto-manages Python)
uv tool install rapid-mlx@latest
# Don't have uv yet? curl -LsSf https://astral.sh/uv/install.sh | sh

# Or one-liner with auto-setup (installs Python if needed)
curl -fsSL https://raullenchai.github.io/Rapid-MLX/install.sh | bash

# Homebrew (Mac-native — needs tap + trust before install on Homebrew 4.x)
brew tap raullenchai/rapid-mlx
brew trust raullenchai/rapid-mlx
brew install rapid-mlx

# pip (requires Python 3.10+ — macOS ships 3.9, so install Python first if needed)
pip install rapid-mlx
```

Upgrade later: `uv tool upgrade rapid-mlx` / `brew upgrade rapid-mlx` / `pip install -U rapid-mlx`.

**2. Chat with a model right now:**

```bash
rapid-mlx chat
```

Defaults to `qwen3.5-4b-4bit`. First run downloads the model (~2.5 GB) — you'll see a progress bar. Drops you into a REPL when it's ready. Type `/help` for slash commands, `/exit` to quit. Pass `--think` to surface chain-of-thought, `--no-think` to skip it for faster replies.

**3. Or serve it for use from other apps:**

```bash
rapid-mlx serve qwen3.5-4b-4bit
```

Starts an OpenAI-compatible HTTP server. Wait for `Ready: http://localhost:8000/v1`, then point Cursor, Claude Code, Aider, LangChain, the OpenAI SDK — anything — at `http://localhost:8000/v1`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say hello"}],
).choices[0].message.content)
```

> **Vision/multimodal models** (Gemma 4, Qwen-VL) need extras: `pip install 'rapid-mlx[vision]'`. Text-only install is ~460 MB; vision adds ~322 MB. See [Optional Extras](#optional-extras).

> **"No matching distribution" from pip?** Your Python is too old. Run `python3 --version` — if it says 3.9, run `brew install python@3.12` then `python3.12 -m pip install rapid-mlx`.

> **`Refusing to load formula ... from untrusted tap`?** Homebrew 4.x requires third-party taps to be explicitly trusted before install. Run `brew trust raullenchai/rapid-mlx` (per-machine, persists across upgrades) then retry `brew install rapid-mlx`.

> **`brew install` stalls on `Tapping homebrew/core`?** Brew 5.x's install sandbox can't auto-tap `homebrew/core` mid-install. Pre-tap it once, then retry:
> ```bash
> brew tap homebrew/core --force   # ~1.3 GB, one-time
> brew tap raullenchai/rapid-mlx
> brew trust raullenchai/rapid-mlx
> brew install rapid-mlx
> ```

> **Not into the terminal?** [**Rapid-MLX Desktop**](https://rapidmlx.com) bundles the same engine inside a one-click Mac app — drag to Applications, pick a model, chat. The CLI here is still the source of truth for serving and scripting; the desktop app is the friendlier on-ramp.

---

## Why Rapid-MLX

If you're comparing against Ollama, LM Studio, `mlx-lm serve`, or `llama.cpp`:

- **Fastest on Apple Silicon, measured.** Under concurrent load (B=4), Rapid-MLX is **2.3× faster than Ollama** on the apples-to-apples GPT-OSS 20B row (identical weights both sides), leads by **1.7–2.4×** across the Qwen3.5 / Qwen3.6 closest-tag Ollama rows, and runs **1.2–1.5× faster than `mlx-lm serve`** on shared MLX weights. Full table in [Benchmarks](#benchmarks).
- **Drop-in OpenAI / Anthropic API.** Same `/v1/chat/completions` and `/v1/responses` as OpenAI, plus `/v1/messages` for the Anthropic SDK and Claude Code. No client-side adapter required.
- **Tool calling that actually works under 4-bit quantization.** 17 parser formats with automatic recovery — when a quantized model emits a malformed tool call as text, Rapid-MLX detects it and rebuilds the structured `tool_calls` object instead of failing silently. Verified end-to-end against Cursor, Codex CLI, Claude Code, Aider, LangChain, PydanticAI, smolagents.
- **Prompt cache, even on hybrid RNN models, sharable across tenants.** Standard transformers get radix-tree prefix cache — N clients hitting the same 2k-token system prompt converge on the same trie node (O(prefix_len) lookup vs. O(log N + LCP_scan) on hash caches). Hybrid models (Qwen3.5 DeltaNet) get RNN state snapshots restored in ~0.1 ms instead of re-running hundreds of tokens through the recurrent layers. 2–5× faster TTFT on every architecture, always on.
- **Long-prompt prefill acceleration.** PFlash scores 32K+ prompts and only prefills the sink + recent tail + query-relevant middle — **3.87–8.5× faster cold-start TTFT** with full needle-in-a-haystack recall. Default-on for verified aliases; `--pflash always` forces it for any model.
- **TurboQuant K8V4 KV codec, default-on for verified MoE.** 9 Qwen3.5/3.6 hero aliases ship with K8V4 (K 8-bit + V 4-bit after Walsh-Hadamard + Lloyd-Max) turned on — codec compresses KV to ~1/2.4 (~58% savings), lossless across the verified matrix. Set `--kv-cache-turboquant none` to force off.
- **Day-0 frontier models.** DeepSeek V4 Flash 158B-A13B, Qwen3.6 with 262K context, DiffusionGemma 26B (non-autoregressive), Gemma 4 vision — all behind the same Chat Completions API.

---

## Use Cases

The same `rapid-mlx` binary covers four very different workflows. Pick the one closest to what you want to do:

### 1. Chat in the terminal

```bash
rapid-mlx chat qwen3.5-9b-4bit
```

A streaming REPL with slash commands (`/help`, `/system`, `/save`, `/clear`). No second process needed. Use `--think` / `--no-think` to control chain-of-thought. Aliased as `rapid-mlx run` for Ollama muscle memory.

### 2. OpenAI-compatible server for your apps

```bash
rapid-mlx serve qwen3.5-9b-4bit
```

Point Cursor, Continue.dev, Open WebUI, LibreChat, the OpenAI Python SDK, LangChain, PydanticAI, smolagents at `http://localhost:8000/v1`. Tool calling, streaming, reasoning separation, structured JSON output, and embeddings are all on the standard endpoints. See [Connect Your Client](#connect-your-client) for one-shot configs.

### 3. Agent backends: Codex CLI, Claude Code, Aider, OpenCode

Rapid-MLX is the *backend* — pair it with an open-source agent CLI for the full Claude Code-like experience.

```bash
rapid-mlx serve qwen3.6-27b-8bit          # or any tool-capable alias
rapid-mlx agents codex --setup            # writes ~/.codex/config.toml
codex                                      # now talks to your local model
```

- **Codex CLI** (OpenAI's official Rust agent, 0.136.0+) hits `/v1/responses`. Verified end-to-end on Qwen3.5-9B / Qwen3.6-27B for chat, file read/write, shell, multi-step, and source analysis. Release gate G7b probes the codex-shape SSE on every tag. ([guide](docs/guides/codex-cli.md))
- **Claude Code** hits `/v1/messages` (Anthropic SDK). `ANTHROPIC_BASE_URL=http://localhost:8000 claude`.
- **Aider / OpenCode / OpenClaude / Goose / Hermes / Claw Code** — `rapid-mlx agents <name> --setup` writes the right config file.

### 4. Benchmark your hardware and contribute to the community

```bash
rapid-mlx bench qwen3.5-9b-4bit --submit
```

Runs the standardized B=1 community benchmark (greedy, 128 + 512 token buckets, 5 rounds each), shows you the JSON payload, asks for consent, and opens the PR via `gh` if you have it (or prints a compare-page URL if you don't). Submitted rows land in [community-benchmarks/submissions/](community-benchmarks/submissions/) and show up on [rapidmlx.com](https://rapidmlx.com) once merged. Your bench helps everyone else pick the right model for their Mac.

### Bonus: share a public URL

```bash
rapid-mlx share qwen3.6-27b-8bit
```

Tunnels your local server through `rapidserver.quicksilverpro.io` over a WebSocket. Prints a public OpenAI-compatible endpoint plus a bearer key — point any chat UI or OpenAI SDK at it. Bearer auth, CORS allowlist, and a 120 RPM rate-limit are wired automatically; closing the terminal tears the tunnel down. The default chat surface is our hosted Big-AGI fork (tool calling, personas, voice — no signup); any OpenAI-compatible client also works. Pick a 27B-class model or larger for a usable share experience.

---

## Connect Your Client

### Tested integrations

**Agent harnesses** (auto-setup with `rapid-mlx agents <name> --setup`):

| Harness | Type | Notes |
|---------|------|-------|
| [Codex CLI](https://github.com/openai/codex) | Agent | OpenAI's official Rust agent — `/v1/responses` shim, verified end-to-end against codex 0.136.0 ([guide](docs/guides/codex-cli.md)) |
| [Claude Code](https://www.anthropic.com/claude-code) | Agent | Anthropic SDK via `/v1/messages` — `ANTHROPIC_BASE_URL=http://localhost:8000 claude` |
| [Aider](https://aider.chat) | Agent | CLI edit-and-commit, architect mode ([test](tests/integrations/test_aider.sh)) |
| [OpenCode](https://github.com/sst/opencode) | TUI Agent | Claude Code-like terminal UX, OpenAI-compat provider |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | Agent | 62 tools, multi-turn ([test](tests/integrations/test_hermes.py)) |
| [Goose](https://github.com/block/goose) | Agent | Ollama provider via `OLLAMA_HOST` |
| [OpenClaude](https://github.com/Gitlawb/openclaude) | Agent | Anthropic SDK, `CLAUDE_CODE_USE_OPENAI=1` ([test](tests/integrations/test_anthropic_sdk.py)) |
| [Claw Code](https://github.com/ultraworkers/claw-code) | Agent | OpenAI & Anthropic endpoints |
| [PydanticAI](https://ai.pydantic.dev) | Framework | Typed agents, structured output ([test](tests/integrations/test_pydantic_ai_full.py)) |
| [LangChain](https://langchain.com) | Framework | `ChatOpenAI`, tools, streaming ([test](tests/integrations/test_langchain.py)) |
| [smolagents](https://github.com/huggingface/smolagents) | Framework | CodeAgent + ToolCallingAgent ([test](tests/integrations/test_smolagents_full.py)) |

**UI / IDE clients:**

| Client | Status | Setup |
|--------|--------|-------|
| [Cursor](https://cursor.com) | Compatible | Settings → OpenAI Base URL |
| [Continue.dev](https://continue.dev) | Compatible | VS Code / JetBrains extension |
| [LibreChat](https://librechat.ai) | Tested | Docker ([test](tests/integrations/test_librechat_docker.py)) |
| [Open WebUI](https://github.com/open-webui/open-webui) | Tested | Docker ([test](tests/integrations/test_openwebui.py)) |
| Any OpenAI-compatible app | Compatible | Point at `http://localhost:8000/v1` |

### Quick setup snippets

**Cursor** — Settings → Models → Add Model:
```
OpenAI API Base:  http://localhost:8000/v1
API Key:          not-needed
Model name:       default          (or qwen3.5-9b-4bit — either works)
```
Cursor's agent/composer mode uses tool calls automatically — Rapid-MLX handles them natively with Qwen3.5 models, no extra flags needed.

**Codex CLI** — `rapid-mlx agents codex --setup` writes `~/.codex/config.toml` for you. Or by hand:
```toml
model = "default"
model_provider = "rapid-mlx"

[model_providers.rapid-mlx]
name = "Rapid-MLX (local)"
base_url = "http://localhost:8000/v1"
# If rapid-mlx was started with --api-key, add env_key = "RAPID_MLX_API_KEY"
# and `export RAPID_MLX_API_KEY=...`. Don't use api_key = "..." — Codex
# CLI's --strict-config rejects inline literals.
```
Then `codex` (or `codex exec '<query>'`) routes through `/v1/responses`. See the [Codex CLI guide](docs/guides/codex-cli.md).

**Claude Code:**
```bash
ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=not-needed claude
```

**Aider:**
```bash
aider --openai-api-base http://localhost:8000/v1 --openai-api-key not-needed
```

**Goose:**
```bash
GOOSE_PROVIDER=ollama OLLAMA_HOST=http://localhost:8000 \
GOOSE_MODEL=default goose run --text "hello"
```

<details>
<summary><strong>More client setup snippets (Continue.dev, OpenCode, LibreChat, Open WebUI, Hermes, PydanticAI, smolagents, Anthropic SDK)</strong></summary>

**Continue.dev** (`~/.continue/config.yaml`):
```yaml
models:
  - name: rapid-mlx
    provider: openai
    model: default
    apiBase: http://localhost:8000/v1
    apiKey: not-needed
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

**Hermes Agent** (`~/.hermes/config.yaml`):
```yaml
model:
  provider: "custom"
  default: "default"
  base_url: "http://localhost:8000/v1"
  context_length: 32768
```

**OpenClaude:**
```bash
CLAUDE_CODE_USE_OPENAI=1 OPENAI_BASE_URL=http://localhost:8000/v1 \
OPENAI_API_KEY=not-needed OPENAI_MODEL=default openclaude -p "hello"
```

**Claw Code:**
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=not-needed
claw --model "openai/default" prompt "summarize this repo"
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
print(Agent(model).run_sync("What is 2+2?").output)
```

**smolagents** (`pip install smolagents`):
```python
from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(
    model_id="default",
    api_base="http://localhost:8000/v1",
    api_key="not-needed",
)
CodeAgent(tools=[], model=model).run("What is 5 multiplied by 7?")
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
| **Qwopus 27B** | **92** | Hermes / PydanticAI / LangChain / smolagents | 100% |
| **Qwen3.5 27B** | **82** | Hermes / PydanticAI / LangChain | 100% |
| **Llama 3.3 70B** | **83** | smolagents (text-based) | 100% |
| **Nemotron Nano 30B** | **59** | PydanticAI / LangChain | 91–93% |
| **Gemma 4 26B** | **62** | Hermes / smolagents | 100% |

Run `rapid-mlx agents` to list supported agents and `python3 scripts/mhi_eval.py` to compute MHI on your own setup.

---

## Command Reference

| Command | What it does |
|---|---|
| `rapid-mlx chat [model]` | Interactive chat REPL (alias: `rapid-mlx run`) |
| `rapid-mlx serve <model>` | Start an OpenAI-compatible HTTP server |
| `rapid-mlx share <model>` | Same as `serve`, but tunneled to a public URL |
| `rapid-mlx agents [name]` | List agent integrations; `--setup` / `--test` per integration |
| `rapid-mlx bench <model>` | Throughput benchmark; `--submit` runs the standardized B=1 community bench and opens a PR |
| `rapid-mlx models` | List all aliases (`--cached` shows only locally-downloaded ones; `ls` is a top-level alias) |
| `rapid-mlx pull <model>` | Download a model to the HF cache without starting a server |
| `rapid-mlx rm <model>` | Remove a cached model |
| `rapid-mlx ps` | List running `rapid-mlx` servers |
| `rapid-mlx info <model>` | Show the per-model profile (parsers, hybrid / MoE flags, DFlash eligibility) |
| `rapid-mlx doctor` | Self-diagnostic — Metal, imports, CLI, inference pipeline |
| `rapid-mlx upgrade` | Detect install method (brew / pip / install.sh) and upgrade |
| `rapid-mlx telemetry <status\|enable\|disable\|preview\|reset>` | Manage anonymous usage telemetry (off by default; see [Telemetry](#telemetry)) |
| `rapid-mlx version` | Print the installed version |

Every subcommand supports `--help`. Tab completion is wired automatically via `argcomplete` — see [shell completion](#shell-completion) if it doesn't fire.

---

## Choose Your Model

### What fits my Mac?

The model has to fit in your Mac's RAM. If your Mac slows down or Activity Monitor shows red memory pressure, pick a smaller model from the table below.

| Your Mac | Best Model | RAM Used | Speed (B=1) | Quality |
|----------|-----------|---------|-------|---------|
| **16 GB** MacBook Air/Pro | [Qwen3.5-4B 4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) | 2.4 GB | 147 tok/s | Good for chat and simple tasks |
| **24 GB** MacBook Pro | [Qwen3.5-9B 4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | 5.1 GB | 101 tok/s | Great all-rounder |
| **32 GB** Mac Mini / Studio | [Qwen3.5-27B 4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | 15.3 GB | 37 tok/s | Solid coding model |
| **32 GB** Mac Mini / Studio | [Gemma 4 12B 4bit](https://huggingface.co/mlx-community/gemma-4-12B-it-4bit) | 7 GB | 64 tok/s | Vision-capable + tool calling |
| **32 GB** Mac Mini / Studio | [GPT-OSS 20B MXFP4](https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q8) | 11 GB | 119 tok/s | Harmony-native, 100% tools |
| **32 GB** Mac Mini / Studio | [Qwen3.6-35B-A3B 4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) | 20 GB | 93 tok/s | 256 MoE experts, 262K context |
| **36 GB** MacBook Pro M3/M4 Pro | [Qwen3.5-27B 4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | 15.3 GB | 37 tok/s | Same as 32 GB — extra headroom for long contexts |
| **48 GB** Mac Mini / Studio | [Qwen3.5-35B-A3B 8bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-8bit) | 37 GB | 80 tok/s | **Sweet spot** — smart + fast |
| **64 GB** Mac Mini / Studio | [Qwen3.5-35B-A3B 8bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-8bit) | 37 GB | 80 tok/s | Same model, more room for KV cache |
| **96 GB** Mac Studio / Pro | [Qwen3.5-122B mxfp4](https://huggingface.co/nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx) | 65 GB | 57 tok/s¹ | Best model, fits comfortably |
| **128 GB** Mac Studio / Pro | [DeepSeek V4 Flash 2-bit DQ](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-2bit-DQ) | 91 GB | 56 tok/s¹ | 158B-A13B frontier MoE, day-0 (chat only) |
| **192 GB** Mac Studio / Pro | [Qwen3.5-122B 8bit](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-8bit) | 130 GB | 44 tok/s¹ | Maximum quality |
| **256 GB** Mac Studio Ultra | [DeepSeek V4 Flash 8-bit](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-8bit) | 136 GB | 31 tok/s¹ | 158B-A13B frontier MoE, 1M context (chat only) |

<sub>Speed = single-user end-to-end throughput (B=1: one request, 256 max output tokens, `output_tokens / wall-clock` including first-token latency), median of 3 rounds. rapid-mlx v0.6.83 (fused top-p sampler) on M3 Ultra 256 GB, 2026-06-09. ¹ Carried over from prior bench (disk-constrained on this refresh).</sub>

> **Not getting the speed in the table?** Most likely the model is reasoning before answering — add `--no-think` to skip chain-of-thought. See [Troubleshooting](#troubleshooting).

> **4bit vs 8bit:** 4bit models are compressed to use less memory (recommended for most users). 8bit models are higher quality but need more RAM. "mxfp4" is a high-quality 4-bit format.

### Alias naming convention

Every alias follows the same template:

`<family>-<version>-<params>-<modality?>-<technique?>-<quant>`

| Segment | Meaning | Examples |
|---|---|---|
| **family** | Model family | `gemma`, `qwen`, `llama`, `mistral`, `deepseek`, `phi` |
| **version** | Major version | `-4`, `3.5`, `3.6`, `-r1`, `-v4-flash` |
| **params** | Parameter count (MoE includes active count) | `12b`, `27b`, `35b-a3b` (35B total / 3B active) |
| **modality** *(optional)* | Non-text variants | `-vl` (vision), `-coder` (code) |
| **technique** *(optional)* | Training-time modifier | `-qat` (Quantization-Aware Training), `-distill`, `-thinking` |
| **quant** *(mandatory)* | Quantization tier | `-4bit`, `-8bit`, `-mxfp4`, `-qat-8bit`, … |

The **quantization suffix is mandatory on every alias** — `qwen3.5-4b-4bit` not `qwen3.5-4b`. This mirrors LM Studio's `…-MLX-4bit` HuggingFace convention so you never guess the bit width. `-qat` is a *technique* suffix, not a quant — it stacks before the quant (so a QAT-trained Gemma 4 12B in 4-bit is `gemma-4-12b-qat-4bit`).

| Quant suffix | Meaning |
|---|---|
| `-4bit` | Standard MLX 4-bit (most common) |
| `-8bit` | Standard MLX 8-bit (higher quality, ~2× RAM) |
| `-2bit`, `-3bit`, `-6bit` | Other bit widths |
| `-mxfp4` | Microscaling FP4 (high-quality 4-bit) |
| `-mxfp4-q8` | MXFP4 weights + Q8 head (GPT-OSS style) |
| `-dwq` | Dynamic Weight Quantization (mlx-community) |
| `-ud` | Unsloth Dynamic (mixed-precision per-layer) |
| `-unpacked` | Original FP16 / BF16 weights, no quantization |

### Full model lineup

**128+ explicit aliases across 30+ families** ship today (`rapid-mlx models`), including audio (13 TTS + 13 STT) and embedding aliases. `rapid-mlx info <alias>` shows the per-alias profile — parser, hybrid / MoE flags, K8V4 eligibility, DFlash / MTP eligibility.

<details>
<summary><strong>Text families at a glance</strong></summary>

| Family | Aliases | Notable |
|---|---|---|
| **Qwen3.5** | `qwen3.5-4b-4bit`, `-4b-8bit`, `-9b-4bit`, `-9b-8bit`, `-9b-6bit`, `-27b-4bit`, `-27b-6bit`, `-27b-8bit`, `-35b-4bit`, `-35b-6bit`, `-35b-8bit`, `-122b-mxfp4`, `-122b-8bit` | DeltaNet hybrid; **27b-8bit DFlash-eligible**; K8V4-verified: `-9b-4bit`, `-9b-8bit`, `-27b-4bit`, `-27b-8bit`, `-35b-6bit` |
| **Qwen3.6** | `qwen3.6-27b-4bit`, `-27b-8bit`, `-27b-ud`, `-35b-4bit`, `-35b-6bit`, `-35b-8bit`, `-35b-dwq`, `-35b-ud` | 262K ctx, 256 MoE experts; **27b-8bit DFlash-eligible**; K8V4-verified: `-35b-4bit`, `-35b-6bit`, `-35b-8bit`, `-35b-dwq` |
| **Qwen3** | `qwen3-0.6b-8bit`, `-4b-8bit`, `-8b-8bit`, `qwen3-coder-4bit`, `qwen3-coder-30b-4bit`, `qwen3-vl-4b-4bit`, `-8b-4bit`, `-30b-4bit` | Coding + vision |
| **Qwopus** | `qwopus-9b-4bit`, `qwopus-27b-4bit`, `qwopus-27b-8bit` | 92 MHI on tool calling |
| **DeepSeek** | `deepseek-r1-8b-4bit`, `-32b-4bit`, `deepseek-coder-v2-lite-16b-4bit`, `deepseek-v4-flash-2bit`, `-4bit`, `-8bit` | R1 reasoning + V4 Flash 158B-A13B day-0 |
| **Gemma** | `gemma-3n-e4b-4bit`, `gemma-4-12b-4bit`, `-12b-8bit`, `-12b-qat-4bit`, `-12b-qat-8bit`, `-26b-4bit`, `-26b-qat-4bit`, `-31b-4bit`, `-31b-8bit`, `-31b-qat-4bit`, `-31b-qat-8bit`, `gemma3-1b-4bit`, `-12b-4bit`, `-27b-4bit` | Vision-capable; QAT variants |
| **Llama / Hermes** | `llama3-1b-4bit`, `-3b-4bit`, `llama-3.1-8b-4bit`, `-8b-8bit`, `hermes3-8b-4bit`, `hermes4-70b-4bit` | |
| **GLM** | `glm4.5-air-4bit`, `glm4.7-9b-4bit` | |
| **GPT-OSS** | `gpt-oss-20b-mxfp4-q8` | Harmony native |
| **MiniMax / Kimi** | `minimax-m2.5-4bit`, `minimax-m2.7-mxfp4`, `kimi-48b-4bit`, `kimi-k2.5-3bit` | |
| **Mistral / Devstral** | `mistral-24b-4bit`, `devstral-24b-4bit`, `devstral-v2-24b-4bit`, `ministral-3b-4bit` | |
| **Phi / Granite / Nemotron / Bonsai / SmolLM** | `phi-4-14b-4bit`, `phi-4-mini-4bit`, `smollm3-3b-4bit`, `nemotron-30b-4bit`, `bonsai-1.7b-unpacked`, `-4b-unpacked`, `-8b-unpacked`, `granite4-tiny-4bit` | |
| **Text-Diffusion** | `diffusion-gemma-26b-4bit`, `diffusion-gemma-26b-8bit` | Non-autoregressive (block denoising); same `/v1/chat/completions` API |
| **Tmax / VibeThinker / Nanbeige / Holo3.1 / Bonsai / SmolLM / EmbeddingGemma / UI** | `tmax-9b-4bit`, `tmax-27b-4bit`, `vibethinker-*`, `nanbeige4.1-*`, `holo3.1-a3b-*`, `ui-*`, `embeddinggemma-*` | Community aliases (MLX-first-mover, agentic, GUI, embeddings) |

TurboQuant **K8V4 KV codec is default-on** for 9 verified Qwen3.5/3.6 MoE aliases (see [Features](#features)); DFlash and MTP speculative drafters are opt-in per alias (`rapid-mlx info <alias>` shows what's eligible).

</details>

### Copy-paste serve commands

```bash
# 16 GB — lightweight, fast
rapid-mlx serve qwen3.5-4b-4bit --port 8000

# 24 GB — best small model
rapid-mlx serve qwen3.5-9b-4bit --port 8000

# 32 GB — Gemma 4 12B (vision-capable, 64 tok/s)
rapid-mlx serve gemma-4-12b-4bit --port 8000

# 32 GB — GPT-OSS 20B (harmony-native, 100% tool calling, 119 tok/s)
rapid-mlx serve gpt-oss-20b-mxfp4-q8 --port 8000

# 32+ GB — Qwen 3.6 35B-A3B (256 experts, 262K context, 93 tok/s)
rapid-mlx serve qwen3.6-35b-4bit --port 8000

# 48+ GB — sweet spot (Qwen3.5-35B-A3B 8bit, 80 tok/s)
rapid-mlx serve qwen3.5-35b-8bit --prefill-step-size 8192 --port 8000

# 96+ GB — frontier (Qwen3.5-122B mxfp4)
rapid-mlx serve qwen3.5-122b-mxfp4 --prefill-step-size 8192 --port 8000

# Coding agent — fast MoE
rapid-mlx serve qwen3-coder-4bit --prefill-step-size 8192 --port 8000

# Vision — image understanding (needs [vision] extras)
rapid-mlx serve qwen3-vl-4b-4bit --mllm --port 8000

# Text-diffusion — DiffusionGemma 26B-A4B (block denoising, needs [vision] for mlx-vlm 0.6.3+)
rapid-mlx serve diffusion-gemma-26b-4bit --port 8000
```

> **Vision deps:** Install into the same environment where rapid-mlx lives:
> - `pip` users: `pip install 'rapid-mlx[vision]'` (in the same venv)
> - `install.sh` users: `~/.rapid-mlx/bin/pip install 'rapid-mlx[vision]'`
> - `brew` users: `$(brew --prefix)/opt/rapid-mlx/libexec/bin/pip install 'rapid-mlx[vision]'`

<details>
<summary><strong>Parser auto-detection &amp; manual overrides</strong></summary>

Parsers are **auto-detected from the model name** — you don't need to specify `--tool-call-parser` or `--reasoning-parser` for supported families. Explicit flags always override auto-detection.

| Model Family | `--tool-call-parser` | `--reasoning-parser` | Notes |
|-------------|---------------------|---------------------|-------|
| Qwen3.5 (all sizes) | `hermes` | `qwen3` | **Recommended** — 100% tool calling |
| Qwen3.6 | `qwen3_coder_xml` | `qwen3` | XML tool format, 262K context |
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

<details>
<summary><strong>Text-Diffusion (DiffusionGemma 26B-A4B)</strong></summary>

DiffusionGemma is a **non-autoregressive** language model — instead of emitting one token at a time, it denoises whole blocks of tokens in parallel via a diffusion process. Rapid-MLX wraps it behind the standard OpenAI Chat Completions API.

```bash
pip install 'rapid-mlx[vision]'       # mlx-vlm 0.6.3+ provides the diffusion runtime
rapid-mlx serve diffusion-gemma-26b-4bit --port 8000
```

**B=1 single-user benchmark** (M3 Ultra 256 GB, mlx-community/diffusiongemma-26B-A4B-it-4bit, median of 3 runs + 1 warmup):

| `max_tokens` | TTFT | E2E | Aggregate tok/s |
|---:|---:|---:|---:|
| 64 | 1.47s | 1.47s | 43 |
| 256 | 6.00s | 6.00s | 43 |
| 1024 | 5.71s | 19.58s | 37 |

Diffusion models emit tokens in **whole denoising blocks**, so the conventional `decode_tok/s = tokens / (e2e − ttft)` metric isn't meaningful here (ttft ≈ e2e for short outputs). The table reports **aggregate** throughput — `tokens / total_wall_time` — i.e. how many tokens actually land in the chat window per second. Throughput climbs with output length because the per-step denoising cost amortizes across more emitted tokens.

Reproduce: `python3.12 scripts/bench_diffusion_gemma.py --port 8000`.

</details>

---

## Benchmarks

Tested on **Mac Studio M3 Ultra (256 GB)**, 2026-06-06. Workload is **B=4 sustained concurrent streaming** (four parallel chat requests, 256 max output tokens each), median of 3 measured rounds after one warmup discard. Engines were swapped sequentially with an 8 s Metal cooldown so contention never crossed engine boundaries.

`chat_template_kwargs.enable_thinking=False` is passed to all engines that honour it (rapid-mlx, mlx-lm, mlx-vlm). Ollama 0.24 ignores that hook for Qwen3 and keeps streaming reasoning chunks — those decode at the same model rate as content tokens, so we count them, and the Qwen3 Ollama numbers reflect chain-of-thought-on throughput in practice. Token counts come from the streaming `usage` chunk (authoritative), not from counting SSE frames.

Versions: rapid-mlx **v0.6.80**, mlx-lm **0.31.3**, Ollama **0.24.0** (latest stable).

Aggregate throughput = sum of output tokens across all four streams ÷ wall-clock seconds — the metric that matters for a server fronting multiple users or a TUI firing parallel sub-agents. Per-user decode is roughly aggregate ÷ 4 on a true batching engine; on Ollama 0.24 (no in-flight batching) the four streams effectively serialize.

| Model (rapid-mlx alias) | rapid-mlx (B=4) | mlx-lm serve | Ollama tag (closest) | Ollama (B=4) | vs mlx-lm | vs Ollama |
|---|---:|---:|---|---:|:-:|:-:|
| **Qwen3.5-4B** | **261** tok/s | 173 | `qwen3:4b`¹ | 120 | **1.51x** | **2.18x** |
| **Qwen3.5-9B** | **180** tok/s | 136 | `qwen3:8b`¹ | 84 | **1.32x** | **2.14x** |
| **Qwen3.5-27B** | **66** tok/s | 55 | `qwen3:32b`² | 27 | **1.20x** | **2.43x** |
| **Gemma 4 12B** | **55** tok/s | crash³ | `gemma3:12b`⁴ | 56 | — | 1.00x |
| **GPT-OSS 20B** | **221** tok/s | 162 | `gpt-oss:20b` ✅ | 97 | **1.36x** | **2.29x** |
| **Qwen3.6-35B-A3B** (4-bit) | **176** tok/s | 129 | `qwen3:30b-a3b`⁵ | 87 | **1.37x** | **2.02x** |
| **Qwen3.5-35B-A3B** (8-bit) | **151** tok/s | 112 | `qwen3:30b-a3b`⁵ | 87 | **1.35x** | **1.74x** |

✅ Direct apples-to-apples: identical weights both sides.

<sub>¹ Ollama Qwen3 base, not Qwen3.5 — DeltaNet hybrid arch isn't on llama.cpp yet. ² Closest dense Qwen3; Unsloth Qwen3.6-27B GGUF fails to load on Ollama 0.24. ³ mlx-lm 0.31.3 has no Gemma 4 loader (it lives in mlx-vlm). ⁴ Gemma 4 not yet on llama.cpp — Gemma 3 is the closest. ⁵ Closest MoE A3B available; Qwen3.5/3.6-35B-A3B don't have a llama.cpp build yet.</sub>

Reproduce the throughput table:

```bash
python3.12 scripts/bench_readme_refresh.py \
  --models qwen3.5-4b-4bit,qwen3.5-9b-4bit,qwen3.5-27b-4bit,gemma-4-12b-4bit,gpt-oss-20b-mxfp4-q8,qwen3.6-35b-4bit,qwen3.5-35b-8bit \
  --engines rapid-mlx,mlx-lm,ollama
```

Raw JSON per round + per-stream tok/s land in `reports/benchmarks/readme-refresh/`. **Want to add your hardware?** Run `rapid-mlx bench <alias> --submit` — it opens a PR for you and your numbers show up on [rapidmlx.com](https://rapidmlx.com).

<details>
<summary><strong>TTFT — Prompt Cache Advantage</strong></summary>

Prompt cache keeps multi-turn conversations fast. For standard transformers, KV cache trimming gives sub-100ms TTFT. For hybrid RNN models (Qwen3.5 DeltaNet), we use state snapshots — the first technique to bring prompt cache to non-trimmable architectures on MLX.

<sub>Numbers below were last verified 2026-04 — the prefix-cache code path has not changed since.</sub>

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
<summary><strong>Capability comparison vs other Apple-Silicon runtimes</strong></summary>

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
<summary><strong>Optimization techniques per model</strong></summary>

| Technique | What it does | Models |
|-----------|-------------|--------|
| **Radix prefix cache** | O(prefix_len) trie lookup + copy-on-write nodes for multi-tenant shared system prompts | All transformer models (default `--prefix-cache-index radix`) |
| **DeltaNet state snapshots** | Deep-copy RNN state at prefix boundary, restore in ~0.1 ms | Qwen3.5 (4B, 9B, 27B, 35B, 122B), Qwen3-Coder-Next |
| **Hybrid cache sync** | Keep trimmable KV + non-trimmable RNN layers in sync | Qwen3.5 (Gated DeltaNet + attention) |
| **PFlash sparse prefill** | Score long prompt, prefill only sink + tail + relevant middle | Default-on for verified aliases; `--pflash always` |
| **Tool logits bias** | Jump-forward decoding — bias logits toward structured tokens | All models with `--enable-tool-logits-bias` |
| **Auto tool recovery** | Detect broken text-format tool calls, convert to structured | All 17 parser formats (incl. Gemma 4) |
| **TurboQuant K8V4 codec** | K 8-bit + V 4-bit after Walsh-Hadamard + Lloyd-Max (~1/2.4 compression, lossless across verified matrix) | Default-on for 9 verified Qwen3.5/3.6 aliases; `--kv-cache-turboquant k8v4\|v4\|none` |
| **KV cache quantization** | Quantize prefix cache entries to reduce memory | All models with `--kv-cache-quantization` |
| **DFlash speculative decoding** | Block-diffusion drafter, parallel draft + verify (single-user) | `qwen3.5-27b-8bit`, `qwen3.6-27b-8bit` with `--enable-dflash` |
| **MTP speculative decoding** | Multi-token prediction head shipped with the model | MTP-trained checkpoints with `--enable-mtp` |
| **SuffixDecoding** | Drafter-free, statistical n-gram lookup speculative decoding | All BatchedEngine models with `--suffix-decoding` |
| **Prefill chunking** | Configurable step size for large-prompt throughput | All models |
| **Cloud routing** | Offload high-token requests to cloud LLM when local is slow | All models with `--cloud-model` |

</details>

<details>
<summary><strong>Eval benchmarks (20 models, 4 suites)</strong></summary>

Tool calling (30 scenarios), coding (HumanEval+), reasoning (MATH-500), general knowledge (MMLU-Pro). Top models:

| Model | Decode (B=1) | Tools | Code | Reason | General | Avg |
|-------|--------|-------|------|--------|---------|-----|
| Qwen3.5-122B 8bit | 44 t/s¹ | 87% | 90% | 90% | 90% | **89%** |
| Qwen3.5-35B 8bit | 59 t/s | 90% | 90% | 80% | 80% | **85%** |
| Qwen3-Coder-Next 4bit | 74 t/s¹ | 90% | 90% | 70% | 70% | **80%** |
| Qwen3.5-27B 4bit | 33 t/s | 83% | 90% | 50% | 80% | **76%** |
| Qwen3.5-9B 4bit | 100 t/s | 83% | 70% | 60% | 70% | **71%** |

<sub>Decode = single-user end-to-end throughput refreshed 2026-06-06 against rapid-mlx v0.6.80. ¹ Carried over from the 2026-04 bench (not re-measured this round).</sub>

Run your own: `bash evals/run_all_models.sh` runs the full quality suite (tool calling, coding, reasoning, general) across every alias and emits a fresh `evals/SCORECARD.md`.

</details>

---

## Features

| Feature | What it does |
|---|---|
| **Tool calling** | OpenAI-compatible, 17 parser formats, automatic recovery when quantized models break (4-bit Qwen, Gemma, Llama, etc.). |
| **Reasoning separation** | Models with chain-of-thought (Qwen3, DeepSeek-R1, MiniMax, GPT-OSS) emit reasoning in a separate `reasoning_content` field, cleanly streamed alongside `content`. Casual chat + tool-call requests auto-disable thinking for lower TTFT (0.8.x+). |
| **Radix prefix cache** | Multi-tenant shared system prompts converge on the same radix node — O(prefix_len) lookup vs. O(log N + LCP_scan) on hash-keyed caches. Default index is `radix`; `--prefix-cache-index hash` for regressions. Always on; disable with `--disable-prefix-cache`. |
| **RNN prompt cache** | For hybrid models (Qwen3.5 DeltaNet), RNN state snapshots restore non-trimmable layers in ~0.1 ms instead of re-running the recurrent stack. 2–5× faster TTFT vs. mlx-lm on the DeltaNet aliases. |
| **PFlash prefill acceleration** | Sparse prefill on 32K+ prompts (attention sink + recent tail + query-relevant middle) — **3.87–8.5× TTFT on cold long prompts** with full needle-in-a-haystack recall. Default-on for verified aliases; `--pflash always` forces it. |
| **TurboQuant K8V4 KV codec** | K is 8-bit + V is 4-bit after Walsh-Hadamard rotation + Lloyd-Max quantization — compresses KV to ~1/2.4 (~58% savings), lossless across the verified matrix. **Default-on for 9 verified Qwen3.5/3.6 aliases** (see below); `--kv-cache-turboquant none` to force off. |
| **Smart cloud routing** | Large-context requests auto-route to a cloud LLM (`--cloud-model openai/gpt-5 --cloud-threshold 20000`) when local prefill would be slow. |
| **Multimodal** | Vision, audio (STT/TTS with bundled Silero VAD silence pre-trim), video understanding, text embeddings — all through the standard OpenAI endpoints. |
| **Speculative decoding** | DFlash (block-diffusion drafter, `--enable-dflash`), MTP (multi-token prediction head, `--enable-mtp`), SuffixDecoding (drafter-free n-gram, `--suffix-decoding`). Opt-in per alias — see below. |
| **Structured output, logprobs, continuous batching, KV quantization** | Standard, no flags or with one flag each. |
| **3300+ tests** | Across reasoning, tool parsing, streaming, agent harnesses, and engine integration. |

**K8V4 default-on aliases** (9 as of 0.9.9): `qwen3.5-9b-4bit`, `qwen3.5-9b-8bit`, `qwen3.5-27b-4bit`, `qwen3.5-27b-8bit`, `qwen3.5-35b-6bit`, `qwen3.6-35b-4bit`, `qwen3.6-35b-6bit`, `qwen3.6-35b-8bit`, `qwen3.6-35b-dwq`. Radix prefix cache is independent and saves additional bytes proportional to inter-request prefix overlap — the two are orthogonal.

<details>
<summary><strong>Speculative decoding — DFlash, MTP, SuffixDecoding</strong></summary>

Three drafters ship in-tree. All are **opt-in** per alias — `rapid-mlx info <alias>` shows what's eligible on the model you're serving.

**DFlash** — block-diffusion drafter (via mlx-vlm), single-user path. Opt in with `--enable-dflash`; requires `pip install 'rapid-mlx[dflash]'`.

| Alias | Drafter | Avg speedup | Min / Max |
|---|---|---|---|
| `qwen3.6-27b-8bit` | `z-lab/Qwen3.6-27B-DFlash` | **1.49×** | 1.06× / 2.07× |
| `qwen3.5-27b-8bit` | `z-lab/Qwen3.5-27B-DFlash` | **1.31×** | 0.59× / 2.15× |

```bash
pip install 'rapid-mlx[dflash]'
rapid-mlx info qwen3.5-27b-8bit       # check per-gate eligibility
rapid-mlx serve qwen3.5-27b-8bit --enable-dflash
```

Workload sensitivity: coding / math / summarization typically see **1.5–2.7×**; high-entropy creative writing and long-form chat can dip to **0.6–0.9×** because the drafter's training distribution diverges from open-ended generation — a known spec-decode literature pattern ([AdaEDL](https://arxiv.org/abs/2410.18351)), not a bug. DFlash mode runs a dedicated single-user server (no batched kernel yet); tool calling, MCP, and embeddings aren't available in that mode — restart without `--enable-dflash` for those.

**MTP** (Multi-Token Prediction) — draft head baked into the model. Opt in with `--enable-mtp` on MTP-trained checkpoints. Auto-tune of `--mtp-num-draft-tokens` per workload is roadmapped for the 0.9.1x line.

**SuffixDecoding** — drafter-free, statistical n-gram lookup over the running suffix. Opt in with `--suffix-decoding`; works on any BatchedEngine alias.

Mutex: DFlash cannot combine with MTP or SuffixDecoding (single-user path). MTP and SuffixDecoding cannot combine with each other (both consume the drafter slot).

</details>

<details>
<summary><strong>Server flags reference</strong></summary>

> You don't need any flags to get started — the defaults work for most setups. These are for advanced tuning.

**Core**

| Flag | Description | Default |
|------|-------------|---------|
| `<model>` | HuggingFace model name, local path, or alias (positional arg) | *(required)* |
| `--host` | Host to bind to (loopback-only by default; pass `0.0.0.0` to expose on LAN) | `127.0.0.1` |
| `--port` | Port to bind to | `8000` |
| `--max-tokens` | Default max tokens for generation | `32768` |

**Tool calling & reasoning**

| Flag | Description | Default |
|------|-------------|---------|
| `--tool-call-parser` | Parser: `hermes`, `minimax`, `qwen`, `llama`, `deepseek`, etc. | *(auto-detected)* |
| `--reasoning-parser` | Parser: `qwen3`, `deepseek_r1`, `minimax`, `gpt_oss`, `harmony`, `glm4`, `gemma4` | *(auto-detected)* |
| `--no-thinking` / `--no-think` | Disable reasoning even if auto-detected; faster, no `<think>` traces | off |
| `--enable-tool-logits-bias` | Jump-forward decoding for faster tool calls | off |

**Performance**

| Flag | Description | Default |
|------|-------------|---------|
| `--prefill-step-size` | Tokens per prefill chunk | `2048` |
| `--kv-cache-turboquant` | TurboQuant KV-cache codec (`k8v4` = K 8-bit + V 4-bit, ~1/2.4 compression; `v4` = legacy V-only; `none` = off). Default-on for 9 verified Qwen3.5/3.6 aliases. | auto (`k8v4` on verified aliases, else off) |
| `--kv-cache-quantization` | Quantize prefix cache entries for memory savings | off |
| `--enable-prefix-cache` / `--disable-prefix-cache` | Cache common prefixes across requests | on |
| `--prefix-cache-index` | Prefix-cache lookup index: `radix` (default) or `hash` | `radix` |
| `--pflash` | PFlash sparse prefill: `auto` / `always` / `off` | `auto` (on for verified aliases) |
| `--enable-dflash` | DFlash speculative decoding (single-user; `qwen3.5-27b-8bit` / `qwen3.6-27b-8bit`) | off |
| `--suffix-decoding` | Drafter-free n-gram speculative decoding (BatchedEngine path) | off |
| `--enable-mtp` | MTP head speculative decoding (requires MTP-trained model) | off |
| `--gpu-memory-utilization` | Fraction of device memory to use (0.0-1.0) | `0.90` |

**Cloud routing**

| Flag | Description | Default |
|------|-------------|---------|
| `--cloud-model` | litellm model string (e.g. `openai/gpt-5`) | *(disabled)* |
| `--cloud-threshold` | New token threshold to trigger cloud routing | `20000` |

**Security & other**

| Flag | Description | Default |
|------|-------------|---------|
| `--api-key` | API key for authentication | *(no auth)* |
| `--rate-limit` | Requests per minute per client | *(unlimited)* |
| `--timeout` | Request timeout in seconds | `1800` |
| `--mllm` / `--no-mllm` | Force / disable multimodal (vision) mode | auto-detect |
| `--force-openai-harmony-streaming` | Force the openai-harmony streaming router on (debug-only) | auto-detect |
| `--no-openai-harmony-streaming` | Disable the openai-harmony streaming router; fall back to the legacy state machine | auto-detect |
| `--mcp-config` | MCP configuration file for tool integration | *(none)* |
| `--embedding-model` | Pre-load embedding model at startup | *(none)* |

</details>

---

## Optional Extras

The base `pip install rapid-mlx` is ~460 MB and covers all text-only models. Vision, audio, and other features ship as opt-in extras:

| Extra | Install | Adds | What it unlocks |
|---|---|---|---|
| `vision` | `pip install 'rapid-mlx[vision]'` | ~322 MB | Gemma 4, Qwen-VL, DiffusionGemma, video understanding (mlx-vlm + opencv + torch) |
| `audio` | `pip install 'rapid-mlx[audio]'` | ~600 MB | 26 TTS/STT aliases (Kokoro, Chatterbox, VibeVoice, VoxCPM, Dia, Whisper, Parakeet) via `/v1/audio/speech` and `/v1/audio/transcriptions`; bundles Silero VAD for silence pre-trim (guards Whisper against silence hallucination) |
| `embeddings` | `pip install 'rapid-mlx[embeddings]'` | ~50 MB | `/v1/embeddings` endpoint (mlx-embeddings) |
| `chat` | `pip install 'rapid-mlx[chat]'` | ~150 MB | Built-in Gradio chat UI |
| `guided` | `pip install 'rapid-mlx[guided]'` | ~80 MB | Schema-constrained JSON generation (outlines) |
| `dflash` | `pip install 'rapid-mlx[dflash]'` | ~50 MB | DFlash speculative decoding runtime (text-only, no torch) |
| `all` | `pip install 'rapid-mlx[all]'` | ~1.1 GB | Vision + audio + chat + embeddings |

If you installed via Homebrew, install extras into the brew-managed Python so the `rapid-mlx` binary picks them up:

```bash
$(brew --prefix)/opt/rapid-mlx/libexec/bin/pip install 'rapid-mlx[vision]'   # or [audio], [embeddings], ...
```

(A separate venv-local `pip install 'rapid-mlx[vision]'` also works, but only affects the `rapid-mlx` inside that venv — the brew-installed binary keeps its own libexec Python.)

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

**Common gotchas:**

- **Getting much lower tok/s than the speed table?** Most often the model is reasoning before answering. Qwen3.5 and Qwen3.6 default to thinking-on, which doubles the work. Add `--no-think` to `chat` or `serve` to skip chain-of-thought. (This is the cause of the most common bug report — see [issue #567](https://github.com/raullenchai/Rapid-MLX/issues/567).)
- **Out of memory or very slow (<5 tok/s).** Model too big for your RAM. Check [What fits my Mac?](#what-fits-my-mac) and pick a smaller quant (4-bit) or smaller model.
- **Empty responses.** Remove `--reasoning-parser` for non-thinking models.
- **Tool calls coming through as plain text.** Set the right `--tool-call-parser` for your model, or rely on auto-detection. Even without it, Rapid-MLX auto-recovers most cases.
- **Slow first response.** Two causes: (1) Qwen3.5/3.6 think before answering — add `--no-think`; (2) cold prefill on long prompts — add `--prefill-step-size 8192`. Subsequent turns hit prompt cache and are 10–30× faster.
- **"parameters not found in model" warnings at startup.** Normal for VLMs — vision weights are auto-skipped.

### Shell completion

Tab completion works automatically when installed via Homebrew. For pip / install.sh, activate it once:

```bash
# zsh / bash
eval "$(register-python-argcomplete rapid-mlx)"

# Or system-wide:
activate-global-python-argcomplete
```

---

## Telemetry

Rapid-MLX **can** send anonymous usage data to help us prioritise the right models and catch regressions. **It is off by default and never starts collecting without your explicit opt-in.**

### What we collect (only if you opt in)

- Subcommand names (`serve` / `chat` / `agents` / `bench` / `doctor`)
- Model alias names (`qwen3.5-9b-4bit`) or canonical HF repo IDs (`mlx-community/...`) — local paths are redacted to `<local>`
- Bucketed counts: prompt/completion tokens, TTFT, tokens/sec — never exact values
- Error categories + a hash fingerprint of the failure site (exception class name + per-frame `file:function:lineno` only — never the message text or absolute paths)
- OS, arch, Apple chip name, RAM (rounded to GB), Python major.minor

### What we never collect

- Prompts, completions, tool-call arguments, file contents, or any user-generated text
- Local file paths, working directory, or model paths beyond their HF repo ID
- IPs or hostnames (Phase 2 routes through a Cloudflare Worker that strips IPs before forwarding; Phase 1 ships no transport at all)
- API keys, environment variable values, auth headers
- Stack trace messages or argument values

### Manage it

```bash
rapid-mlx telemetry status     # show current state and why
rapid-mlx telemetry preview    # print the exact JSON payload that would be sent
rapid-mlx telemetry enable     # opt in
rapid-mlx telemetry disable    # opt out
rapid-mlx telemetry reset      # delete consent + client-id files (re-prompts on next run)
```

### Force-disable in scripts / CI

Either of these always wins, regardless of stored consent:

```bash
RAPID_MLX_TELEMETRY=0 rapid-mlx serve qwen3.5-9b-4bit
rapid-mlx --no-telemetry serve qwen3.5-9b-4bit
```

There is intentionally **no env-var equivalent for force-on** — opting in must be an explicit one-time `rapid-mlx telemetry enable`. CI agents will never silently contribute. Code lives in [`vllm_mlx/telemetry/`](vllm_mlx/telemetry/); tracking issue [#236](https://github.com/raullenchai/Rapid-MLX/issues/236).

---

## Development

```bash
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX
pip install -e ".[dev]"
```

**Test commands:**

| Command | What | Time | Needs server? |
|---------|------|------|---------------|
| `make lint` | ruff lint | ~10s | No |
| `make test` | pytest unit suite (3300+ tests) | ~30s | No |
| `make smoke` | lint + unit | ~1 min | No |
| `make stress` | 8-scenario stress test | ~5 min | Yes |
| `make soak` | 10-min agent soak test | 10 min | Yes |
| `make check` | 1-model regression harness (auto starts server) | ~10 min | auto |
| `make full` | 3 models + 12 agent profiles | ~1 hr | auto |

For stress/soak, start a server first:

```bash
rapid-mlx serve qwen3.5-4b-4bit
# In another terminal:
make stress
```

Or use the script directly for more options: `python scripts/dev_test.py {smoke,stress,full}`.

<details>
<summary><strong>Architecture overview</strong></summary>

```
vllm_mlx/
  server.py              # App factory + model loading + CLI entry
  config/                # ServerConfig singleton
  service/
    helpers.py           # Shared request helpers
    postprocessor.py     # Streaming pipeline (100% test coverage)
  routes/
    chat.py              # /v1/chat/completions
    completions.py       # /v1/completions
    responses.py         # /v1/responses (Codex CLI)
    anthropic.py         # /v1/messages (Anthropic API)
    health.py, models.py, embeddings.py, audio.py, mcp_routes.py
  engine/                # BatchedEngine (continuous batching)
  reasoning/             # 7 reasoning parsers (Qwen3, DeepSeek, MiniMax, ...)
  tool_parsers/          # 17 tool call parsers
  speculative/           # DFlash, SuffixDecoding, MTP drafters
  agents/                # 12 agent profiles (YAML)
  runtime/               # Model registry, cache persistence
  doctor/                # User self-diagnostic
scripts/                 # Dev-only (NOT shipped with pip)
tests/                   # pytest unit tests (3300+)
harness/                 # Regression baselines + thresholds
```

</details>

---

## Roadmap

| Technique | Expected Gain | Status |
|-----------|---------------|--------|
| [DFlash](https://arxiv.org/abs/2602.06036) — block-diffusion drafter, single-user | 1.3–2× decode (workload-dependent) | Shipping, opt-in (`--enable-dflash`, `[dflash]` extra; qwen3.5-27b-8bit, qwen3.6-27b-8bit) |
| [SuffixDecoding](https://arxiv.org/abs/2411.04975) — drafter-free n-gram speculative | 1.1–1.5× decode | Shipping (`--suffix-decoding`, per-model tier sweep ongoing) |
| MTP — Multi-Token Prediction head | 1.4–1.7× decode | Shipping, opt-in (`--enable-mtp`, MTP-trained checkpoints); per-workload auto-tune of `--mtp-num-draft-tokens` landing in the 0.9.1x line |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) — feature-level draft on Metal | 3–6.5× decode | Not started |
| [ReDrafter](https://arxiv.org/abs/2403.09919) — Apple's RNN draft head | 1.4–1.5× decode | Not started |

See [ROADMAP.md](ROADMAP.md) for the full backlog.

---

## Contributing

We welcome contributions of all sizes! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

**Easy first contributions** (no model download needed):
- [Add a model alias](https://github.com/raullenchai/Rapid-MLX/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) — map a short name to a HuggingFace model ID
- [Request model support](https://github.com/raullenchai/Rapid-MLX/issues/new?template=model_support.yml) — tell us which model you want

**Testing contributions** (needs a Mac with Apple Silicon):
- **Share your hardware's benchmark numbers** — one command:
  ```bash
  rapid-mlx bench qwen3.5-9b-4bit --submit
  ```
  Runs the standardized B=1 bench (greedy, 128 + 512 token buckets, 5 rounds each), shows you the JSON payload, asks for consent, and opens the PR for you via `gh`. If you don't have `gh`, it prints the JSON path + a deep-link to GitHub's compare page so you can open the PR in your browser. Submitted rows land in [community-benchmarks/submissions/](community-benchmarks/submissions/) and show up on https://rapidmlx.com once merged.
- Test with your favorite AI client (Cursor, Aider, LangChain, etc.)
- [Report a bug](https://github.com/raullenchai/Rapid-MLX/issues/new?template=bug_report.yml)

### Contributors

<a href="https://github.com/raullenchai/Rapid-MLX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=raullenchai/Rapid-MLX" />
</a>

## Star History

<a href="https://star-history.com/#raullenchai/Rapid-MLX&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=raullenchai/Rapid-MLX&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=raullenchai/Rapid-MLX&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=raullenchai/Rapid-MLX&type=Date" />
  </picture>
</a>

## License

Apache 2.0 — see [LICENSE](LICENSE).
