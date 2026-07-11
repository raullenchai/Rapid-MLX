<img width="1600" height="800" alt="banner" src="https://github.com/user-attachments/assets/f3743bb7-7287-4b24-ac97-a7037974396f" />

<h1 align="center">Rapid-MLX</h1>

<p align="center">
  <strong>The fastest local AI engine for Apple Silicon.</strong>
  <br>
  <em>Drop-in OpenAI / Anthropic API · 2–4× faster than Ollama · Runs on any M-series Mac.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/rapid-mlx/"><img src="https://img.shields.io/pypi/v/rapid-mlx?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://github.com/raullenchai/homebrew-rapid-mlx"><img src="https://img.shields.io/badge/Homebrew-raullenchai%2Frapid--mlx-orange?logo=homebrew" alt="Homebrew tap"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://support.apple.com/en-us/HT211814"><img src="https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple" alt="Apple Silicon"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://github.com/raullenchai/Rapid-MLX/stargazers"><img src="https://img.shields.io/github/stars/raullenchai/Rapid-MLX?style=social" alt="GitHub stars"></a>
</p>

<p align="center">
  <sub>
    <a href="https://rapidmlx.com"><b>rapidmlx.com</b></a> ·
    <a href="https://rapidmlx.com/docs/">Docs</a> ·
    <a href="https://models.rapidmlx.com/">Model mirror</a> ·
    <a href="https://rapidmlx.com/desktop">Desktop app</a>
  </sub>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/docs/assets/demo.gif" alt="Rapid-MLX demo — install, serve Gemma 4, chat, tool calling" width="700">
</p>

---

## Quick Start (60 seconds)

**1. Install** (one command, detects your RAM, picks a starter model):

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
```

Installs Python 3.10+ if missing, creates an isolated venv at `~/.rapid-mlx/`, symlinks the `rapid-mlx` CLI into `~/.local/bin/`, and prints a serve command sized to your Mac (8–23 GB → `qwen3.5-4b-4bit`; 24–47 GB → `gpt-oss-20b-mxfp4-q8`; 48–95 GB → `qwen3.6-35b-8bit`; 96 GB+ → `gpt-oss-120b-mxfp4-q8`).

> **`curl | bash` security.** `install.sh` is served over HTTPS (HSTS-preload) from `rapidmlx.com` and is a byte-identical mirror of [`install.sh`](install.sh) at the current release commit — read it before running if you like. Two verified alternatives:
> - **Pin to a commit hash** — `curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/<commit>/install.sh -o install.sh && shasum -a 256 install.sh && bash install.sh`
> - **Skip the shell script entirely** — use Homebrew, `uv`, or `pip` below.

See [Alternative install methods](#alternative-install-methods) for the non-curl paths.

**2. Chat with a model right now:**

```bash
rapid-mlx chat
```

Defaults to `qwen3.5-4b-4bit`. First run downloads the weights (~2.5 GB) with a progress bar and drops you into a REPL. Type `/help` for slash commands, `/exit` to quit.

**3. Or serve it for use from other apps:**

```bash
rapid-mlx serve qwen3.5-4b-4bit
```

Starts an OpenAI-compatible HTTP server bound to `http://localhost:8000`. Point any OpenAI SDK / client (Cursor, Aider, LangChain, OpenCode, PydanticAI, your own scripts) at **`http://localhost:8000/v1`**; Claude Code / Anthropic SDK uses **`http://localhost:8000`** (the Anthropic messages route lives at `/v1/messages` under the same host).

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

> **Vision / audio / diffusion models?** Base install is text-only (~460 MB). Vision, audio, embeddings, and DFlash speculative decoding ship as opt-in extras. → [Optional extras](https://rapidmlx.com/docs/extras.html)

> **Not into the terminal?** [**Rapid-MLX Desktop**](https://rapidmlx.com/desktop) bundles the same engine inside a one-click Mac app.

---

## Why Rapid-MLX

| | |
|---|---|
| **Apple-Silicon-native** | Pure MLX kernels — no llama.cpp fallback, no Metal shim. Continuous batching, prompt cache (radix + DeltaNet RNN snapshots), and TurboQuant K8V4 KV codec run at native MLX bandwidth on M1 → M4. |
| **Drop-in OpenAI / Anthropic API** | `/v1/chat/completions`, `/v1/responses` (Codex CLI), `/v1/messages` (Anthropic SDK / Claude Code), `/v1/embeddings`, `/v1/audio/*` — same wire as ChatGPT / Claude, no client adapter. |
| **Tier-1 ecosystem coverage** | 8 agent CLIs and 3 Python frameworks are wire-verified against real weights every release — Codex CLI, Claude Code, OpenCode, Qwen Code, OpenHands, Hermes Agent, Aider, Kilo Code + LangChain, PydanticAI, smolagents. |

→ [Full feature breakdown](https://rapidmlx.com/docs/index.html)

---

## Use Cases

| | | |
|---|---|---|
| **Chat in the terminal** | `rapid-mlx chat qwen3.5-9b-4bit` | Streaming REPL, `/help` for slash commands, `--think` / `--no-think` to control CoT. |
| **OpenAI server for your apps** | `rapid-mlx serve qwen3.5-9b-4bit` | Point Cursor, Aider, LibreChat, Open WebUI, LangChain at `http://localhost:8000/v1`. |
| **Agent backends** | `rapid-mlx serve qwen3.6-35b-8bit &`<br>`rapid-mlx agents codex --setup && codex` | 8 Tier-1 agents auto-configure once the server is up — see [Tier-1 support](#tier-1-support). |
| **Benchmark your Mac** | `rapid-mlx bench qwen3.5-9b-4bit --submit` | Standardized B=1 bench, opens a PR to publish your row on [rapidmlx.com](https://rapidmlx.com). |

→ [One-shot IDE setup](https://rapidmlx.com/docs/cli.html#launch) with `rapid-mlx launch <cursor|claude-code|cline|continue-dev>`

---

## Tier-1 Support

Every row below has a `rapid-mlx agents <name> --setup` config template (except Claude Code, which is one env-var) *and* an integration test that drives the same wire the real client drives against a live server.

| Agents (8) | Frameworks (3) |
|---|---|
| [Codex CLI](https://github.com/openai/codex) · [Claude Code](https://www.anthropic.com/claude-code) · [OpenCode](https://github.com/sst/opencode) · [Qwen Code](https://github.com/QwenLM/qwen-code) · [OpenHands](https://github.com/All-Hands-AI/OpenHands) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Aider](https://aider.chat) · [Kilo Code](https://github.com/Kilo-Org/kilocode) | [LangChain](https://langchain.com) (+ [LangGraph](https://langchain-ai.github.io/langgraph/)) · [PydanticAI](https://ai.pydantic.dev) · [smolagents](https://github.com/huggingface/smolagents) |

Also compatible with any OpenAI-compatible client via `http://localhost:8000/v1` — Cursor, LibreChat, Open WebUI, and more plug in with a single URL change.

→ [Full 8×3 agent matrix + 3×3 framework matrix (test cells + xfail reasons)](https://rapidmlx.com/docs/matrix.html)
→ [Codex CLI](https://rapidmlx.com/docs/matrix.html#agent-codex-cli) · [Claude Code](https://rapidmlx.com/docs/matrix.html#agent-claude-code) · [OpenCode](https://rapidmlx.com/docs/matrix.html#agent-opencode) · [Qwen Code](https://rapidmlx.com/docs/matrix.html#agent-qwen-code) · [OpenHands](https://rapidmlx.com/docs/matrix.html#agent-openhands) · [Hermes](https://rapidmlx.com/docs/matrix.html#agent-hermes-agent) · [Aider](https://rapidmlx.com/docs/matrix.html#agent-aider) · [Kilo Code](https://rapidmlx.com/docs/matrix.html#agent-kilo-code)

---

## Choose Your Model

The installer's RAM detector picks a sensible default. If you want to shop the full catalog: `rapid-mlx models` lists every alias, `rapid-mlx info <alias>` shows the per-alias profile (parser, MoE / hybrid flags, KV codec eligibility, speculative-decoding gates).

| RAM | Recommended | One-shot |
|---|---|---|
| **8–23 GB** MacBook Air/Pro | `qwen3.5-4b-4bit` | `rapid-mlx serve qwen3.5-4b-4bit` |
| **24–47 GB** MacBook Pro / Mac Mini | `gpt-oss-20b-mxfp4-q8` | `rapid-mlx serve gpt-oss-20b-mxfp4-q8` |
| **48–95 GB** Mac Studio | `qwen3.6-35b-8bit` | `rapid-mlx serve qwen3.6-35b-8bit` |
| **96 GB+** Mac Studio / Pro | `gpt-oss-120b-mxfp4-q8` | `rapid-mlx serve gpt-oss-120b-mxfp4-q8` |

→ [Full RAM tier map + serve flags per tier](https://rapidmlx.com/docs/hardware-tiers.html)
→ [Every alias, quant, and family (128+ aliases across 30+ families)](https://rapidmlx.com/docs/aliases.html) · interactive at [models.rapidmlx.com](https://models.rapidmlx.com/)

---

## Alternative install methods

The curl one-liner above wraps all of these — reach for these only if you already manage Python yourself.

<details>
<summary><strong>Homebrew</strong> — Mac-native, tap + trust required on Homebrew 4.x</summary>

```bash
brew tap raullenchai/rapid-mlx
brew trust raullenchai/rapid-mlx
brew install rapid-mlx
```

Upgrade with `brew upgrade rapid-mlx`. If `brew install` stalls on `Tapping homebrew/core`, run `brew tap homebrew/core --force` once (one-time ~1.3 GB download) and retry.

</details>

<details>
<summary><strong>uv</strong> — isolated tool install, auto-manages Python</summary>

```bash
uv tool install rapid-mlx@latest
```

Don't have uv yet? `curl -LsSf https://astral.sh/uv/install.sh | sh`. Upgrade with `uv tool upgrade rapid-mlx`.

</details>

<details>
<summary><strong>pip</strong> — requires Python 3.10+ (macOS ships 3.9)</summary>

```bash
python3.12 -m pip install rapid-mlx
```

If `pip install rapid-mlx` says "no matching distribution", your Python is too old. `brew install python@3.12` first. Upgrade with `pip install -U rapid-mlx`.

For image-input / VLM models (Qwen-VL, true multimodal), install the vision extra: `pip install 'rapid-mlx[vision]'` — see [Optional extras](https://rapidmlx.com/docs/extras.html).

</details>

---

## Command Reference

```bash
rapid-mlx --help                    # top-level command list
rapid-mlx <subcommand> --help       # per-subcommand flags
```

Covers chat, serve, share, agents (setup / test), bench, models, pull, rm, ps, info, doctor, upgrade, telemetry, launch, and jlens.

→ [Full CLI reference with every flag](https://rapidmlx.com/docs/cli.html)

---

## Troubleshooting

Run the built-in self-check first:

```bash
rapid-mlx doctor
```

Top three things that go wrong:

- **Much slower than expected.** Qwen3.5 / 3.6 default to thinking-on — add `--no-think` to skip chain-of-thought. → [Slow tok/s](https://rapidmlx.com/docs/troubleshooting.html#issue-slow-tps)
- **Out of memory.** Model too big for your RAM — pick a smaller quant from [Choose Your Model](#choose-your-model) or the [full tier map](https://rapidmlx.com/docs/hardware-tiers.html). → [OOM guide](https://rapidmlx.com/docs/troubleshooting.html#issue-oom)
- **Tool calls arriving as plain text.** Auto-recovery handles most cases; if not, set `--tool-call-parser` explicitly for your model. → [Tool-call recovery](https://rapidmlx.com/docs/troubleshooting.html#issue-tool-call-text)

→ [All troubleshooting entries](https://rapidmlx.com/docs/troubleshooting.html) (OOM, empty responses, slow TTFT, port taken, shell completion, HF cache, and more)

---

## Community & Contributing

- **Report a bug or request a model:** [Issues](https://github.com/raullenchai/Rapid-MLX/issues/new/choose)
- **Ask a question or share a build:** [Discussions](https://github.com/raullenchai/Rapid-MLX/discussions)
- **Contribute code, aliases, or docs:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Add your hardware to the public benchmark:** `rapid-mlx bench <alias> --submit` opens the PR for you

Rapid-MLX ships **opt-in anonymous telemetry** (off by default; explicit `rapid-mlx telemetry enable` required). No prompts, completions, paths, IPs, or API keys are ever collected. → [What we do and don't collect](https://rapidmlx.com/docs/telemetry.html)

### Star History

<a href="https://star-history.com/#raullenchai/Rapid-MLX&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=raullenchai/Rapid-MLX&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=raullenchai/Rapid-MLX&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=raullenchai/Rapid-MLX&type=Date" />
  </picture>
</a>

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
