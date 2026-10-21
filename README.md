<h1 align="center">Rapid-MLX</h1>

<p align="center"><strong>Run Codex, Claude Code, OpenCode, Pi, Hermes, Crush or Goose with a local model on Mac:</strong><br><code>rapid-mlx launch &lt;agent&gt;</code></p>

Install on a stock Apple Silicon Mac with `curl -fsSL https://rapidmlx.com/install.sh | bash`, restart your terminal (or run `export PATH="$HOME/.local/bin:$PATH"`), then install your chosen coding agent and run, for example, `rapid-mlx launch codex`. Run the same command tomorrow; `rapid-mlx launch codex --remove` removes Rapid's private profile without changing your existing Codex settings. [One-command guides for each agent](https://rapidmlx.com/docs/agents/). A first reply can take longer while a model downloads, loads, or processes the agent's opening prompt.


<p align="center">
  <strong>Rapid-MLX is an open-source (Apache 2.0) OpenAI- and Anthropic-compatible LLM inference server for Apple Silicon, built on MLX, focused on reliable tool calling for coding agents.</strong>
  <br>
  <strong>Use local models through an OpenAI- or Anthropic-compatible API.</strong>
</p>



<p align="center">
  <a href="https://pypi.org/project/rapid-mlx/"><img src="https://img.shields.io/pypi/v/rapid-mlx?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://formulae.brew.sh/formula/rapid-mlx"><img src="https://img.shields.io/badge/Homebrew-core-orange?logo=homebrew" alt="Homebrew core"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://support.apple.com/en-us/HT211814"><img src="https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple" alt="Apple Silicon"></a>
  <a href="https://github.com/raullenchai/Rapid-MLX/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://github.com/raullenchai/Rapid-MLX/actions/workflows/ci.yml"><img src="https://github.com/raullenchai/Rapid-MLX/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/raullenchai/Rapid-MLX/stargazers"><img src="https://img.shields.io/github/stars/raullenchai/Rapid-MLX?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/raullenchai/Rapid-MLX/graphs/contributors"><img src="https://img.shields.io/github/contributors/raullenchai/Rapid-MLX?color=orange" alt="Contributors"></a>
  <a href="https://github.com/raullenchai/Rapid-MLX/commits/main"><img src="https://img.shields.io/github/last-commit/raullenchai/Rapid-MLX?color=orange" alt="Last commit"></a>
  <a href="https://discord.gg/nZcXkUjY5R"><img src="https://img.shields.io/discord/1540051732279599116?color=5865F2&label=Discord&logo=discord&logoColor=white" alt="Join the Rapid-MLX Discord"></a>
  <a href="https://deepwiki.com/raullenchai/Rapid-MLX"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

<p align="center">
  <sub>
    <a href="https://rapidmlx.com"><b>rapidmlx.com</b></a> ·
    <a href="https://rapidmlx.com/docs/">Docs</a> ·
    <a href="https://models.rapidmlx.com/">Model mirror</a> ·
    <a href="https://rapidmlx.com/desktop">Desktop app</a> ·
    <a href="https://discord.gg/nZcXkUjY5R">Discord</a>
  </sub>
</p>

---



## Works with your AI stack

Run `rapid-mlx launch <agent>` for Codex, Claude Code, OpenCode, Pi, Hermes, Crush, or Goose. For apps and SDKs that accept a local endpoint, use `rapid-mlx serve <model>` and the printed address. See the [agent guides](https://rapidmlx.com/docs/agents/) and [API guide](https://rapidmlx.com/docs/server.html).

Claude Desktop connects through its official in-app Gateway settings. Follow the [Claude Desktop guide](https://rapidmlx.com/docs/claude-desktop/).

## Install

### Desktop — macOS (Apple Silicon)

The easiest way to chat locally, manage models, and use vision, files, voice,
and image generation from one app.

- [Download Rapid-MLX Desktop](https://rapidmlx.com/desktop)
- [Browse signed Desktop releases](https://github.com/raullenchai/Rapid-MLX/releases?q=rapid-mac-v)
- Requires an M-series Mac; Windows and Linux desktop builds are not available yet
- [Share Compute setup and credit ledger](https://github.com/raullenchai/Rapid-MLX/blob/main/docs/guides/share-compute.md)

### CLI and server — macOS (Apple Silicon)

```bash
# Homebrew — prebuilt bottle from homebrew-core
brew install rapid-mlx

# Or the guided installer — detects RAM and recommends a starter model
curl -fsSL https://rapidmlx.com/install.sh | bash
```

Both install the same `rapid-mlx` CLI. Prefer `uv` or `pip`, or want to verify
the installer before running it? See [alternative install methods](#alternative-install-methods)
and [install security](https://github.com/raullenchai/Rapid-MLX/blob/main/SECURITY.md).

The guided installer prefers a runnable model already cached on this Mac when
it fits the RAM tier. Otherwise its quick first-chat download is
`lfm2.5-1b-4bit` below 16 GB and `qwen3.5-4b-4bit` at 16 GB or above; larger
quality picks remain available through `rapid-mlx recipe` and the model picker.

---

## Submit a Community Benchmark

Help other Mac users choose a model by sharing a standardized result from your
own Apple Silicon Mac. Runs stay local until you explicitly choose to publish;
you do not need a GitHub account or a benchmark JSON pull request.

- **Desktop:** Enable **Community Benchmark** in **Settings → Experimental**,
  then choose a model in the Community Benchmark tab, run it, review the result
  in **My Results**, and select **Publish** if you want to share it.
- **CLI:** Run `rapid-mlx benchmark catalog`, then
  `rapid-mlx benchmark run <model>`. Review the saved result and publish it
  with `rapid-mlx benchmark share <run-id>` only if you choose to.

Explore the [Community Benchmark leaderboard](https://rapidmlx.com/leaderboard)
or read the [protocol and privacy details](https://github.com/raullenchai/Rapid-MLX/blob/main/community-benchmarks/README.md).

---

## Quick Start

**1. Chat with a model right now:**

```bash
rapid-mlx chat
```

Defaults to `qwen3.5-4b-4bit`. First run downloads the weights and opens an interactive chat. Type `/help` for slash commands, `/exit` to quit.

**2. Or serve it for use from other apps:**

```bash
rapid-mlx serve qwen3.5-4b-4bit
```

Starts an OpenAI-compatible HTTP server on the first free port in `8000`–`8009` when `--port` is omitted. An explicit `--port` never falls back. The examples below assume port `8000` was selected. Point any client that supports a local custom endpoint (Aider, LangChain, OpenCode, PydanticAI, your own scripts) at **`http://localhost:8000/v1`**; Claude Code / Anthropic SDK uses **`http://localhost:8000`** (the Anthropic messages route lives at `/v1/messages` under the same host).

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

**3. Or wire up your coding agent — one command:**

```bash
rapid-mlx launch claude-code
```

This saves a private Claude Code profile, starts a local server if needed, and launches Claude Code without editing `~/.claude/settings.json`. The same command works after a restart. Swap in `codex`, `opencode`, `pi`, `hermes`, `crush`, or `goose`. See `rapid-mlx launch list` for other integrations.

[Agent setup details](docs/guides/ai-clients.md#coding-agents).

> **Cursor:** Cursor currently routes BYOK requests through its own servers, so its servers cannot reach a Rapid-MLX endpoint on `localhost`. Rapid-MLX therefore does not generate a Cursor localhost config. If you intentionally expose the server through a public HTTPS tunnel, set `RAPID_MLX_API_KEY=your-secret` for both `rapid-mlx serve ...` and `rapid-mlx launch cursor --server-url https://your-public-host`. This sends requests beyond your Mac; never expose an unauthenticated server. Rapid-MLX rejects explicit local/private addresses but cannot verify reachability from Cursor's network, whose DNS view may differ from your Mac.

> **Vision / image / video models?** They work out of the box with the official installer above (and Rapid-MLX Desktop), which include the vision, image-generation and video-generation runtimes (image and video generation need Python 3.11+). Audio (TTS, STT, voice cloning) and embeddings ship as opt-in extras. → [Optional extras](https://rapidmlx.com/docs/extras.html)

> **Not into the terminal?** [**Rapid-MLX Desktop**](https://rapidmlx.com/desktop) bundles the same engine inside a one-click Mac app.



## More local workloads

Rapid-MLX also supports [image and vision workflows](docs/guides/multimodal.md), [video generation](docs/guides/video-generation.md), [speech and transcription](docs/guides/audio.md), and [typed decisions](docs/guides/system-one.md). Start with the linked guide for each workload's models, memory needs, and commands.

---

## Use Rapid-MLX

- **Coding agents:** Run `rapid-mlx launch codex`. Replace `codex` with `claude-code`, `opencode`, `pi`, `hermes`, `crush`, or `goose`. The command starts a local model and opens the agent with a private profile. See the [agent guides](https://rapidmlx.com/docs/agents/).
- **Terminal chat:** Run `rapid-mlx chat` for an interactive conversation.
- **Local API:** Run `rapid-mlx serve qwen3.5-4b-4bit`, then connect a client to `http://localhost:8000/v1`. The server prints its address if it selects another free port.
- **Claude Desktop:** Follow the [in-app Gateway guide](https://rapidmlx.com/docs/claude-desktop/) after starting a local server.
- **Community benchmarks:** Run `rapid-mlx benchmark catalog`, then `rapid-mlx benchmark run <model>`. Results stay local until you choose to share them.

See the [CLI reference](https://rapidmlx.com/docs/cli.html) and [model catalog](https://rapidmlx.com/docs/aliases.html).

## Choose Your Model

Run `rapid-mlx recipe` for this Mac's Smart and Fast picks. Run `rapid-mlx models` to browse available aliases or `rapid-mlx info <alias>` for one model's profile. The [hardware guide](https://rapidmlx.com/docs/hardware-tiers.html) explains memory requirements.

Current Smart picks by unified memory:

- **8 GB and up:** `lfm2.5-2.6b-4bit` — `rapid-mlx serve lfm2.5-2.6b-4bit`
- **16 GB and up:** `qwen3.5-4b-4bit` — `rapid-mlx serve qwen3.5-4b-4bit`
- **18 GB and up:** `qwen3.5-9b-4bit` — `rapid-mlx serve qwen3.5-9b-4bit`
- **24 GB and up:** `bonsai-27b-2bit` — `rapid-mlx serve bonsai-27b-2bit`
- **32 GB and up:** `qwen3.8-27b-4bit` — `rapid-mlx serve qwen3.8-27b-4bit`

## Alternative install methods

The two paths above cover most users — reach for these only if you already manage Python yourself.

<details>
<summary><strong>Homebrew</strong> — Mac-native, one command, prebuilt bottle from <code>homebrew/core</code></summary>

```bash
brew install rapid-mlx
```

The package is available in homebrew-core. Upgrade with `brew upgrade rapid-mlx`.

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

If `python3.12` is missing, use the official installer above or install Python 3.12 with Homebrew first. Upgrade a Python 3.12 installation with `python3.12 -m pip install -U rapid-mlx`.

Image-input / VLM models (Qwen-VL, Gemma 4, true multimodal) need no extra step — the vision runtime ships with the base install. See [Optional extras](https://rapidmlx.com/docs/extras.html) for audio and embeddings.

For the complete feature set — chat, embeddings, audio, and System One on top of the base install — install the `[all]` extra: `python3.12 -m pip install 'rapid-mlx[all]'`. Audio alone is `python3.12 -m pip install 'rapid-mlx[audio]'`; see [Optional extras](https://rapidmlx.com/docs/extras.html).

</details>

---

## Command Reference

```bash
rapid-mlx --help                    # top-level command list
rapid-mlx <subcommand> --help       # per-subcommand flags
```

Covers chat, serve, system-one, share, agents (setup / test), bench, recipe, models, ls, pull, rm, alias, ps, info, connect, doctor, upgrade, telemetry, and launch.

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

## Community & Support

- **Discord:** [Join the Rapid-MLX community](https://discord.gg/nZcXkUjY5R) for live help and local-AI discussion.
- **Tell us what you want:** run `rapid-mlx feedback` (or `--no-open` to just print the link) — it opens that same Discord invite. Telemetry can only show us what people do; the Discord is where you tell us why.
- **Twitter / X:** Follow [@rapidmlx](https://x.com/rapidmlx) for releases, benchmarks, and project updates.
- **Questions & builds:** Ask or share in [GitHub Discussions](https://github.com/raullenchai/Rapid-MLX/discussions).
- **Feedback & ideas:** [Report a bug, request a model, or propose a feature](https://github.com/raullenchai/Rapid-MLX/issues/new/choose).
- **Security:** Send sensitive reports through a [private advisory](https://github.com/raullenchai/Rapid-MLX/security/advisories/new); see [SECURITY.md](https://github.com/raullenchai/Rapid-MLX/blob/main/SECURITY.md).
- **Contributing:** Start with [CONTRIBUTING.md](https://github.com/raullenchai/Rapid-MLX/blob/main/CONTRIBUTING.md). Explore compatible benchmark models with `rapid-mlx benchmark catalog`; runs stay local unless you explicitly share one in the internal beta.
- **Show support:** [Star this repository](https://github.com/raullenchai/Rapid-MLX) to follow releases and help others discover the project.

**Privacy:** Anonymous usage reporting can be turned off in Settings →
Privacy or with `rapid-mlx telemetry off`. See the
[privacy guide](https://rapidmlx.com/docs/telemetry.html) for current details.

---

## Contributors

Every avatar here shipped something in Rapid-MLX — model support, tool-call parsers, fixes, docs, and benchmark submissions. Thank you.

<a href="https://github.com/raullenchai/Rapid-MLX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=raullenchai/Rapid-MLX" alt="Rapid-MLX contributors" />
</a>

---

## Acknowledgements

Rapid-MLX began as **[vLLM-MLX](https://github.com/waybarrios/vllm-mlx)** by
[Wayner Barrios](https://github.com/waybarrios), which is where this repository's
history starts and where the engine's paged KV cache, prefix cache, and
continuous batching were first built. It was renamed to Rapid-MLX on 2026-03-13,
now ships as the `rapid-mlx` package (import name `rapid_mlx`), and has been
extensively modified and in large part rewritten since (see [NOTICE](https://github.com/raullenchai/Rapid-MLX/blob/main/NOTICE)).
Thank you.

It stands on Apple's MLX stack and the runtimes built around it:

- **[MLX](https://github.com/ml-explore/mlx)** — Apple's array framework for Apple Silicon
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)** — LLM inference, KV cache, quantization
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)** — vision-language models
- **[mlx-audio](https://github.com/Blaizzy/mlx-audio)** — speech and audio models

Vendored third-party components and their licenses are listed in
[NOTICE](https://github.com/raullenchai/Rapid-MLX/blob/main/NOTICE); what the macOS app ships is enumerated in
[apps/rapid-mac/THIRD_PARTY.md](https://github.com/raullenchai/Rapid-MLX/blob/main/apps/rapid-mac/THIRD_PARTY.md).

## License

Apache 2.0 — see [LICENSE](https://github.com/raullenchai/Rapid-MLX/blob/main/LICENSE) and [NOTICE](https://github.com/raullenchai/Rapid-MLX/blob/main/NOTICE).
