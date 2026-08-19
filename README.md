<img width="1600" height="800" alt="banner" src="https://github.com/user-attachments/assets/f3743bb7-7287-4b24-ac97-a7037974396f" />

<p align="center">
  <strong>The fastest local AI engine for Apple Silicon.</strong>
  <br>
  <em>Drop-in OpenAI / Anthropic API · up to 3× Ollama's throughput (<a href="https://rapidmlx.com/blog/rapid-mlx-vs-ollama-benchmark">measured</a>) · Runs on any M-series Mac.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/rapid-mlx/"><img src="https://img.shields.io/pypi/v/rapid-mlx?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://formulae.brew.sh/formula/rapid-mlx"><img src="https://img.shields.io/badge/Homebrew-core-orange?logo=homebrew" alt="Homebrew core"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://support.apple.com/en-us/HT211814"><img src="https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple" alt="Apple Silicon"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://github.com/raullenchai/Rapid-MLX/actions/workflows/ci.yml"><img src="https://github.com/raullenchai/Rapid-MLX/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/raullenchai/Rapid-MLX/stargazers"><img src="https://img.shields.io/github/stars/raullenchai/Rapid-MLX?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/raullenchai/Rapid-MLX/graphs/contributors"><img src="https://img.shields.io/github/contributors/raullenchai/Rapid-MLX?color=orange" alt="Contributors"></a>
  <a href="https://github.com/raullenchai/Rapid-MLX/commits/main"><img src="https://img.shields.io/github/last-commit/raullenchai/Rapid-MLX?color=orange" alt="Last commit"></a>
  <a href="https://deepwiki.com/raullenchai/Rapid-MLX"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

<p align="center">
  <sub>
    <a href="https://rapidmlx.com"><b>rapidmlx.com</b></a> ·
    <a href="https://rapidmlx.com/docs/">Docs</a> ·
    <a href="https://models.rapidmlx.com/">Model mirror</a> ·
    <a href="https://rapidmlx.com/desktop">Desktop app</a>
  </sub>
</p>

---

## Quick Start (60 seconds)

**1. Install** — pick one path (run only one of these):

One-liner — detects your RAM, picks a starter model (recommended):

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
```

or Homebrew — prebuilt bottle straight from homebrew-core:

```bash
brew install rapid-mlx
```

Both land the same `rapid-mlx` CLI. The curl installer additionally installs Python 3.10+ if missing, creates an isolated venv at `~/.rapid-mlx/`, symlinks the `rapid-mlx` CLI into `~/.local/bin/`, and prints a serve command sized to your Mac (8–15 GB → `lfm2.5-2.6b-4bit`; 16–17 GB → `qwen3.5-4b-4bit`; 18–23 GB → `qwen3.5-9b-4bit`; 24–31 GB → `bonsai-27b-2bit`; 32 GB+ → `qwen3.8-27b-4bit`).

> **Install security.** `install.sh` is served over HTTPS (HSTS-preload) from `rapidmlx.com` and is a byte-identical mirror of [`install.sh`](install.sh) at the release commit — read it before running if you like. If you want a cryptographically verified installer rather than trusting the website pipe, don't `curl | bash` the URL above: instead download the release's `install.sh` asset, verify it against the cosign-signed `SHA256SUMS.txt` shipped alongside it, and run that verified copy — full recipe in [SECURITY.md](SECURITY.md). PyPI artifacts additionally carry Sigstore attestations (PEP 740). Two more low-trust paths:
> - **Pin to a commit hash** — `curl -fsSL https://raw.githubusercontent.com/raullenchai/Rapid-MLX/<commit>/install.sh -o install.sh && shasum -a 256 install.sh && bash install.sh`
> - **Skip the shell script entirely** — use Homebrew, `uv`, or `pip` below.

See [Alternative install methods](#alternative-install-methods) for the non-curl paths.

**2. Chat with a model right now:**

```bash
rapid-mlx chat
```

Defaults to `qwen3.5-4b-4bit`. First run downloads the weights (~3 GB) with a progress bar and drops you into a REPL. Type `/help` for slash commands, `/exit` to quit.

**3. Or serve it for use from other apps:**

```bash
rapid-mlx serve qwen3.5-4b-4bit
```

Starts an OpenAI-compatible HTTP server bound to `http://localhost:8000`. Point any client that supports a local custom endpoint (Aider, LangChain, OpenCode, PydanticAI, your own scripts) at **`http://localhost:8000/v1`**; Claude Code / Anthropic SDK uses **`http://localhost:8000`** (the Anthropic messages route lives at `/v1/messages` under the same host).

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

**4. Or wire up your coding agent — one command:**

```bash
rapid-mlx launch claude-code
```

With a server running (step 3), this patches Claude Code's local config (`~/.claude/settings.json`) to route at `http://localhost:8000` — no manual env vars, no editing JSON by hand. You get a fully local Claude Code: `$0` per token, nothing leaves your Mac. Swap in `cline` or `continue-dev` for the other IDE clients, or run `rapid-mlx launch list` to see what's detected on this machine.

> **Cursor:** Cursor currently routes BYOK requests through its own servers, so its servers cannot reach a Rapid-MLX endpoint on `localhost`. Rapid-MLX therefore does not generate a Cursor localhost config. If you intentionally expose the server through a public HTTPS tunnel, set `RAPID_MLX_API_KEY=your-secret` for both `rapid-mlx serve ...` and `rapid-mlx launch cursor --server-url https://your-public-host`. This is no longer a fully local connection; never expose an unauthenticated server. Rapid-MLX rejects explicit local/private addresses but cannot verify reachability from Cursor's network, whose DNS view may differ from your Mac.

> **Vision / audio / video / diffusion models?** Base install is text-only (~460 MB). Vision, audio (TTS, STT, voice cloning), video generation, embeddings, and DFlash speculative decoding ship as opt-in extras. → [Optional extras](https://rapidmlx.com/docs/extras.html)

> **Not into the terminal?** [**Rapid-MLX Desktop**](https://rapidmlx.com/desktop) bundles the same engine inside a one-click Mac app.

---

## Video generation

Run text-to-video or image-to-video locally through the OpenAI-compatible
Videos API. Three backends ship — **Wan 2.1 / 2.2**, **CogVideoX-Fun** and
**LTX-2.3** — across 8 registered checkpoints. `wan2.2-ti2v-5b-q8` is the
recommended starting point: smallest of the Wan set, and TI2V means one
checkpoint does both text-to-video and image-to-video.

Requires Python 3.11+ (the video runtime does not support 3.10; core text and
audio still do) and `ffmpeg` for the final MP4 mux.

```bash
pip install 'rapid-mlx[video]'
brew install ffmpeg
rapid-mlx serve wan2.2-ti2v-5b-q8
```

Create and download a clip:

```bash
curl http://localhost:8000/v1/videos \
  -F model=wan2.2-ti2v-5b-q8 \
  -F 'prompt=A fox running through fresh snow, cinematic tracking shot' \
  -F seconds=1 \
  -F size=832x512

# Poll until GET /v1/videos/VIDEO_ID reports "status": "completed", then:
curl http://localhost:8000/v1/videos/VIDEO_ID/content -o output.mp4
```

The create call returns a job immediately. Poll `GET /v1/videos/VIDEO_ID`
until `status` is `completed`. Add `-F input_reference=@start.png` for
image-to-video.

Generation is serialized — one clip at a time — because two diffusion
pipelines resident at once will exhaust unified memory. Expect minutes of
compute per second of footage, not real time.

→ [Every checkpoint, RAM requirement and tuning knob](https://rapidmlx.com/docs/models/families/video.html)

---

## Audio: speech, transcription, voice cloning

44 audio aliases behind the OpenAI-compatible `/v1/audio/*` endpoints — any
OpenAI SDK works unchanged.

```bash
pip install 'rapid-mlx[audio]'

# Text to speech
rapid-mlx serve kokoro
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"hello from rapid-mlx"}' --output hello.wav

# Transcription (Whisper / Parakeet / SenseVoice)
rapid-mlx serve whisper-large-v3-turbo
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@hello.wav -F model=whisper-large-v3-turbo
```

Beyond the basics, three things you may not expect to run locally:

- **Zero-shot voice cloning** from a reference clip. `indextts` is the only
  one that takes the clip alone; `qwen3-tts-clone`, `f5-tts-zh` and
  `chatterbox` all require `ref_text` (the clip's exact transcript) paired
  with `ref_audio`, and the request is rejected before generation if it is
  missing.
- **Voice design** — `qwen3-tts-voicedesign` has no named speakers at all.
  Describe the voice you want in natural language via `instructions`
  (timbre, gender, age, accent, emotion, prosody) and it synthesises it.
- **Forced alignment** — `qwen3-aligner` takes audio *plus the transcript you
  already have* and returns per-character timings. It never guesses at the
  words, so it cannot mis-hear them; that is what karaoke captions and
  beat-synced editing need.

Also: word-level timestamps on transcription, and local text-to-music at
`/v1/audio/music`.

→ [All 44 aliases across 13 families](https://rapidmlx.com/docs/models/families/audio.html)

---

## Why Rapid-MLX

| | |
|---|---|
| **Apple-Silicon-native** | Pure MLX kernels — no llama.cpp fallback, no Metal shim. Continuous batching, prompt cache (radix + DeltaNet RNN snapshots), and a quantized live KV cache (int4/int8 on the continuous-batching cache + TurboQuant K8V4 codec) run at native MLX bandwidth on M1 → M4. |
| **Drop-in OpenAI / Anthropic API** | `/v1/chat/completions`, `/v1/responses` (Codex CLI), `/v1/messages` (Anthropic SDK / Claude Code), `/v1/embeddings`, `/v1/audio/*`, `/v1/videos` — same wire as ChatGPT / Claude, no client adapter. |
| **First-class ecosystem coverage** | 12 agent CLIs and 3 Python frameworks are wire-verified against real weights every release (5 are Tier-1, re-verified on current binaries) — Codex CLI, Claude Code, OpenCode, Qwen Code, OpenHands, Hermes Agent, Aider, Kilo Code, DeepSeek Harness, GitHub Copilot, Factory Droid, Moonshot Kimi Code + LangChain, PydanticAI, smolagents. |

→ [Full feature breakdown](https://rapidmlx.com/docs/index.html)

---

## Use Cases

| | | |
|---|---|---|
| **Chat in the terminal** | `rapid-mlx chat qwen3.5-9b-4bit` | Streaming REPL, `/help` for slash commands, `--think` / `--no-think` to control CoT. |
| **OpenAI server for your apps** | `rapid-mlx serve qwen3.5-9b-4bit` | Point Aider, LibreChat, Open WebUI, or LangChain at `http://localhost:8000/v1`. |
| **Agent backends** | `rapid-mlx serve qwen3.6-35b-8bit &`<br>`rapid-mlx agents codex --setup && codex` | 10 agents auto-configure via `agents <name> --setup` once the server is up (12 wire-verified total, 5 Tier-1) — see [Agent support](#agent-support). |
| **Benchmark your Mac** | `rapid-mlx bench qwen3.5-9b-4bit --submit` | Standardized B=1 bench, opens a PR to publish your row on [rapidmlx.com](https://rapidmlx.com). |

→ [One-shot IDE setup](https://rapidmlx.com/docs/cli.html#launch) with `rapid-mlx launch <claude-code|cline|continue-dev>`

---

## Agent Support

All 12 agents below are wire-verified against real weights every release via their own integration-test cell. Of these, five are **Tier-1** — **Claude Code, Codex CLI, Hermes, Aider, and DeepSeek Harness** — re-verified end-to-end against the *current* client binary every release, with one guardian per API wire (Anthropic `/v1/messages`, OpenAI `/v1/responses`, and `/v1/chat/completions` covered for tool-calling depth, reach, and DeepSeek's own harness protocol). The other seven are **Tier-2**: wire-verified in the matrix and configured on-demand. The first nine agents each ship a `rapid-mlx agents <name> --setup` config template (except Claude Code, which is one env-var), and Continue.dev gets the same one-command setup via `rapid-mlx agents continue --setup` (ten setup-capable clients in all, though Continue.dev is not part of the wire-verified matrix below); GitHub Copilot, Factory Droid, and Moonshot Kimi Code plug in through their own documented BYOK config (auth-gated, so the matrix cell is a wire smoke).

Tier-1 is not a label — it is a job that blocks the release. `tests/integrations/agent_smoke.sh` drives each of the five through the same real multi-step bug-fix task against a local 35B model and asserts the repo's own test suite goes green afterwards; if any one of them fails, the version cannot tag or publish.

**Tier-1 (5):** Claude Code · Codex CLI · Hermes · Aider — last re-verified end-to-end 2026-07-28 on current binaries (claude 2.1.211, codex 0.145.0, hermes 0.9.0, aider 0.86.2). DeepSeek Harness — promoted 2026-08-17, verified on dsh 0.1.0-rc.7 against `qwen3.6-35b-8bit`.
**Tier-2 (7):** OpenCode · Qwen Code · OpenHands · Kilo Code · GitHub Copilot · Factory Droid · Moonshot Kimi Code.

| Agents (12) | Frameworks (3) |
|---|---|
| [Codex CLI](https://github.com/openai/codex) · [Claude Code](https://www.anthropic.com/claude-code) · [OpenCode](https://github.com/sst/opencode) · [Qwen Code](https://github.com/QwenLM/qwen-code) · [OpenHands](https://github.com/All-Hands-AI/OpenHands) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [Aider](https://aider.chat) · [Kilo Code](https://github.com/Kilo-Org/kilocode) · [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) · [GitHub Copilot](https://github.com/features/copilot) · [Factory Droid](https://factory.ai) · [Moonshot Kimi Code](https://github.com/MoonshotAI/kimi-cli) | [LangChain](https://langchain.com) (+ [LangGraph](https://langchain-ai.github.io/langgraph/)) · [PydanticAI](https://ai.pydantic.dev) · [smolagents](https://github.com/huggingface/smolagents) |

Also compatible with OpenAI-compatible clients that allow direct local endpoints via `http://localhost:8000/v1` — LibreChat, Open WebUI, and more plug in with a single URL change.

→ [Full 12-agent + 3-framework matrix (test cells + xfail reasons)](https://rapidmlx.com/docs/matrix.html)
→ [Codex CLI](https://rapidmlx.com/docs/matrix.html#agent-codex-cli) · [Claude Code](https://rapidmlx.com/docs/matrix.html#agent-claude-code) · [OpenCode](https://rapidmlx.com/docs/matrix.html#agent-opencode) · [Qwen Code](https://rapidmlx.com/docs/matrix.html#agent-qwen-code) · [OpenHands](https://rapidmlx.com/docs/matrix.html#agent-openhands) · [Hermes](https://rapidmlx.com/docs/matrix.html#agent-hermes-agent) · [Aider](https://rapidmlx.com/docs/matrix.html#agent-aider) · [Kilo Code](https://rapidmlx.com/docs/matrix.html#agent-kilo-code) · [DeepSeek Harness](https://rapidmlx.com/docs/matrix.html#agent-deepseek-harness) · [Copilot](https://rapidmlx.com/docs/matrix.html#agent-copilot) · [Droid](https://rapidmlx.com/docs/matrix.html#agent-droid) · [Kimi Code](https://rapidmlx.com/docs/matrix.html#agent-kimi-code)

---

## Choose Your Model

The installer and desktop app use the same RAM-tier recommendation catalog. Run `rapid-mlx recipe` to see its Smart and Fast picks for this Mac (`--max-ram 32` simulates another tier; `--json` is machine-readable). If you want to shop the full catalog: `rapid-mlx models` lists every alias, `rapid-mlx info <alias>` shows the per-alias profile (parser, MoE / hybrid flags, KV codec eligibility, speculative-decoding gates).

This table is the same one the desktop app's picker reads, and the installer
prints the matching line for your Mac — a CI test parses both files and fails
if they drift apart. Measured rows use the standard ~8K prompt peak of the
complete `rapid-mlx serve` process tree on an M2 Pro 32 GB Mac mini (the 32 GB+ row: M3 Ultra, 2026-08-18 — footprint is config-bound, speed reads lower on smaller chips).

| RAM | Recommended | Peak RSS | One-shot |
|---|---|---:|---|
| **8–15 GB** MacBook Air / base Mini | `lfm2.5-2.6b-4bit` | 3.0 GB | `rapid-mlx serve lfm2.5-2.6b-4bit` |
| **16–17 GB** MacBook Air / Pro | `qwen3.5-4b-4bit` | 6.0 GB | `rapid-mlx serve qwen3.5-4b-4bit` |
| **18–23 GB** MacBook Pro | `qwen3.5-9b-4bit` | 8.7 GB | `rapid-mlx serve qwen3.5-9b-4bit` |
| **24–31 GB** Mac Mini / MacBook Pro | `bonsai-27b-2bit` | 13.0 GB | `rapid-mlx serve bonsai-27b-2bit` |
| **32 GB+** Mac Studio / MacBook Pro | `qwen3.8-27b-4bit` | 20.0 GB | `rapid-mlx serve qwen3.8-27b-4bit` |

Every Mac from 32 GB up gets the same pick, and that is the point: Qwen3.8-27B
scores 52 on the Artificial Analysis Intelligence Index (2026-08-18) —
GPT-5.6-class, the highest of any open-weights model we serve, ahead of the
much larger 122B (33) and 35B (32) it replaces (the index scores the
full-precision release; our 4-bit build's deltas are unmeasured — the
standing caveat for every quantized pick here). Multi-token prediction is on by
default (~40 tok/s decode, 8K prefill at ~324 tok/s, zero swap at every tier
budget).

→ [Full RAM tier map + serve flags per tier](https://rapidmlx.com/docs/hardware-tiers.html)
→ [Every alias, quant, and family (170 text + 2 text-diffusion + 2 image + 8 video + 44 audio aliases, 226 total)](https://rapidmlx.com/docs/aliases.html) · interactive at [models.rapidmlx.com](https://models.rapidmlx.com/)

---

## Alternative install methods

The two paths above cover most users — reach for these only if you already manage Python yourself.

<details>
<summary><strong>Homebrew</strong> — Mac-native, one command, prebuilt bottle from <code>homebrew/core</code></summary>

```bash
brew install rapid-mlx
```

Ships in homebrew-core since 0.10.12 — no tap, no trust prompt. Upgrade with `brew upgrade rapid-mlx`. If you previously installed from the legacy `raullenchai/rapid-mlx` tap, switch once: `brew uninstall rapid-mlx && brew untap raullenchai/rapid-mlx && brew install rapid-mlx`.

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

For the complete feature set — vision, chat, embeddings, and audio — install the `[all]` extra: `pip install 'rapid-mlx[all]'`. Audio alone is `pip install 'rapid-mlx[audio]'`; see [Optional extras](https://rapidmlx.com/docs/extras.html).

</details>

---

## Command Reference

```bash
rapid-mlx --help                    # top-level command list
rapid-mlx <subcommand> --help       # per-subcommand flags
```

Covers chat, serve, share, agents (setup / test), bench, recipe, models, ls, pull, rm, alias, ps, info, connect, doctor, upgrade, telemetry, and launch.

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

## See it in action

<p align="center">
  <img src="https://raw.githubusercontent.com/raullenchai/Rapid-MLX/main/docs/assets/demo.gif" alt="Rapid-MLX demo — install, serve Gemma 4, chat, tool calling" width="700">
</p>

## Community & Contributing

- **Report a bug or request a model:** [Issues](https://github.com/raullenchai/Rapid-MLX/issues/new/choose)
- **Report a security issue:** [Private advisory](https://github.com/raullenchai/Rapid-MLX/security/advisories/new) — see [SECURITY.md](SECURITY.md)
- **Ask a question or share a build:** [Discussions](https://github.com/raullenchai/Rapid-MLX/discussions)
- **Contribute code, aliases, or docs:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Add your hardware to the public benchmark:** `rapid-mlx bench <alias> --submit` opens the PR for you

Rapid-MLX ships **opt-in anonymous telemetry** (off by default; explicit `rapid-mlx telemetry enable` required). No prompts, completions, paths, IPs, or API keys are ever collected. → [What we do and don't collect](https://rapidmlx.com/docs/telemetry.html)

### 🚀 Contributors

Every avatar here shipped something in rapid-mlx — model support, tool-call parsers, fixes, docs, and benchmark submissions. Thank you.

<a href="https://github.com/raullenchai/Rapid-MLX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=raullenchai/Rapid-MLX" alt="rapid-mlx contributors" />
</a>

### Star History

<a href="https://star-history.com/#raullenchai/Rapid-MLX&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=raullenchai/Rapid-MLX&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=raullenchai/Rapid-MLX&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=raullenchai/Rapid-MLX&type=Date" />
  </picture>
</a>

---

## Acknowledgements

Rapid-MLX began as **[vLLM-MLX](https://github.com/waybarrios/vllm-mlx)** by
[Wayner Barrios](https://github.com/waybarrios), which is where this repository's
history starts and where the engine's paged KV cache, prefix cache, and
continuous batching were first built. It was renamed to Rapid-MLX in March 2026
and has been heavily modified since. Thank you.

It stands on Apple's MLX stack and the runtimes built around it:

- **[MLX](https://github.com/ml-explore/mlx)** — Apple's array framework for Apple Silicon
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)** — LLM inference, KV cache, quantization
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)** — vision-language models
- **[mlx-audio](https://github.com/Blaizzy/mlx-audio)** — speech and audio models

Vendored third-party components and their licenses are listed in
[NOTICE](NOTICE); what the macOS app ships is enumerated in
[apps/rapid-mac/THIRD_PARTY.md](apps/rapid-mac/THIRD_PARTY.md).

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
