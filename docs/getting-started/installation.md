# Installation

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## Install with uv (recommended)

```bash
uv tool install rapid-mlx@latest
```

One command, isolated tool venv, no Python-version juggling — uv finds (or
installs) the right Python automatically. Upgrade later with
`uv tool upgrade rapid-mlx`. If you don't have uv yet, install it first:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

## One-liner install script

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
```

Auto-installs Python (via Homebrew) if needed, then creates a self-contained
virtual environment at `~/.rapid-mlx` and symlinks the CLI entry points
(`rapid-mlx`, `rapid-mlx-chat`, `rapid-mlx-bench`) into `~/.local/bin`. Good
fallback if you don't want to install `uv` first.

## Install with Homebrew

```bash
brew install rapid-mlx
```

`rapid-mlx` is in **homebrew/core** — no tap, no trust, just one command.
Upgrade later with `brew upgrade rapid-mlx`.

## Install with pip

```bash
pip install rapid-mlx
```

If `python3 --version` reports 3.9 (macOS default), install a newer Python
first: `brew install python@3.12` then `python3.12 -m pip install rapid-mlx`.

### From source (for development)

```bash
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX
pip install -e .
```

## Optional Extras

The base install includes the vision (mlx-vlm), image-generation (mflux) and
video-generation (mlx-video) runtimes, so multimodal, image and video models
start without an extra step. Image and video generation need Python 3.11+. On
our M2 Pro (Python 3.11, arm64, fresh venv) the base install is about 360 MB to
download and 1.7 GB on disk. Audio, embeddings and the other features below
ship as opt-in extras. `vision`, `image`, `video`, `dflash` and `mtp` remain
accepted extra names for existing scripts; on macOS they add nothing.

| Extra | Install | Adds |
|---|---|---|
| `audio` | `pip install 'rapid-mlx[audio]'` | mlx-audio + spacy + scipy (~600 MB) for TTS / STT |
| `embeddings` | `pip install 'rapid-mlx[embeddings]'` | mlx-embeddings (~50 MB) for `/v1/embeddings` |
| `chat` | `pip install 'rapid-mlx[chat]'` | Gradio web UI (~150 MB) |
| `guided` | `pip install 'rapid-mlx[guided]'` | Legacy no-op kept for compatibility — llguidance ships in the core install (it replaced outlines in 0.10) |
| `all` | `pip install 'rapid-mlx[all]'` | audio + embeddings + chat + System One + Computer Use on top of the base install |

When `rapid-mlx serve` finds that an optional runtime is absent, it prints a
version-pinned repair command matched to the detected install method. For a
runtime that ships with the base install (vision, image, video), a missing
module means the environment is damaged, and the command reinstalls the pinned
`rapid-mlx` package itself. pip and
install.sh environments can offer to install into the current interpreter and
restart the original command after success; the prompt defaults to no after 30
seconds, and `--yes` (or `-y`) accepts non-interactively. uv tool, pipx, and
Homebrew repairs are print-only so a running manager-owned environment is never
replaced underneath the process. Broken or incompatible runtimes also remain
print-only; a broken pip-based runtime is shown a forced reinstall command,
while an absent runtime uses an ordinary pinned install. Prompt telemetry keeps
an explicit no distinct from timeout/EOF/read failure and Ctrl-C, and Ctrl-C
retains normal interrupt exit behavior after the failure is recorded.

Homebrew builds its own formula and does not provide Python extras or, unless
the formula carries them, the vision/image/video runtimes.
The formula is built as a Homebrew-managed virtualenv, but optional PyPI
dependencies are not formula resources and an in-place pip mutation is not a
supported, upgrade-stable repair. Switch to an isolated tool install instead:

```bash
brew uninstall rapid-mlx && uv tool install 'rapid-mlx==<rapid-mlx-version>'
```

## Verify Installation

```bash
# Check CLI
rapid-mlx --help
rapid-mlx version

# Self-diagnostic (works without downloading a model)
rapid-mlx doctor

# Smallest interactive smoke test (downloads ~3 GB on first run)
rapid-mlx chat qwen3.5-4b-4bit
```

## Troubleshooting

### MLX not found

Ensure you're on Apple Silicon:
```bash
uname -m  # Should output "arm64"
```

### Model download fails

Check your internet connection and HuggingFace access. Some models require authentication:
```bash
huggingface-cli login
```

### Out of memory

Use a smaller quantized model:
```bash
rapid-mlx serve qwen3.5-4b-4bit
```

### `brew install` fails with `Operation not permitted`

Brew's install sandbox sometimes can't auto-tap `homebrew/core` mid-install.
Pre-tap it once, then retry:

```bash
brew tap homebrew/core --force   # ~1.3 GB, one-time
brew install rapid-mlx
```
