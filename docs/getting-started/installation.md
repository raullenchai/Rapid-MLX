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

The base text-only install is ~460 MB. Vision/audio/etc. ship as opt-in extras.

| Extra | Install | Adds |
|---|---|---|
| `vision` | `pip install 'rapid-mlx[vision]'` | mlx-vlm + opencv + torch (~322 MB) for VLMs (Gemma 4, Qwen-VL, video) |
| `dflash` | `pip install 'rapid-mlx[dflash]'` | mlx-vlm for DFlash speculative decoding on supported 8-bit aliases |
| `audio` | `pip install 'rapid-mlx[audio]'` | mlx-audio + spacy + scipy (~600 MB) for TTS / STT |
| `embeddings` | `pip install 'rapid-mlx[embeddings]'` | mlx-embeddings (~50 MB) for `/v1/embeddings` |
| `chat` | `pip install 'rapid-mlx[chat]'` | Gradio web UI (~150 MB) |
| `video` | `pip install 'rapid-mlx[video]'` | mlx-video + imageio for video generation (LTX-2.3 T2V/I2V with audio); requires Python 3.11+ |
| `image` | `pip install 'rapid-mlx[image]'` | mflux for text-to-image / image edit (FLUX.1-schnell, Qwen-Image); requires Python 3.11+ |
| `mtp` | `pip install 'rapid-mlx[mtp]'` | pillow for MTP speculative decoding (Gemma 4 assistant drafters) |
| `guided` | `pip install 'rapid-mlx[guided]'` | Legacy no-op kept for compatibility — llguidance ships in the core install (it replaced outlines in 0.10) |
| `all` | `pip install 'rapid-mlx[all]'` | vision + dflash + audio + embeddings + chat (~1.1 GB); `video` / `image` / `mtp` are installed separately |

Homebrew installs the text-only package and does not provide Python extras.
To switch a Homebrew installation to DFlash, use an isolated tool install:

```bash
brew uninstall rapid-mlx
uv tool install 'rapid-mlx[dflash]'
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
