# Installation

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## Install with Homebrew (recommended)

```bash
brew tap raullenchai/rapid-mlx
brew install rapid-mlx
```

The `brew tap` step is required: Homebrew 4.x refuses one-shot installs from
third-party taps (`Refusing to load formula ... from untrusted tap`). The tap
only needs to be added once per machine; `brew upgrade rapid-mlx` after that
works directly.

## Install with pip

```bash
pip install rapid-mlx
```

If `python3 --version` reports 3.9 (macOS default), install a newer Python
first: `brew install python@3.12` then `python3.12 -m pip install rapid-mlx`.

### One-liner with auto-setup

```bash
curl -fsSL https://raullenchai.github.io/Rapid-MLX/install.sh | bash
```

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
| `audio` | `pip install 'rapid-mlx[audio]'` | mlx-audio + spacy + scipy (~600 MB) for TTS / STT |
| `embeddings` | `pip install 'rapid-mlx[embeddings]'` | mlx-embeddings (~50 MB) for `/v1/embeddings` |
| `chat` | `pip install 'rapid-mlx[chat]'` | Gradio web UI (~150 MB) |
| `guided` | `pip install 'rapid-mlx[guided]'` | outlines (~80 MB) for schema-constrained JSON |
| `all` | `pip install 'rapid-mlx[all]'` | Everything above (~1.1 GB) |

## Verify Installation

```bash
# Check CLI
rapid-mlx --help
rapid-mlx version

# Self-diagnostic (works without downloading a model)
rapid-mlx doctor

# Smallest interactive smoke test (downloads ~2.5 GB on first run)
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

### `Refusing to load formula ... from untrusted tap`

Homebrew 4.x refuses one-shot installs from third-party taps. Use the
two-step install at the top of this page (`brew tap raullenchai/rapid-mlx`
then `brew install rapid-mlx`). Only needs to be done once per machine.

### `brew install` fails with `Operation not permitted`

Brew 5.x's install sandbox sometimes can't auto-tap `homebrew/core` mid-install.
Pre-tap it once, then retry:

```bash
brew tap homebrew/core --force   # ~1.3 GB, one-time
brew tap raullenchai/rapid-mlx
brew install rapid-mlx
```
