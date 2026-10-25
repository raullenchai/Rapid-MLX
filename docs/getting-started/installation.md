# Installation

Install on an Apple Silicon Mac, then start a local chat:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx chat
```

If the command is unavailable in the current shell, run
`export PATH="$HOME/.local/bin:$PATH"`.

## Requirements

- macOS on Apple Silicon
- Python 3.10+ for a manual Python install; image and video need Python 3.11+

## Install with uv

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
python3.12 -m pip install rapid-mlx
```

If `python3 --version` reports 3.9 (macOS default), install a newer Python
first: `brew install python@3.12` then `python3.12 -m pip install rapid-mlx`.

### From source (for development)

Follow the [contributor setup](../../CONTRIBUTING.md) to create a supported
Python environment and install the repository in editable mode.

## Optional Extras

The base install includes the vision (mlx-vlm), image-generation (mflux) and
video-generation (mlx-video) runtimes, so multimodal, image and video models
start without an extra step. Image and video generation need Python 3.11+.
Audio, embeddings and the other features below
ship as opt-in extras. `vision`, `image`, `video`, `dflash` and `mtp` remain
accepted extra names for existing scripts; on macOS they add nothing.

After the official installer, add an extra to its Python environment:

- **Audio:** `~/.rapid-mlx/bin/python -m pip install 'rapid-mlx[audio]'` adds speech and transcription.
- **Embeddings:** `~/.rapid-mlx/bin/python -m pip install 'rapid-mlx[embeddings]'` adds embeddings.
- **Browser chat:** `~/.rapid-mlx/bin/python -m pip install 'rapid-mlx[chat]'` adds the web UI.
- **All optional runtimes:** `~/.rapid-mlx/bin/python -m pip install 'rapid-mlx[all]'`.

If you installed with another Python environment, use that environment's
Python with `-m pip` instead.

When `rapid-mlx serve` finds that an optional runtime is absent, it prints a
repair command matched to the detected install method. For a
runtime that ships with the base install (vision, image, video), a missing
module means the environment is damaged. The pip repair forcibly reinstalls
the running version; tool managers reinstall with their recorded options. pip and
install.sh, uv tool, and pipx environments offer to run the displayed repair
when stdin is a TTY, then restart the original serve command after success.
Press Enter at `Install now? [Y/n]` to accept, or use `--yes` (or `-y`) in a
non-interactive session. The prompt times out after 30 seconds without making
changes. Homebrew, broken runtimes, and incompatible Python versions show
manual guidance. A broken pip-based runtime uses a forced reinstall; an absent
opt-in runtime uses an ordinary pinned install.

The Homebrew formula ships the text runtime only: it omits the vision, image
and video runtimes and does not provide Python extras.
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

Check your internet connection and Hugging Face access. Some models require
account access; follow the instructions on the model page.

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

### Recovering a damaged Python environment

If startup cannot import a required dependency, Rapid-MLX prints one repair
command for the active installation: uv tool, pipx, Homebrew, or the running
Python environment. In a terminal, enter `y` at `Run this repair now? [y/N]`
to run it. Enter or any other answer declines. After the repair, Rapid-MLX
verifies the failed import in a fresh Python process and retries startup once.
Scripts receive the command and a nonzero exit status without an installer
being run automatically.

`rapid-mlx doctor` also checks declared required modules. Use
`rapid-mlx doctor --fix` to review and apply a dependency repair, or
`rapid-mlx doctor --fix --dry-run` to inspect it. Desktop's bundled Python is
never modified: update or reinstall the app instead.

This recovery needs Python to start. If the executable or the launcher's
interpreter has been deleted, reinstall Rapid-MLX with the installation method
used above to restore the launcher first.
