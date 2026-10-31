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
- Python 3.12 or newer for pip installs; the installer uses Python 3.12 or 3.13 (and installs Python 3.12 automatically if neither is found)

## Install with uv

```bash
uv tool install rapid-mlx@latest
```

One command, isolated tool venv, no Python-version juggling — uv finds (or
installs) the right Python automatically. Upgrade later with
`uv tool upgrade rapid-mlx` (if the tool was installed on Python 3.10/3.11, see
[Upgrading from Python 3.10 or 3.11](#upgrading-from-python-310-or-311)). If you don't have uv yet, install it first:
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
python3.12 -m venv ~/.venvs/rapid-mlx
~/.venvs/rapid-mlx/bin/pip install rapid-mlx
```

Add `~/.venvs/rapid-mlx/bin` to your PATH or call the executable directly.

If `python3 --version` reports 3.9 (macOS default), install a newer Python
first: `brew install python@3.12`, then create a virtual environment and install
there: `python3.12 -m venv ~/.venvs/rapid-mlx && ~/.venvs/rapid-mlx/bin/pip install rapid-mlx`.

### From source (for development)

Follow the [contributor setup](../../CONTRIBUTING.md) to create a supported
Python environment and install the repository in editable mode.

## Upgrading from Python 3.10 or 3.11

Rapid-MLX 1.0.1 requires Python 3.12 or newer, so an older Python environment cannot upgrade to it.
Check `rapid-mlx --version`, then run `rapid-mlx doctor --only python`: its first check line (for example `✓ Python 3.11.9`) is the Python your install actually runs.

- **One-line installer:** rerun `curl -fsSL https://rapidmlx.com/install.sh | bash`. It rebuilds `~/.rapid-mlx` using a supported Python (installing Python 3.12 if needed), reinstalls optional features it finds, and keeps settings, credentials, and working `~/.local/bin` symlinks.
- **uv:** `uv tool upgrade --python 3.12 rapid-mlx`.
- **pipx:** if `python3.12` is not found, install it first with `brew install python@3.12`, then run `pipx reinstall --python python3.12 rapid-mlx`.
- **pip:** install Python 3.12 if needed (`brew install python@3.12`), then create a new virtual environment and install there: `python3.12 -m venv ~/.venvs/rapid-mlx-py312 && ~/.venvs/rapid-mlx-py312/bin/pip install rapid-mlx`. Add its `bin` directory to `PATH` or call it directly. Remove the old Python 3.10/3.11 environment only after checking that the new one works.
- **Homebrew:** its formula has its own Python; run `brew upgrade rapid-mlx` once the formula is updated.
- **Desktop:** the app bundles Python 3.12 and updates itself; no Python upgrade is needed.

Re-add any extras installed in a new pip, uv, or pipx environment, for example `~/.venvs/rapid-mlx-py312/bin/pip install "rapid-mlx[audio]"`. Downloaded models remain in the Hugging Face cache and are not re-downloaded.

**Symptoms:**

- `pip install -U rapid-mlx` or `rapid-mlx upgrade` reports `Requirement already satisfied ... (1.0.0)` and stays on 1.0.0: use the fix for your install method above.
- A pinned pip install reports `ERROR: Package 'rapid-mlx' requires a different Python: 3.11.x not in '>=3.12'` or `No matching distribution found`: use the pip fix above.
- `uv tool upgrade rapid-mlx` reports `Nothing to upgrade`: use the uv fix above.
- `pipx upgrade rapid-mlx` reports `already at latest version 1.0.0`: use the pipx fix above.

## Optional Extras

The base install includes the vision (mlx-vlm), image-generation (mflux) and
video-generation (mlx-video) runtimes, so multimodal, image and video models
start without an extra step.
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
