# Contributing to Rapid-MLX

Thanks for your interest! Here's how to get started.

## Development Setup

```bash
# Clone and install in dev mode
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# Start a dev server
rapid-mlx serve qwen3.5-4b --port 8000
```

**Requirements:** Python 3.11+, macOS with Apple Silicon (M1/M2/M3/M4).

## Running Tests

```bash
# Run all unit tests (no model needed)
python3.12 -m pytest tests/ -x -q

# Run a specific test file
python3.12 -m pytest tests/test_tool_calling.py -v
```

Most tests run without a model. Tests that require a running server are in `tests/test_event_loop.py` — start a server first if you want to run those.

## Pull Request Workflow

1. Fork the repo and create a branch: `feat/`, `fix/`, `docs/`, `refactor/`
2. Make your changes with tests if applicable
3. Run `ruff check` and `ruff format` before committing
4. Open a PR against `main` with a clear description

## What We Need

- **Hardware benchmarks** — test on your Mac and share results
- **Model reports** — which models work well, which don't
- **Client verifications** — test with your favorite AI tool (Cursor, Continue, Aider, etc.)
- **Bug fixes and features** — check issues tagged `good first issue`

## Code Style

- We use `ruff` for linting and formatting
- Type hints are encouraged but not required
- Keep changes focused — one feature/fix per PR
