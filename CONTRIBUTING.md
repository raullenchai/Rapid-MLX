# Contributing to Rapid-MLX

Thanks for your interest! Here's how to get started.

## Development Setup

```bash
# Clone and install in dev mode
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest ruff        # dev tools for testing and linting

# Start a dev server
rapid-mlx serve qwen3.5-4b --port 8000
```

**Requirements:** Python 3.11+, macOS with Apple Silicon (M1/M2/M3/M4).

## Running Tests

```bash
# Run all unit tests (no model needed)
python3 -m pytest tests/ -x -q

# Run a specific test file
python3 -m pytest tests/test_tool_calling.py -v

# Lint and format
ruff check .
ruff format --check .
```

Most tests run without a model. Tests in `tests/test_event_loop.py` require a running server.

## Pull Request Workflow

1. Fork the repo and create a branch: `feat/`, `fix/`, `docs/`, `refactor/`
2. Make your changes with tests if applicable
3. Run `ruff check` and `ruff format` before committing
4. **Self-validate your PR** (see below) — saves a round trip with maintainers
5. Open a PR against `main` with a clear description

## Self-Validating Your PR

Before opening (or after pushing fixes to) your PR, run our validation pipeline against it. The same script is what maintainers run before merging — running it yourself catches the easy stuff before review and signals you've done your homework.

```bash
python3 -m scripts.pr_validate.pr_validate <PR#>
```

The script grades your PR through 7 steps and prints a strict markdown scorecard. Exit code 0 = `MERGE-SAFE`, exit code 1 = at least one step failed.

| step | what it does | when |
|---|---|---|
| `fetch` | pulls your PR + diff, classifies blast radius | always |
| `deepseek_review` | adversarial code review (skipped if no API key) | when `DEEPSEEK_API_KEY` is set and `PR_VALIDATE_NO_DEEPSEEK` is unset |
| `supply_chain` | flags new deps, install hooks, `eval`/`exec`/`shell=True`, hardcoded URLs | always |
| `lint` | `ruff check` + `ruff format --check` | when diff has `.py` |
| `targeted_tests` | runs tests touching the files you changed; **negative-control** filters pre-existing flakes | when diff has `.py` |
| `full_unit` | full pytest suite minus integrations | medium/high blast |
| `stress_e2e_bench` | boots a server, runs stress + agent integrations + bench vs baseline | high blast (engine/scheduler/memory_cache) |

**You don't need every step to pass for a clean PR**, but the more green checks you have, the faster review goes. In particular:

- **`lint` and `targeted_tests` are non-negotiable** — run these locally even without the full pipeline.
- **`supply_chain` warnings** mean a maintainer will read your changes carefully (especially if you touched `setup.py`, `.github/workflows/`, `Makefile`, or added a new dep). That's not a problem — just be ready to explain the why.
- **`stress_e2e_bench` requires Apple Silicon + enough RAM** to load a small model (≥6GB free). If you don't have the hardware, opt out with `PR_VALIDATE_NO_STRESS=1` — maintainers will run it for you on merge.
- **`deepseek_review` needs an API key** — opt out with `PR_VALIDATE_NO_DEEPSEEK=1` if you don't have one. Maintainers will run it for you.

```bash
# Quick local check (no DeepSeek, no stress) — covers the "did I break anything obvious" case in <1 minute for most PRs:
PR_VALIDATE_NO_DEEPSEEK=1 PR_VALIDATE_NO_STRESS=1 \
    python3 -m scripts.pr_validate.pr_validate <PR#>
```

Full step list, gating logic, and how to add steps: [`scripts/pr_validate/README.md`](scripts/pr_validate/README.md).

### What if my PR fails on a pre-existing main bug?

`targeted_tests` already handles this — it re-runs failures on your PR's base commit and reclassifies "fails on main too" as pre-existing (not a regression). For `full_unit` you'll currently see the failure surfaced; mention it in the PR comment ("`test_X` is failing on main too — see issue #123") and a maintainer will confirm.

### What if `pr_validate` itself misbehaves?

It's still new. File an issue with `[pr_validate]` in the title and the artifacts under `/tmp/pr_validate/pr-<N>/` attached.

## Ways to Contribute

### 🟢 Easy — No model download needed

- **Add a model alias** — Add a short name to `vllm_mlx/aliases.json` so users can `rapid-mlx serve <alias>` instead of typing a full HuggingFace path. See [open model-support issues](https://github.com/raullenchai/Rapid-MLX/issues?q=is%3Aissue+is%3Aopen+label%3Amodel-support).

- **Fix a `good first issue`** — Check the [good first issue](https://github.com/raullenchai/Rapid-MLX/labels/good%20first%20issue) label.

### 🟡 Medium — Needs a model + some testing

- **Test a model and report results** — Download a model, run benchmarks, report what works. Use the "Model Support Request" issue template.

- **Add parser auto-detection** — Add a regex pattern to `vllm_mlx/model_auto_config.py` so a new model family gets the right tool/reasoning parser automatically.

- **Verify client integrations** — Test Rapid-MLX with your favorite AI tool (Cursor, Continue, Aider, LangChain, etc.) and report results.

### 🔴 Advanced

- **Write a new tool call parser** — Add support for a new tool call format in `vllm_mlx/tool_parsers/`.
- **Performance optimization** — Profiling, kernel improvements, caching strategies.
- **BatchedEngine / continuous batching** — Multi-user serving improvements.

## How to Add a Model Alias

The easiest contribution — no model download needed!

**File:** `vllm_mlx/aliases.json`

```json
{
  "my-model-7b": "mlx-community/My-Model-7B-Instruct-4bit"
}
```

That's it. Find the MLX model on [HuggingFace mlx-community](https://huggingface.co/mlx-community) and add the mapping. Convention: `<family>-<size>` in lowercase (e.g., `qwen3.5-9b`, `gemma-4-26b`).

## How to Add Parser Auto-Detection

When users serve a model without `--tool-call-parser`, Rapid-MLX auto-detects the right parser from the model name.

**File:** `vllm_mlx/model_auto_config.py`

```python
# Add your pattern (order matters — more specific first):
(re.compile(r"my-model", re.IGNORECASE), ModelConfig(
    tool_call_parser="hermes",    # most common format
    reasoning_parser=None,        # set if model has thinking tags
)),
```

Common tool parsers: `hermes`, `llama`, `deepseek`, `gemma4`, `glm47`, `minimax`, `kimi`.
Common reasoning parsers: `qwen3`, `deepseek_r1`, `gemma4`, `minimax`.

**How to figure out the right parser:** Check the model's chat template for tool call format. Most models use Hermes-style `<tool_call>` tags. If unsure, try `hermes` first.

## Code Style

- We use `ruff` for linting and formatting
- Type hints are encouraged but not required
- Keep changes focused — one feature/fix per PR

## Releasing

The release pipeline is fully automated from a single commit on `main`. Push a commit with subject `chore: bump version to X.Y.Z` (matching the new `pyproject.toml` version) and the rest happens on its own: tag → GitHub Release → PyPI → Homebrew formula PR.

If your PR adds a model alias, capability profile, or CLI flag, the `version-check.yml` workflow requires you to bump `pyproject.toml` in the same PR (or set the `skip-version-bump` label for pure refactors). This is the safety net that prevents stale `rapid-mlx models` after a merge.

Full details, escape hatches, and rationale: [`docs/development/releasing.md`](docs/development/releasing.md).
