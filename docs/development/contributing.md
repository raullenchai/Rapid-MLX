# Contributing

We welcome contributions to rapid-mlx!

## Getting Started

```bash
# Clone the repository
git clone https://github.com/raullenchai/Rapid-MLX.git
cd Rapid-MLX

# Install with dev dependencies
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_paged_cache.py -v

# Run with coverage
pytest --cov=vllm_mlx tests/
```

### Test Precision Policy

Repo-wide rule when picking which model variant to use in a test. Three buckets, in order of strictness:

> 1. **Correctness tests** — use **8-bit (or higher)**. Quant noise must not be a confounder.
> 2. **Performance tests** — use **4-bit**. ~80% of rapid-mlx users run 4-bit on M-series machines, so perf numbers must come from 4-bit to represent real user experience.
> 3. **Smoke / boot sanity** — small 4-bit model is acceptable for speed (the test only proves "the engine starts and emits a sane response," not strict format conformance). Currently applies only to `make check`.

A correctness test asks *"does the model + our code produce the right output?"* A performance test asks *"how fast / how much memory?"* A smoke test asks *"did anything explode?"* — the third bucket exists because forcing every test into one of the first two would either slow down per-PR feedback (running 8-bit smoke on every push) or pollute the perf signal (running correctness on 4-bit). Keep the smoke bucket small and well-justified; default to one of the first two.

| Suite | Bucket | Model used today |
|---|---|---|
| `tests/` unit + integration | correctness | `mlx-community/Qwen3-0.6B-8bit` |
| `scripts/pr_validate/` stress + agent matrix | correctness | per `scripts/pr_validate/golden_models.yaml` (all 8-bit) |
| `scripts/bench_dflash.py`, `scripts/bench_suffix_decoding_integrated.py`, `harness/runs/` | perf | 4-bit aliases (user reality) |
| `make check` doctor smoke | smoke / boot sanity | `mlx-community/Qwen3.5-4B-MLX-4bit` (4-bit, ~30s boot) |
| `make full` doctor harness | mixed | 8-bit for correctness suites, 4-bit for bench suites; separate baselines per precision |
| `evals/run_all_models.sh` scorecard | scoring + perf column | scoring on 8-bit; perf column on 4-bit |

**Why the split matters in practice.** Quant noise on a 4-bit model produces failures that look like engine bugs but aren't. Reproducible example: `mlx-community/Qwen3.6-27B-4bit` with thinking enabled and a 2-tool composition prompt (`Compute (3+4)*5 using add and multiply`) reliably generates a 4000+ token natural-language ramble without ever emitting a valid `<tool_call><function=...>` XML, hitting the 300s client timeout in PydanticAI's multi-tool test. The 8-bit variant (`unsloth/Qwen3.6-27B-MLX-8bit`) emits both tool calls in 286 tokens. Same engine code in both runs — the only variable is quant noise interacting with the model's strict-format tool-call output under deliberation. If a correctness gate runs 4-bit, the failure looks like an engine regression; running 8-bit attributes it cleanly to where it belongs (the 4-bit quant + multi-tool capability ceiling, not rapid-mlx).

**Hardware constraints:**

- GitHub `test-apple-silicon` (macos-14, M1/M2, 16 GB RAM) — large 8-bit models don't fit. Stick to 8-bit *small* models on CI (`mlx-community/Qwen3-0.6B-8bit`, `mlx-community/Qwen3.5-4B-MLX-8bit` at most). The big-model 8-bit correctness gate runs in `pr_validate` on the maintainer's M-series box, not on GitHub CI.
- Local M-series with 64 GB+ — no constraint, run anything.

**When adding a new test:**

- New correctness test → pick 8-bit. If your family has no 8-bit option (rare), document why in the test file.
- New perf bench → pick 4-bit (it's what users run).
- New smoke test → justify why it can't be a correctness test (usually: "boot speed matters more than precision here"). The smoke bucket should not grow without reason.
- A test that's "kind of both" → split it into two test files, one per bucket. Mixed-purpose tests collapse the signal.

### Code Style

```bash
# Lint and format
ruff check .
ruff format --check .
```

### Running Benchmarks

```bash
# LLM benchmark — short alias works
rapid-mlx bench qwen3.5-4b-4bit

# Or by full HF repo
rapid-mlx bench mlx-community/Qwen3.5-9B-4bit
```

Run `rapid-mlx bench --help` for the full flag list. For multimodal (image /
video) benchmarks, use `scripts/` (e.g. `scripts/bench_*` for the dev-only
benchmarks not shipped with pip).

## Areas for Contribution

- **Bug fixes** - Fix issues and improve stability
- **Performance optimizations** - Improve inference speed
- **New features** - Add functionality
- **Documentation** - Improve docs and examples
- **Benchmarks** - Test on different Apple Silicon chips
- **Model support** - Test and add new models

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure they pass
5. Submit a pull request

## Code Structure

See [Architecture](architecture.md) for details on the codebase structure.

## Testing on Different Hardware

If you have access to different Apple Silicon chips (M1, M2, M3, M4), benchmark results are valuable:

```bash
rapid-mlx bench qwen3.5-4b-4bit | tee results_m4.txt
```

## Questions?

Open an issue at [GitHub Issues](https://github.com/raullenchai/Rapid-MLX/issues).
