# Rapid-MLX Doctor — regression harness

A four-tier "code health checkup" for Rapid-MLX:

```
rapid-mlx doctor smoke       # ~2 min,  no model         — pre-commit
rapid-mlx doctor check       # ~15 min, qwen3.5-35b      — pre-PR / big change
rapid-mlx doctor full        # ~2-3 hr, 3 models         — pre-release / refactor
rapid-mlx doctor benchmark   # overnight, all models     — periodic / promo material
```

Pick the smallest tier that catches the kind of regression you're worried
about. Smoke is for "did I break the build". Check is for "did I regress
performance or break the API". Full is for "did I regress any of the
models or agents we promise to support". Benchmark is for "what does the
cross-model scorecard look like now".

> **Where:** `vllm_mlx/doctor/` (code) + `harness/` (baselines, thresholds,
> per-run artefacts).

## Quick start

From a source checkout (the doctor refuses to run from a pip install —
it needs `tests/`, `harness/`, and `pyproject.toml`):

```bash
# Pre-commit — no model required
make smoke                            # or: rapid-mlx doctor smoke

# Pre-PR — boots qwen3.5-35b, runs API + perf checks, diffs vs baseline.
# 35B 8-bit is the smallest model we trust to ~never err on the eval
# suite, so failures cleanly attribute to rapid-mlx bugs.
HF_HUB_CACHE=... make check           # or: rapid-mlx doctor check

# Pre-release — three models + all 11 agent profiles
HF_HUB_CACHE=... make full

# Re-record baselines (after intentional perf changes)
HF_HUB_CACHE=... make update-baselines TIER=check

# Cross-model scorecard (overnight)
HF_HUB_CACHE=... make benchmark
```

`HF_HUB_CACHE` is a vanilla Hugging Face env var — use it if your models
live somewhere other than `~/.cache/huggingface`. The doctor inherits
your environment when spawning the server.

Run `make help` for the full target list. The Makefile auto-detects a
suitable Python interpreter (active venv → python3.13/12/11/10) and
respects `make smoke PY=python3.X` for explicit override.

## Exit codes

The doctor's exit code is a stable contract for hooks/CI:

| Code | Meaning |
| --- | --- |
| 0 | All checks pass |
| 1 | At least one **performance regression** detected (vs baseline) |
| 2 | At least one **functional failure** (a check actually broke) |

A run with both a regression and a fail returns 2 (worse signal wins).

## Tier reference

### `smoke` (~2 min, no model)

Cheap static checks. Safe to run from anywhere — no Metal, no model load.
Designed to be invoked from a pre-commit hook or `make` target.

| Check | What it does |
| --- | --- |
| `repo_layout` | Sanity-check `pyproject.toml`, `aliases.json`, `agents/profiles/` |
| `imports` | Import lightweight modules — catches syntax errors fast |
| `ruff` | Lint (binary or `python -m ruff`, gracefully skips if neither) |
| `cli_sanity` | `rapid-mlx --help / models / agents` actually run |
| `pytest` | Full unit suite (~45s, ~2070 tests) excluding `tests/integrations/` and `test_event_loop.py` |

### `check` (~15 min, qwen3.5-35b)

Spins up a real server with `qwen3.5-35b` (Qwen3.5-35B-A3B-8bit — A3B
MoE so decode is fast despite the 35B param count), runs API + perf
checks, diffs against `harness/baselines/check-qwen3.5-35b.json`.

Why 35B-8bit and not a smaller 4-bit model: validation needs the model
itself to ~never err so a failure cleanly attributes to a rapid-mlx
bug rather than quant noise / small-model flakiness. 4B at 4-bit was
the old default and made bug triage ambiguous.

| Check | What it does |
| --- | --- |
| `repo_layout`, `imports` | Same as smoke (cheap fail-fast) |
| `regression_suite` | `tests/regression_suite.py` (10 API contract cases) |
| `smoke_matrix` | `tests/test_smoke_matrix.sh` (emoji/CJK/thinking/leak) |
| `autoresearch` | `scripts/autoresearch_bench.py --json` (13 perf metrics) |
| `baseline_diff` | Compare metrics, flag regressions per `harness/thresholds.yaml` |

Override the model with `--model qwen3.6-35b` (will need its own baseline).

### `full` (~2-3 hr, 3 models × 11 agent profiles)

Loops the check tier across `qwen3.5-35b` and `qwen3.6-35b`
(real-capacity Qwen lines — both 8-bit, both go through the Hermes
parser path that most users hit). For each model, also runs all 11
agent profiles' auto-generated test plans.

> Gemma 4 was previously in the default list for orthogonal coverage
> but was dropped after PR #208 validation showed it fails multiple
> agent tests due to model-side instruction-following gaps (writes
> essays for "Count to 5", refuses to call tools, drops multi-turn
> context). It can still be passed explicitly via `--models` for
> manual investigation.

Override the model list:

```bash
rapid-mlx doctor full --models qwen3.5-35b,qwen3.6-35b
```

### `benchmark` (overnight, all local models)

Sweeps every model with locally-present weights and produces a single
scorecard markdown:

```bash
# Auto-discovers models in HF_HUB_CACHE / $HF_HOME/hub / ~/.cache/huggingface / ~/.lmstudio
HF_HUB_CACHE=... rapid-mlx doctor benchmark

# Or be explicit (forces inclusion even if cache probe misses):
rapid-mlx doctor benchmark --models qwen3.5-35b,qwen3.6-35b
```

Output:

- `harness/scorecard/scorecard-{ts}.md` — timestamped, gitignored
- `harness/scorecard/latest.md` — always points at the most recent run
- `harness/runs/{ts}-benchmark/scorecard.md` — copy in the run dir for
  self-containment alongside server logs

Scorecard columns (kept narrow on purpose — wide markdown tables are
unreadable):

| Model | Decode TPS | Cold TTFT | Cached TTFT | Tool % | Score | Status |

Models that fail to boot or whose autoresearch returns all-zero
metrics get a `FAIL — <reason>` row instead of being silently dropped,
so the scorecard always covers every model the user asked about.

> v1 only sweeps the Simple engine. Cross-engine columns
> (Simple/Batched/Hybrid) are planned for v2 once BatchedEngine
> stabilises (see issue #105).

## Baselines

Baselines live at `harness/baselines/{tier}-{model}.json` and are checked
into git. Per-model file because comparing decode-tps across model
sizes is meaningless. Filename uses URL percent-encoding so model IDs
containing `/` (e.g. `mlx-community/Qwen3.5-35B-A3B-8bit`) don't collide
with names that happen to contain `__`.

Baseline file shape:

```json
{
  "captured_at": "2026-04-15T21:36:32",
  "rapid_mlx_version": "0.5.1",
  "model": "qwen3.5-35b",
  "metrics": {
    "decode_tps": 49.67,
    "cold_ttft_ms": 313.63,
    "tc_success_rate": 1.0,
    "...": "..."
  }
}
```

### Recording / updating baselines

```bash
# Record a fresh baseline (after intentional perf change, or first time)
HF_HUB_CACHE=... rapid-mlx doctor check --update-baselines

# Same for full tier — writes a baseline per model in --models
HF_HUB_CACHE=... rapid-mlx doctor full --update-baselines
```

`--update-baselines` is the only path that writes to `harness/baselines/`.
**Always inspect the diff before committing** — this is the moment to
catch a real regression masquerading as "just a metric change". Workflow:

```bash
# 1. Record what you see now
rapid-mlx doctor check --update-baselines

# 2. Inspect the diff
git diff harness/baselines/

# 3. If the change is justified, commit; otherwise revert + investigate
git commit harness/baselines/check-qwen3.5-35b.json -m \
  "chore(doctor): bump qwen3.5-35b decode_tps baseline (mlx 0.31 SDPA gains)"
```

## Thresholds

`harness/thresholds.yaml` controls when a metric change becomes a
"regression" or "improvement". Two knobs per metric:

```yaml
decode_tps:
  regression_pct: 5     # current 5%+ slower than baseline → REGRESSION
  improvement_pct: 10   # current 10%+ faster → IMPROVEMENT (consider --update-baselines)
```

**Sign convention:** the comparator inverts the sign for lower-is-better
metrics (latency, memory). So a positive Δ% always means "better" in the
report, and a negative Δ% always means "worse" — regardless of whether
the underlying metric is throughput or latency.

Defaults (used when a metric isn't listed):

- Performance: `regression_pct=5`, `improvement_pct=10`
- Accuracy: `regression_pct=0`, `improvement_pct=5` (zero tolerance)

## Run artefacts

Every run writes to `harness/runs/{ts}-{tier}/` (gitignored — local only):

```
harness/runs/2026-04-15-220614-check/
├── report.md                  # human-readable summary table
├── result.json                # machine-readable, full per-check detail
├── diff.md                    # combined delta tables across all models
├── diff-qwen3.5-35b.md        # per-model delta table (full tier)
└── server-qwen3.5-35b.log     # server stdout/stderr for post-mortem
```

The directory name uses second precision plus a numeric suffix on
collision, so concurrent invocations never overwrite each other's
artefacts.

## Common failure modes

### "this command requires a source checkout"

You're running from a pip-installed wheel. The doctor needs `tests/`,
`harness/`, and `pyproject.toml` (none of which ship in the wheel).
Clone the repo and run from there.

### "all primary metrics zero — server probably rejected requests"

The model name in the request body didn't match what the server
registered. If you're hacking on `autoresearch_bench.py`, make sure the
`"model"` field uses `"default"` or the alias the server actually knows.

### "baseline model mismatch"

The baseline file's recorded `model` field doesn't match the model you
asked the doctor to test. Either you renamed a model alias, or someone
copied a baseline file. Re-record with `--update-baselines` after
checking what changed.

### "no baseline found for model=X"

You're running this model for the first time. Run with
`--update-baselines` to record the current metrics as the new baseline.

## Adding a new check

1. Drop a function into `vllm_mlx/doctor/checks/` that returns a
   `CheckResult` — see `checks/smoke.py` for the simplest examples and
   `checks/perf.py` for one that captures metrics for baseline diff.
2. Add a line to the relevant tier in `vllm_mlx/doctor/cli.py`
   (`run_smoke_tier` / `_run_per_model_block`).
3. If your check produces metrics, add them to `harness/thresholds.yaml`
   so they get the right regression threshold (otherwise the default
   5%/10% applies).

## TODO

- `benchmark` v2: cross-engine columns (Simple / Batched / Hybrid) once
  BatchedEngine stabilises (issue #105)
- Pre-push git hook integration (optional, opt-in)
