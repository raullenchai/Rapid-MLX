# Performance baselines

`harness/baselines/` contains the committed comparison points used by
`pr_validate`'s live-model benchmark. The candidate inventory comes from
`scripts/pr_validate/golden_models.yaml`: every candidate that may be selected
for a machine's RAM must have a baseline, including fallback candidates.

This directory no longer belongs to `rapid-mlx doctor`. The old
`doctor check/full --update-baselines` workflow was removed in v0.7.22, and its
`full-*.json` files are not compatible with the current benchmark.

## Audit

```bash
python3.12 scripts/release_baselines.py
python3.12 scripts/release_baselines.py --strict-stale
```

The default audit fails for missing, invalid, mismatched, or orphaned files. A
baseline older than the latest release is printed as a warning so age cannot
be missed, but does not fail by default. `make release-check-m3` runs the
default audit before it downloads or boots a model.

Baseline files use `harness/baseline.schema.json` and record:

- exact model ID, Hugging Face revision, quantization, and engine;
- chip, memory, macOS, Python, Rapid-MLX commit, and MLX toolchain versions;
- the cold/warm request medians consumed by `stress_e2e_bench`;
- the number of independent sample runs and their per-metric median;
- reviewed cold/warm regression thresholds (5% for stable AR paths; higher
  only when repeated captures document a wider noise floor).

## Capture state: a FRESH server, always

Both sides of the comparison must be measured the same way, so
`stress_e2e_bench` benches **first** — on a server that has served
nothing else — before the stress battery and the SDK matrix touch it.
Capture baselines the same way.

("SDK matrix" = the `agents:` list in
`scripts/pr_validate/golden_models.yaml`, which is Anthropic SDK /
LangChain / Pydantic-AI scripts. It is *not* the coding-agent matrix in
`docs/agents/matrix.md`, which pr_validate does not run.)

This is load-bearing, not a detail. Measured on Qwen3.5-35B-A3B-8bit /
M3 Ultra, each group highly reproducible within itself:

| capture state | cold median |
|---|---|
| after stress + SDK matrix | 287.6, 288.2 ms |
| fresh server | 252.8, 253.1, 252.0 ms |

A ~14% cold gap with under 0.5% spread inside each group. While the
bench ran last, every PR was measured post-stress against a baseline
captured fresh, so the 5% threshold could not survive the mismatch: the
gate reported a "regression" for a change that only edits prompt
assembly and a regex, and the identical delta showed up on main. Warm
moves the other way (~4% faster once the engine is hot), which is what
made the symptom read as noise.

`tests/test_bench_runs_on_a_clean_server.py` pins the ordering.

## Refreshing a baseline

`pr_validate` writes each fresh measurement to its run directory as
`bench-<model>.json`. Treat that output as a **candidate**, not an automatic
update:

1. Run the benchmark on the documented hardware profile.
2. Compare the candidate with the committed baseline and investigate any
   slowdown before changing the baseline.
3. Copy the accepted metrics into the schema-v1 file and update all capture
   metadata, including the exact model revision and Rapid-MLX commit.
4. Run `python3.12 scripts/release_baselines.py --strict-stale` and inspect the
   Git diff.

A capture taken on a busy machine is not a candidate. The cold metric is
the one exposed to it: on a quiet M3 Ultra, eight independent fresh-server
medians of `Qwen3.5-35B-A3B-8bit` span 2.4%, while a single gate run on the
same commit with macOS's animated desktop decoding video on the GPU came out
+37%. Close other GPU consumers before capturing, and check
`bench-ab-<model>.json` when one is emitted: a flagged bench is settled by a
counterbalanced A/B against the base ref measured in the same session, and that
artifact records both arms' captures, their `capture_spread_pct`, and the
resulting delta. A wide spread means the number is not usable, whichever
direction it points.

There is deliberately no `--update-baselines` command. Automatically replacing
the comparison point after every run would turn a real regression into the new
normal before a maintainer reviews it.

The current thresholds live beside the consumer in
`scripts/pr_validate/steps/stress_e2e_bench.py`.
