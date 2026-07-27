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

There is deliberately no `--update-baselines` command. Automatically replacing
the comparison point after every run would turn a real regression into the new
normal before a maintainer reviews it.

The current thresholds live beside the consumer in
`scripts/pr_validate/steps/stress_e2e_bench.py`.
