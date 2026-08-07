# Measured capability gaps

This release sweep fills catalog slots only for aliases that lacked compatible
published benchmark rows. It does not replace existing official model-card
scores. Raw counts and the exact engine/hardware metadata are committed in
[`model-capability-measurements.json`](model-capability-measurements.json).

All models ran sequentially through `rapid-mlx serve` and
`evals/run_eval.py` on the 32 GB M2 Pro Mac mini. Scores are small release
regression suites, not claims of parity with full MMLU-Pro, LiveCodeBench, or
BFCL. Instruction following was not tested and therefore remains explicitly
`Untested` in the app.

| Model | Tool (30) | Code (10) | Reasoning (10) | General (10) | Four-suite mean | 8K decode |
|---|---:|---:|---:|---:|---:|---:|
| `lfm2.5-1b-4bit` | 47% | 50% | 40% | 50% | 46.75% | 124.33 tok/s |
| `lfm2.5-2.6b-4bit` | 77% | 50% | 50% | 80% | 64.25% | 64.95 tok/s |
| `lfm2.5-8b-a1b-4bit` | 73% | 40% | 40% | 80% | 58.25% | 82.52 tok/s |
| `bonsai-27b-2bit` | 93% | 90% | 70% | 90% | 85.75% | 15.00 tok/s |
| `qwen3.5-35b-4bit` | 97% | 100% | 60% | 80% | 84.25% | 21.14 tok/s |

The compact catalog quality meter uses the mean of the local Reasoning and
General suites for these fallback rows. The detailed tooltip retains the
measured Code and Tool axes separately. No value is inferred for a suite that
was not run.
