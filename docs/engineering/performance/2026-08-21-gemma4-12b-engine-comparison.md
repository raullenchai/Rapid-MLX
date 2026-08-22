# Gemma 4 12B engine comparison on M2 Pro

Measured 2026-08-21 on a Mac mini (Apple M2 Pro, 10 CPU cores, 32 GB unified
memory) running macOS 26.5.2. The comparison covers three MLX Community Gemma 4
12B 4-bit checkpoints across Rapid-MLX, mlx-vlm, and oMLX.

## Results

Each cell is the median decode throughput across 16 samples: eight prompts
times two independent model loads. Pooled throughput is included in
parentheses.

| Checkpoint | On-disk size | Rapid-MLX | mlx-vlm | oMLX | Rapid vs mlx-vlm | Rapid vs oMLX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard 4-bit | 6.3 GiB | **23.06** (23.06) | 22.92 (22.91) | 22.34 (22.19) | +0.6% | +3.2% |
| QAT 4-bit | 10 GiB | **15.72** (15.72) | 15.69 (15.69) | 15.43 (15.42) | +0.2% | +1.9% |
| OptiQ 4-bit | 8.4 GiB | **18.70** (18.67) | 18.57 (18.65) | 18.19 (18.21) | +0.7% | +2.8% |

Units are generated tokens per second. The practical conclusion is parity with
a small Rapid-MLX lead. QAT's 0.2% difference from mlx-vlm is a tie for any
public claim; none of these results supports a broad claim that Rapid-MLX is
materially faster on every Gemma 4 workload.

The best configuration tested is the standard checkpoint at **23.06 tok/s**.
It is also both the smallest checkpoint and the lowest-memory option in this
matrix. For a rounded public number, use **23.1 tok/s** on M2 Pro.

## Versions and checkpoints

| Engine | Version | MLX stack |
| --- | --- | --- |
| Rapid-MLX | 0.12.18, source `a3a0d02bbc050c37923b8a1aeb3773f0e3390f94` | mlx 0.31.2, mlx-lm 0.31.3 |
| mlx-vlm | 0.6.15, source `72f37ca46ace7bb8f8b3fd91d1b6c75e20c77b40` | mlx 0.32.1 |
| oMLX | 0.6.3rc2, source `2df39bfcdd9c8fb80847b2869d7f2d62a162f673` | mlx 0.32.0, mlx-vlm 0.6.3, mlx-lm 0.31.3 |

Model revisions:

- `mlx-community/gemma-4-12B-it-4bit`: `73bcf09092aa277861d5a191b989b666f7f32e8f`
- `mlx-community/gemma-4-12B-it-qat-4bit`: `e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6`
- `mlx-community/gemma-4-12B-it-OptiQ-4bit`: `c5183df90d827c09764547a74955e6cb21a97db9`

## Workload and controls

- Batch/concurrency: 1
- Eight prompts: coding, explanation, JSON, reflection, dialogue, summary,
  reasoning, and translation
- Two complete runs per engine/checkpoint; model reloaded between runs
- Maximum generation: 128 tokens per prompt
- Sampling: temperature 0, deterministic seed 0, thinking disabled
- Warmup: one four-token generation before each measured run
- MLX cache cleared between Rapid-MLX and mlx-vlm prompts
- oMLX cache storage disabled
- Statistic: median of per-prompt decode throughput; pooled throughput is total
  generated tokens divided by total decode time
- Before valid runs, no process consumed more than 20% CPU; macOS reported no
  thermal or performance warning

An initial run while Chrome had seven helpers consuming 60–100% CPU produced
an invalid 10.26 tok/s Rapid result. That run was rejected, Chrome was closed
with human authorization, and the valid standard-checkpoint repeats measured
23.05 and 23.06 tok/s with per-run ranges of 23.01–23.09 tok/s. This is why the
busy-machine gate is part of the methodology.

## Memory and loading observations

Peak MLX memory after warmup and during generation:

| Checkpoint | Rapid-MLX | mlx-vlm |
| --- | ---: | ---: |
| standard 4-bit | **6.90 GB** | 6.95 GB |
| QAT 4-bit | **11.11 GB** | 11.16 GB |
| OptiQ 4-bit | **9.05 GB** | 9.16 GB |

Median cached model-load time was 2.74/2.53/3.40 seconds for Rapid-MLX and
5.14/6.40/5.89 seconds for mlx-vlm (standard/QAT/OptiQ). These are direct
in-process load timings, not cold-download or server-ready timings. oMLX memory
and model-load time were not captured by this harness, so no cross-engine
memory or loading claim should include oMLX.

## Reproduction

The benchmark workspace on the Mac mini is `~/gemma12-perf`. The direct
Rapid-MLX and mlx-vlm command shape was:

```bash
cd ~/gemma12-perf

venvs/rapid/bin/python bench_gemma4_direct.py \
  --engine rapid \
  --model mlx-community/gemma-4-12B-it-4bit \
  --max-tokens 128 \
  --output results/rapid-base-r1.json

venvs/mlx-vlm/bin/python bench_gemma4_direct.py \
  --engine mlx-vlm \
  --model mlx-community/gemma-4-12B-it-4bit \
  --max-tokens 128 \
  --output results/mlx-vlm-base-r1.json

venvs/omlx/bin/python bench_omlx_engine.py \
  --model mlx-community/gemma-4-12B-it-4bit \
  --condition ar \
  --max-tokens 128 \
  --output results/omlx-base-r1.json
```

Replace `base` and the model ID with `qat` / `...-qat-4bit` or `optiq` /
`...-OptiQ-4bit`, and repeat with `r2`, to reproduce the full matrix.

The finalized direct benchmark script SHA-256 is
`758560ad60ffff0f588c69e545e228749b97ae8190510affc8490d56652afbef`;
the oMLX script SHA-256 is
`a8d53333f92881cce03255d6680eaca6a13376281606056c51d712380ccd26d3`.
Raw result hashes remain on the benchmark host alongside the JSON files.

## Limitations

- This is short-prompt, single-stream, direct decode. It does not measure HTTP
  overhead, TTFT, long-context prefill, prefix-cache reuse, image/audio input,
  concurrent throughput, or suffix decoding on code-edit traffic.
- The engines use their supported dependency stacks rather than one identical
  MLX version. This reflects real installations but does not isolate framework
  overhead from MLX-version effects.
- The editable Rapid-MLX checkout had local changes in the Qwen-only
  `bench_spec_decode_mtp.py` and `qwen3_5_inject.py` paths. Gemma autoregressive
  loading and generation do not use those changes, but the checkout was not
  byte-for-byte clean and the fact is recorded for reproducibility.
- Rapid-MLX and mlx-vlm did not produce token-identical output for every prompt
  despite deterministic sampling, so these numbers compare engine behavior,
  not identical token-by-token execution traces. All direct runs generated the
  same 128-token budget per prompt.
- oMLX stopped early on some prompts; median per-prompt throughput remains the
  primary comparison, while pooled results should be read with that caveat.
- Do not combine these M2 Pro numbers with the M3 Pro suffix-decoding result in
  `docs/guides/gemma4-12b-18gb.md`; that is a different workload and hardware.
