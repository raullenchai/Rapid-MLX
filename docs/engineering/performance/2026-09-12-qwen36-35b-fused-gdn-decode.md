# Qwen3.6-35B-A3B fused GDN decode qualification

Date: 2026-09-12

Target: `mlx-community/Qwen3.6-35B-A3B-4bit` at revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46`

Host: Mac Studio, Apple M3 Ultra, 256 GB unified memory

OS: macOS 26.5.2 (25F84)

Runtime: MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.6.17

Measured code base: Rapid-MLX PR #3381 at `1d662e5f1`; the production change
was subsequently rebased onto `main` after #3381 landed.

## Decision

Ship a default-on, fail-closed Metal specialization for the Qwen3.5-family
GatedDeltaNet recurrence at single-token text decode. It combines causal
convolution and cache shift, Q/K normalization, gated-delta state update, and
gated RMSNorm. The already-qualified fused input projection feeds the kernel;
the quantized output projection remains stock.

Only BF16 `(batch=1, length=1)` inputs with initialized, non-ragged caches and
the qualified 16-key-head / 32-value-head / 128-dimension / four-tap geometry
are eligible. Prefill, batching, masks, training, sharding, MTP verification,
unknown geometry, and failed probes remain stock. Set
`RAPID_MLX_QWEN35_FUSED_GDN_DECODE=0` to disable.

## Numerical qualification

Mathematical equivalence was not accepted as sufficient. The implementation
reproduces the stock operation order at every rounding boundary:

- Q/K RMSNorm accumulates mean-of-squares in FP32, casts to BF16, then applies
  the distinct query and key scales in BF16;
- the output epilogue forms the FP32 SiLU gate first and multiplies the
  normalized value second;
- convolution state remains BF16 and recurrent state remains FP32.

The install-time Metal probe runs eight sequential states and compares output,
convolution cache, and recurrent state with `array_equal`. Its input includes a
sigmoid edge value that distinguishes fast and precise exponential forms.

Additional real-weight evidence:

- 32 sequential single-layer steps had exact output and both cache slots;
- a 20-token whole-model audit compared every eligible layer at every decode
  transition and found no output or state difference;
- five complete greedy chat-template cases emitted the same token sequence
  with the path on and off: coding, creative writing, arithmetic reasoning,
  compact JSON, and a tool-call instruction.

The five cases establish preservation relative to the existing checkpoint;
they are not a standalone claim about the checkpoint's absolute task quality.

## Performance

The model was loaded once. Existing gate/up, GDN input-projection, and router
fusions were enabled on both sides. Four alternating warmup generations were
excluded, followed by six adjacent stock/fused pairs. Each measured generation
used greedy decoding, concurrency one, and a 128-token ceiling.

| Metric | Stock | Fused |
| --- | ---: | ---: |
| Median decode | 114.65 tok/s | 128.00 tok/s |
| Mean decode | 114.62 tok/s | 128.09 tok/s |
| Observed range | 113.85–115.14 tok/s | 127.76–128.80 tok/s |
| Positive adjacent pairs |  | 6 / 6 |

The median of the six paired speedup ratios was **1.117x (+11.7%)**; the mean
was **1.117x (+11.7%)**. Compared with the pre-optimization checkpoint path of
about 89 tok/s, the combined gate/up, router, projection, and recurrence work
now reaches about 128 tok/s, roughly **+44%** end to end on this host.

## Rejected experiments

- Fusing the shared expert's dense gate/up projections preserved all five
  token sequences but regressed median decode by about 1%; it is not included.
- A custom routed-expert weighted reduction was about 15% faster in isolation
  but could not reproduce the stock BF16/FP16 reduction bit-for-bit across
  serial, reverse, tree, and FP32 accumulation variants; it is not included.
- The first fused-GDN prototype used a mathematically equivalent FP32 multiply
  association in its output gate. A real hidden-state audit caught a one-BF16-
  ULP layer-output difference that eventually changed greedy tokens. Matching
  the stock association removed the difference before qualification.

## Reproduction

Resolve the target through the configured Hugging Face cache and pass only its
immutable local snapshot path:

```bash
PYTHONPATH=. python3.12 scripts/large-model-run.py \
  --working-set-gb 32 --reserve-gb 12 -- \
  python3.12 scripts/benchmark_qwen35_fused_gdn_decode.py \
    --model /path/to/immutable/snapshot \
    --pairs 6 --max-tokens 128 --quality-max-tokens 512
```

The benchmark exits nonzero unless all five quality cases are token-identical.
It also rejects a performance run when either side has more than 5% throughput
coefficient of variation or fewer than five of six adjacent pairs improve.
Absolute throughput should be measured on an otherwise idle GPU; adjacent
within-process ratios are the landing metric.
