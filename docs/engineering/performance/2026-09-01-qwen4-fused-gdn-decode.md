# Qwen4 fused GDN single-token decode

## Scope

This experiment fuses the Qwen4-Exp single-token Gated DeltaNet recurrence into
one Metal dispatch. It includes the causal-convolution state update, SiLU,
query/key normalization, decay and beta gates, recurrent update, and gated
RMSNorm. The affine-q4 output projection remains on Rapid's stock path.

The implementation is opt-in through
`RAPID_MLX_QWEN4_FUSED_GDN_DECODE=1` or the resident `stock|fused` selector.
Prefill, batching, masks, ragged caches, training, and speculative rollback use
the stock implementation.

## Prior qualification

The same recurrence-only kernel was qualified in the mlx-uag source tree on an
M5 Max before this Rapid port:

- twelve 32-token trajectories across 1K, 4K, 8K, 16K, 32K, and 64K contexts;
- exact full logits, convolution cache, and fp32 recurrent state;
- 13,824 of 13,824 eligible fused layer calls with zero fallback;
- median decode improvement between 6.39% and 6.93% at every context rung.

The separately tested GDN plus affine-q4 output-projection epilogue was slower
and is intentionally not included.

## Rapid port verification

Rapid-specific validation used an M3 Ultra with 256 GB unified memory,
Python 3.12.14, MLX 0.32.2, and
`rapid-mlx/Qwen3.8-Flash-Next-4bit` at revision
`dcf657e4acda2aae72da99cde65b6c491cd96998`. No other model was resident.

The real-weight layer gate passed 32 sequential steps with exact output,
convolution cache, and fp32 recurrent state. All 32 eligible calls used the
fused kernel with zero fallback. Across eight interleaved 64-step observations,
the stock path took 6.521 ms at the median and the fused path took 5.164 ms, a
26.28% improvement for one complete GDN layer including its input and output
projections.

The same resident model then ran three interleaved 256-token observations per
mode. All six token sequences had the same SHA-256. Median decode throughput
was 25.43 tok/s for stock and 27.04 tok/s for fused, a 6.35% end-to-end gain.
Each fused observation recorded 9,252 eligible calls. The 36 prefill calls
fell back to the stock path as designed.

The focused admission, dispatch, resident-selector, cache-update, fallback,
and real-Metal numerical contracts pass. The Apple-enrolled Metal test runs
all four probe candidates (threadgroup Y of 32, 16, 8, and 4), reproduces the
stock computation for 32 sequential synthetic steps, and compares output and
both cache slots bit-for-bit.

Reproduce the real-weight layer gate on an idle GPU:

```bash
MLX_ENABLE_TF32=0 PYTHONPATH=. python scripts/bench_qwen4_fused_gdn_decode.py \
  --execute-metal \
  --model /path/to/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --output /tmp/qwen4-fused-gdn-decode.json
```

Reproduce the end-to-end interleaved gate without reloading the model between
variants:

```bash
MLX_ENABLE_TF32=0 PYTHONPATH=. python \
  scripts/bench_qwen4_fused_gdn_end_to_end.py \
  --model /path/to/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --repeats 3 \
  --output /tmp/qwen4-fused-gdn-end-to-end.json
```

The environment flag remains disabled by default. Prefill, multi-request
batching, ragged caches, and speculative rollback are explicit stock-path
fallbacks rather than unqualified extensions of this result.

## Failure lifecycle

The capability probe compile-and-runs the exact BF16 kernel specialization
before the fused path can be selected. Unsupported shapes and synchronous
probe or dispatch failures leave the request cache untouched and use stock.
A later Metal command-buffer failure is handled at Rapid's generation boundary:
the scheduler aborts the affected running cohort, closes its `BatchGenerator`,
drops the request-owned caches, and clears Metal state. It is not safe to retry
that token against any partially executed model graph.

Forcing `mx.eval` inside every GDN layer was rejected because it inserts a host
synchronization boundary between sequential model layers and turns the optional
fast path into a regression. The request lifecycle boundary preserves the
engine's existing fatal-device-error semantics without serializing decode.
