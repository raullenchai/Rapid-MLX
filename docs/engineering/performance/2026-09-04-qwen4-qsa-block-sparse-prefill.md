# Qwen4 block-sparse QSA prefill qualification

## Outcome

Rapid-MLX now has an opt-in Metal path that consumes QSA's compact selected
blocks directly during long-context prefill. It avoids rebuilding a dense mask
and avoids visiting every physical K/V row. The route remains off by default
behind `RAPID_MLX_QSA_BLOCK_SPARSE=1`; decode, training, short queries, short
contexts, non-Metal systems, and unsupported layouts retain dense attention.

This is deliberately not an automatic promotion. The sparse kernel changes the
softmax reduction order relative to dense masked attention. A small fp64 probe
is favorable, but model-scale correctness and performance must be requalified
for each MLX build before changing the default.

## Production-path contract

Each QSA attention layer records cumulative `kernel_calls`, `declines`, and a
complete `decline_reasons` histogram. `qwen4_qsa_block_sparse_stats(model)`
aggregates the receipt without evaluating MLX arrays. Synchronous construction
or dispatch failures fail closed to dense attention and retain the exception
class, not its potentially sensitive message.

The existing fused-GDN receipt now also retains every fallback reason instead
of only the most recent reason per layer. Its end-to-end gate subtracts the
cumulative histograms and requires the exact expected prefill fallback. A fast
path is therefore not qualified merely because an engine-direct microbenchmark
can call it.

## Reproducible isolated measurement

- Candidate: worktree diff on `origin/main@d6c50526a85d50346af4126c1dca9f149aaa9fbe`
- Hardware: Apple M3 Ultra, 256 GiB unified memory
- OS: macOS 26.5.2 (25F84)
- Python: 3.12.13 from `.venv`
- MLX / mlx-lm / NumPy: 0.32.1 / 0.31.3 / 2.5.2
- Precision: bf16 performance geometry; fp16 inputs with an fp64 NumPy oracle
  for the numerical probe; TF32 disabled before importing MLX
- Warmup / samples: two warmups and five interleaved sparse/dense observations;
  three warmups and 100 synchronized dispatch-floor observations
- Artifact: `/tmp/qsa-qualification-production.json`, SHA-256
  `3eedcab197d6609a83279cac40b2b4e0796a6c94a2b8d487c720d79eff2fd821`

```bash
MLX_ENABLE_TF32=0 .venv/bin/python \
  scripts/bench_qwen4_qsa_block_sparse.py \
  --query-length 64 --key-length 16384 \
  --block-topk 512 --block-size 4 \
  --query-heads 24 --kv-heads 2 --head-dim 256 \
  --warmup 2 --repeats 5 --dispatch-repeats 100 \
  --output /tmp/qsa-qualification-production.json
```

At 64 queries over 16,384 physical keys, selecting 2,048 tokens:

| Measurement | Result |
| --- | ---: |
| Dense masked attention median | 2.828 ms |
| Direct block-sparse median | 1.468 ms |
| Isolated speedup | 1.93x |
| Exact binary synchronized dispatch floor | 177.4 us median |
| Sparse max absolute error vs fp64 | 0.000336 |
| Dense max absolute error vs fp64 | 0.001031 |
| Sparse RMS error vs fp64 | 0.000118 |
| Dense RMS error vs fp64 | 0.000303 |
| Sparse max absolute delta vs dense | 0.000977 |

The dispatch floor is about 12% of the sparse call at this smallest admitted
production geometry. This supports a coarse long-context gate and rejects a
design that would split the same work into many launches.

## Existing end-to-end evidence and limits

The repository's earlier M3 Ultra full-model campaign measured a 32K prompt at
523.4 to 579.2 prefill tokens/s (+10.7%) and 62.539 to 56.516 seconds TTFT,
with unchanged decode and byte-identical normalized outputs over five user
paths. That campaign used the same kernel design but predates this rebased
integration and its complete path receipts, so it is supporting evidence rather
than an exact-head promotion gate.

Before automatic enablement, rerun the full served route at 16K, 32K, and 64K
on the pinned release dependency build. Require three or more settled paired
observations, stable stock output, the expected nonzero kernel-call count, no
unexpected decline reasons, peak-memory evidence, and the model-scale
correctness gate. Keep the feature opt-in if any requirement is missing.
