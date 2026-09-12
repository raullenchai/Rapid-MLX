# DeepSeek V4.1 Flash Engram SSD offload qualification

Date: 2026-09-11

Status: qualified as the default product load path for the 256 GiB DeepSeek
V4.1 Flash REAP 2-bit target. Selected Engram rows are prefetched from the
checkpoint mmap while preceding transformer layers execute.

## Boundary

This experiment changes only the residency and scheduling of the two
affine-quantized Engram embedding tables. Tensor validation is strict and the
same selected rows are dequantized with the same MLX affine operation. Target
weights, DSpark weights, verifier, K=4 policy, sampling, API, and catalog are
unchanged.

## Environment

- Apple M3 Ultra, 256 GiB unified memory (`hw.memsize=274877906944`)
- macOS 26.5.2 (25F84)
- Rapid base commit `c152b0451`
- Target: local 212.93 GB REAP12.5 native affine 2-bit checkpoint
- DSpark: pinned 4-bit-dense/2-bit-expert sidecar, 4.62 GB
- Four fixed prompts: code, arithmetic reasoning, JSON-only structured output,
  and Chinese
- Greedy speculative decode, K=4, one 16-token warmup, 64 output tokens per
  prompt

The benchmark used the product runtime loader and recorded MLX active and peak
memory. The resident control changed only `engram_ssd_offload=False`; prompts,
process configuration, target, sidecar, and decode loop were identical.

## Result

| Metric | Resident Engram | Prefetched SSD Engram | Change |
| --- | ---: | ---: | ---: |
| Load time | 266.72 s | 170.02 s | -36.3% |
| Peak MLX memory | 217.95 GB | 156.78 GB | -61.17 GB (-28.1%) |
| Weighted decode | 16.65 tok/s | 16.47 tok/s | -1.0% |

Per-domain prefetched SSD results:

| Domain | tok/s | Accepted tokens/block |
| --- | ---: | ---: |
| Code | 19.25 | 1.91 |
| Reasoning | 17.02 | 1.56 |
| Structured | 21.10 | 2.15 |
| Chinese | 11.80 | 0.78 |

The resident control produced 19.63, 16.63, 21.55, and 12.08 tok/s for the
same prompts and identical acceptance counts. SSD prefetch therefore preserves
98.95% of resident weighted throughput while leaving roughly 61 GB more memory
headroom. Without prefetch, the same SSD path measured 9.25 tok/s; overlapping
selected-row I/O with model computation recovered 78.0% throughput.

## Correctness and operational checks

- Direct resident-versus-disk affine row parity passes.
- Prefetched and synchronous paths return identical arrays.
- A mismatched prefetch is discarded and the requested rows are read instead.
- Tensor dtype, shape, byte length, file boundary, row bounds, and index mapping
  are validated before use.
- The raw-row cache is bounded; close is idempotent and shuts down pending I/O.
- Two clean product loads completed without memory-pressure termination. A
  prior resident full-model experiment was killed under memory pressure, while
  the controlled resident run completed near 218 GB peak.

## Decision and limitation

Use prefetched Engram SSD offload in the DeepSeek V4.1 product runtime on the
qualified 256 GiB path. The 1.0% weighted decode cost is justified by the 28.1%
peak-memory reduction and 36.3% faster load.

This does not establish a 40 tok/s result. The per-domain spread tracks draft
acceptance much more strongly than Engram residency, so the next optimization
should improve draft acceptance and target verification cost rather than add
more SSD caching complexity.
