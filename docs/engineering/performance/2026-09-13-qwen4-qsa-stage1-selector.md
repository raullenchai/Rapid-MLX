# Qwen4 QSA stage-one selector qualification

Date: 2026-09-13

## Result

An opt-in Metal radix selector replaces the `mx.argpartition` portion of
Qwen4/Qwen3.8-Flash-Next QSA stage one after the existing FP32 score
calculation. The route is deliberately narrow:

- `RAPID_MLX_QSA_STAGE1=1`
- MLX 0.32.2
- Apple GPU architectures `applegpu_g14s` (M2 Pro) and `applegpu_g15d`
  (M3 Ultra)
- batch size 1
- query width at least 64
- physical KV length at least 65,024
- inference only

All other cases retain the existing eager score plus `mx.argpartition` path.

## Reproduction

Both hosts ran macOS 26.5.2 and MLX 0.32.2. Inputs use seed 19 for the
selector microbenchmark and seed 23 for the integrated indexer benchmark.
The production QSA geometry is four index heads, head dimension 128,
compression ratio four, and top-k 512.

```bash
PYTHONPATH=. python scripts/bench_qwen4_qsa_stage1.py \
  --tokens 16384 65024 98304 --query-length 512 \
  --warmups 5 --runs 21

PYTHONPATH=. python scripts/bench_qwen4_qsa_stage1_indexer.py \
  --context 65024 --query-length 512 --warmups 5 --runs 21

PYTHONPATH=. python scripts/bench_qwen4_qsa_stage1_indexer.py \
  --context 65024 --query-length 64 --warmups 5 --runs 21
```

### Isolated selector, Apple M2 Pro

| Context | Eager median | Native median | Speedup | Peak memory, eager/native |
| ---: | ---: | ---: | ---: | ---: |
| 16K | 3.404 ms | 3.021 ms | 1.13x | 113.8 / 79.2 MB |
| 65,024 | 13.040 ms | 7.966 ms | 1.64x | 447.0 / 306.4 MB |
| 98,304 | 20.810 ms | 11.729 ms | 1.77x | 674.9 / 461.9 MB |

The selected block sets were exactly equal at every size. Peak temporary
memory fell by about 31% at the two long-context points. These are the final
numbers after review changed the score producer back to Rapid's exact bf16
matmul-then-FP32-reduction order.

### Full QSAIndexer, Apple M2 Pro

At 65,024 physical tokens and a 512-token query chunk, the full indexer path
(projection, normalization, RoPE, compressed-cache update, scoring,
selection, and compact output construction) improved from 13.072 ms to
8.249 ms, or **1.58x**. The 512 selected block IDs per query were set-equal.
The opt-in route recorded 26 constructions; the off route recorded 26
`disabled` declines.

At the lower admitted query width of 64, the integrated path measured
5.089 ms eager versus 3.227 ms native (1.58x). This run showed allocator
warm-up drift in both arms, so it supports the crossover but is not used as a
headline number.

### Compatibility signal, Apple M3 Ultra

The reviewed exact-score implementation compiled, selected the same sets, and
remained positive at 65K and 98K. The Studio was not exclusive: sample latency
showed substantial interference even when the known serving jobs had exited.
At 98K/L=512 the least noisy cell measured 7.173 ms eager versus 5.023 ms
native (1.43x); the integrated 65K indexer median was 39.984 versus 28.755 ms
(1.39x), but both arms were noisy. These qualify compatibility and direction,
not a headline throughput claim. The route therefore remains opt-in.

## Correctness and model identity

- Direct tests compare the selected set against the existing FP32-score plus
  `mx.argpartition` oracle across padding, causal tails, exact ties, and the
  production 4x128/top-k-512 geometry.
- The integrated harness compares all 512 selected blocks for all 512 query
  rows. Sets are identical. Storage order differs, which is semantically
  neutral: dense attention scatters a set and both sparse attention consumers
  sort the physical block starts before dispatch.
- 215 focused QSA/Qwen4, packaging, benchmark-contract, and environment-policy
  tests pass, including the existing dense, block-sparse, and indexed split-K
  paths.

The cached `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` checkpoint must **not** be used
as a Qwen4 E2E receipt: its `config.json` declares `model_type: qwen3_5` and it
uses `mlx_lm.models.qwen3_next.Qwen3NextAttention`, not QSA. An off/on 65,536
token run of that checkpoint produced the same greedy token hash but zero
stage-one constructions, so its timing was excluded. The only cached
`qwen4_exp` checkpoint is the original unquantized model and cannot be loaded
safely within this host's memory/storage envelope. No model was downloaded.

## Rejected NAX stage-two route

The upstream NAX attention kernel was also spiked. It compiled on M3 Ultra,
but a compile-only availability probe is a false positive on pre-M5 hardware.
Against a gathered dense reference at the production 24-query-head,
2-KV-head, head-dimension-256 geometry, the maximum absolute deviation was
2.887 and the mean absolute deviation was 0.406. This is corruption, not a
bf16 tolerance difference. Rapid therefore does not absorb or expose that
route on M1-M4. A future M5 qualification must use a numerical oracle, not
only successful compilation.

## Decision

Land only as an opt-in, fail-closed route until a quantized `qwen4_exp`
checkpoint can provide full-model long-context output and TTFT evidence.
The measured selector/indexer gain and lower temporary memory are strong, but
they are not substitutes for an E2E production-model receipt.
