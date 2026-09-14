# Qwen4 QSA stage-one selector

- Receiving role: Atlas
- Owner/host: Vector / Studio, with an Apple M2 Pro qualification run on Mini
- Branch: `vector/qsa-stage1-fusion`
- PR: #3440

## Verified facts

- The opt-in `RAPID_MLX_QSA_STAGE1=1` route preserves Rapid's existing
  activation-dtype score matmul and FP32 reduction, then replaces only
  `mx.argpartition` with an exact Metal radix selector.
- On Mini (Apple M2 Pro, 32 GB, macOS 26.5.2, MLX 0.32.2), the complete
  synthetic production-geometry QSA indexer at 65,024 tokens improved from
  13.072 ms to 8.249 ms for Q=512 and from 5.089 ms to 3.227 ms for Q=64.
  Both are 1.58x, with identical selected block sets.
- The isolated selector reached 1.64x at 65,024 tokens and 1.77x at 98,304
  tokens; peak temporary memory fell about 31% at those sizes.
- The route is qualified only for MLX 0.32.2, batch one, inference, long
  context, and the measured M2 Pro/M3 Ultra GPU architectures. It is off by
  default and all declines retain the existing eager implementation.
- Focused QSA/Qwen4, packaging, benchmark-contract, and environment-governance
  tests pass. Reproduction commands and raw methodology are distilled in
  `docs/engineering/performance/2026-09-13-qwen4-qsa-stage1-selector.md`.

## Unresolved questions and risks

- The cached `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` is a `qwen3_5` checkpoint,
  not `qwen4_exp`, so it cannot exercise this route. The cached upstream
  `qwen4_exp` checkpoint is unquantized and cannot be loaded safely within the
  current host envelope. Full-model long-context TTFT/output qualification is
  therefore still missing; do not enable this route by default yet.
- The upstream NAX stage-two attention path is numerically corrupt on M3 Ultra
  despite compiling. It was rejected and is not included in this branch.

## Next concrete action

After a quantized `qwen4_exp` checkpoint is available without a new large
download, run a fixed-seed long-context off/on output-parity and TTFT suite. If
that receipt stays positive, Atlas can decide whether to broaden the hardware
matrix and promote the selector from opt-in.
