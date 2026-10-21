# DeepSeek V4.1 exact vocabulary-head GEMV (2026-10-20)

## Finding

The V4.1 LM head keeps the reference's FP32 logits: `head(h.astype(float32))`
with an unquantized BF16 weight of 129280 x 5120 (1.32 GB). MLX promotes the
whole weight to FP32 on every call before the GEMV, so a single decode token
reads 1.3 GB, writes 2.6 GB and reads that back. On an M5 Max this made the head
alone cost 17.4 ms per token, far more than the bandwidth floor of the BF16
weight (~3 ms).

Feeding the head BF16 activations would be fast (3.2 ms) but changes logits by
up to 0.17 and is not the reference numerics. Keeping a resident FP32 copy is
bit-exact but costs another 2.6 GB and still reads 2.6 GB per token (6.4 ms).

## Change

`rapid_mlx/models/deepseek_v41_native/head_gemv.py` adds a Metal GEMV that
converts BF16 weights to FP32 in registers and reproduces the stock FP32 GEMV
operation order for this shape (32 lanes across K, four contiguous values per
lane, fused multiply-add in element order, shuffle-down reduction). The stock
lane geometry was identified by exhaustive comparison against the stock path
and is asserted bit-exact by `tests/test_deepseek_v41_head_gemv.py` on Metal.

- 1 row (decode): bit-exact with the stock projection.
- 2-8 rows (batched lanes, speculative verify): each row equals its own
  single-row result. For the vocabulary head (at least
  `SHARED_READ_MIN_OUTPUTS` outputs) these rows share one read of the weight;
  the small router weights keep one threadgroup per row. The stock multi-row GEMM may use a different kernel (on
  M5-class GPUs a reduced-precision one, max abs error ~3e-3 vs a float64
  reference, while this kernel stays at ~1e-6).
- More than 8 rows (prefill), CPU, quantized/biased heads, or a K that is not
  a multiple of 128: stock path unchanged.
- The stock order is an MLX/GPU implementation detail, so the first call per
  head weight runs a one-time self-check: an 8-row seeded probe on the real
  weight through every kernel variant that weight can select, each row
  compared with its single-row stock projection; a bit
  difference or a launch failure keeps that head on the stock path (logged as
  a warning), and a replaced weight is checked again.
- `head_gemv.ENABLED = False` restores the stock path everywhere.

## Measurements

Relative A/B on a 4-layer truncation of the published V4.1 Flash build (layers
0-3, Engram layer 1, model head and embedding), M5 Max 36 GB, mlx 0.32.4,
1024-token prompt, median of 150 decode steps after 10 warm-up steps:

| Tree | decode step (ms) | p10 (ms) |
| --- | ---: | ---: |
| main (run 1) | 22.60 | 22.25 |
| exact head (run 1) | 8.58 | 8.44 |
| main (run 2) | 22.59 | 22.21 |
| exact head (run 2) | 8.60 | 8.44 |

Isolated head projection, 1 x 5120 -> 129280: stock 17.4 ms, exact GEMV
3.2 ms; 5 rows 3.6 ms. Greedy 64-token continuation from the same 1024-token
prompt is token-identical to main. Prefill is unchanged (it uses the stock
path).

The truncated model exaggerates the share of the head; on the full 40-layer
model the absolute saving per token is the same order (the head cost is
independent of depth) and must be confirmed on the full checkpoint before
publishing an end-to-end speedup.

## Shared weight read for multi-row calls (2026-10-21)

The first kernel launched one threadgroup column per row, so a 5-row
speculative verify read the 1.3 GB head five times. The vocabulary head now
uses a second kernel for 2-8 rows that loads each weight block once and
accumulates it into each row's own sum in the unchanged order; each row stays
bit-identical to its single-row result (asserted on Metal for 2-8 rows and by
the first-use self-check). One row and the 384-output routers keep the
per-row kernel: the routers are latency bound, and a shared read measured
slower there (0.20 -> 0.26 ms at 5 rows).

Isolated head projection, M3 Ultra, 5120 -> 129280:

| Rows | per-row kernel (ms) | shared read (ms) |
| ---: | ---: | ---: |
| 1 | 2.14 | (per-row kernel) |
| 2 | 4.08 | 2.31 |
| 5 | 9.63 | 2.94 |
| 8 | 15.43 | 4.35 |

Full V4.1 Flash 1017 checkpoint (revision c95bb8ad), M3 Ultra 256 GB, target
forward of N rows at a ~60-token rendered chat prompt with rollback after each
call, median of 12 after 3 warm-up calls, base and change interleaved twice in
the same GPU window:

| Rows | base (ms) | shared read (ms) |
| ---: | ---: | ---: |
| 1 | 51.3-51.8 | 51.2-51.3 |
| 2 | 67.5-68.1 | 65.3-65.4 |
| 5 | 97.8-97.9 | 91.7-91.7 |
| 8 | 130.8-131.1 | 121.1-121.2 |

A 5-row verify (four draft tokens) is 6.2% cheaper; decode is unchanged.
