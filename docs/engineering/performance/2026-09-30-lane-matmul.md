# Row-invariant lane matmul for multi-row decode (opt-in)

`RAPID_MLX_LANE_MATMUL=crossover|exact` installs a row-invariant "lane"
matmul on a loaded dense model's linear projections
(`rapid_mlx/kernels/lane_matmul/`). The design is the TensorFold team's
(github.com/ashhart/TensorFold). This implementation is adapted from mlx2
(github.com/pierre427/mlx2, `src/mlx2/runtime/lane/`).

## What it does

Stock MLX picks its quantized-matmul kernel by row count, and its cost grows
with the rows. A decode step for B concurrent requests, or a verify of K+1
drafted tokens, sends B or K+1 rows through every projection. The lane
matmul uses one arithmetic for every row count and reads each weight once
for all rows, so 16 rows cost about as much as one.

- **Backends:**
  - `mpp` on M5 (Metal 4 tensor units): affine 2–8-bit and bf16/fp16 weights.
  - `simd` on M1–M4 (simdgroup matrix units): TensorFold's row-exact
    `simd_qmm` family, for affine 2–8-bit weights from bf16 checkpoints.
- **`crossover`:** calls of 8 or more rows (16 or more for bf16/fp16
  weights) take the lane kernels. Fewer rows keep stock kernels: one-token
  decode, and a K=3 MTP verify, which is 4 rows.
- **Stacked siblings:** sibling projections (q/k/v, gate/up, GatedDeltaNet
  `in_proj_*`) are stacked into one launch. Below the crossover they run as
  one stock `quantized_matmul` over the stack. That is what
  `gdn_in_proj_fusion` does, extended to q/k/v and gate/up, and it runs only
  where an install-time probe proves the result bitwise equal to separate
  calls for every such row count, in bf16 and fp16.
- **`exact`:** every call of 1–32 rows takes the lane arithmetic, so each
  row's projections equal that row decoded alone.
- **Scope:**
  - Mixture-of-experts models are skipped, because `gather_qmm` is not
    covered.
  - Projections attached after load, such as an injected MTP head, keep
    stock kernels.
- **Cache identity:** the installed law ID is appended to the prefix-cache
  identity, so state computed under it is never served to a stock process.

## Serving A/B

| Setting | Value |
|---|---|
| Commit | `feat/qwen-lane-verify-matmul` on e9f4a8998 |
| Model | `Qwen3.8-27B-MLX-4bit`: dense, 4-bit group 64, no MTP head, so plain batched decode |
| Server | `rapid-mlx --no-telemetry serve <model> --disable-prefix-cache` |
| Environment | `RAPID_MLX_LANE_MATMUL` unset (off) or `crossover`; mlx 0.32.3 and mlx-lm 0.31.3 from PyPI; Python 3.12 |
| Load | N concurrent greedy chat requests (`enable_thinking: false`), distinct code-writing prompts, one warm-up request per server |
| Ordering | Six server runs per host in ABBAAB order, 90 s cooldown, each concurrency level played twice per run (n=6 per cell) |
| Swap | No swap-outs during any run |

Aggregate decode tok/s, median (min–max):

**M5 Max, 128 GB** (`mpp` backend), 256 tokens per request:

| Concurrent requests | Off | Crossover | Change |
|---:|---|---|---:|
| 1 | 33.2 (33.0–33.3) | 33.1 (32.8–33.3) | −0.5% |
| 4 | 101.1 (95.5–104.7) | 97.9 (93.5–99.6) | −3.1% |
| 8 | 109.7 (102.6–111.7) | 165.1 (162.0–169.3) | **+50%** |
| 16 | 152.5 (145.8–154.1) | 275.9 (268.8–279.6) | **+81%** |

**M3 Pro, 36 GB** (`simd` backend), 128 tokens per request:

| Concurrent requests | Off | Crossover | Change |
|---:|---|---|---:|
| 1 | 8.7 (8.7–8.7) | 8.6 (8.6–8.6) | −0.3% |
| 4 | 30.1 (29.9–30.1) | 30.0 (29.9–30.0) | −0.4% |
| 8 | 31.7 (31.7–31.7) | 50.4 (49.4–50.5) | **+59%** |
| 16 | 38.3 (38.3–38.4) | 58.4 (58.3–58.5) | **+52%** |

The M5 result at 4 concurrent requests is inside the run-to-run spread. In a
single process, a 4-row forward measured 39.5–40.3 ms against 39.6–39.9 ms
for the default path with `gdn_in_proj_fusion`, three runs each.

## Correctness

- **Row invariance.** For every tested row count (1–48, including the
  partial last 32-row block), each row of a multi-row call is bitwise equal
  to the row computed alone. Every multi-row input was freshly allocated at
  exactly that size.
- **Coverage.** The M5 kernel was tested on 2/3/4/5/6/8-bit weights at
  group sizes 32/64/128 and on bf16/fp16. The simd kernels were tested on
  the same formats forced on an M5 and natively on an M3 Pro, including
  Qwen3.8-27B shapes up to the 5120×248320 head.
- **Accuracy.** Error against an fp32 dequantized reference is within 2× of
  stock MLX's, and at or below it on the layer shapes.
- **Whole model, M3 Pro, `exact`.** Measured in mlx2 with the same kernels
  on plain mlx-lm (`scripts/lane_mlxlm_forward_ab.py` there). Over 16 verify
  rows against 16 one-token stock decode steps, top-1 agreement was 16/16 at
  2K context (stock's own 16-row forward: 16/16) and 16/16 at 16K (stock:
  15/16).
- **Tests.** `tests/test_lane_matmul.py` runs on Metal, so the Apple CI leg
  exercises the simd kernels on an M1.

## Limitations

- **No gain at 4 or fewer rows:** single-stream decode and a K=3 MTP verify.
  The gains start at 8 rows: 8 or more concurrent requests, or a batched MTP
  verify of 2 or more lanes at K=3.
- **Hardware tested:** M5 Max and M3 Pro only; no M1, M2 or M4.
- **M1/M2 thread limits:** the probing that steps a launch down on those
  chips is covered only by a unit test.
- **Arithmetic differs from stock.** The lane law is not bitwise equal to
  stock MLX, so greedy output at 8 or more concurrent requests can differ
  from lane-off output by near-ties. Stock multi-row kernels already differ
  from one-row decode the same way.
- **Checkpoints:** fp16 checkpoints on M1–M4 and MoE models are not covered.
