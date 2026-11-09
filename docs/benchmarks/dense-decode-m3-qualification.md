# Dense singleton decode qualification

November 2026 development measurements, based on main `5a9fbf56b`.
MLX/MLX-Metal 0.32.4, MLX-LM 0.31.3, MLX-VLM 0.7.2, Python 3.12.

## Production scope

Dense Qwen3.5 multimodal loads in resolved `spec_decode="none"` now enroll
existing guarded input-projection and recurrent-decode fusion. The recurrent
kernel depends on projected head/state geometry, not the layer's input width.
Its explicit `allow_dense` opt-in preserves other callers' enrollment scope.

This keeps FP32 recurrent state and the existing device/full-state parity
probe, real-weight projection checks, singleton BF16 admission and stock
fallbacks for unsupported calls. Prefill, batches, masks, speculative modes,
unknown projected geometry and failed probes retain their existing paths.
Independent rollback switches are `RAPID_MLX_GDN_IN_PROJ_FUSION=0` and
`RAPID_MLX_QWEN35_FUSED_GDN_DECODE=0`.

Final actual-server qualification figures appear below. Earlier projection-only
serving measured 166.95→170.86 tok/s on M3 Ultra and 81.68→82.62 on M4 Pro.
LFM is a negative control: the production patch does not enroll its layers.
M5 Max qualification is pending machine availability; no all-chip
non-regression claim is made.

## General prototypes

[The development tool](../../bench/dense_decode/README.md) measures grouped
same-input projections, residual/RMS seams, submission cadence and memory
budgets. These graph-sharing mutations are benchmark-only. Concurrent forwards,
cancellation, speculative rollback and pending-output teardown remain
unqualified. No captured Metal dispatch replay is included.

Three accepted repetitions per cell, median decode tok/s, same cached weights,
fresh prompt caches, greedy 256-token raw loop, prefill excluded:

| Chip / checkpoint | Stock | Full residual/RMS seams | Grouped projections + full seams |
|---|---:|---:|---:|
| M3 Ultra / Qwen3.5 4B |168.20|172.60|176.12 (+4.71%)|
| M3 Ultra / LFM2.5 1.2B |533.35|567.41|568.22 (+6.54%)|
| M4 Pro / Qwen3.5 4B |82.08|82.70|83.29 (+1.48%)|
| M4 Pro / LFM2.5 1.2B |289.54|296.70|296.46 (+2.39%)|

All use the shared GPU lock, CPU≤15%, GPU≤12%, VM CPU≤20% before/after
samples; contaminated attempts are rejected. Raw-loop timing includes a final
submitted step and drain, so absolute rates are not interchangeable with HTTP
client rates. M3 macOS 26.5.2; M4 Pro 48 GB macOS 26.5.1. Sequential-arm sub-percent
changes are inconclusive. Later stock LFM serving cohorts drifted from the
initial campaign; only paired cohorts are compared.

Checkpoint snapshots:

- `mlx-community/Qwen3.5-4B-MLX-4bit`:
  `32f3e8ecf65426fc3306969496342d504bfa13f3`.
- `mlx-community/LFM2.5-1.2B-Instruct-4bit`:
  `dee2f8a2786e6648bb644a7ca40652842490034b`.

The Qwen prompt has 564 templated positions; the LFM raw-loop prompt has 562.
Output digests match stock in all retained raw-loop arms. Both general
prototypes match stock token-for-token on four fixed prompts on M3; additional
192 Qwen + 96 LFM seam checks per arm compare BF16/FP16 bit patterns on real norm
weights. M4 half-seam four-prompt parity and all 288 seam checks pass; M4
full-seam story digests match over three repetitions. These are measured parity
checks, not a universal bitwise guarantee. No intentional arithmetic change
was adopted, so a perplexity-relaxation gate was not used.

## Attribution limits

Stock actual-server dispatch counts are Qwen 868 and LFM 273 per token.
Projection-only Qwen enrollment reduces 868→803 and qmv 249→177, adding seven
BF16 copy dispatches. Quantized matvec on M3 uses threadgroup(32,2,1) in the
captured graphs. There is no evidence of a missing Ultra-specific qmv tiling
selection. Submission depth2/4, per-layer submissions, command-buffer budgets,
wiring limits and library substitution did not produce a general improvement.
The graph's projection, residual/norm and recurrent seams are more promising.

M3 dispatch-boundary timestamp sampling is unsupported. The shader-timing table
of the native Metal System Trace was empty; only command-buffer intervals and
actual dispatch names/geometry were usable. Per-op GPU-time attribution remains
incomplete. Boundary timers include backend waits and must not be described
as pure Python CPU costs or summed into a GPU-time partition.

Local research receipts and the full cross-engine/integrity report are in the
Round4 scratch and report directories. Generated text, token IDs, credentials
and lease files are excluded from this repository. The residual kernel's
Apple copyright and MIT notice are retained in
[LICENSE-RMS](../../bench/dense_decode/LICENSE-RMS).

## Final production qualification

Three accepted fresh-server repetitions per arm, median decode tok/s:

| Chip | Stock | Default, projection + recurrence | Explicit no-spec | Default gain |
|---|---:|---:|---:|---:|
| M3 Ultra | 166.89 (166.48–166.97) | 187.86 (187.29–188.23) | 187.73 (186.68–188.01) | +12.57% |
| M4 Pro | 81.76 (81.56–81.78) | 85.68 (85.58–85.73) | 85.66 (85.63–85.69) | +4.79% |

Cold and repeated response text digests separately match stock across all arms.
All four fixed prompts match stock token-for-token on each chip, 282 emitted
positions including EOS. No intentional arithmetic change was adopted.

Final M3 Qwen graph: **868→460 dispatches**, **249→177 qmv**, **153→81
standalone RMS**, **19→13 command buffers** per token. Each of three captured
complete cycles has 24 fused recurrent dispatches. Buffer counts use the index
difference between argmax boundaries, excluding double-counting of a shared
boundary buffer. The output state remains FP32.

The pinned raw-loop public tool reports 189.19 tok/s (188.71–189.30) for
`gdnfull` over three accepted reps with the stock token digest. This is a
different measurement boundary from serving.

Targeted tests:

```bash
python -m pytest tests/test_dense_gdn_loader.py \
  tests/test_gdn_in_proj_fusion.py tests/test_qwen35_fused_gdn_decode.py -q
```

119 passed in 3.43 seconds on M3 Ultra under the shared GPU lock and idle gate.
Ruff check and format checks pass for all changed Python files. No full-suite,
M5 Max or all-dense-model non-regression claim is made.

Combining guarded GDN enrollment with the benchmark-only gate/up and full
residual seams reaches 193.49 tok/s (193.13–194.13) over three valid M3 raw-loop
reps, with exact four-prompt token parity. This combined experiment is not a
production or cross-chip qualification. It retains the general prototypes'
lifetime limitations.
