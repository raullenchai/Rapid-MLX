# Dense decode graph fusions

Compatible no-spec loads enroll functional residual-add/RMSNorm and grouped
same-input projections by default. Set `RAPID_MLX_DENSE_DECODE_FUSION=0` before
loading to restore the stock implementation. This changes the graph of the
selected model; it does not select a model, scheduler, or speculative policy.

Qwen3.5 dense and LFM2 carry the residual and normalized activation as local
values through the decoder. LFM3 retains its post-operator and post-FFN sandwich
norms before each residual add. Gemma 4 retains its attention/FFN branch norms,
routed expert combination, and layer scalar; only the attention residual add
and the following FFN input norm share a kernel. Gemma per-layer input gating
remains stock. The Gemma layer seam sits after attention, so it forwards the
mask, cache, shared KV, offset, and shared cache to the original attention
unchanged; the backbone-level mask and cache restrictions below apply to the
Qwen3.5/LFM residual path.

The kernel rounds the residual sum to the input dtype before the FP32 square
reduction, retains the stock four-values-per-thread reduction, and preserves
the norm's cast and weight-multiplication boundaries. Its two outputs are fresh
arrays owned by the call. No request activation, pending projection result, or
next-layer norm result is stored on a module or in a process-wide map.

Enrollment probes each real norm weight at three input scales and each eligible
projection group at two deterministic inputs. A complete decoder probe compares
stock and fused outputs on independent caches. Failed decoder enrollment restores
all staged instance classes, parameter views, and attributes. Unsupported classes,
quantization layouts, dtypes, widths, and probe failures retain stock execution.
The adapted numerical source's permission notice is packaged in
`rapid_mlx/kernels/LICENSE-RMS`.

Grouped gate/up projections use the existing activation and down projection.
Compatible Qwen3.5/LFM2 vision-language attention also groups Q/K/V before calling
its existing projected-attention preparation helper. Split projection parameters
remain row-slice views of the grouped weights, so wider calls keep the original
matmuls without retaining a second copy of the weights.

The residual path admits only B=1, S=1 FP16/BF16 inputs up to width 4096, ordinary
cache types without left padding, and calls without hidden-state capture or an
explicit mask. Other calls invoke the original backbone. Multi-row projection
calls use the original projection functions. Installers run on the model-load
worker before publication and enroll model instances rather than changing shared
upstream classes. Explicit MTP and unresolved speculative modes do not enroll.
Cancellation and unload need no fusion-specific cleanup: only the ordinary
request caches and lazy graph own request buffers. Prefix-cache tensors and
numerical identity retain their existing contracts.

`tests/test_dense_decode_fusion.py` uses small synthetic models and covers exact
seams, both decoder libraries, sandwich and routed branches, cache progression,
stock fallbacks, the kill switch, concurrent calls, cancelled graphs, unload,
failed enrollment restoration, and license packaging. It is included in the
hosted Apple Silicon coverage list; loader tests exercise both production entry
points. Hardware qualification must separately check real weights, fixed greedy
tokens, fresh-server performance, and the active speculative policy.

## Qualification (2026-11-09)

Three fresh-server repetitions per arm, greedy 564 prompt / 256 completion
positions, idle-gated on M3 Ultra and M4 Pro. Median (min–max):

| Chip | Model | Stock tok/s | Default candidate tok/s | Median change |
|---|---|---:|---:|---:|
| M3 Ultra | Qwen3.5 4B | 187.71 (187.32–188.13) | 193.91 (193.39–193.96) | +3.30% |
| M3 Ultra | LFM2.5 1.2B | 515.12 (514.20–518.09) | 546.86 (536.14–547.17) | +6.16% |
| M3 Ultra | LFM3 1.7B | 216.26 (216.20–216.73) | 224.10 (223.71–224.46) | +3.63% |
| M3 Ultra | Gemma 4 26B A4B | 115.96 (115.85–116.51) | 118.58 (118.50–118.64) | +2.26% |
| M4 Pro | Qwen3.5 4B | 85.59 (85.40–85.72) | 86.21 (86.19–86.27) | +0.72% |
| M4 Pro | LFM2.5 1.2B | 300.46 (300.33–300.57) | 304.32 (304.14–304.71) | +1.28% |
| M4 Pro | LFM3 1.7B | 116.11 (115.74–116.49) | 116.92 (116.46–117.19) | +0.70% |
| M4 Pro | Gemma 4 26B A4B | 75.11 (75.03–75.20) | 75.52 (75.50–75.54) | +0.55% |

All selected cold/repeated request digests match stock. Each model/chip also
matches 256 fixed-prompt argmax positions; 238 real-activation seams per chip
match bitwise. The focused suite passes 38 tests on each chip; 310 targeted
regressions pass locally, with 100% coverage of the 300 changed production
lines. The focused suite peaks below 0.5 GB process memory. Hosted macOS-15
execution remains pending.

The MTP-default Qwen 9B control does not enroll fusions. M4 cold throughput is
57.34→57.32 tok/s. M3 cold throughput is unresolved: 118.76→113.35 initially
and 116.48→112.50 on confirmation; repeated-request medians are neutral
(112.74→112.74 and 112.95→112.99). Output digests match. This is not a passed
strict M3 MTP performance gate. M5 Max (cold, three fresh boots versus main):
Qwen 4B 140.83→141.60, LFM2.5 473.37→479.00, LFM3 186.45→188.37, Gemma
113.00→115.50 tok/s; the MTP-default 9B control stays within range and exact
request digests match.
The broader all-model/all-chip throughput goal is not established.
