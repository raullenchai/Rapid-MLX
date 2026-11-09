# Selected affine-4 MoE decode and verification

`RAPID_MLX_QWEN35_AFFINE4=1` enables selected-expert gate/up/SwiGLU,
down/weighted reduction and parallel routing for qualified Qwen blocks.
The path remains opt-in: unset or `0` disables enrollment and execution.
Round 3 improves default MTP on M5/M3, but M4 default MTP still regresses.
This fails the cross-chip default-on gate.

The loaders enroll compatible blocks after a deterministic relative-RMS
probe capped at 2%. Eligibility requires 256 experts, top-8 normalized
routing, BF16 input, hidden size 2048, expert intermediate size 512,
affine 4-bit expert banks, group size 64 and BF16 scales/biases. The shared
gate may use affine 4-bit or 8-bit weights. Calls with batch size one and
widths 1–9 use independent token-local expert IDs, scores and shared gates.
Larger widths/batches, unsupported layouts/dtypes/families and sharding
retain their existing implementation. Dense models enroll no blocks.

Each projection uses the same tile at every supported token width. M5/M3
use one output row per SIMD group for both expert projections. Apple M4 Pro
uses two gate/up rows and four down rows; threadgroups contain 64 threads.
The chip choice is fixed during kernel initialization. No speculative
controller constants, class thresholds or policy change is included.

Down projections and route-weighted terms round to BF16 before the FP32
route sum and final BF16 cast, preserving the composed path's rounding
boundaries. Gate/up accumulation remains approximate. Tests check exact
singleton-versus-multi-token expert/router arithmetic with distinct token
routes, including widths 2, 3, 5 and 9 and both production tile geometries.
This does not make full-model output bitwise identical to stock.

Full banks remain explicit Metal inputs, retained by the pending MLX graph.
Class closures retain only the original method; enrollment and shared-gate
weights belong to each instance, and output scratch belongs to each call.
A false switch value also disables an already enrolled instance immediately.

## Accuracy qualification

Qwen3.6-35B-A3B-4bit revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46`, MLX/MLX-Metal 0.32.4,
MLX-LM 0.31.3 and MLX-VLM 0.7.2. Fifty thousand teacher-forced positions,
in independent 1024-position contexts, give perplexity 4.8826757102 stock
versus 4.8888548369 fused: +0.126552%, within the 0.5% relative gate.
Greedy top-1 agreement is 93.620% (46,810/50,000). Arithmetic, JSON and
Python-function prompts pass. Default-MTP functional and concurrent checks
pass on stock/fused across M5 Max, M4 Pro and M3 Ultra.

The fixed downloaded excerpt of
[public-domain prose](https://www.gutenberg.org/ebooks/2701.txt.utf-8) has
SHA-256 `4caa34e6e54399f212ff26f09e8bf48f8801e5a614f4714e84f6dff8d96e84da`.
Scored token-stream SHA-256:
`577ad617d469d4db5f88e4ae974acd99d5d4f63615b536d3a9987035f17f048d`.
Retain that exact excerpt to reproduce the corpus hash. The earlier
2048-position screen failed its local threshold; the required 50k run is
the qualification result. These checks cover this checkpoint/text/prompts,
not broad task quality or exact-output parity.

Run `scripts/qualify_qwen35_affine4.py --model <snapshot> --corpus <text>
--output <directory>` on an idle GPU. It scores both arms one token at a
time and stores NLLs, top-1 IDs, functional outputs and hashes.

## Serving qualification (2026-11-09)

Rebased onto main `7b3bdd902`, including MTP park fix `890a6e2` (#5674).
The round-2 base did not include that fix. Every cell below has three valid
fresh-process repetitions, 564 templated prompt tokens, 256 client-counted
output tokens, greedy sampling, thinking disabled and excluded warmup.
Rates are medians with min–max ranges. Stock disables only this expert path.
Default MTP uses default policy parameters and an explicitly selected cached
sidecar for private-HOME isolation. M5/M4/M3 have 36/48/256 GiB RAM.

| Chip | Mode | Stock tok/s (min–max) | Fused tok/s (min–max) | Change |
|---|---|---:|---:|---:|
| M5 Max | No spec | 150.69 (150.52–150.96) | 167.03 (166.83–167.40) | +10.84% |
| M5 Max | Default MTP | 164.62 (164.43–164.76) | 180.02 (180.00–180.05) | +9.35% |
| M4 Pro | No spec | 97.72 (97.60–97.86) | 99.89 (99.81–99.99) | +2.22% |
| M4 Pro | Default MTP | 100.69 (95.32–101.09) | 96.14 (95.98–96.18) | -4.52% |
| M3 Ultra | No spec | 136.13 (133.55–136.28) | 157.28 (156.25–157.44) | +15.54% |
| M3 Ultra | Default MTP | 150.65 (149.26–152.36) | 168.09 (163.90–168.23) | +11.58% |

Dense q9 negative control (zero enrolled blocks):

| Chip | Mode | Stock tok/s (min–max) | Fused tok/s (min–max) | Change |
|---|---|---:|---:|---:|
| M5 Max | No spec | 80.62 (80.59–80.67) | 80.52 (80.24–80.56) | -0.12% |
| M5 Max | Default MTP | 94.55 (94.36–94.66) | 94.89 (94.65–95.01) | +0.36% |
| M4 Pro | No spec | 49.32 (49.31–49.33) | 49.30 (49.29–49.35) | -0.04% |
| M4 Pro | Default MTP | 60.29 (60.21–60.33) | 60.27 (60.23–60.29) | -0.03% |
| M3 Ultra | No spec | 111.40 (111.35–111.78) | 111.60 (111.46–111.63) | +0.18% |
| M3 Ultra | Default MTP | 112.07 (111.93–114.57) | 114.53 (111.94–114.59) | +2.20% |

Dense no-spec digests match across arms on each chip; M3 default-MTP ranges
overlap and do not establish a tight sub-percent bound. The M4 q35 table is
the final mixed-tile campaign. Its q9 controls come from the earlier complete
campaign; dense geometry never executes the changed expert tiles.

Default MTP now uses the fused experts at verify width two. M5/M3 classify
both arms as the default verification-cost class. M4 stock boots classified
default/steep/default; fused boots all classified steep. Its cheaper singleton
step and still-costly verification change that ratio enough to lose stock's
faster MTP regime. In a separate common-steep diagnostic, stock/fused parked
rounds cost 10.395/10.176 ms and drafting rounds 15.889/16.208 ms. Per-request
cold-story acceptance was 16/21 stock versus 10/15 fused; sparse probes and
different generated trajectories limit this comparison. M5 acceptance was
100/146 versus 105/146; M3 was 102/149 versus 107/143. These are observations,
not controlled same-prefix quality attribution.

M4 launch tuning improves no-spec to 99.89 tok/s and keeps the tested story
output digests unchanged. It does not close the supplied ~104.7 tok/s target
or qualify default MTP. A one-repetition larger-buffer-budget diagnostic
reached 99.93 fused tok/s and did not demonstrate a useful remedy; no global
buffer setting is changed. Repeated-block profiles identify tile costs but
do not establish GPU memory-bandwidth saturation or a complete explanation
of the remaining engine gap.

Studio GPU work used the shared validation lock. M4 held a machine lease,
waited for idle CPU/GPU/VM boundaries and preserved CI VMs. Contaminated
samples were excluded/retried. Final relevant affine/router/gate-up/GDN/MTP
suites passed 396 tests on each host, along with lint/format/whitespace checks.
The full repository suite and broad model/task qualification are not claimed.
