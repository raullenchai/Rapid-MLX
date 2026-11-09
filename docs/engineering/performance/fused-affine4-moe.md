# Selected affine-4 MoE decode and verification

Selected-expert gate/up/SwiGLU, down/weighted reduction and parallel routing
are enabled by default for qualified Qwen blocks. Set
`RAPID_MLX_QWEN35_AFFINE4=0` to disable enrollment and execution, including
already enrolled instances. All shape, quantization and numerical gates
below continue to apply.

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
The chip choice is fixed during kernel initialization. Expert arithmetic
and launch geometry are unchanged by the default switch.

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

## Round 3 serving qualification (2026-11-09)

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

## Continuous MTP cost calibration

Greedy requests used to choose one of two permanent cost curves from seven
round-robin timing samples. A two-row target forward at 1.33 times the
one-row cost crossed the class threshold, changing the K=1 round price from
1.272 to 1.87 plain rounds. An optimization to one-row decode could therefore
park profitable drafts even when verification itself became faster.

Calibration now uses 21 timed passes after two warmup passes. It takes the
median of each pass's verify/step and draft/step ratios, retaining the actual
measured curve instead of replacing it with a class proxy. The existing
normalized draft and host-overhead estimate supplies the round costs. The
request's own acceptance and existing expected-value controller/probes then
choose depth, including zero where drafting cannot pay. There are no model
name checks or forced depths/classes in this policy.

The measured profile stays frozen for the process/model/depth key. Greedy
requests never consume another request's acceptance or live wall times.
This preserves repeated-request scheduling within a process; exact output
across restarts remains limited by width-sensitive floating-point ties.
Seeded sampling and explicit fixed-depth configuration keep their existing
behavior. Copy-draft verification uses the same frozen forward costs at
timed widths; beyond those widths, the existing tiled lookup proxy is scaled
to the final measured width without selecting a class.

The Round 3 tables above are historical evidence from before this controller
change and do not qualify the new default.

### Round 5 M4 qualification

Base `85373e7d7`, same pinned Qwen35 target/sidecar and serving protocol as
above. Stock uses the original controller with experts disabled; candidate
uses the measured controller and the unset default-on switch. Three fresh
processes per cell; 24/24 primary cells passed idle/contamination gates.

| Model / mode | Stock median (min–max) tok/s | Candidate median (min–max) tok/s |
|---|---:|---:|
| Qwen35 default MTP | 95.29 (95.20–101.16) | 102.62 (102.60–102.73) |
| Qwen35 no spec | 97.68 (97.56–97.80) | 99.96 (99.92–100.06) |
| Dense 4B default MTP | 94.84 (94.67–94.89) | 94.58 (94.56–94.67) |
| Dense 9B default MTP | 60.27 (60.19–60.41) | 60.28 (60.24–60.30) |

Stock's original-controller boots selected default/steep/steep. Even its
fastest boot is below the candidate's slowest. The 4B measurement is a small
negative (-0.27%); separate instrumented diagnostics found identical full
K/acceptance sequences on both dense controls (4B: six parks, 159 drafting
rounds, 91 accepted drafts; 9B: six parks, 153 drafting rounds, 98 accepted).
Diagnostic timings are excluded from the throughput table.

The selected numerical/controller/greedy/park/head-history suites pass 430
checks on M4. Arithmetic, JSON and Python-function prompts, repeated twice,
match no-spec text and actual generated content IDs on all three models;
candidate MTP IDs also match stock MTP. Dense MTP includes a terminal EOS ID
that dense no-spec omits internally, on both arms. Raw Qwen35 IDs match
without that normalization. This fixed-prompt result does not imply arbitrary
long-form bitwise parity; width-sensitive and prefix-cache differences still
exist. The kernel's earlier 50k qualification is inherited, not rerun.

The 102.62 tok/s M4 median is still 1.99% below the supplied 104.7 reference
target. This is a stability/rollout improvement, not evidence that every
supplied target has been reached.

### M5 Max qualification

M5 Max (36 GiB) used the same fresh-server protocol against current main:
three valid boots per cell, 564-token prompt, 256 greedy tokens, cold and
repeated requests. Qwen35 fixed-request content IDs and text match on both
modes; dense controls stay within run-to-run range.

| Cell | Base median (range) | Candidate median (range) | Delta |
|---|---:|---:|---:|
| Qwen35 default MTP, cold | 164.56 (162.28–164.74) | 179.93 (179.91–180.16) | +9.34% |
| Qwen35 default MTP, repeat | 158.71 (158.65–158.72) | 170.33 (170.28–170.33) | +7.32% |
| Qwen35 no spec, cold | 150.86 (150.76–151.19) | 167.20 (166.96–167.22) | +10.83% |
| Dense 4B default MTP, cold | 144.66 (144.45–144.75) | 144.21 (143.92–144.47) | -0.31% |
| Dense 9B default MTP, cold | 96.08 (95.14–96.14) | 95.53 (94.95–96.21) | -0.57% |

The M5 Qwen35 default-MTP median reaches the supplied 178 tok/s target.

### Round 5 Studio qualification

Studio's M3 Ultra uses the same snapshots, runtime versions and protocol,
under a persistent `lockf -k` whose held inode is checked against the path.
All 24/24 primary cells passed their idle/contamination gates.

| Model / mode | Stock median (min–max) tok/s | Candidate median (min–max) tok/s |
|---|---:|---:|
| Qwen35 default MTP | 153.96 (153.62–154.25) | 169.01 (168.02–169.14) |
| Qwen35 no spec | 138.71 (138.35–139.29) | 157.85 (156.51–158.28) |
| Dense 4B default MTP | 154.11 (152.41–154.62) | 153.52 (152.81–153.61) |
| Dense 9B default MTP | 111.50 (110.77–111.52) | 114.94 (113.60–115.30) |

Qwen35 gains 9.78% with default MTP and 13.80% without speculation. Its
candidate calibration ratios (1.3286, 1.3307, 1.3270) straddle the old class
boundary, but the measured K=1 prices stay close (1.3722, 1.3764, 1.3717) and
all three cold-prose output digests match. Stock has one fixed default class
on all three boots. The 169.01 tok/s median remains 11.79% below the supplied
191.6 target.

Dense 4B measures -0.38% with overlapping run ranges; its instrumented full
K/acceptance sequence is identical across arms, as on M4. Dense 9B measures
+3.09%. Both dense models produce identical cold-prose text across all stock
and candidate boots on both chips. These measurements do not establish
sub-percent performance equivalence.

Studio also passes 430 selected tests and all fixed-prompt emitted-content
ID/text comparisons, including repeated requests and candidate-versus-stock
MTP. The same unchanged terminal-EOS reporting distinction applies to dense
models. Both hosts enroll 40 eligible expert blocks under the unset switch;
dense controls enroll none. The M5 Max controller qualification is recorded
in the M5 Max section above.
