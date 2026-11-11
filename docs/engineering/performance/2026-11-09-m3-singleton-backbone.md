# Singleton backbone submission, residual carry, and device lattice policy

Qwen3.6-35B-A3B plain decoding spent avoidable time submitting small command
buffers. Two changes address the measured path without changing its arithmetic:

- Qualified affine-4 expert kernels bind zero-offset prefix views backed by
  complete weight allocations for width-one calls. Wider verification keeps
  full-bank bindings. Views are cached outside the parameter tree, rebuilt
  when projection arrays change, and formed only after making the **complete**
  source bank contiguous. Pending graphs retain their backing allocations.
- The qualified plain backbone carries residual operands through the existing
  exact residual/RMSNorm seam. This removes separate residual-add launches.
  Enrollment probes every real norm and compares a complete initialized
  width-one output and cache state byte-for-byte. Unsupported calls retain
  the original backbone. No projection weights are rewritten by this helper.

Round-14 defaults enabled both knobs only on M3 Ultra / MLX 0.32.4
and the existing 35B MoE geometry; the round-15 combination policy is below.
The residual helper enrolls only when serving resolves to `spec_decode="none"`.
Hidden capture, extra arguments, training, wider inputs, unsupported cache
classes, and ragged recurrent caches retain the original call. Existing
verification, rollback, sampling, and cache-materialization barriers remain.

Set `RAPID_MLX_QWEN35_AFFINE4_BANK_VIEWS=0` and
`RAPID_MLX_QWEN35_RESIDUAL_NORM=0` before server startup to restore the previous
path. These implementation switches do not select a different model or route.
The residual switch also restores the original call on already-enrolled models.

## Measurements

Apple M3 Ultra, 256 GiB, macOS 26.5.2; Python 3.12.13, MLX 0.32.4,
mlx-lm 0.31.3, mlx-vlm 0.7.2. Target is the existing four-bit snapshot
`38740b847e4cb78f352aba30aa41c76e08e6eb46`; the MTP control uses head snapshot
`0295b81421bf4d0fccca9a7c0fcfb1418dda3516`. No model download was needed.

The scratch qualification used baseline `6ec62b76542d1b3de4c6029eb789ccf81df0c9e7`,
564 templated prompt tokens, 256 greedy completion tokens, thinking disabled,
one fresh server per cell, excluded short warmup, and three interleaved
repetitions with reversed middle order. Primary rates use only cold requests:
`(retokenized tokens - first-chunk tokens) / (last-content time - first-content time)`.
Repeated-prompt results are kept separate. All GPU work holds the shared
validation lock and checks its held/path inode before and after execution.
Admission requires CPU <=30% and GPU <=45% throughout at least ten seconds.
Background workloads were admitted under those thresholds; the machine was
not reserved exclusively.

| Plain arm | Repetitions, tok/s | Median | Range / median |
| --- | --- | ---: | ---: |
| Previous path | 162.21, 160.81, 162.56 | 162.21 | 1.08% |
| Singleton expert bindings | 170.92, 171.60, 170.55 | 170.92 | 0.61% |
| Bindings plus residual carry | 177.10, 177.13, 178.52 | 177.13 | 0.80% |

The combined prototype improves median throughput by 9.2%, saving about
0.519 ms/token. Cold output digests agree in every arm. The singleton-binding
MTP control gives 169.86 versus 170.43 tok/s medians, spreads 0.61% and 0.73%,
with matching output digests. The residual prototype does not run for MTP.
A broader binding prototype that also changed verification was rejected:
its MTP median was 1.34% lower with roughly 4% spread in both arms.

A separate observational trace of the expert-binding change gives:

| Plain trace | GPU busy, ms/token | GPU inactive, ms/token | Buffers/token |
| --- | ---: | ---: | ---: |
| Previous bindings | 5.854 | 0.401 | 47.38 |
| Reduced bindings | 5.799 | 0.050 | 22.24 |

The large recovery is submission idle time. During the previous path,
0.334 ms/token of GPU inactivity overlaps the terminal asynchronous submission
call, falling to 0.020 with reduced bindings. Current-token/logprob evaluation
is approximately 0.006 ms/token of host time, mostly overlapped; recurrent
barriers overlap only approximately 0.002 ms/token of inactivity. These host
phase correlations do not identify individual GPU kernels inside a buffer.

The actual plain route shares the loaded vision-language backbone with the
native-cache text scheduler. The existing eager-layer hook wraps a different
text-backbone class. This plain trace has one terminal asynchronous submission
per token and no eager-layer calls. Disabling that hook is therefore not an
explanation for the plain improvement. The buffer difference comes from
internal encoding/accounting splits despite unchanged explicit eval counts.

## Component evidence and limits

Real-weight, initialized-cache microbenchmarks use five warmups and 31 timed
materialized repetitions per component. Sum of median GPU busy times is
5.983 ms versus 5.870 for the complete step, a 1.9% difference. MoE contributes
2.486 ms; GDN 1.434; full attention 0.821; norms/residuals 0.768; head 0.365;
normalized greedy sampling 0.096. Embedding and final norm complete the sum.
Isolated service times sum to 50.59 ms versus a 7.84 ms fully materialized step:
extra fences make wall times non-additive for production performance.

The fused GDN body is timed as a unit. Standalone convolution, Q/K norm,
recurrence, and gated norm measurements describe an alternative unfused
composition and must not be added to the fused-body result. Likewise the
expert down kernel includes shared gating and combination; those operations
cannot be independently added again. These results do not establish that all
remaining GPU work is avoidable.

## Verification and reproduction

Focused tests cover bit-exact full/prefix expert results, the final expert,
materialized views surviving owner deletion, noncontiguous source banks,
weight replacement, device and environment gates, verification-width fallback,
model-local enrollment, probe refusal, unsupported calls, and load-path wiring.
The focused roster passed 249 tests; dense/native-cache neighbors passed 136
tests in a fresh process. Combining the existing eager-dispatch fixture with
those neighbors produces four stub-constructor failures on unchanged base too;
this ordering issue is recorded separately and is not repaired here.
Run the affected tests with a Metal-capable interpreter:

```sh
python -m pytest -q tests/test_qwen35_affine4.py tests/test_qwen35_residual_norm.py \
  tests/test_no_out_of_band_routing.py tests/test_dense_decode_fusion.py \
  tests/test_qwen36_native_text_cache.py
```

The local fresh-server protocol and raw receipts are retained in
`/private/tmp/rapid-mlx-gap-round13/`. Its wrapper acquires the lock itself;
do not wrap it in another lock invocation:

```sh
python /private/tmp/rapid-mlx-gap-round13/runner.py \
  --arms rapid-main-plain,rapid-new-plain,rapid-main-mtp,rapid-new-mtp \
  --out committed-rerun --reps 3
python /private/tmp/rapid-mlx-gap-round13/runner.py --model-key q4 \
  --arms rapid-main-mtp,rapid-new-mtp --out q4-rerun --reps 3
python /private/tmp/rapid-mlx-gap-round13/runner.py --model-key q9 \
  --arms rapid-main-mtp,rapid-new-mtp --out q9-rerun --reps 3
```

Those arms select the same implementation checkout with both new switches off
or on. The final committed-source measurements, raw emitted-ID comparisons,
and source/lock verification receipt belong to that run's report. No generated
text, model files, private caches, or raw benchmark artifacts are committed.

## Round 14: per-chip policy qualification

The device gates now resolve through [measured lattice policy](rapid-lattice.md).
The checked-in table keys exact chip variant and MLX version. Both defaults
remain on for M3 Ultra / MLX 0.32.4; M4 Pro and M5 Max / MLX 0.32.4 are
explicitly **measured-off** for both knobs. All other device/version combinations
use the conservative off default. Truthy overrides request a qualified trial;
falsey overrides win over measured defaults. The output/cache probes and
geometry restrictions are unchanged.

M4 Pro (48 GiB, macOS 26.5.1) and M5 Max (36 GiB, macOS 27.0.1) used Python
3.12, MLX 0.32.4, mlx-lm 0.31.3, mlx-vlm 0.7.2, and the same cached target/head
revisions, prompt, greedy harness, warmup exclusion and streaming timing above.
Exact clean source: `e79576e4bd94a19f3e299fb86d70d1dad1565996`. Final follow-up
changes only add typed measurement records, tests and documentation; the gate
and resolver functions are unchanged. The off/on arms use this same source
with explicit switch values, rather than different source checkouts.

Each chip has three interleaved fresh-server repetitions of four plain arms
and two default-MTP arms, reversing the middle repetition. `main` disables
both knobs, `bank` enables only bank views, `residual` enables only residual
carry, and `both` forces both on. Residual carry is still excluded from MTP.
Primary rates are cold requests; repeated requests are retained separately.

| Chip | Arm | Repetitions, tok/s | Median | Delta vs mode baseline | Range / median |
| --- | --- | --- | ---: | ---: | ---: |
| M4 Pro | main-plain | 100.59, 100.29, 100.16 | 100.29 | +0.00% | 0.43% |
| M4 Pro | bank-plain | 100.19, 100.40, 100.07 | 100.19 | -0.10% | 0.33% |
| M4 Pro | residual-plain | 101.47, 101.42, 101.42 | 101.42 | +1.13% | 0.05% |
| M4 Pro | both-plain | 101.41, 101.57, 101.56 | 101.56 | +1.27% | 0.16% |
| M4 Pro | main-mtp | 103.18, 103.15, 103.18 | 103.18 | +0.00% | 0.03% |
| M4 Pro | both-mtp | 103.17, 103.05, 103.18 | 103.17 | -0.01% | 0.13% |
| M5 Max | main-plain | 167.07, 167.46, 167.27 | 167.27 | +0.00% | 0.23% |
| M5 Max | bank-plain | 168.88, 168.94, 169.06 | 168.94 | +1.00% | 0.11% |
| M5 Max | residual-plain | 168.91, 169.49, 169.06 | 169.06 | +1.07% | 0.34% |
| M5 Max | both-plain | 171.81, 172.13, 171.72 | 171.81 | +2.71% | 0.24% |
| M5 Max | main-mtp | 180.10, 179.94, 180.03 | 180.03 | +0.00% | 0.09% |
| M5 Max | both-mtp | 179.39, 179.93, 180.02 | 179.93 | -0.06% | 0.35% |

M3 Ultra retains round-13 committed evidence: 161.93 -> 178.21 tok/s plain
(+10.05%, three reps), pooled MTP 168.68 -> 169.17 (+0.29%, six reps).
Its isolated prototype evidence is 162.21 -> 170.92 for bank views (+5.37%),
then 170.92 -> 177.13 with residual carry (+3.63%). These prototype rates
are separate from the committed campaign and must not be mixed into its median.

Neither isolated knob reaches the required 2% gain on M4 Pro or M5 Max.
M5 Max's combined +2.71% is recorded, but does not establish a >=2% gain
for either isolated knob or either marginal addition: adding bank views to
residual-only gives +1.63%; adding residual carry to bank-only gives +1.70%.
It therefore does not justify enabling either knob under the per-knob rule.
MTP differences (-0.01% M4, -0.06% M5) are smaller than observed candidate
spreads; no positive MTP gain is claimed. Both measured-off decisions are
retained in `rapid_mlx/lattice/defaults.py` with commit/date/rate provenance.

All 36 cells have exact 256-token cold and repeated-request raw ID equality
against their chip/mode baseline. All 12 residual-enabled plain enrollments
across the two chips passed the unchanged real norm/output/cache-byte probes
(12 enrollments: two residual-enabled arms x three reps x two chips).
Cold-versus-repeat equality is not asserted: the existing MTP prefix-state
behavior remains separately documented in the round-13 report.

Every cell passed ten-second CPU/GPU admission, held/path lock-inode equality,
clean source identity, expected output count and intended enrollment checks.
No model download, CI VM change or foreign process signal occurred. Remote
scratch was removed after verified evidence transfer; the M5 lease was released.
Raw scripts, manifests, rows, IDs, logs, lock receipts, summaries and cleanup
receipts live under `/private/tmp/rapid-mlx-gap-round14/`, including `REPORT.md`.

The final focused roster passes 276 tests, plus 136 dense/native-cache neighbors
in a fresh process: 412 total (the original 385 plus 27 new cases).
Scoped Ruff check/format, lattice-module type check and `git diff --check` pass.

## Round 15: combination policy

The [Rapid Lattice framework](../decisions/rapid-lattice.md) permits
a measured configuration to set a combination of knobs. Under this rule the
existing M5 Max three-repetition evidence above qualifies **the pair** for
MLX 0.32.4: plain +2.71%, baseline range 167.07–167.46 versus pair
171.72–172.13 tok/s, with no overlap. MTP -0.06% is within observed spread.
The table therefore enables the pair on M5 Max, while residual carry remains
plain-only. This supersedes the round-14 per-knob decision without claiming
an isolated >=2% gain. M4 Pro's +1.27% pair remains measured-off.
