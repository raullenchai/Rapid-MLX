# M3 Ultra singleton backbone submission and residual carry

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

Both defaults are limited to Apple M3 Ultra and the existing 35B MoE geometry.
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
