# Qwen4 fp32-input fast RMSNorm qualification

Date: 2026-09-06

Owner: Vector

Branch: `vector/qwen4-fast-rmsnorm`, based on `c18faeb0f`

## Outcome

The fp32-input `mx.fast.rms_norm` route is a real narrow-decode kernel lever
for the Qwen4-Exp architecture used by Qwen3.8 Flash-Next. Across the three
production tensor shapes, an interleaved microbenchmark measured 1.105--1.117x
on an M3 Ultra and 1.103--1.126x on an M2 Pro. The route remains opt-in and is
limited to forward widths at or below eight; wider prefill stays on the stock
explicit fp32 reduction.

This is exactness class 3 because the fast reduction may reorder floating-point
operations. Against an fp64 reference over 1,310,720 elements, the candidate's
RMS-error ratio to stock was `1.000000000066`. The rejected bf16-input variant
was `1.413971716665x` farther from fp64 and changed 25.3912% of elements. The
candidate changed 0.000229% of elements. This independently reproduces the
discriminator in the external performance handoff and is why the upcast must
remain before `mx.fast.rms_norm`.

The handoff's real-model end-to-end results are +4.2% decode at 1K and +3.9% at
16K, with prefill unchanged and 147 RMSNorm calls per token. Those two numbers
are inherited evidence, not an independent Rapid rerun. The immutable
Flash-Next artifact is 28 shards totaling about 106 GB; it was no longer local,
the Studio system volume had 46 GB free, and both attached external volumes
blocked on basic writes. A Mini download plus transfer was abandoned after its
measured transfer rate projected roughly 2.5 hours. No end-to-end number from
that incomplete attempt is claimed here.

## Implementation boundary

- `RAPID_MLX_QWEN4_FAST_RMSNORM=1` selects `fast_fp32`; the default is `stock`.
- Every eligible input is upcast to fp32 before entering the fast kernel.
- The additive zero-centered checkpoint weight is multiplied in fp32 after the
  kernel, preserving grouped HC and PLE weights.
- The route is admitted only for sequence widths `<= 8`.
- QSA compressed-key normalization passes the parent forward width explicitly;
  its synthetic singleton axis cannot accidentally admit prefill.
- Mode counts, fast-call counts, and wide-input declines provide a mechanism
  receipt without synchronizing device arrays.
- The change is confined to the vendored Qwen4-Exp text model. Gemma diffusion
  and other model families are unaffected.

## Environment and method

| Component | Studio | Mini |
| --- | --- | --- |
| Hardware | Apple M3 Ultra, 256 GB | Apple M2 Pro, 32 GB |
| macOS | 26.5.2 (25F84) | 26.5.2 (25F84) |
| Python | 3.12.13 | 3.12 via `uv` |
| MLX | 0.32.2 | 0.32.2 |
| Input / weight dtype | bf16 / bf16 | bf16 / bf16 |
| Seed | 3058 | 3058 |
| Warmup | 30 calls per arm | 30 calls per arm |
| Timing | 7 AB/BA repeats, 400 evaluated calls each | same |
| Accuracy | 128 cases, 1,310,720 elements, fp64 reference | same |

The three shapes represent grouped hyperconnection/PLE normalization
(`[1,1,4,2560]`), hidden-state normalization (`[1,1,2560]`), and 24-head QSA
normalization (`[1,1,24,256]`). Each timing fences the returned array with
`mx.eval`, so it measures an individual eager call rather than constructing one
large lazy graph.

Reproduce on either host:

```bash
python3.12 scripts/bench_qwen4_fast_rmsnorm.py \
  --micro-only \
  --micro-repeats 7 \
  --micro-iterations 400 \
  --error-cases 128 \
  --output /private/tmp/qwen4-fast-rmsnorm.json
```

The uncompleted real-model gate is retained for a future host with the artifact
already resident:

```bash
python3.12 scripts/bench_qwen4_fast_rmsnorm.py \
  --model /path/to/dcf657e4acda2aae72da99cde65b6c491cd96998 \
  --prompt-tokens 1024 16384 \
  --max-tokens 128 \
  --repeats 3 \
  --output /private/tmp/qwen4-fast-rmsnorm-e2e.json
```

That gate alternates stock/fast order, requires identical greedy token digests,
records TTFT and decode separately, and asserts that fast calls occur only in
the candidate arm while the wide prefill decline is observed.

## Results

Times are median microseconds per evaluated call. Parentheses contain the raw
sample range; speedup is `stock / fast_fp32`.

| Host / shape | Stock us | Fast fp32 us | Speedup |
| --- | ---: | ---: | ---: |
| M3 Ultra / HC 4x2560 | 236.346 (231.745--297.598) | 213.844 (200.372--229.139) | 1.105x |
| M3 Ultra / hidden 2560 | 240.289 (233.455--283.664) | 216.599 (208.727--263.329) | 1.109x |
| M3 Ultra / QSA 24x256 | 243.489 (235.179--274.660) | 217.908 (206.258--223.772) | 1.117x |
| M2 Pro / HC 4x2560 | 189.730 (186.219--229.266) | 171.952 (170.413--181.603) | 1.103x |
| M2 Pro / hidden 2560 | 187.832 (185.466--193.445) | 166.862 (164.843--175.263) | 1.126x |
| M2 Pro / QSA 24x256 | 193.117 (183.082--194.403) | 171.867 (164.301--174.919) | 1.124x |

| Accuracy result | Value |
| --- | ---: |
| Stock RMS error to fp64 | 0.001663547813861649 |
| Fast fp32 RMS error to fp64 | 0.001663547813971471 |
| Bad bf16 RMS error to fp64 | 0.002352209558120544 |
| Fast fp32 / stock error | 1.000000000066x |
| Bad bf16 / stock error | 1.413971716665x |
| Fast fp32 elements different from stock | 0.0002289% |
| Bad bf16 elements different from stock | 25.3911591% |

## Review gates

The first adversarial pass rejected a non-interleaved version of the micro
harness after a busy Studio run made one shape appear 0.6% slower. Alternating
AB/BA order restored stable positive results on all six host/shape pairs. The
second pass checked every call site and found the synthetic QSA singleton; the
explicit parent-width override is the resulting fix. The third pass removed
counter and sequence-width work from the default stock path.

Promotion beyond opt-in requires a resident-artifact real-model rerun and named
acceptance of the class-3 scope. The current evidence supports landing the
guarded implementation and benchmark tooling, but not enabling it by default.
