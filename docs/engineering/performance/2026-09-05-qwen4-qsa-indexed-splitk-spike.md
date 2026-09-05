# Qwen4 indexed split-K QSA qualification

Date: 2026-09-05

Owner: Vector

Host: Mac Studio, M3 Ultra 256 GB, `applegpu_g15d`

Originating branch: `vector/qsa-indexed-splitk`. The implementation and its K=2
qualification are now combined in PR #3087 on top of `main`; prerequisite PR
#3055 is merged.

## Outcome

The initial standalone spike was not ready to promote because Qwen3.8 rejected
K=2 and the active K=1 target width M=2 was intentionally unqualified. The
follow-up in PR #3087 adds explicit-only K=2 and completes real-model
qualification. The production gate admits only M=3 from 16K and M=1 from 64K,
requires batch size one, and pins both MLX 0.32.2 and the measured M3 Ultra
Metal architecture. It also fail-closes on any tensor or selection geometry
other than the measured BF16 QH=24, KVH=2, D=256, block-size-4, top-K-512
layout. Explicit K=2 activates the route by default while an explicit
environment opt-out remains available.

Do not broaden the gate to M=2 merely to make the current MTP path exercise the
kernel. A first cold M=2 served run regressed, and there is no settled hot
end-to-end comparison for that geometry.

## Implementation and numerical evidence

The spike implements two Metal passes over sorted compact QSA block/tail
indices. K/V remain in the physical cache. The implementation clones MLX
`sdpa_vector_2pass`'s float accumulators, `metal::fast::exp`, bf16 partial
boundary, and pass-two transpose/reduction order.

An early D=32 smoke passed while the production D=256 layout was wrong: pass
one used interleaved lane dimensions but pass two expected MLX's contiguous
`D / 32` lane chunks. The production-shape adversarial comparison caught it.
After the fix, at B=1, QH=24, KVH=2, M=3, N=16K, D=256, block size 4 and 512
selected blocks:

| Comparison | Max absolute error | Mean absolute error |
| --- | ---: | ---: |
| indexed split-K vs dense masked | 1.2207e-4 | 1.0857e-5 |
| indexed split-K vs independent fp64 | 8.4140e-5 | 1.0021e-5 |
| dense masked vs independent fp64 | 7.5344e-5 | 1.0404e-5 |

Distinct selections were used for all three query rows. Split counts 32, 64,
and 512 all passed the fp64 tolerance. The production schedule uses 128 splits,
which is bit-identical (`max_abs 0.0`) to MLX 0.32.2 attention over physically
gathered 2,048-token K/V. A 512-split schedule was faster in one microbench but
changed 9,261 bf16 outputs, so it was rejected. Additional tests cover opt-in,
version/architecture gates, exact production layout/selection admission,
malformed int32 indices, empty selections, route priority/receipts, and graph
identity when both sparse routes are disabled.

The GitHub M1/M2 lane confirmed why bit identity is part of the qualification
boundary rather than a cross-architecture property: its direct comparison to
native attention differed in 7,824 of 18,432 BF16 elements, with maximum
absolute difference 9.7656e-4. Those devices remain production-ineligible;
their independent FP64-oracle test stays enabled, while the bit-exact assertion
now runs only on the qualified MLX 0.32.2 M3 Ultra combination.

The final stride-aware implementation was measured with K/V sliced from a
larger capacity buffer, matching the non-row-contiguous decode-cache layout:

| Query / physical KV | Dense mask | Indexed, 128 splits | Isolated speedup |
| ---: | ---: | ---: | ---: |
| M=3 / 16K | 2.226 ms | 0.391 ms | 5.70x |
| M=1 / 64K | 0.864 ms | 0.304 ms | 2.84x |

Each arm used 12 warmups and 80 serialized timings; p90 was 2.242 / 0.418 ms
for M=3 and 0.906 / 0.315 ms for M=1. These are go/no-go microbenchmarks, not
end-to-end claims. Earlier prototypes used `ensure_row_contiguous=True`, which
copied the full KV capacity slice on every layer and invalidated the indexed
read advantage. The final pass consumes MLX-provided input strides directly.

## Served dogfood and why it is not a promotion receipt

Artifact:
`rapid-mlx/Qwen3.8-Flash-Next-4bit` at immutable revision
`dcf657e4acda2aae72da99cde65b6c491cd96998`; MLX 0.32.2; MLX-LM 0.31.3;
fixed native MTP k=1; temperature zero; 16,348 reported prompt tokens.

The first candidate request constructed and executed the indexed route and
completed 64 output tokens: TTFT 22.112 s, decode 23.58 tok/s, peak RSS 55.72
GiB. It was a cold JIT run on the unqualified M=2 route and preceded the
stride-aware no-copy fix, so it is a debugging receipt rather than performance
evidence for the final code. The first baseline was
34.29 tok/s; later baseline samples were 18.33 and 35.95 tok/s while a desktop
model service repeatedly restarted. Multiple fresh-process loads then failed
in weight `mx.eval` with a Metal GPU timeout before the candidate kernel could
run. Those failures are host-state failures, not kernel failures, but the
resulting samples are too contaminated to compare. Artifacts:

- `/private/tmp/qsa-indexed-candidate-k1-16k.json`
- `/private/tmp/qsa-indexed-baseline-k1-16k.json`
- `/private/tmp/qsa-indexed-baseline-k1-settled-16k.json`

## Promotion gate disposition

The follow-up requalification completed the clean Metal session, three samples
per K=1/K=2 arm at 16K/32K/64K, real indexed-route construction, rollback tests,
same-seed coherent-output comparison, the 45-case release battery, sampled
decoding, and cancellation recovery. The strict 64K comparison kept indexed
QSA enabled for both arms and measured K=2 at 33.72 tok/s versus K=1 at 29.23
tok/s (+15.4%). See
`2026-09-05-qwen38-k2-indexed-requalification.md` for the complete evidence and
the shorter-context non-win.

The MLX version and Metal architecture pins remain mandatory. Production tensor
captures must be rerun before broadening either pin or admitting M=2.
