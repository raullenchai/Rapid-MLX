# Qwen4 indexed split-K QSA spike (not promoted)

Date: 2026-09-05

Owner: Vector

Host: Mac Studio, M3 Ultra 256 GB, `applegpu_g15d`

Branch: `vector/qsa-indexed-splitk`, stacked on PR #3055 head
`8ef208607ececfaf9646da18575a1c7b95f7f4da`

## Outcome

The direct compact-index Metal path is promising for the geometry that the
source handoff qualified, but it is **not ready to promote or queue** in Rapid.
The handoff's end-to-end result is for self-MTP k=2 (target verify width M=3),
while the current Qwen3.8 Flash-Next native-MTP runtime rejects k=2 and supports
k=1 only. Its target verify width is M=2, which was not qualified by the source
experiment. The production gate therefore admits only M=3 from 16K and M=1
from 64K, remains opt-in, requires batch size one, and pins both MLX 0.32.2 and
the measured M3 Ultra Metal architecture.

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
changed 9,261 bf16 outputs, so it was rejected. Additional tests cover opt-in and
version/architecture gates, malformed int32 indices, empty selections, route
priority/receipts, and graph identity when both sparse routes are disabled.

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

## Reopen / promotion gate

1. Enable and independently qualify Qwen3.8 Flash-Next native MTP k=2, or run
   a real 64K M=1 decode campaign.
2. Start from a clean Metal session with no other model server resident.
3. Warm the new pipelines before timing, then collect at least three samples
   per arm at 16K, 32K, and 64K with identical prompts and seeds.
4. Require nonzero indexed route receipts, zero unexpected declines/fallbacks,
   successful state rollback transactions, and a same-seed output comparison.
5. Re-pin the exact MLX build and rerun production tensor captures on every
   MLX-core bump.

Until all five conditions pass, do not open/queue a performance PR and do not
enable this route by default.
