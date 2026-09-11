# Vector handoff: DeepSeek V4.1 exact affine-2bit MoE kernel

- Owner: Vector
- Host: Studio (M3 Ultra, 256 GiB unified memory)
- Branch: `vector/deepseek-v41-moe-megakernel`
- Base: `vector/deepseek-v41-speed-20tps` / PR #3307
- Worktree: `/private/tmp/rapid-mlx-deepseek-v41-moe-megakernel`
- Status: exact target optimization qualified; combined DSpark rerun pending

## Intention and boundary

Reduce the target verification bottleneck with an exact affine 2-bit expert
kernel. Keep this PR to benchmark tooling, correctness tests, and reproducible
performance evidence. Server, catalog, GUI, model conversion, downloads, and
default runtime changes are explicit non-goals until the complete path passes
the 12 tok/s product floor.

## Verified facts

- The route-direct kernel specializes only FP32 activation into affine 2-bit,
  group-64 expert down projections with 1--36 routed rows.
- Real layer 20 is element-for-element equal to the stock projection for 1, 6,
  24, 30, and 36 routes.
- Layer speedups are 2.65x at six routes, 1.94x at 24, 1.82x at 30, and 1.72x
  at 36.
- Two real 32-token target-only runs reach 9.66--9.83 tok/s from 7.90--7.92
  tok/s (+22.2% to +24.0%), preserve the exact token sequence, and leave peak
  MLX memory unchanged at 213.5387 GB.
- Target oracle throughput reaches 17.91--18.13/25.38--25.49/32.39--33.86/
  34.49--36.32 rows/s at K2--K5.
- The prior trusted checkpoint runtime was ephemeral and is no longer present.
  Based on its earlier timing split, K4 is estimated at 14--15 tok/s with this
  kernel, but the combined result is not measured and must not be advertised.

## Reference check (internal only)

- vLLM and SGLang were checked first. Their current V4.1 speculative paths use
  dedicated draft networks, batched/ragged verification, and fused target MoE
  execution; neither exposes a drop-in MLX kernel for this artifact.
- Current oMLX was checked next. It has V4.1 DSpark and adaptive acceptance, but
  its converted tensor namespace is incompatible with the existing 212.93 GB
  Rapid artifact. Rewriting another full copy would exceed the scoped storage
  and time budget.
- Its block-shaped affine kernels use row tiles that are inefficient when
  24--36 routes are spread across 336 experts. This experiment instead assigns
  one QMV grid to each real route and preserves stock arithmetic order.

## Next concrete action

Recover or reconstruct a trusted DSpark adapter without copying the target
artifact, then rerun K4 end to end with `--direct-down-qmv`. Product integration
requires measured throughput of at least 12 tok/s plus multi-domain 128-token
quality and long-context cache qualification.
