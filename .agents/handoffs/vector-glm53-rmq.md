# Vector handoff: GLM-5.3 RMQ

- Receiving role: Atlas
- Branch: `vector/glm53-rmq-mvp`
- PR: #3372
- Host: Studio (M3 Ultra, 256 GB)

## Verified

- RMQ planner emits fusion-safe MLX affine Q4/Q8 group-64 decisions.
- Official GLM E4M3 128x128 block-FP8 is supported tensor-by-tensor.
- Real official layer-45 dequantization and Q4 output match mlx-vlm exactly.
- Unit tests pass (14/14); ruff and diff checks pass.
- Full-plan payload projects to 186,198,375,288 bytes, 4.20 GiB above the
  current uniform-Q4 payload.
- Uniform-Q4 MTP reached 37.15 tok/s versus 28.63 tok/s autoregressive on the
  first 128-token probe (+29.8%), with identical greedy output.
- A standalone higher-precision MTP did not improve acceptance against a Q4
  target. Target/draft precision symmetry is now an explicit design rule.

## Remaining and risks

- The complete official 62-shard source cannot fit the currently available HF
  cache. Do not redirect the cache or delete other models to bypass policy.
- The tested GLM MTP runtime is mlx-vlm post-0.7 commit `d2a1434a...`; Rapid is
  pinned to 0.6.17. Product integration needs a tagged dependency or an Atlas
  decision to vendor it.
- Tiny-fixture quality results validate numerics, not full-model intelligence.
- Short-prompt prefill cost regresses when a separate drafter is loaded.
- The two current E2E probes are decode diagnostics, not a realistic quality
  suite. Coding execution, knowledge answer keys, instruction checks, blind
  creative-writing review, and long-context tasks remain mandatory.

## Next action

Atlas should review the format/product boundary. Once source capacity and a
tagged GLM MTP runtime are available, Vector should emit the full target and
target-matched sidecar, then run the release gate recorded in
`docs/engineering/performance/2026-09-12-glm53-rmq-mvp.md`.
