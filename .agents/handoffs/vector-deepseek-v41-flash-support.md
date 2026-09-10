# Vector handoff — DeepSeek V4.1 Flash performance qualification

## PR-start FYI

- Owner/host: Vector, Studio
- Branch/worktree: `vector/deepseek-v41-flash-support`,
  `/private/tmp/rapid-mlx-deepseek-v41-flash-support`
- User-visible goal: determine whether DeepSeek V4.1 Flash has a credible path
  to at least 20 tok/s on a 256 GiB Mac before exposing unsupported catalog or
  product claims.
- Scope: official artifact/config qualification, a reproducible synthetic MLX
  MoE scheduling benchmark, quantization layout comparison, and a bounded next
  real-weight gate.
- Non-goals: inventing an alias for an unpublished checkpoint, cloud-provider
  integration, default-model changes, unrelated V4 performance work, or silent
  fallback from native vision to text-only behavior.
- Constraints: single standard Hugging Face cache under the Studio storage
  policy; no cache relocation or deletion; no production/release operation.
- Verification: official-source and architecture check, existing-runtime
  precedent audit, config-level compatibility tests before any large download,
  real Server/Desktop path dogfood if runnable weights exist, self-adversarial
  review, focused/full tests, and PR validation.

The Orca agent messaging channel is unavailable in this session. This tracked
handoff carries the equivalent non-blocking start FYI for Atlas, Pixel, Harbor,
Echo, and ds0731.

## PR-complete FYI

- Outcome: experiment complete; do not add catalog/runtime/Desktop support yet.
- Storage blocker: the 222.4 GiB community 2-bit checkpoint is absent and the
  policy-managed Hugging Face cache has only about 80 GiB free. No shards were
  downloaded, no models were deleted, and no cache was redirected.
- Evidence: on M3 Ultra, 256 GiB, 2-bit/group-64 synthetic published shapes,
  pre-packed active-expert batching reduced one-layer MoE time from 0.601 ms to
  0.454 ms. Stacking separately stored experts per call took 0.796 ms and was
  slower than the serial path. Removing 40 layer syncs reduced the serial
  synthetic MoE graph from 24.18 ms to 11.36 ms.
- Guardrail: these are MoE-only synthetic measurements and not end-to-end model
  throughput claims. Full results and excluded work are documented in
  `docs/engineering/performance/2026-09-10-deepseek-v41-flash-moe-experiment.md`.
- Verification: benchmark outputs matched bit-for-bit; Ruff passed; 33 benchmark
  metadata tests passed; diff check passed.
- Prepared gate: `scripts/bench_deepseek_v41_runtime.py` can run the unchanged
  local checkpoint, capture continuous warm decode latency, existing eval-barrier
  waits, Engram/disk counters, and memory. It refuses
  downloads and checkpoint Python execution is explicit opt-in. Its focused and
  metadata suites pass 42 tests, including a tiny local-runtime end-to-end run.
- Safety finding: whole-process Metal capture is forbidden for this model. A
  one-token capture grew to about 112 GiB before it was terminated and removed;
  system free space recovered fully. Use isolated layer/kernel benchmarks.
- Next owner/action: Vector, when at least 223 GiB of policy-compliant cache
  capacity exists, run the unchanged real checkpoint with per-layer profiling,
  then validate a one-layer expert-major conversion and batched kernel.
