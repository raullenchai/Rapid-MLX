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
  policy; delete only caches explicitly authorized by the administrator; no
  cache relocation and no production/release operation.
- Verification: official-source and architecture check, existing-runtime
  precedent audit, config-level compatibility tests before any large download,
  real Server/Desktop path dogfood if runnable weights exist, self-adversarial
  review, focused/full tests, and PR validation.

The Orca agent messaging channel is unavailable in this session. This tracked
handoff carries the equivalent non-blocking start FYI for Atlas, Pixel, Harbor,
Echo, and ds0731.

## PR-complete FYI

- Outcome: real-weight qualification complete; do not add catalog/runtime or
  Desktop support yet because 20+ tok/s is not demonstrated.
- Artifact: all 238,796,133,496 indexed bytes, 48 shards, and 143,982 tensors
  are present in the standard cache at revision
  `802f1a00982705d81b79ad1c83aa0ccc0b863ebc`; `hf cache verify` checked all 64
  files. No cache was redirected.
- Baseline: M3 Ultra 60-core GPU, 256 GiB, MLX 0.32.2, resident text backbone.
  The unchanged 32-warmup/128-measure control reached 6.055 tok/s and 165.96 ms
  median latency. Resident weights occupy about 173.5 GB active memory.
- Safe candidate: expert-local gate/up fusion is bit-identical in the focused
  quantized test, preserves the greedy chain, and reaches 6.216 tok/s (+2.65%)
  with 160.55 ms median latency (-3.26%). Packing takes 4.76 seconds without a
  memory spike.
- Rejected candidate: all-expert `gather_qmm` is 2.02x faster for one isolated
  real layer but collapses to 0.436 tok/s at 40 layers because 4.25 GB layer
  buffers destroy full-model Metal memory locality.
- Quality-blocked candidate: fused pipelined mHC reaches 8.030 tok/s (+32.6%)
  but only 43.75% top-1 agreement over 16 teacher-forced tokens, with 15.80
  maximum logit error after 40-layer amplification. Keep benchmark-only.
- MTP: the bundled three-stage DSpark verifier advances target tokens serially
  and is documented as slower. The next material gate is parallel target block
  verification with commit-on-accepted-prefix KV semantics.
- Guardrail: these are MoE-only synthetic measurements and not end-to-end model
  throughput claims unless explicitly labeled as real-weight results. Full
  results and excluded work are documented in
  `docs/engineering/performance/2026-09-10-deepseek-v41-flash-moe-experiment.md`.
- Verification: focused quantized fusion output is bit-identical; Ruff and 43
  profiler/benchmark-metadata tests pass; diff check passes.
- Prepared gate: `scripts/bench_deepseek_v41_runtime.py` can run the unchanged
  local checkpoint, capture continuous warm decode latency, existing eval-barrier
  waits, Engram/disk counters, memory, and explicit experimental HC/MoE
  overrides. It refuses downloads and checkpoint Python execution is explicit
  opt-in.
- Safety finding: whole-process Metal capture is forbidden for this model. A
  one-token capture grew to about 112 GiB before it was terminated and removed;
  system free space recovered fully. Use isolated layer/kernel benchmarks.
- Next owner/action: Vector, design a bounded multi-token target forward for
  DSpark block verification. Catalog/GUI/server work remains out of scope until
  throughput and quality gates pass.
