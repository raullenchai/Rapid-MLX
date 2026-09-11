# Vector handoff — DeepSeek V4.1 drafter precision experiment

## 2026-09-11 — starting

- Owner: Vector on Studio. Branch `vector/deepseek-v41-drafter-precision`,
  worktree `/private/tmp/rapid-mlx-deepseek-v41-drafter-precision`, based on PR
  #3325 exact head `9abd21d9d` because the experiment uses that candidate's
  product-owned DSpark K4 runtime and benchmark contract.
- Intention: determine whether the official source-precision DeepSeek V4.1
  three-stage DSpark tensors increase accepted draft length and end-to-end TPS
  relative to the currently pinned affine 2-bit head, then derive the smallest
  mixed-precision head justified by measured quality and memory. Success is a
  reproducible multi-domain gain with target-authoritative output, not drafter
  microbenchmark speed alone.
- Scope: source-head extraction tooling, immutable artifact manifests, isolated
  A/B benchmarks, memory accounting, and internal performance evidence. Explicit
  non-goals: training, changing the released/default model or K, catalog/GUI/API
  changes, target-model requantization, release, deployment, and deleting other
  cached models.
- Storage constraint: the enforced Hugging Face cache currently has about 5.1
  GiB free while the three official source shards total about 7.4 GiB. Downloads
  must use the global cache and stop on quota failure. Extraction may stream one
  task-owned shard at a time into the allowed warm-tier model directory, but
  must not remove unrelated cache entries or redirect Hub downloads.
- Verification: fingerprint every source revision/shard/tensor; compare 2-bit,
  source precision, and any selected mixed-precision candidate on code,
  reasoning, structured output, Chinese, and longer-context prompts; report
  accepted tokens per block, K2-K6 profitability, end-to-end decode TPS,
  target-authority/parity, peak MLX memory, and failure/regression rows. Run
  focused tests, scope-locked adversarial self-review, and PR validation only if
  a shippable artifact/tooling change survives the performance gate.
- Private precedent check: current vLLM DeepSeek V4.1 DSpark keeps a dedicated
  three-stage model and gates adaptive verification on confidence-head support;
  current SGLang DSpark separates draft, verification-window planning,
  confidence/accept estimation, and calibration. MLX-LM and oMLX expose no
  directly reusable V4.1 higher-precision sidecar path. This experiment retains
  Rapid's existing lossless target-authoritative verifier and changes only
  drafter weights; controller/calibration work remains a follow-up.
- Start FYI for Atlas, Pixel, Harbor, and Echo could not be delivered through
  Orca because this shell has no stable Orca pane identity. This tracked handoff
  records the required fallback; implementation continues because the FYI is
  non-blocking.

## 2026-09-11 — experiment complete and experimental sidecar published

- Draft PR: https://github.com/raullenchai/Rapid-MLX/pull/3328 at commit
  `0e9b2abf2` (dependent on #3325).
- Extracted the revision-pinned official DSpark tensors to affine 4-bit, then
  composed a 4-bit-dense/2-bit-routed-expert candidate. The mixed artifact is
  4,617,792,648 bytes, 157,710,560 bytes larger than the current 2-bit head.
  It passes strict runtime loading and peaks at 218.232 GB with the target.
- A uniform affine 4-bit head is 8,015,180,738 bytes. Two load attempts were
  killed at target/head residency, so it is rejected for the 256 GiB boundary.
- Mixed K4, four domains x two 128-token repeats: 19.390 tok/s weighted versus
  9.580 AR (2.024x), 1.850 accepted tokens/block, and 4/4 repeat-stable outputs.
  This is 80.9% above the previous head's conservative 10.72 tok/s suite result.
- Mixed K5: 18.488 tok/s versus 9.565 AR (1.933x), 2.136 accepted/block, and 4/4
  repeat-stable. Extra verification cost outweighs acceptance, so K4 remains
  the measured fixed-K sweet spot and no product default changes here.
- The target-authoritative fixed-batch output contract is retained, but none of
  the four K4 streams is bitwise equal to sequential AR because target batch
  shape crosses low-margin numerical decisions. Manual task inspection clears
  only the Chinese sample; code, reasoning, and structured samples are poor or
  repetitive. Similar defects exist in AR/current-head probes, so the speed
  experiment passes but publication is explicitly gated on a broader comparative
  task-quality qualification.
- Added a fail-closed offline extractor/composer with source/output hashes and
  immutable precision metadata, extended the suite to bounded K2-K6 experiments,
  and recorded full reproducibility evidence in
  `docs/engineering/performance/2026-09-11-deepseek-v41-dspark-mixed-precision.md`.
  Focused result: 27 tests pass. The complete 12-file DeepSeek/DSpark regression
  set passes 128/128; ruff and `git diff --check` pass.
- Adversarial self-review fixes: rejected mislabeled resume outputs instead of
  inferring their precision, validated source weight-map structure and state
  filenames, enforced the qualified 2-bit/4-bit pair, added config/index hashes,
  kept target-shared embed/head explicitly 2-bit, and removed the K4-hardcoded
  stability error.
- The owner explicitly expanded scope after reviewing the evidence. Published
  the head-only public repository
  `rapid-mlx/DeepSeek-V4.1-Flash-DSpark-4d2e-MLX` at immutable revision
  `9530d6d2bf59e0d05177bd538095d5704ded1488`; it contains no target weights.
  The Model Card states the required target/revision, K4 recommendation,
  measured approximately-20-tok/s boundary, memory floor, non-standalone
  behavior, deterministic target-authoritative contract, and quality caveat.
- Rapid's artifact contract now pins that repo/revision plus the exact sizes
  and SHA-256 values for config, index, manifest, and all three stage shards.
  User documentation now reports the mixed-head result instead of the replaced
  2-bit head result.
- Next concrete action: compare mixed/current/AR task correctness on a larger,
  scored prompt set before removing the experimental label or widening
  admission. Publication does not make this an unconditional default.
- Completion FYI is recorded here because Orca messaging remains unavailable
  from this shell. Receiving roles: Atlas (publication decision), Harbor
  (artifact/rollout integrity), Echo (dependent PR tracking), Pixel (awareness;
  no GUI impact).
