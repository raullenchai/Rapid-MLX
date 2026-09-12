# Vector to Atlas: GLM-5.3 native-MTP dependency decision

## Receiving role

Atlas (architecture, dependency compatibility, and release disposition).

## Current branch

Original compatibility work: `vector/glm53-native-mtp`. Follow-up qualification:
`vector/glm53-adaptive-mtp`, rebased onto `origin/main@f18255fb2`.

## Verified facts

- Upstream mlx-vlm PR #2127 reports 29.48 to 43.67 tok/s (1.481x) at
  batch one, 512-token context, 90.5% MTP acceptance, exact parity, M3 Ultra.
- The optimized GLM rewrite is after the v0.7.0 tag. PyPI 0.7.0 does not contain
  it; pinning `mlx-vlm==0.7.0` would not absorb the reported path.
- Rapid's old GLM runtime overlay crashed on that rewrite because
  `Glm5NextSparseAttention` no longer exists.
- This branch makes the overlay self-retire only when the complete native GLM
  class family is present. Pinned-runtime tests pass (33), and the post-v0.7
  checkout strictly loads the cached five-layer 0.1B GLM fixture through both
  mlx-vlm and Rapid's MLLM wrapper (with only the version gate bypassed).
- Rapid's previous full-model K=1 probe was a no-go: 31.94 to 31.59 tok/s,
  72.97% acceptance, +5.71 GiB. The upstream result uses a substantially newer
  transactional/batched verifier and native block size three.
- The current target revision is `76add2a341a1cd90ad0e86bb69839ea9c35827c6`;
  the benchmarked `06d6a...` revision no longer resolves on the Hub. Layer 45
  now spans source shards 1 and 2. The upstream splitter produced a strict-load
  3.9 GiB q4-g64 drafter. Its real one-token block measured 1.541 ms median and
  3.896 GiB active memory. Including the checkpoint's q4 output head and argmax
  over 154,880 tokens measured 2.141 ms median, so the old K=1 regression is
  not explained by an expensive draft proposal alone.
- The full target is now cached. Same-artifact 8K prefill medians were Rapid
  351.627 tok/s, native-extension oMLX 342.754, and post-0.7 mlx-vlm 337.128.
  The public oMLX 449.6 tok/s result is primarily a different oQ4e checkpoint,
  not a runtime advantage on Rapid's uniform-q4 artifact.
- A full-model Rapid multi-boundary rollback spike ran K=1/2/3. K=1 was
  31.431 to 33.453 tok/s (1.064x, 76.39% acceptance), K=2 was 33.154 tok/s
  (1.055x, 50.79%), and K=3 was 30.329 to 26.760 tok/s (0.882x, 32.31%).
  All depths first diverged from serial greedy output at token 109. The spike
  was discarded and the alias remains fail-closed.

## Unresolved

- Choose between waiting for the next tagged mlx-vlm release and vendoring the
  GLM-specific drafter/verifier. The upstream PR touches a broad cache,
  quantized-verifier, model, and speculative-runtime surface.
- Qualify `dfp-official/GLM-5.3-Flash-oQ4e-mtp` once the policy-controlled HF
  cache has capacity. It had 77 GiB free during this campaign versus the
  repository's approximately 182 GB size; no download was attempted.
- Decide whether a future proven path is experimental opt-in or default-on.

## Risks

- Accepting the untagged source by its reported `0.7.0` version is not
  reproducible.
- Enabling MTP from acceptance alone repeats the old K=1 failure; verifier cost
  and workload-by-workload throughput are release gates.
- The third-party DFlash2 checkpoint is non-commercial by default and does not
  have benchmark evidence sufficient for Rapid's sustained gate.

## Recommended next action

Keep native MTP disabled. Wait for a tagged post-v0.7 mlx-vlm artifact and
policy-compliant oQ4e cache capacity, then re-run the exact workload gate in
`docs/engineering/performance/2026-09-12-glm53-flash-next-tier.md` before any
alias capability or dependency change.
