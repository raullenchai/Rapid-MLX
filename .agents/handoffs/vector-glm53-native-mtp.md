# Vector to Atlas: GLM-5.3 native-MTP dependency decision

## Receiving role

Atlas (architecture, dependency compatibility, and release disposition).

## Current branch

`vector/glm53-native-mtp`, based on `origin/main@233439e88`.

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

## Unresolved

- Choose between waiting for the next tagged mlx-vlm release and vendoring the
  GLM-specific drafter/verifier. The upstream PR touches a broad cache,
  quantized-verifier, model, and speculative-runtime surface.
- Re-run the full 181.7 GB Rapid target. It is absent from the HF cache; Studio
  has about 65 GiB free and Mini about 104 GiB free, so policy-compliant restore
  is currently blocked without storage capacity becoming available.
- Decide whether a future proven path is experimental opt-in or default-on.

## Risks

- Accepting the untagged source by its reported `0.7.0` version is not
  reproducible.
- Enabling MTP from acceptance alone repeats the old K=1 failure; verifier cost
  and workload-by-workload throughput are release gates.
- The third-party DFlash2 checkpoint is non-commercial by default and does not
  have benchmark evidence sufficient for Rapid's sustained gate.

## Recommended next action

Approve the fail-closed compatibility seam independently. Wait for a tagged
post-v0.7 mlx-vlm artifact, then run the exact paired K=0/1/2/3 server campaign
defined in
`docs/engineering/performance/2026-09-12-glm53-flash-next-tier.md` before any
alias capability or dependency change.
