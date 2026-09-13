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
- A subsequent qualification of mlx-vlm #2206 + #2231 + #2234 on the exact
  same target supersedes that legacy-injector rejection: 12/12 repeated MTP
  tasks passed, every reasoning/final byte matched a 6/6 same-branch AR
  control, median paired throughput improved 1.221x, and median category
  throughput reached 33.655 tok/s versus 31.813 tok/s for the same-width oMLX
  control. Peak Metal was 188.499 GB versus 184.147 GB for AR.
- #2234 was subsequently closed without merge. A rebuilt candidate containing
  current mlx-vlm main + #2206 + #2231, but no #2234, retained 6/6 task and
  complete reasoning/final byte parity against its 6/6 AR control. Clean-run
  per-task gains were 1.264x, 1.234x, 1.187x, 1.323x, 1.194x, and 1.070x
  (1.214x median); median category throughput was 32.887 tok/s, 3.4% above the
  same-width oMLX control. Peak Metal was 188.432 GB versus 184.141 GB for AR.
  Two timing-contaminated repeats were excluded, but retained 12/12 exact task
  and byte parity. #2234 contributed about another 2.8% in its first clean run
  and is optional rather than a release dependency.
- A same-load K=3 spike remained exact and improved median task throughput
  1.030x over K=2, but regressed coding and creative writing by 3.6-3.7% while
  improving instruction and knowledge by 7.4-8.6%. Keep K=2 as the qualified
  default; treat adaptive depth as a separate follow-up.

## Unresolved

- Wait for #2206 and #2231 to merge and appear in a tagged mlx-vlm release. Do
  not vendor only part of the cache transaction. #2234 is closed and is not a
  dependency of the qualified path.
- Connect that released path to Rapid's GLM serving lane and repeat the six-task
  gate through the Rapid server; direct mlx-vlm qualification is not sufficient
  evidence that the product integration preserves the gain.
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

Keep the legacy injector disabled. Once an mlx-vlm release contains #2206 and
#2231, update Rapid's pin, connect the cache-owned path, and run the
exact six-task server gate in
`docs/engineering/performance/2026-09-12-glm53-real-task-mtp.md` before any
alias capability change. Treat oQ4e as a separate quantization-quality and
prefill-throughput follow-up, not the blocker for uniform-Q4 MTP.
