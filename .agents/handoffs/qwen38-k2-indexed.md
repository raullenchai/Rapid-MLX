# Qwen3.8 K=2 indexed-QSA requalification handoff

Receiving role: Atlas

Owner: Vector

Host: Mac Studio, Apple M3 Ultra 256 GB

Branch: `vector/qwen38-mtp-k2-indexed`

PR: #3087 (Draft)

## Verified facts

- PR #3087 is a self-contained indexed-QSA plus K=2 change rebased onto current
  `main`; prerequisite PR #3055 is merged.
- Qwen3.8 keeps implicit MTP at K=1, admits only explicit K=2, and rejects K=3.
- Existing generic chain-of-K and Qwen-specific rollback logic is unchanged.
- Focused MTP, Qwen4, indexed-QSA, and CLI suites pass (357 tests); changed
  Python files pass Ruff.
- Adversarial review found that the initial gate admitted unmeasured tensor and
  selection geometries. The fixed gate now admits only the dogfooded BF16
  QH=24, KVH=2, D=256, block-size-4, top-K-512 layout; rejection tests cover
  every pinned field.
- A later Apple CI run showed that M1/M2 are numerically close but not
  bit-identical to the M3 Ultra/native reduction. The production gate was
  already M3-Ultra-only; the FP64 numerical oracle remains cross-architecture,
  while bit-exactness is now asserted only on the qualified host and MLX
  version. The follow-up adversarial review returned LGTM.
- A fresh-process isolated A/B collected three accepted samples per arm. K=2
  was -7.1% at 16K, -0.7% at 32K, and +15.4% at 64K. Every 64K K=2 sample beat
  every K=1 sample.
- The final 64K K=1 comparator also enabled indexed QSA, so the result isolates
  speculative depth from the M=1 kernel toggle. Explicit K=2 now activates the
  qualified kernel automatically unless the operator explicitly opts out.
- The old gathered-K/V K=2 long-context collapse was not reproduced. The value
  is specifically an ultra-long-context crossover, not a universal speedup.
- The K=2 real-model release battery retained the established K=1 vector at
  42/45 with exactly the same three known failures and no new regressions.
- Real-service sampled decoding completed twice at temperature 0.8/top-p 0.9.
  A client-disconnected stream was removed from the running batch, and the
  immediate recovery request returned the expected output.
- Full commands, environment, exclusions, and raw result summary are recorded
  in `docs/engineering/performance/2026-09-05-qwen38-k2-indexed-requalification.md`.

## Unresolved questions and risks

- K=2 regresses 16K throughput and is only neutral at 32K. It must remain an
  explicit operator choice; these results do not justify a default change.
- Real K=1 and K=2 greedy captures can diverge at near-tied logits because the
  target block uses a different Metal accumulation shape. Both continuations
  were coherent. The release battery, sampled decoding, and cancellation
  recovery found no new failure.
- The local chunked model-load launcher worked around a pre-inference Metal
  watchdog timeout. It is not part of the branch and must not become an
  undocumented production dependency.
- Atlas must decide whether explicit-only K=2 is a supported public surface or
  remains experimental after performance qualification.
- Full local `pr_validate` reached 22,604 passes but reported seven Bonsai tests
  failing on an installed `TilingConfig` signature and a DiffusionGemma golden
  checkpoint that now identifies as the old `diffusion` family. Both failures
  reproduce unchanged from `origin/main` (`c0e09b560`) in clean worktrees and
  are outside this PR. Qwen3.5/Qwen3.6 stress A/B showed -0.1%/-0.5% warm
  deltas, classified by the validator as not this PR.

## Next concrete action

Complete the scoped validation rerun and wait for the final GitHub Apple lane.
With explicit human authorization already given, apply the Mac-required queue
label only after every current-head relevant check is green.
