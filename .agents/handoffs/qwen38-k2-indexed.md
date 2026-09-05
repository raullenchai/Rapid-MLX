# Qwen3.8 K=2 indexed-QSA requalification handoff

Receiving role: Atlas

Owner: Vector

Host: Mac Studio, Apple M3 Ultra 256 GB

Branch: `vector/qwen38-mtp-k2-indexed`

PR: #3087 (Draft)

## Verified facts

- The branch is stacked on indexed split-K spike `f620f0d84`, itself based on
  PR #3055 head `8ef208607`.
- Qwen3.8 keeps implicit MTP at K=1, admits only explicit K=2, and rejects K=3.
- Existing generic chain-of-K and Qwen-specific rollback logic is unchanged.
- Focused MTP, Qwen4, indexed-QSA, and CLI suites pass; changed Python files
  pass Ruff.
- A fresh-process isolated A/B collected three accepted samples per arm. K=2
  was -7.1% at 16K, -0.7% at 32K, and +16.0% at 64K. Every 64K K=2 sample beat
  every K=1 sample.
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

## Next concrete action

Complete the independent review loop and repository PR validation. Atlas should
approve the explicit-only 64K-oriented contract before #3087 enters the merge
queue.
