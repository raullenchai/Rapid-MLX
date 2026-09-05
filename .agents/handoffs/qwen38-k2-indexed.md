# Qwen3.8 K=2 indexed-QSA requalification handoff

Receiving role: Atlas

Owner: Vector

Host: Mac Studio, Apple M3 Ultra 256 GB

Branch: `vector/qwen38-mtp-k2-indexed`

PR: none; this branch is not ready for production validation

## Verified facts

- The branch is stacked on indexed split-K spike `f620f0d84`, itself based on
  PR #3055 head `8ef208607`.
- Qwen3.8 keeps implicit MTP at K=1, admits only explicit K=2, and rejects K=3.
- Existing generic chain-of-K and Qwen-specific rollback logic is unchanged.
- Focused MTP, Qwen4, indexed-QSA, and CLI suites pass; changed Python files
  pass Ruff.
- On the released Qwen3.8 Flash-Next 4-bit artifact, three isolated 32K K=2
  samples had a 35.64 tok/s median. The only clean same-code K=1 comparator was
  34.29 tok/s, a provisional +3.9% signal.
- A clean 64K K=2 request reached 37.80 tok/s, so the old gathered-K/V K=2
  long-context collapse was not reproduced.
- Full commands, environment, exclusions, and raw result summary are recorded
  in `docs/engineering/performance/2026-09-05-qwen38-k2-indexed-requalification.md`.

## Unresolved questions and risks

- Host contention left only one clean K=1 comparator and one clean 64K K=2
  sample. The measured delta is not yet statistically actionable.
- Real K=1 and K=2 greedy captures can diverge at near-tied logits because the
  target block uses a different Metal accumulation shape. Both continuations
  were coherent, but the release correctness battery has not been rerun.
- The local chunked model-load launcher worked around a pre-inference Metal
  watchdog timeout. It is not part of the branch and must not become an
  undocumented production dependency.
- Atlas must decide whether explicit-only K=2 is a supported public surface or
  remains experimental after performance qualification.

## Next concrete action

Reserve an isolated M3 Ultra window, collect at least three fresh-process K=1
and K=2 samples at 16K, 32K, and 64K, then run the 45-case Flash-Next release
battery plus cancellation, EOS, and sampled-decoding coverage. Open and enqueue
a production PR only if K=2 wins beyond run-to-run noise and correctness passes.
