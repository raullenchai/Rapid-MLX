# Vector to Atlas: APCv2 host prompt plane

- Owner: Vector
- Receiving role: Atlas
- Host: Studio (M3 Ultra, 256 GB)
- Branch: `vector/apcv2-segments`
- PR: `#3446`

## Verified facts

- Exact chat render and tokenization results now share one bounded host LRU
  between `BatchedEngine` and the text `Scheduler`.
- The tokenizer remains on the existing MLX worker thread; no long-prompt work
  moved onto the asyncio event loop.
- Cache limits are 64 entries / 64 MiB; manual prefix-cache clear and unload
  clear it, and
  `RAPID_MLX_PROMPT_HOST_CACHE=0` is a rollback switch.
- A cached Qwen3 tokenizer benchmark measured 18.67x to 30.67x lower repeated
  host preparation latency across 719 to 21,708 prompt tokens with exact
  string/token equality.
- The generic device-side B1-to-B2 broadcast idea was rejected: at 65,536
  tokens it ran at 0.46x physical-B2 speed and allocated 134.8 MB over baseline.
- Focused engine/server/scheduler/template tests pass (437/437). After the
  declared optional test extras were installed, full unit passed 24,438 tests.
- `pr_validate` returned `MERGE-SAFE`: targeted 224 passed, supply-chain and
  lint gates passed. Its first run's Apple Silicon stress matrix passed four
  model families across three integration surfaces; the repeated run skipped
  stress and reused that receipt. Advisory diff coverage timed out and skipped
  without affecting the verdict.

## Risks and unresolved questions

- The feature accelerates exact repeated host inputs; it does not incrementally
  tokenize a changed multi-turn suffix.
- Static Qwen4 MTP cohort batching remains a larger cross-cutting port and needs
  a real baseline-vs-candidate server throughput receipt before product work.
- Atlas should review the additive stats surface and default-on/rollback policy
  as part of release integration.

## Next concrete action

Atlas should merge/queue PR #3446 after hosted CI is green. After merge,
benchmark static Qwen4 cohort batching as a separate spike rather than
extending this branch.
