# DeepSeek V4.1 draft depth and timing

`RAPID_MLX_DSV41_DRAFT_DEPTH` accepts `0`, `1`, `2`, `3`, `4`, or
`auto`. The default is `4`; invalid values log a warning and use `4`.
The setting is read at request start. Fixed depths are explicit overrides:
`0` skips draft execution and context projection, and `4` retains the previous
seed-plus-four verification width. All positive depths retain all five draft
decoder positions, including their noise embeddings. Only the consumed Markov
predictions are generated. The serving path skips confidence projection.

Auto mode owns its acceptance and cost estimates per request. It maximizes
`(1 + a + ... + a**K) / (verify_ms[K] + draft_ms)` for positive K, comparing
against `1 / verify_ms[0]`. Accepted predictions and the first rejection update
an exponential estimate with weight 0.25. Unchecked suffixes do not count as
failures. The initial estimate is 0.5. Switching positive depths requires a 10%
gain; any proposed positive depth must beat target-only by 5%. Unprofitable
drafting is disabled immediately. Every 32 target-only rounds, a probe uses
the largest depth permitted by the remaining output budget and cache rings.
The probe refreshes acceptance and measured costs even while drafting is disabled.

## Cost provenance and online learning

The initial verify priors are the current measured v2 curve on M3 Ultra:
synchronized end-to-end verification with a 40-token context. Rows include the
seed plus K draft predictions, so `VERIFY_MS[K]` uses K + 1 rows.

| Verify rows | Milliseconds |
| --- | --- |
| 1 | 28.7 |
| 2 | 46.2 |
| 3 | 58.0 |
| 4 | 69.5 |
| 5 | 81.7 |

The draft prior remains 21 ms from the pre-optimization M3 Ultra measurement;
it does not assume the unverified 5 ms optimization target. These constants
are hardware/workload priors only. Real costs are learned online, separately
for each depth and request. The first measurement replaces its prior; later
measurements update an EMA with weight 0.25. `rate()` uses measured costs when
available and the priors for depths or phases not yet measured.

Phase costs may be learned separately only when existing host sync points
already bound completed work. Serving submits draft work asynchronously and
completes both draft and verify at the existing verify `mx.eval`. It therefore
records the whole round from before draft submission through that boundary,
including deferred work completed there, and attributes the total to the depth
actually used (including K=0 and budget-limited depths). This learned total
replaces both phase priors: nothing is subtracted, and no draft prior is added
again. No `mx.eval` or synchronization is added for timing. Time spent yielding
tokens to the caller is outside this interval.

For independent Metal recalibration, prefill a representative prompt once,
verify a fixed token sequence of each width with rollback to the original offset
after each call, warm each width three times, and record twelve synchronized
elapsed times. Retain hardware, context length, methodology and the measured
curve with the results. The environment default stays at 4 pending deployment
Metal measurements.

This policy estimates steady-state throughput. Fixed overrides, acceptance
changes, calibration error, startup, short requests, and periodic probes can
still cost more than target-only. A strict per-request no-slowdown guarantee
cannot coexist with forced positive depths or exploratory probes. Auto mode
provides a profitability fallback, not that guarantee.

## Reproduction

No checkpoint is needed for the CPU fixture smoke:

```sh
RAPID_MLX_DSV41_DRAFT_DEPTH=auto python scripts/bench_deepseek_v41_draft.py \
  --tokens 64 --repeats 3
```

For Metal, provide local checkpoint paths with `--model` and `--draft`, run
fixed depths 0 through 4 and auto on both prose and coding prompts, and retain
the emitted environment, methodology hash, throughput, acceptance histogram,
and phase times. Check `greedy_equal` on every run. Record hardware, model
revision, prompt, MLX version and commands with the results.

Benchmark phase boundaries evaluate results and synchronize before stopping
their timer. Round timing also waits for deferred context updates. This
intentionally serializes benchmark phases so `draft_ms` counts completed
work; normal serving submits draft work asynchronously while building verify.
Phase sums therefore measure isolated work and cannot quantify serving overlap.

The proposal fixtures in `test_deepseek_v41_draft_numerics.py` record the
pre-change proposal IDs at commit `59460430b8dd951a684a946b25e807ae33ce8350`
for three deterministic CPU fixtures. CPU tests require exact proposal prefixes
and exact greedy tokens at every depth and in auto mode. Metal performance and
token equality require the deployment hardware run; CPU timings do not establish
the 5 ms draft target.
