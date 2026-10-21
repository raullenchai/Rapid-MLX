# DeepSeek V4.1 draft depth and timing

`RAPID_MLX_DSV41_DRAFT_DEPTH` accepts `0`, `1`, `2`, `3`, `4`, or
`auto`. The default is `4`; invalid values log a warning and use `4`.
The setting is read at request start. Fixed depths are explicit overrides:
`0` skips draft execution and context projection, and `4` retains the previous
seed-plus-four verification width. All positive depths retain all five draft
decoder positions, including their noise embeddings. Only the consumed Markov
predictions are generated. The serving path skips confidence projection.

Auto mode owns its acceptance and cost estimates per request. For depth K it
maximizes `(E[min(a, K)] + 1) / round_cost(K)`, where a is the accepted prefix
length observed at the request's maximum depth (normally 4). A decayed histogram
of accepted counts preserves bimodal distributions; every full-depth observation
multiplies existing weights by 0.9 and adds one to its accepted-count bin. Smaller
depths, including budget-limited tails, are excluded from the histogram because
their suffix is censored; they still update costs. EOS-truncated rounds are also
excluded. Cache rings may lower the request's maximum depth, in which case only
rounds at that maximum update the histogram.

The first eight maximum-depth rounds warm up the histogram, then the policy
selects the highest estimated rate. Switching to a positive depth requires a 10%
gain over the current choice; an unprofitable current depth is discarded
immediately. After 32 completed plain steps at K=0, two consecutive rounds probe
at maximum depth, even if the first probe changes the preferred rate. The output
budget and cache rings cap all choices, including warm-up and probes.

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

The depth-independent draft prior is 19.6 ms: all positive depths execute the
same five decoder positions. Host overhead starts at 3 ms. One shared draft EMA
learns from every positive depth. Verify costs are learned per observed row count;
unobserved costs use their prior multiplied by the arithmetic mean of the
learned/prior ratios at observed depths. Observed costs are never overwritten by
this scaling. Cost EMAs use weight 0.25, with the first phase measurement replacing
its prior. K=0 uses its learned whole plain-step cost directly, falling back to
the scaled one-row verify prior plus host overhead until a plain step is observed.

Auto serving completes candidate IDs and the finite flag with an `mx.eval` before
submitting verification, so the draft EMA measures completed work instead of
asynchronous submission time. The existing verify eval then measures the verify
phase independently. This adds a completion boundary only in auto mode; fixed
depths keep their asynchronous draft/verify path. Whole-round timing ends after
matching, rollback and draft-context observation, before tokens are yielded to
the caller. The residual after measured draft and verify phases learns host
overhead (clamped at zero for timing noise). No phase is inferred by subtracting
a prior from an asynchronous combined measurement. The extra auto completion
boundary needs server throughput validation on the target hardware.

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
