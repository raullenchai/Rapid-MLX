# DeepSeek V4.1 draft depth and timing

`RAPID_MLX_DSV41_DRAFT_DEPTH` accepts `0`, `1`, `2`, `3`, `4`, or
`auto`. The default is `2`; invalid values log a warning and use `2`.
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

## Measured defaults

Environment: M3 Ultra 256 GB, v2 checkpoint revision `c521050a`, server single
stream. Results are per-stream decode tok/s with 256 max tokens, greedy decoding,
and 1 warm-up + 2 measured rounds. Prompts include the 1008-token summarisation
prompt `p1008` plus `qa`, `summarize`, `code`, and `structured` chat prompts.

| Depth | p1008 | qa | summarize | code | structured |
| --- | --- | --- | --- | --- | --- |
| 2 | 35.2 | 34.3 | 36.2 | 42.7 | 42.1 |
| 4 | 31.2 | 33.1 | 33.6 | 49.0 | 46.5 |
| auto | 31.2 | 31.1 | 32.0 | 49.1 | 47.0 |

Depth 2 is the default because it is the best single setting across prompt
classes. It leads on `p1008`, `qa`, and `summarize`, while depth 4 and auto are
faster on `code` and `structured`. Auto remains opt-in.

Draft cost per round fell from 19.6 ms before this change set to 10.1 ms; the
head share fell from 8.8 ms to 2.1 ms. Acceptance histograms were unchanged
versus the stock draft path.

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
curve with the results. The environment default is 2 based on the server
measurements above.

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
for three deterministic CPU fixtures. Stock draft CPU tests require exact proposal prefixes; optimized draft tests
require finite proposals and exact greedy output tokens. Output tokens must
match at every depth and in auto mode. Metal performance and
token equality require the deployment hardware run; CPU timings do not establish
the 5 ms draft target.

## Stage A2: acceptance-qualified draft kernels

The fused draft MoE and draft head GEMV are **acceptance-qualified, not
bit-exact** against the batched stock draft. Supported layouts, dtypes, and row
counts use the optimized kernels directly, without numerical equality probes.
Draft rounding can change proposals and acceptance, but the target verification
forward decides every emitted token. The target path and its exactness
qualification remain unchanged. The correctness contract is token equality to
draft-off greedy on the fixtures, rather than equality of draft proposals,
logits, or confidence.

Each optimized component records a device finite flag. These flags and the
final draft logits are evaluated at the existing per-round verify boundary;
checking them adds no host synchronization to successful rounds. A component
that produces NaN/inf from finite input is permanently disabled independently
of other layers and the head, with one WARNING identifying the layer/head and
bit combination (head dtype). Non-finite values propagated from an earlier
component do not disable downstream components. Before retrying a failed round,
serving rolls back target cache offsets and restores overwritten window slots;
the round is recomputed using the stock draft path for the failed component.
Unsupported kernel layouts retain the stock draft path. Non-finite stock draft
results still fail the request.

## Negative result: multi-row attention projection GEMV (shelved)

A custom Metal GEMV shared weight loads across 1–8 rows for five affine attention
projections: `wq_a` 1280×5120, `wq_b` 32768×1280, `wkv` 512×5120, grouped
`wo_a` 8×1024×4096, and `wo_b` 5120×8192 (4/8-bit, group size 64). Each row
was bit-identical to the stock single-row projection. The motivation was a
roofline estimate of about 5.5 ms/token recoverable in attention projections.

A/B environment: M3 Ultra 256 GB, v2 checkpoint `c521050a`, in-process
synchronized timers, depth 4, 256 tokens, greedy decoding, same GPU window.
Baseline main `95f2215d4` was compared with `vector/dsv41-attn-gemv` at
`14c9bba8c`. Times below are milliseconds.

| Phase | Prompt | Baseline | Candidate |
| --- | --- | --- | --- |
| Verify (5 rows) | p1008 | 79.6 | 89.4 |
| Verify (5 rows) | qa | 76.1 | 87.7 |
| Verify (5 rows) | code | 75.9 | 87.4 |
| Target-only step | p1008 | 37.0 | 37.8 |
| Target-only step | qa | 30.5 | 33.2 |
| Target-only step | code | 30.4 | 34.0 |

Server single-stream results at default depth 2, per-stream decode tok/s over
two rounds:

| Prompt | Baseline | Candidate |
| --- | --- | --- |
| p1008 | 36.5 / 30.4 | 29.6 / 28.8 |
| qa | 28.7 / 31.8 | 31.4 / 31.7 |

These shared-machine results are noisier than the measured-defaults table;
the candidate showed no throughput gain.

The synthetic per-projection microbenchmark on the shelved branch,
`scripts/bench_deepseek_v41_attention_gemv.py`, used synchronized timers,
5 warm-up calls, 20 repeats, and float32 activations. Candidate/stock speed was
0.73–0.99× for almost all projection/bit/row combinations with rows in {1, 5}.
Exceptions at 5 rows were `wq_b` (1.20× at 8-bit, 1.35× at 4-bit) and `wo_b`
8-bit (1.07×). Per-call times were 0.25–0.44 ms, with effective weight bandwidth
of 5–150 GB/s.

The results indicate launch/occupancy limits rather than weight-bandwidth limits:
the 5.5 ms roofline estimate does not apply, and replacing the GEMV kernel does
not recover it. Per-projection GEMV kernel work is shelved; further attention-side
gains require fewer dispatches through fusion or graph compilation, rather than
better bandwidth. Branches `vector/dsv41-attn-gemv` and
`vector/dsv41-attn-gemv-test` are retained for reference. The latter includes a
tiny-model test showing spec-on equals spec-off when every attention projection
uses the per-row kernel; it passes on Metal.

### Speculative vs plain greedy output

On the real checkpoint, greedy speculative output is not guaranteed token-identical
to plain decoding. Batched verify uses different multi-row reduction orders from
single-row decode, so near-ties can fork. At depth 4 on main `95f2215d4`, the
first divergence was token 38 (`p1008`), 63 (`qa`), and 237 (`code`); plain-decode
token hashes were unchanged by the candidate. This is expected numerics, not a
correctness bug; token identity is not a product invariant. Every emitted token
is still the target model's greedy choice under the verify computation.
