# GLM-5.3 real-task MTP qualification

Date: 2026-09-12

Owner: Vector

Host: Studio, Apple M3 Ultra, 256 GiB, macOS 26.5.2 (25F84)

## Outcome

The cached uniform-Q4 GLM-5.3 target and its target-matched Q4 MTP head have a
real decode win. On six deterministic tasks, block-total 2 MTP preserved both
the reasoning and final response exactly and raised median decode throughput
by 1.240x. Median end-to-end completion throughput rose 1.191x, and the slowest
task still rose 1.127x.

The release gate nevertheless failed. Without a thinking budget, coding and
creative-writing prompts spent the complete 1,024-token allowance in reasoning
and produced no final answer. The AR server passed all six tasks when given the
per-task thinking budgets in the harness, but the speculative server rejects
the same request with HTTP 500:

```text
thinking_budget is not supported with speculative decoding in the server.
```

This was a product-quality blocker, not benchmark noise. A follow-up
transactional implementation now passes the budgeted gate, preserves every AR
byte, and keeps a measured 1.145x median task-throughput gain. It is proposed
upstream in mlx-vlm PRs #2231, #2232, and #2233; it is not Rapid production
behavior until those changes are released and Rapid updates its dependency.

## Environment

- target: `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, revision
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6`
- Q4 drafter: layer-45 sidecar produced from the same revision, 3.9 GiB
- mlx-vlm source: `d2a1434a03e4c9975b0d505e7178e0cfc4082a83`
- Python 3.12.13, MLX 0.32.2, mlx-lm 0.31.3
- batch one, greedy temperature zero, thinking enabled, no thinking budget
- one server process at a time; one warm-up request before measured tasks
- MTP `block_size=2`: one draft token plus the verifier bonus

The full target and drafter were already local. The campaign ran offline and
did not download or relocate Hugging Face data.

## Results

Every MTP response had byte-identical reasoning and final content to AR.
`Total ratio` is baseline request latency divided by candidate request latency,
so values above one are faster. Coding and creative writing correctly remain
failed rows because both modes exhausted the output limit without a final
answer under the comparable no-budget policy.

| Task | AR pass | MTP pass | AR decode | MTP decode | Decode ratio | AR total | MTP total | Total ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coding, hidden tests | no | no | 26.24 | 32.48 | 1.238x | 39.69 s | 32.14 s | 1.235x |
| Creative constraints | no | no | 27.50 | 32.56 | 1.184x | 37.80 s | 32.01 s | 1.181x |
| Instruction following | yes | yes | 29.01 | 36.03 | 1.242x | 5.33 s | 4.44 s | 1.201x |
| Closed-book knowledge | yes | yes | 28.83 | 36.04 | 1.250x | 8.20 s | 6.68 s | 1.229x |
| Long-context contract | yes | yes | 25.03 | 33.95 | 1.356x | 17.00 s | 14.67 s | 1.158x |
| Multi-step math | yes | yes | 28.75 | 32.60 | 1.134x | 12.56 s | 11.14 s | 1.127x |

Peak Metal memory was 184.141 GB for the AR process and 188.679 GB after the
MTP long-context task, a 4.538 GB increase. The six-task smoke is intended to
catch trajectory and completion regressions; it is not a broad intelligence
benchmark or an RMQ quality claim.

## Budgeted AR control

With the harness per-task thinking budgets enabled, the same AR checkpoint
passed 6/6:

- coding: emitted a complete function and passed 8/8 hidden cases;
- knowledge: 5/5 atomic facts;
- math: 4/4 recurrence values;
- instruction following: 3/3 exact fields;
- creative writing: 6/6 machine-checkable constraints, with blind style review
  still required for a release claim;
- long context: 4/4 fields retrieved from clause 47 among 80 clauses.

The first coding control with a 512-token maximum was truncated mid-function.
The harness therefore treats `finish_reason=length` and incomplete executable
output as failures even when the visible prefix appears correct.

## Budgeted speculative follow-up

The follow-up implementation snapshots `ThinkingBudgetCriteria` by absolute
output position and restores it after a draft rejection. A forced
`\n</think>` therefore participates in the same target/drafter transaction as
ordinary tokens. Only singleton greedy MTP takes this path; sampled requests
and multi-row budget cohorts retain the complete-request AR fallback from
mlx-vlm PR #2232.

Candidate source: mlx-vlm commit
`f45792a96b5ae0a62c2a474977910e26e6877315` (PR #2233), stacked on safe-fallback
commit `acc638ab9bf777dbae8a4c99f8c70cdfd5457198` (PR #2232). The strict
quantized `lm_head` loader is PR #2231. Target, Q4 sidecar, host, temperature,
task prompts, budgets, and 1,024-token ceiling were unchanged from the AR
control. The per-task budgets were 256, 128, 192, 96, 192, and 128 tokens in
table order.

Three independent warm runs passed all six tasks. All 18 reasoning strings and
all 18 final answers were byte-identical to budgeted AR.

| Task | MTP run 1 | MTP run 2 | MTP run 3 | MTP median | AR | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coding, hidden tests | 33.249 | 29.702 | 28.741 | 29.702 | 27.788 | 1.069x |
| Closed-book knowledge | 31.914 | 31.746 | 31.703 | 31.746 | 26.572 | 1.195x |
| Multi-step math | 30.399 | 30.419 | 30.247 | 30.399 | 26.072 | 1.166x |
| Instruction following | 30.541 | 30.649 | 30.617 | 30.617 | 24.996 | 1.225x |
| Creative constraints | 29.759 | 29.978 | 29.865 | 29.865 | 27.227 | 1.097x |
| Long-context contract | 11.485 | 11.491 | 11.509 | 11.491 | 10.025 | 1.146x |

The median of the six per-task medians was 30.132 tok/s versus 26.322 tok/s for
AR, a 1.145x gain. Every category improved; the range was 1.069x to 1.225x.
Peak Metal memory was 188.674 GB.

The relevant regression set completed with 1,374 passed and 1 skipped. It
covers forced-close rollback, reset of a pending forced token, sampled and
multi-row fallback, and single-projection positioned target sampling.

### Current oMLX comparison

oMLX `0.7.0.dev2` at commit `b390b31` was run from a clean local server against
the same target checkpoint, prompts, per-task budgets, temperature, and output
limits. Native Lightning MTP was capped at draft depth 2. That is a useful
best-runtime comparison, but not a same-width comparison: oMLX can draft two
tokens plus a verifier bonus, while the qualified Rapid candidate uses one
draft token plus a bonus.

| Task | oMLX run 1 | oMLX run 2 | oMLX run 3 | oMLX median | Task passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coding, hidden tests | 34.239 | 34.191 | 34.292 | 34.239 | 3/3 |
| Closed-book knowledge | 35.002 | 34.881 | 32.615 | 34.881 | 3/3 |
| Multi-step math | 32.768 | 32.914 | 32.799 | 32.799 | 3/3 |
| Instruction following | 30.243 | 30.530 | 30.619 | 30.530 | 3/3 |
| Creative constraints | 29.500 | 29.305 | 30.715 | 29.500 | 1/3 |
| Long-context contract | 12.283 | 11.847 | 11.829 | 11.847 | 3/3 |

The median of oMLX's per-task medians was 31.664 tok/s, 1.051x the Rapid
candidate. That number is not a quality-qualified win: oMLX passed 6/6, 5/6,
and 5/6 across the three runs. Its creative score fell to 0.833 and then 0.667;
the last run exhausted all 1,024 tokens. Despite temperature zero, every
reasoning string differed from AR and three of six final answers differed in
the first run. Output lengths also changed between repeated oMLX runs. Its log
showed adaptive depth changes. oMLX also installs a custom verify-QMM route,
but its source gates that route to M=3..6; the same-width depth-1 result below
uses M=2 and therefore does not exercise that custom kernel.

### Same-width oMLX control and causal split

The same-width oMLX depth-1 rerun completed on an idle Studio. It uses one
draft token plus the verifier bonus, matching the qualified Rapid block-total
2 width. All six tasks passed, with a 31.813 tok/s median of category
throughputs, 1.056x the Rapid candidate.

| Task | oMLX AR, two-run mean | oMLX depth 1 | Depth-1 / own AR | Final equals own AR |
| --- | ---: | ---: | ---: | --- |
| Coding, hidden tests | 28.792 | 31.909 | 1.108x | no |
| Closed-book knowledge | 27.775 | 33.917 | 1.221x | no |
| Multi-step math | 27.513 | 32.228 | 1.171x | yes |
| Instruction following | 26.248 | 31.717 | 1.208x | yes |
| Creative constraints | 28.550 | 28.008 | 0.981x | no |
| Long-context contract | 11.051 | 12.622 | 1.142x | yes |

oMLX AR itself was deterministic: two consecutive warm runs produced identical
reasoning and final output on all six tasks. Its AR median was 27.644 tok/s,
1.050x mlx-vlm AR. Depth-1 MTP raised that to 31.813 tok/s, a 1.151x gain;
Rapid's exact MTP raised its own AR from 26.322 to 30.132 tok/s, a comparable
1.145x gain. The approximately 5% absolute oMLX lead therefore comes primarily
from its different GLM backbone/runtime, not from deeper drafting.

That speed is not trajectory-equivalent. Relative to oMLX's own deterministic
AR, depth-1 changed all six reasoning strings and three of six final answers.
All tasks passed in this single depth-1 run, but creative writing was also the
only category slower than oMLX AR. The result supports investigating exact
backbone dispatch cost; it does not support replacing the qualified Rapid
transaction with oMLX's MTP path.

### Rejected follow-up spikes

Two narrowly scoped Rapid spikes did not clear the performance gate:

- Adaptive block-total 3 started at 2 and expanded only after eight rounds
  with at least 65% configured-prefix hits. One run remained 6/6 and AR-exact,
  but its category median was 29.005 tok/s, 0.963x the qualified block-total 2
  median. It was rejected without additional runs.
- Compiling each GLM single-token FFN block remained AR-exact across 18/18
  tasks. After both variants reached a steady warm state, the six category
  differences versus an immediate uncompiled control were 0.0% to 0.8% and
  did not establish a material sustained win. The apparent early gain was
  consistent with lazy-kernel warm-up, so no runtime patch was proposed.
- Compiling the elementwise work immediately around the recurrent kernel made
  each of two production-shaped sub-operations 1.026x faster in isolation
  (243.1 to 236.9 microseconds and 222.3 to 216.7 microseconds). The complete
  five-layer GLM structural fixture did not retain that signal: six interleaved
  512-token pairs measured 321.8 versus 325.6 tok/s, a 0.995x paired median,
  with individual ratios from 0.931x to 1.024x. All sampled greedy fingerprints
  matched. The full-model effect was both immaterial and unstable, so the
  fusion was rejected.
- Shapeless-compiling the exact FP32 index-score expression preserved every
  score element across 64 BF16/FP16 cases spanning batch 1/2, query lengths
  1/2/8/32, and multiple pool lengths. The isolated score call improved by
  1.122x, 1.103x, 1.081x, and 1.062x at pooled lengths 2K, 8K, 16K, and 32K.
  That saving did not survive the complete request: a same-host paired
  six-task run kept all outputs and reasoning byte-identical, but delivered a
  0.998x median throughput ratio. Long context gained only 1.014x and the
  noisiest category fell to 0.959x. The compile wrapper was therefore
  rejected rather than promoting another micro-only result.
- Slicing the recurrent prefill into 512-token pieces was also re-measured
  against current mlx-vlm rather than inferred from an older vendored runtime.
  A production-shaped Q4 BF16 layer measured 41.17 versus 40.94 ms at 2,048
  tokens (1.005x) and 82.88 versus 80.85 ms at 4,096 tokens (1.025x). At 1,024
  tokens it regressed to 0.986x. For every width above 512, the maximum output
  difference was 1.5259e-5 and the final recurrent cache was not bit-identical.
  The current kernel therefore does not reproduce the older claim of a
  bit-exact 30% win; no slicing patch was proposed.
- Four-row blocking in the vector-gated recurrent Metal kernel was bit-exact
  and strong in isolation: 1.219x at 512 tokens, 1.313x at 2,048, and 1.789x
  at 8,192. On the complete model it improved the 3,126-token prefill by only
  1.2% to 1.6%, while repeated full-task runs did not establish a net request
  throughput win under the host's concurrent CPU load. Restricting it to
  prefill preserved all task outputs but did not clear the product gate, so
  the spike was rejected rather than exposing a new runtime knob.

### Accepted gate/up storage fusion

One backbone reduction did clear the gate and is proposed upstream as
mlx-vlm PR #2234. GLM's eight routed experts previously read the same hidden
vector and routing indices through separate gate and up gathered QMMs. The
candidate stores their affine Q4 tensors gate-first in one
`QuantizedSwitchLinear`, performs one gathered projection, and then splits the
unchanged results before LimitedSwiGLU. Both raw per-expert target checkpoints
and older pre-stacked MTP sidecars migrate at load time.

At production geometry (H=4,096, I=2,048, 288 experts, top-8), the complete
routed SwitchGLU call improved from 0.583 to 0.532 ms, or 1.086x. A real
checkpoint sparse layer measured 1.082x, 1.057x, 1.042x, 1.067x, and 1.018x at
T=1, 2, 4, 8, and 16 respectively. Every tested projection and layer output
was bit-exact.

Three warm full-model runs retained all 18/18 task passes and byte-identical
reasoning/final strings. Median server decode gains by category were +2.8%,
+3.6%, +2.5%, +2.2%, +2.7%, and +1.5%. The median per-category decode gain
was 2.65%; the median paired end-to-end category gain was 2.49%, and the
median of category medians moved from 30.132 to 31.156 tok/s (1.034x). Peak
Metal memory remained 188.674 GB.

The first load-time implementation concatenated each expert separately and
was killed on first materialization. The accepted implementation instead uses
one flat stack and a zero-copy reshape, eliminating thousands of intermediate
expressions. The existing 3.9 GB Q4 sidecar subsequently loaded with
`strict=True`, and the 184 GB target completed the full qualification without
exceeding the prior suite's peak.

### Cache-owned MTP follow-up

The maintainer's cache-owned MTP rebuild in mlx-vlm PR #2206 was subsequently
tested with the same Q4 target, the existing 3.9 GB Q4 sidecar, PR #2231's
strict-load fix, and the gate/up storage change from #2234. This matters
because #2206's published measurements used a GLM FP8 target; Q4 compatibility
and product-task quality were previously unverified.

The Q4 combination passed all six real tasks in two consecutive MTP runs and
was deterministic between runs. Its two-run per-task medians were 36.674,
35.025, 32.781, 33.815, 33.494, and 11.534 tok/s, for a 33.655 tok/s median of
category medians. One same-branch AR control also passed 6/6. MTP matched that
control's complete reasoning and final response byte-for-byte while improving
the six tasks by 1.253x, 1.306x, 1.188x, 1.297x, 1.179x, and 1.082x. The
median paired gain was 1.221x.

Peak Metal memory was 184.147 GB for the AR control and 188.499 GB for MTP,
within the previously qualified working-set envelope.

Relative to the previously qualified positioned-MTP plus gate/up result, the
first cache-owned run moved median client throughput from 31.390 to 33.821
tok/s (1.077x) and median server decode from 35.461 to 38.564 tok/s (1.087x).
It also exceeded the same-width oMLX result of 31.813 tok/s by 6.3%, while
retaining exact equivalence to its own AR control. Short-task reasoning differs
from the older Rapid artifacts because #2206 enforces the thinking boundary
inside the target distribution; the matching same-branch AR control shows
that this is the branch's budget policy rather than speculative drift.

This is a qualification result for the upstream combination, not a Rapid
dependency update. #2206 remains under maintainer review and must land before
the release integration can use it.

The 184-189 GB working set also makes host health part of the benchmark gate.
A follow-up stock run produced only 0.382 tok/s while the 256 GiB host was
actively swapping after concurrent large-model campaigns; it was discarded.
Future measurements on this target must check both competing model processes
and memory pressure, and should use the shared large-model lock. Process
isolation alone is not enough when inactive model pages still push the target
over physical memory.

## Reproduction

Start post-0.7 mlx-vlm once without a drafter and once with the target-matched
Q4 drafter. Use the same immutable target path for both server preload and the
request `model` field; a short, unknown model name asks mlx-vlm to unload and
switch models.

```bash
python scripts/benchmark_glm53_real_tasks.py \
  --model /path/to/immutable/target/snapshot \
  --label ar-no-thinking-budget \
  --omit-thinking-budget \
  --output /private/tmp/glm53-real-ar.json

python scripts/benchmark_glm53_real_tasks.py \
  --model /path/to/immutable/target/snapshot \
  --label mtp-block2-q4-no-thinking-budget \
  --omit-thinking-budget \
  --output /private/tmp/glm53-real-mtp.json

python scripts/benchmark_glm53_real_tasks.py \
  --compare /private/tmp/glm53-real-ar.json \
  /private/tmp/glm53-real-mtp.json
```

The budgeted positioned-MTP server used:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/path/to/mlx-vlm-at-f45792a \
python -m mlx_vlm.server \
  --model /path/to/immutable/target/snapshot \
  --draft-model /path/to/target-matched-q4-mtp-sidecar \
  --draft-kind mtp --draft-block-size 2 \
  --host 127.0.0.1 --port 8465 --max-tokens 1024 --enable-thinking

python scripts/benchmark_glm53_real_tasks.py \
  --base-url http://127.0.0.1:8465/v1 \
  --model /path/to/immutable/target/snapshot \
  --label mtp-budget-positioned \
  --output /private/tmp/glm53-real-mtp-budget-positioned.json
```

The oMLX comparison used commit `b390b31`, cache disabled, concurrency one,
thinking enabled, and the same local target exposed as `glm53-target`. The AR
control set `mtp_enabled=false`; the same-width run set `mtp_enabled=true` and
`mtp_num_draft_tokens=1`. For each server mode, the same harness command was:

```bash
python scripts/benchmark_glm53_real_tasks.py \
  --base-url http://127.0.0.1:8466/v1 \
  --model glm53-target \
  --label omlx-ar-or-depth1 \
  --output /private/tmp/glm53-real-omlx.json
```

The comparator fails unless every baseline and candidate task passes, complete
reasoning and output are identical, quality does not regress, and every task
retains at least 95% of baseline end-to-end completion throughput.

The coding grader is Apple-host-only and fails closed without `sandbox-exec`.
Generated code is AST-filtered, run with imports and dangerous builtins denied,
denied network and writes outside its temporary directory, and bounded by CPU,
file-size, process-count, descriptor, and wall-time limits.

## Next engineering gate

The cache-owned transaction now exists upstream. Atlas should not vendor a
partial copy or point a release at an untagged Git commit. Once mlx-vlm ships a
release containing #2206, #2231, and #2234, update Rapid's pin, run this exact
six-task gate through the Rapid server, and only then enable GLM MTP. The
legacy #2232/#2233 chain remains a smaller fallback if #2206 does not land.
The next performance investigation should target exact backbone dispatch cost;
the same-width result shows that deeper drafting is not the main competitor
gap. Do not adopt a numerically different verify path without first recovering
repeated temperature-zero determinism and 18/18 quality.
