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
showed adaptive depth changes and a custom M=2..6 verify-QMM path; this report
does not claim which one caused the trajectory drift.

A same-width oMLX depth-1 run was attempted, but the Studio CI runner started a
37 GB Qwen3.6-35B-8bit server during the suite. Throughput collapsed mid-run,
so the complete artifact was marked contended and excluded rather than folded
into the comparison.

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

The comparator fails unless every baseline and candidate task passes, complete
reasoning and output are identical, quality does not regress, and every task
retains at least 95% of baseline end-to-end completion throughput.

The coding grader is Apple-host-only and fails closed without `sandbox-exec`.
Generated code is AST-filtered, run with imports and dangerous builtins denied,
denied network and writes outside its temporary directory, and bounded by CPU,
file-size, process-count, descriptor, and wall-time limits.

## Next engineering gate

The transaction and safe fallback now exist upstream. Atlas should not vendor a
partial copy or point a release at an untagged Git commit. Once mlx-vlm ships a
release containing #2231, #2232, and #2233, update Rapid's pin, run this exact
six-task gate through the Rapid server, and only then enable GLM MTP. The next
performance investigation should target exact verify/backbone dispatch cost;
do not adopt oMLX's deeper/custom-QMM path without first recovering repeated
temperature-zero determinism and 18/18 quality.
