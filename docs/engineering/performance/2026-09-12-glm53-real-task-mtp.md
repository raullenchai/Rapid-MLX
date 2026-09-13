# GLM-5.3 real-task MTP qualification

Date: 2026-09-12

Owner: Vector

Host: Studio, Apple M3 Ultra, 256 GiB, macOS 26.5.2 (25F84)

## Outcome

The cached uniform-Q4 GLM-5.3 target and its target-matched Q4 MTP head have a
real decode win, but the current post-0.7 mlx-vlm runtime is not yet eligible
for Rapid production. On six deterministic tasks, block-total 2 MTP preserved
both the reasoning and final response exactly and raised median decode
throughput by 1.240x. Median end-to-end completion throughput rose 1.191x, and
the slowest task still rose 1.127x.

The release gate nevertheless failed. Without a thinking budget, coding and
creative-writing prompts spent the complete 1,024-token allowance in reasoning
and produced no final answer. The AR server passed all six tasks when given the
per-task thinking budgets in the harness, but the speculative server rejects
the same request with HTTP 500:

```text
thinking_budget is not supported with speculative decoding in the server.
```

This is a product-quality blocker, not benchmark noise. Do not trade complete
answers for a higher token rate, and do not enable GLM MTP by default until the
speculative transaction can enforce the thinking boundary or safely fall back
to AR for that request.

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

The comparator fails unless every baseline and candidate task passes, complete
reasoning and output are identical, quality does not regress, and every task
retains at least 95% of baseline end-to-end completion throughput.

The coding grader is Apple-host-only and fails closed without `sandbox-exec`.
Generated code is AST-filtered, run with imports and dangerous builtins denied,
denied network and writes outside its temporary directory, and bounded by CPU,
file-size, process-count, descriptor, and wall-time limits.

## Next engineering gate

Supporting a forced thinking-end token inside a speculative block is not a
one-line removal of the server guard. A block can already have drafted and
verified tokens beyond the budget boundary; forcing the next token requires a
transactionally correct truncate/rollback or a request-scoped transition to AR.
Atlas should choose one of these designs when the tagged GLM MTP runtime is
integrated. Until then the alias remains fail-closed.
