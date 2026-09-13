# Vector handoff: GLM-5.3 real-task MTP qualification

## Receiving role

Atlas, for runtime architecture and dependency integration.

## Branch

`vector/glm53-real-eval`, based on `origin/main@e28f68c41`.

## Verified facts

- A reproducible six-category harness now executes coding hidden tests and
  grades knowledge, math, instruction following, creative constraints, and
  long-context retrieval.
- Same-target Q4 MTP block-total 2 preserved reasoning and final content on
  6/6 deterministic tasks.
- Median decode improvement was 1.240x; median end-to-end completion
  throughput improvement was 1.191x; the minimum task gain was 1.127x.
- Comparable no-budget AR and MTP each passed only 4/6 because coding and
  creative writing exhausted 1,024 tokens in reasoning without a final answer.
- Budgeted AR passed 6/6. The speculative server rejects `thinking_budget`
  with HTTP 500 before generation.
- MTP long-context peak Metal memory was 188.679 GB versus 184.141 GB for AR.

## Unresolved

- The post-0.7 GLM runtime is not in the currently pinned release dependency.
- Thinking-budget enforcement inside a speculative block needs transactional
  truncation/rollback, or a safe request-scoped AR fallback.
- Creative prose still needs blind human review before any quality claim.

## Risks

Removing the server guard without teaching the speculative batch about the
forced thinking-end token can silently cross the budget boundary and corrupt
the drafter/target cache transaction. High acceptance and exact no-budget
output do not make that safe.

## Next action

Atlas should select the integration design after a tagged mlx-vlm release is
available. Re-run this harness with per-task thinking budgets on the integrated
server; require 6/6 exact-output parity before enabling GLM MTP experimentally.
