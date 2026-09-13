# Vector handoff: GLM-5.3 real-task MTP qualification

## Receiving role

Atlas, for runtime architecture and dependency integration.

## Branch

`docs/glm53-budget-mtp-qualification`, based on
`origin/main@a42e5fbf8`.

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
- Budgeted AR passed 6/6. A new positioned-target transaction passed 18/18
  task runs with byte-identical reasoning and final output versus AR.
- Three-run per-category MTP medians were 29.702, 31.746, 30.399, 30.617,
  29.865, and 11.491 tok/s. Their median was 30.132 tok/s versus 26.322 for AR
  (1.145x); every category improved.
- Upstream mlx-vlm PR #2231 fixes strict quantized `lm_head` loading, #2232
  provides the safe AR fallback, and #2233 keeps singleton greedy MTP active
  through the thinking budget. Their CI and maintainer review remain upstream.
- oMLX depth-2 reached a 31.664 tok/s median of category medians, 1.051x the
  Rapid candidate, but passed only 6/6, 5/6, and 5/6 across repeated runs and
  changed temperature-zero outputs. Rapid remained 18/18 and AR-exact.
- MTP long-context peak Metal memory was 188.679 GB versus 184.141 GB for AR.

## Unresolved

- The post-0.7 GLM runtime is not in the currently pinned release dependency.
- Rapid still pins a released mlx-vlm version without these three upstream
  changes. No release dependency is available to integrate yet.
- A same-width oMLX depth-1 rerun is still needed on an idle Studio; the first
  attempt was invalidated by a concurrent 37 GB CI inference server.
- Creative prose still needs blind human review before any quality claim.

## Risks

Do not infer quality from oMLX's small throughput lead: its deeper/custom verify
path changed greedy trajectories and failed the creative task twice. Do not
vendor only part of the upstream three-PR chain; strict loading, safe fallback,
and positioned rollback are separate necessary pieces.

## Next action

Atlas should update the Rapid dependency only after a tagged mlx-vlm release
contains #2231, #2232, and #2233. Re-run the six-task harness through Rapid and
require 18/18 across three runs with AR-exact reasoning/final output before
enabling GLM MTP experimentally.
