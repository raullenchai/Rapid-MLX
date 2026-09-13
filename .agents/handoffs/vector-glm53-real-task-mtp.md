# Vector handoff: GLM-5.3 real-task MTP qualification

## Receiving role

Atlas, for runtime architecture and dependency integration.

## Branch

`docs/glm53-competitor-causal`, based on
`origin/main@69cdecf7e` after qualification PR #3385 merged.

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
- An idle-host, same-width oMLX depth-1 run reached 31.813 tok/s, 1.056x the
  Rapid candidate. Against its own two-run deterministic AR control, oMLX MTP
  gained 1.151x versus Rapid's comparable 1.145x gain. oMLX's approximately 5%
  absolute lead is primarily its backbone/runtime baseline.
- oMLX depth-1 changed 6/6 reasoning strings and 3/6 final answers relative to
  its own AR. The custom verify-QMM is not the cause at depth 1 because its
  implementation gates to M=3..6 while depth 1 verifies at M=2.
- Adaptive Rapid block-total 3 was AR-exact but reached only 29.005 tok/s, or
  0.963x the qualified block-total 2 median. A single-token FFN compile spike
  was also AR-exact but converged within 0.0% to 0.8% of a warm uncompiled
  control and was rejected as immaterial.
- MTP long-context peak Metal memory was 188.679 GB versus 184.141 GB for AR.

## Unresolved

- The post-0.7 GLM runtime is not in the currently pinned release dependency.
- Rapid still pins a released mlx-vlm version without these three upstream
  changes. No release dependency is available to integrate yet.
- Creative prose still needs blind human review before any quality claim.

## Risks

Do not infer trajectory equivalence from oMLX's small throughput lead: even its
same-width depth-1 path changed greedy reasoning and half the final answers.
Do not vendor only part of the upstream three-PR chain; strict loading, safe
fallback, and positioned rollback are separate necessary pieces.

## Next action

Atlas should update the Rapid dependency only after a tagged mlx-vlm release
contains #2231, #2232, and #2233. Re-run the six-task harness through Rapid and
require 18/18 across three runs with AR-exact reasoning/final output before
enabling GLM MTP experimentally.
