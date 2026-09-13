# Vector handoff: GLM-5.3 real-task MTP qualification

## Receiving role

Atlas, for runtime architecture and dependency integration.

## Branch

`docs/glm53-followup-findings`, based on `origin/main@aa92dc238` after the
competitor-causal record merged in PR #3389.

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
- Recurrent elementwise compilation was rejected after a six-pair structural
  fixture run measured 0.995x overall despite two isolated 1.026x sub-operation
  wins. Recurrent prefill slicing was rejected against current mlx-vlm: it was
  only 1.005x at 2K and 1.025x at 4K on a production-shaped Q4 BF16 layer, and
  changed output by 1.5259e-5 plus the final cache above 512 tokens.
- Upstream mlx-vlm PR #2234 fuses GLM routed gate/up Q4 storage. Three complete
  runs passed 18/18 with byte-identical reasoning/final output. Median
  per-category decode improved 2.65%, median paired end-to-end category
  throughput improved 2.49%, and the median of category medians moved from
  30.132 to 31.156 tok/s (1.034x). Peak Metal memory remained 188.674 GB.
- The gate/up microbenchmark measured 0.583 to 0.532 ms (1.086x) at production
  geometry. A real checkpoint layer remained bit-exact from T=1 through T=16
  and measured 1.018x to 1.082x depending on width. Both raw target weights
  and the existing pre-stacked Q4 sidecar load into the fused representation.
- A vector-gated recurrent R=4 spike was bit-exact and 1.219x/1.313x/1.789x at
  512/2K/8K in isolation, but complete-model 3K prefill improved only 1.2% to
  1.6% and full-request throughput did not clear the gate. It was rejected.
- The maintainer's cache-owned MTP rebuild in mlx-vlm #2206 also works with the
  real Q4 target, Q4 sidecar, #2231, and #2234. Two MTP runs passed 12/12 and
  were byte-identical to each other. A same-branch AR control passed 6/6; MTP
  matched every reasoning/final byte and improved the tasks by
  1.253x/1.306x/1.188x/1.297x/1.179x/1.082x (1.221x median).
- The cache-owned combination's two-run median of category medians was 33.655
  tok/s. Its first run improved median client/server-decode throughput by
  7.7%/8.7% over positioned MTP plus #2234 and exceeded same-width oMLX's
  31.813 tok/s by 6.3% while retaining own-AR equivalence.
- The cache-owned AR/MTP peak Metal readings were 184.147/188.499 GB, within
  the existing product qualification envelope.
- MTP long-context peak Metal memory was 188.679 GB versus 184.141 GB for AR.

## Unresolved

- The post-0.7 GLM runtime is not in the currently pinned release dependency.
- Rapid still pins a released mlx-vlm version without the qualified upstream
  changes. PRs #2206, #2231, and #2234 are upstream-only, so no release
  dependency is available to integrate yet.
- Creative prose still needs blind human review before any quality claim.
- The Q4 target plus MTP reaches 188.679 GB peak Metal memory. The Studio had
  only 23 GiB of cache-volume headroom during the follow-up, so a complete new
  mixed-quant derivative cannot be staged without an explicit storage plan.

## Risks

Do not infer trajectory equivalence from oMLX's small throughput lead: even its
same-width depth-1 path changed greedy reasoning and half the final answers.
Do not vendor an ad-hoc subset of the upstream work. #2206 replaces the legacy
speculative transaction and its own AR control must remain the equivalence
oracle; #2231 and #2234 supply the Q4 strict-load and storage paths used here.
Do not transplant oMLX's older recurrent-slicing result without re-measuring
current mlx-vlm; the optimized current kernel has a different crossover and
did not preserve bit-exact state in the production-shape check.

## Next action

Atlas should prefer the cache-owned #2206 route over the legacy #2232/#2233
chain once a tagged mlx-vlm release contains #2206, #2231, and #2234. Re-run
the six-task harness through Rapid and require 18/18 across three runs with
same-release AR-exact reasoning/final output before enabling GLM MTP
experimentally. The next backbone experiment should pursue a material
resident-size or dispatch reduction rather than another small graph compile;
do not delete the only cached Q4 control to make room without explicit human
authorization and a recovery plan.
