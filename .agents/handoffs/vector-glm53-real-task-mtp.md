# Vector handoff: GLM-5.3 real-task MTP qualification

## Receiving role

Atlas, for runtime architecture and dependency integration.

## Branch

Current implementation: `vector/glm53-native-runtime-v1`, based on
`origin/main@12fbee034`. Rapid now owns the narrow cache transaction and
generation-hook seam; mlx-vlm PR #2206 still supplies the model/cache protocol
and GLM drafter architecture used for qualification.

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
- #2234 later closed without merge. Current mlx-vlm main + #2206 + #2231,
  without #2234, passed a clean 6/6 AR/MTP pair with complete reasoning/final
  byte parity. Per-task ratios were 1.264x/1.234x/1.187x/1.323x/1.194x/1.070x
  (1.214x median); median category throughput was 32.887 tok/s, 3.4% above the
  31.813 tok/s same-width oMLX control. AR/MTP peak Metal was
  184.141/188.432 GB. #2234's incremental first-run contribution was about
  2.8%, so it is optional rather than a release dependency.
- Same-load K=3 remained exact and had a 1.030x task median versus K=2, but
  regressed coding/creative 3.6-3.7% while improving instruction/knowledge
  7.4-8.6%. Keep K=2 qualified and pursue adaptive depth separately.
- A request-local K2/K3 gate retained K=3 except when a rolling 64-round
  acceptance window fell below 65%. Two runs passed 12/12 with identical
  decisions and byte-identical reasoning/final output. Category medians were
  34.552 and 34.610 tok/s; paired task throughput improved 1.298x over the
  26.490 tok/s same-branch AR control and exceeded the 31.813 tok/s same-width
  oMLX control by 8.6% in the first run. The threshold remains a Q4-specific
  experiment, not a generic default.
- Launching the replay-head seed asynchronously before the verified block's
  final yield improved the adaptive path by a further 1.016x paired median,
  with all six task ratios at least 1.003x. Two repeats reached a 1.322x and
  1.320x paired median over AR; both were 6/6, byte-identical, and had exactly
  the same round/proposal/accept counters as the non-overlapped adaptive run.
- The threshold-free, request-local EV controller reached a 34.725-34.785
  tok/s category median with replay overlap and preserved 12/12 exact output,
  but did not materially beat the simpler gate. A cross-request cost-EWMA
  follow-up was rejected: identical coding requests changed from 199 to 250
  rounds, and instruction throughput fell from 36.27 to 34.27 tok/s because
  lazy replay cost was attributed across request boundaries.
- Upstream mlx-vlm PR #2241 isolates replay overlap directly on #2206's head.
  Fixed K=3 paired medians improved 1.011x and 1.005x in two runs with exact
  output. Its exact head `b762dba5` is mergeable; 3,544 full-suite tests, 128
  subtests, Ruff, and diff checks passed. Rapid PR #3409 merged response-level
  speculative counters into the six-task artifact so future acceptance claims
  remain reproducible.
- MTP long-context peak Metal memory was 188.679 GB versus 184.141 GB for AR.
- The cache-owned candidate now passes Rapid's own serial OpenAI-compatible
  Server path. A warm budgeted AR/MTP pair passed 6/6 in each mode with all six
  reasoning/final responses byte-identical. Per-task MTP gains were
  1.204x/1.387x/1.313x/1.395x/1.283x/1.101x (1.298x paired median), and the
  median category throughput moved from 26.58 to 35.09 tok/s.
- That integration gate exposed a shared serial-server bug: DFlash/native-MTP
  prompt rendering honored `enable_thinking`, but generation did not receive
  it and silently dropped `reasoning_max_tokens`. The fix maps the public Rapid
  cap to mlx-vlm's `thinking_budget`; the formerly failing creative task
  contracted from a truncated 1,024 tokens to a passing 388-token completion.
  A cap without a duplicate `enable_thinking=true` now opts into bounded
  thinking, while an explicit false value and `--no-thinking` remain dominant.
- Rapid's own transaction passed two further six-task server runs (12/12),
  byte-identical to the same AR control. Run-2 task gains were
  1.405x/1.377x/1.306x/1.390x/1.285x/1.106x, a 1.341x paired median. Median
  category throughput repeated at 35.544 and 35.541 tok/s.
- Streaming dogfood found and fixed the GLM-4/GLM-5 protocol mismatch. The new
  GLM-5 parser treats the template-primed prefix as reasoning until
  `</think>`; a live request returned exactly `STREAM_OK` as content, kept the
  trace only in `reasoning_content`, and completed with `[DONE]`.
- The qualified Q4 sidecar is now public at
  `rapid-mlx/GLM-5.3-Flash-MTP-4bit@e9d62773d3e5272fb298830e8e06fadc4137ae2c`.
  Its 4,183,323,401-byte safetensors file has SHA-256
  `369cf9c0f9cdf3ae5f1b9f72d3db8e65ad3026b8e416b8de5618e9117d3f00ca`.
  The activation branch pins both it and the target by immutable revision.

## Unresolved

- The post-0.7 GLM runtime is not in the currently pinned release dependency.
- Rapid still pins a released mlx-vlm version without the qualified upstream
  changes. PRs #2206 and #2231 are upstream-only, so no release
  dependency is available to integrate yet.
- The Rapid serial-server thinking-budget fix is independently releasable, but
  GLM MTP capability metadata must remain disabled until a tagged mlx-vlm
  release contains the qualified cache-owned runtime and Q4 head loader.
- The immutable sidecar gate is cleared. Runtime availability is now the only
  external activation blocker.
- #2241 is intentionally stacked on #2206's source branch and cannot be
  retargeted to main until #2206's current conflict is resolved.
- Creative prose still needs blind human review before any quality claim.
- The Q4 target plus MTP reaches 188.679 GB peak Metal memory. The Studio had
  only 23 GiB of cache-volume headroom during the follow-up, so a complete new
  mixed-quant derivative cannot be staged without an explicit storage plan.

## Risks

Do not infer trajectory equivalence from oMLX's small throughput lead: even its
same-width depth-1 path changed greedy reasoning and half the final answers.
Do not vendor an ad-hoc subset of the upstream work. #2206 replaces the legacy
speculative transaction and its own AR control must remain the equivalence
oracle; #2231 supplies the required Q4 strict-load path. #2234 is closed and
must not be treated as a release dependency.
Do not transplant oMLX's older recurrent-slicing result without re-measuring
current mlx-vlm; the optimized current kernel has a different crossover and
did not preserve bit-exact state in the production-shape check.

## Next action

Atlas should wait for a tagged mlx-vlm release containing the required
structural seams, update the exact dependency pin, and re-run the six-task
harness through that released wheel. The alias activation is already
fail-closed: an older runtime keeps an unflagged serve on AR, while an explicit
native request fails before loading either checkpoint. Include #2241's replay
scheduling change when released, but do not block on experimental K2/K3
adaptation. Do not delete the cached Q4 control without explicit human
authorization and a recovery plan.
