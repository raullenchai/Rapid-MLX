# Planner + GUI-Actor-Verifier browser POC

Date: 2026-09-25  
Owner: Atlas  
Status: Experimental; no product claim

## Question

For a real browser shopping-research task, is the limiting factor the 9B
planner/reflection model or the 2B GUI grounding verifier?

The task was: find a highly rated flashlight on Amazon with enough reviews to
make the rating credible, compare candidates, and stop on the recommended
product page. Cart, checkout, account, credential, and payment actions were
blocked.

## Setup

- Mac Studio, Apple M3 Ultra, 256 GB unified memory
- macOS 26.5.2
- Rapid-MLX base revision `e68bb534f`
- Planner: `mlx-community/Qwen3.5-9B-4bit`, revision
  `8b2b98c00a6b4d291155e4890773ca8f769aee53`
- Verifier: `microsoft/GUI-Actor-Verifier-2B`, revision
  `30dd0db468762d45df20b5a01c4084c5a3ed3ca3`
- MLX 0.32.2, mlx-vlm 0.7.2
- Visible Google Chrome, 1280x800 viewport, clean temporary profile
- Planner screenshots resized to at most 960x600; verifier used the original
  1280x800 screenshot.

The planner received the screenshot plus compact visible DOM text. It produced
one semantic action and three normalized coordinate candidates for the same
target. The verifier saw each point as a red circle and returned the probability
of `True` versus `False`. Execution remained coordinate based. DOM hit-testing
was recorded only for attribution and the safety guard.

## Controls

The verifier separated an obvious synthetic search-field point from a wrong
point:

| Point | P(True) | Output |
|---|---:|---|
| Search field | 0.9669 | True |
| Empty lower page | 0.0097 | False |

On the real Amazon search field, three points inside the field scored 0.953,
0.984, and 0.974. The selected point focused the field and the deterministic
search sequence reached `/s?k=flashlight`.

On a real product page, an oracle point on the `199 reviews` link scored 0.9959.
Wrong points on the product title and image scored 0.2018 and 0.0953. This shows
that the verifier can solve these individual grounding cases when the correct
candidate is present.

## Unguided run

The first run started from the Amazon home page without an oracle action
sequence.

1. Qwen typed `flashlight` before focusing the search field. Reflection correctly
   returned `no_effect`.
2. The search field was off screen, but Qwen proposed three points on a Cinnamon
   card. The verifier incorrectly scored them 0.593-0.706 and selected one.
3. Qwen then treated the cinnamon product as a flashlight candidate. The
   verifier accurately grounded that incorrect local instruction at 0.905-0.988;
   it had no access to the overall-goal validity of the instruction.
4. Qwen recognized the wrong page and found the real search field, but entered a
   focus/type loop and never pressed Enter. Reflections noticed `no_effect`, but
   did not prevent repeating the same failed sequence.

The run was stopped after eight actions. It never reached flashlight results.
An earlier raw attempt also returned an invalid click schema. A later attempt
repeated output until Rapid's repetition guard stopped at 696 tokens and both
the original JSON and model repair were malformed.

## Controlled search run

To isolate behavior after navigation, a deterministic focus/type/Enter sequence
was added. The verifier still selected the search-field coordinate. Strict JSON
schema and screenshot downscaling removed the protocol and 8,192-token vision
budget failures.

| Step | Planner result | Verifier result | Reflection |
|---|---|---|---|
| Search bootstrap | Oracle semantic sequence | Correct field, 0.984 | Search results reached |
| R-300 product | Claimed all points were the product; only one was | Wrong carousel arrow 0.893 beat correct product 0.881 | Correctly `no_effect` |
| S-600 product | Proposed the visible S-600, 4.7 stars / 199 reviews | Correct candidate ranked first at 0.881 | Correctly `success` |
| Reviews link | All three proposed points landed on the title | No correct candidate; selected 0.531 | Correctly `no_effect` |
| Reviews retry | Repeated the same grounding error | No correct candidate; selected 0.731 | Correctly `no_effect` |

Qwen called the R-300 a non-sponsored result even though it was in a sponsored
carousel. It also declared the S-600 the best result without completing the
requested three-product comparison. The run stopped on the S-600 detail page,
but did not establish that it was the best-reviewed flashlight.

## Latency

For the four-step controlled run:

| Component | Mean | Median | Range / count |
|---|---:|---:|---|
| Qwen plan | 38.68 s | 31.46 s | 28.89-62.91 s, n=4 |
| Qwen reflection | 8.30 s | 8.52 s | 5.47-10.71 s, n=4 |
| GUI verifier per candidate | 1.83 s | 1.89 s | 1.56-2.17 s, n=15 |

Three serial verifier candidates add about 5.5 seconds per click. The planner is
still the dominant latency source.

## Attribution

The primary blocker is the 9B planner:

- it orders focus/type/submit incorrectly;
- it fails to preserve the overall goal;
- it fabricates coordinate alternatives that do not share one target;
- it makes unsupported product-comparison claims;
- after a correct reflection, it repeats the failed action instead of applying
  a bounded recovery policy;
- without constrained decoding, its JSON output is not reliable enough for an
  action loop.

The verifier is useful but insufficient:

- it is strong when a correct candidate is present and spatially distinct;
- it can give high confidence to an adjacent carousel control;
- it cannot reject a locally well-grounded instruction that conflicts with the
  user's overall goal;
- it has no built-in abstention rule, candidate-set consistency check, or memory.

## Recommended architecture

1. The planner should choose a semantic target from stable Accessibility/DOM
   node IDs, not generate three raw coordinates.
2. Candidate boxes should come from Accessibility, DOM geometry, or a dedicated
   GUI-Actor action head. The verifier should rerank those grounded candidates.
3. Reject the entire candidate set when the winning margin is small, all points
   resolve to unrelated elements, or the points do not share the requested
   semantic target.
4. Add a separate goal-level action critic before the coordinate verifier. The
   coordinate verifier must not be treated as a policy or safety model.
5. Encode focus -> type -> submit and other compound UI actions as atomic typed
   operations.
6. Make reflection operational: block an identical action after `no_effect`,
   cap retries, and require a different recovery action.
7. Batch verifier candidates or reuse the screenshot vision encoding before
   considering an interactive product experience.

The current combination is a useful research harness. It is not reliable enough
for general browser Computer Use or shopping recommendations.

## GLM-5.3 planner comparison

The same harness was then run against the user's existing two-node Spark vLLM
service:

- planner: `GLM-5.3-Flash-EXL3`, immutable checkpoint revision
  `25a44fdbf16862a46b7cc9921142c6c81350af2f`;
- speculative draft: `incoai/GLM-5.3-Flash-DFlash2`, revision
  `7d74cdd881ed7e32c31175984a67823127b66cfe`;
- `reasoning_effort=low`, strict JSON schema, and the same screenshot input
  format, prompt, verifier, viewport, browser profile policy, and safety guard.

### Controlled search

With the deterministic focus/type/submit bootstrap, GLM completed the remaining
task in four model steps: two scrolls, one product click, then `done`. It examined
visible candidates including 4.7 stars / about 2K ratings, 4.6 / 81.9K, 4.6 /
50.8K, 4.5 / 6.9K, and 4.5 / 16.2K. It rejected a 4.8-star result with only 24
ratings and selected the organic Lepro result with 4.6 stars and 50,823 ratings.
All three proposed points for the product resolved to the same title link and
scored 0.980-0.998. This is a material improvement over Qwen's mixed-target
candidate sets and unsupported comparison.

### Unguided run

GLM also completed the task from the Amazon home page, without the bootstrap,
in 11 model steps. The result was substantively useful and stopped on the same
Lepro product without adding it to the cart. However, four steps were wasted on
repeated search-field clicks:

1. The first plan selected `action=type` while its instruction said to click and
   type. The executor correctly performed only the declared action, so typing
   had no effect because the field was not focused.
2. GLM then proposed three valid points inside the search input. The verifier
   scored them 0.982-0.989 and execution focused the input.
3. Screenshot-only reflection could not observe keyboard focus, reported
   `no_effect`, and GLM repeated the same successful focus action three times.
4. GLM eventually typed the query, submitted it, compared the visible results,
   and opened the selected product.

GLM occasionally described sponsored results as organic while collecting the
comparison set. The final selected Lepro result was visibly organic, but the
wording shows that page-state claims still need structured attribution.

| Component | Qwen controlled | GLM controlled | GLM unguided |
|---|---:|---:|---:|
| Plan mean | 38.68 s | 9.88 s | 7.65 s |
| Reflection mean | 8.30 s | 4.50 s | 4.53 s |
| Model steps | 4, incomplete comparison | 4, completed | 11, completed |
| End-to-end task | Failed | Completed after bootstrap | Completed |

These runs change the attribution. A stronger planner fixes most goal tracking,
comparison, candidate-generation, and latency failures. The remaining dominant
problem is the agent protocol: `type` does not carry a target, focus is not
represented in observable state, compound operations are not atomic, and a
visual reflection cannot reliably detect focus. The verifier remains effective
for clear candidates but still cannot validate sponsored status or the overall
shopping policy.

Before testing another larger planner, the next POC should add typed semantic
actions such as `fill(target_node_id, text)` and `submit(target_node_id)`, record
the focused Accessibility/DOM node after each action, and make `no_effect`
recovery consume that structured state. GLM at low reasoning is already strong
enough to expose this interface bottleneck.
