# Semantic action protocol browser CUA POC

Date: 2026-09-25
Owner: Atlas
Status: Experimental; no product claim

## Goal

Test whether a typed agent action protocol plus a local fast decision model can
turn the earlier GLM-5.3 + GUI-Actor-Verifier browser experiment into a smoother
end-to-end flow. The task starts from a clean Amazon home page, searches for a
flashlight, compares visible organic results by rating and review count, and
stops on the recommended product detail page. Cart, checkout, account,
credential, payment, and purchase actions remain blocked.

## Architecture

The POC uses three local-model roles:

1. `GLM-5.3-Flash-EXL3` at `reasoning_effort=low` performs visual intent
   resolution and evidence-based product selection through an OpenAI-compatible
   endpoint.
2. `convaiinnovations/laya` runs on the Mac Studio CPU through Rapid's System
   One `/v1/rank` endpoint. It classifies structured action outcomes and
   pre-ranks visible product cards for GLM review.
3. `microsoft/GUI-Actor-Verifier-2B` confirms click points on the Mac Studio.

The executor publishes only current-page semantic target IDs. The planner emits
`click(target_id)`, `fill(target_id, text, submit)`, `submit(target_id)`,
`scroll`, `wait`, or `done`. It cannot emit raw coordinates. Click candidates
come from the target's live bounding box and must still resolve to the same
target at execution time. The center point is executed immediately when it
passes the verifier threshold; alternate points are evaluated only on failure.

Each action returns URL, visible-text hash, scroll position, focused target,
active input value, and execution error. A typed postcondition settles clear
successes. Laya handles ambiguous structured outcomes; GLM visual reflection is
the final fallback.

Page text is explicitly treated as untrusted prompt data. Navigation targets
must remain on the start site's domain, and the shopping terminal guard accepts
completion only on a product detail URL.

For the shopping fast path, the controller scrolls until at least three visible
organic product cards exist. Laya pre-ranks those cards, then GLM makes the
final choice. A verified navigation to a product detail URL terminates with the
comparison generated during that choice, avoiding another GLM call.

## Environment

- Mac Studio, Apple M3 Ultra, 256 GB unified memory
- macOS 26.5.2
- MLX 0.32.2, mlx-vlm 0.7.2, laya-mlx 0.2.0
- POC base commit before this change: `7f6067158`
- Laya revision: `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`
- GUI verifier revision: `30dd0db468762d45df20b5a01c4084c5a3ed3ca3`
- GLM revision: `25a44fdbf16862a46b7cc9921142c6c81350af2f`
- GLM speculative draft revision:
  `7d74cdd881ed7e32c31175984a67823127b66cfe`
- Visible system Chrome, clean temporary profile, 1280x800 viewport

The GLM endpoint was the user's existing two-node Spark vLLM service, forwarded
to a literal loopback address. No model server configuration was changed.

## Results

All runs started from a new browser profile and completed on the Lepro LED White
Light Flashlight detail page without a cart or purchase action.

| Variant | Completed | Model steps | GLM plan calls | GLM reflections | In-loop wall time |
|---|---:|---:|---:|---:|---:|
| Previous raw GLM loop | yes | 11 | 11 | 10 | 189.74 s |
| Semantic actions, run 1 | yes | 5 | 5 | 0 | 82.12 s |
| Semantic actions, run 2 | yes | 5 | 5 | 0 | 75.19 s |
| Fast controller before context fix | yes | 4 | 3 | 0 | 45.44 s |
| Fast controller after context fix | yes | 3 | 2 | 0 | **37.37 s** |
| Final same-site/terminal-guard rerun | yes | 3 | 2 | 0 | **37.93 s** |

The timer begins after local verifier loading and initial browser launch, so it
measures the interactive loop rather than cold startup.

The final run performed:

1. GLM: atomic `fill(searchbox, "flashlight", submit=true)`;
2. fast controller: one scroll to gather organic candidates;
3. Laya: pre-rank four organic cards; GLM: compare and select; verifier: confirm
   the derived click point; executor: navigate and terminate.

The final rerun's two GLM plan calls consumed 27.37 seconds. Laya outcome checks
took 0.292-0.503 seconds, its four-product pre-ranking took 0.255 seconds, and
the single GUI verification took 1.59 seconds with `P(True)=0.905`. No visual GLM
reflection was needed.

GLM compared four organic cards and explicitly excluded sponsored OLIGHT
results. The recommendation evidence was:

- Lepro: 4.6 stars, 50.8K ratings, $9.99, Overall Pick;
- TrixHub: 4.5 stars, 6.9K ratings, $22.79;
- Victoper: 4.5 stars, 16.2K ratings, $9.95;
- D511 clip light: 4.2 stars, 5 ratings.

Laya also ranked Lepro first, but its probability was only 0.3024. The fast
ranking is therefore advisory evidence for the strategist, not authority to
click or make the final product claim.

## Bugs found and fixed

- A free `type` action allowed models to type into no focused target. Atomic
  `fill` now carries the target and optional submit operation.
- Screenshot reflection could not observe keyboard focus and caused repeated
  clicks. Focus and input value are now structured postconditions.
- Model-generated coordinate alternatives referred to unrelated controls.
  Candidates now come only from one semantic target's bounding box and are
  checked again against that target before execution.
- Three serial verifier calls added about 5.5 seconds. The common path now uses
  one center-point verification and expands only when necessary.
- The first fast-path run put Laya's pre-ranking after a 6,500-character page
  context and it was truncated before reaching GLM. Pre-ranking now precedes
  the page context; GLM then selected immediately instead of scrolling again.
- Same-site navigation and product-detail terminal guards close two prompt
  injection paths that otherwise allowed page text to redirect or end the task.

## Purchase-mode dogfood (2026-09-26)

A `--purchase` mode relaxes the guard to allow cart/checkout/place-order while
still hard-blocking credential, password, CVV, and card-number entry. Two human
gates make the money moment explicit: the runner pauses at an Amazon sign-in
page until the human signs in inside the Chrome window and touches `RESUME`, and
it pauses before any place-order click, writes `order-summary.json`, and waits
for `CONFIRM_ORDER` (or the pause timeout, which ends the run with
`awaiting_human` recorded). Termination requires the Amazon order-confirmation
page (`/buy/confirmation` or "Order placed" text).

Three dogfood runs exercised the pipeline (GLM-5.3-Flash low reasoning + laya
pre-ranking + GUI-Actor-Verifier-2B, signed-in persistent profile):

- Search, compare, product selection, buy-box scroll, and add-to-cart all ran
  without human input. Add-to-cart was verified by the typed cart-count
  postcondition (`#nav-cart-count` 0 → 1, and 1 → 2 on a re-run). Mean GLM
  planning latency was 9.3–11.2 s per step across the runs.
- The cart page's Prime Video ad banner was misread by the planner as the
  checkout button three times; GUI-Actor-Verifier correctly rejected every
  click (P(True) 0.47–0.56 < 0.65), after which the planner looped on the wrong
  target. Fix: expose DOM ids in target context plus a deterministic cart-page
  checkout bootstrap that still passes verifier review.
- The executor-side research guard was still applied inside `_verified_click`,
  rejecting add-to-cart clicks in purchase mode; threading `forbidden_re`
  through verifier-click and bootstrap paths fixed it.
- One background run died silently between gates (no traceback); restarts with
  the persistent profile resumed cleanly. Gates, timeouts, and graceful
  `awaiting_human` exits behaved as designed.

No order was placed: per operator decision the dogfood stopped at a populated
shopping cart, and the sign-in gate timeout ended the final run before checkout.
The end-to-end purchase path remains gated behind `CONFIRM_ORDER` for a
human-authorized run.

## Product assessment

The resulting browser movement is coherent: one search operation, one scroll,
and one verified product click. The loop is materially smoother than the raw
GLM agent and recovered 80% of its wall time. It is still an experimental
Amazon-specific fast path. The user waits about 10 seconds for intent resolution
and 16 seconds for the final GLM comparison, so the model checkpoints remain the
dominant visible pauses.

The protocol is model-independent. A future Qwen 27B planner can replace GLM
through the same OpenAI-compatible interface and should be evaluated against
the same evidence trace. General desktop Computer Use still needs native macOS
Accessibility targets, non-browser app coverage, typed drag/select operations,
and a held-out task suite.

## Reproduction

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  rapid-mlx system-one convaiinnovations/laya \
  --backend laya --device cpu --port 18700

python tools/gui_verifier_cua_poc/run_poc.py \
  --planner-url http://127.0.0.1:18888/v1/chat/completions \
  --planner-model GLM-5.3-Flash-EXL3 \
  --reasoning-effort low \
  --fast-ranker-url http://127.0.0.1:18700/v1/rank \
  --fast-ranker-model convaiinnovations/laya \
  --shopping-fast-path \
  --start-url https://www.amazon.com/ \
  --max-steps 10
```

Run artifacts remain under `/private/tmp`; screenshots and clean-browser
profiles are not committed.
