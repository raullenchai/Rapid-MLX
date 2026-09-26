# GUI verifier Computer Use POC

This experiment separates four possible failure sources in a browser task:

1. **Planner:** an OpenAI-compatible vision model chooses the next semantic
   action using current-page target IDs.
2. **Fast controller:** a local System One ranker handles structured outcome
   routing and pre-ranks visible product candidates.
3. **Grounding verifier:** GUI-Actor-Verifier-2B confirms coordinates compiled
   from the selected target's live bounding box.
4. **Action protocol:** typed actions return URL, DOM, scroll, focus, and input
   value deltas. The planner is used for visual reflection only when those
   postconditions and the fast ranker cannot classify the result.

`fill(target_id, text, submit=true)` atomically focuses, replaces, and submits an
input. The model never generates coordinates. For clicks, the runner first
tries the target center and evaluates more points only when the center fails the
verifier threshold or no longer resolves to the intended element.

The optional shopping fast path lets the local controller gather enough organic
result cards, pre-rank them, and call the planner only for intent resolution and
the final evidence-based choice.

The default shopping task stops at a product detail page. A hard guard rejects
cart, checkout, purchase, account, credential, and payment actions.

With `--purchase`, the guard relaxes to cart/checkout/purchase actions only:
credential and payment-data entry stays blocked, and two human gates make the
money moment explicit. When Amazon asks for sign-in the runner pauses until the
human signs in inside the opened Chrome window and touches `RESUME` in the run
directory (or the pause timeout ends the run). Before executing any place-order
click the runner writes `order-summary.json`, then pauses until the human
touches `CONFIRM_ORDER`. The run terminates on the Amazon order-confirmation
page. Use `--profile-dir` to reuse a signed-in browser profile across runs;
never commit profile directories or order summaries.

## Run

Install the optional vision runtime and browser driver into the active Rapid
development environment. The runner uses the installed system Chrome and does
not download a Playwright browser bundle.

```bash
uv pip install -e '.[vision,system-one]' playwright
```

Start the local fast ranker:

```bash
rapid-mlx system-one convaiinnovations/laya \
  --backend laya --device cpu --port 18700
```

Start or forward an OpenAI-compatible vision planner to a loopback port, then
run the POC:

```bash
python tools/gui_verifier_cua_poc/run_poc.py \
  --planner-url http://127.0.0.1:18888/v1/chat/completions \
  --planner-model GLM-5.3-Flash-EXL3 \
  --reasoning-effort low \
  --fast-ranker-url http://127.0.0.1:18700/v1/rank \
  --shopping-fast-path \
  --start-url https://www.amazon.com/ \
  --goal '在 Amazon 上找到评价最好且评价数量足够可信的手电筒，比较前几个结果，停在推荐商品详情页。不要加入购物车或购买。'
```

End-to-end purchase variant (human-gated sign-in and order placement):

```bash
python tools/gui_verifier_cua_poc/run_poc.py \
  --planner-url http://127.0.0.1:18888/v1/chat/completions \
  --planner-model GLM-5.3-Flash-EXL3 \
  --reasoning-effort low \
  --fast-ranker-url http://127.0.0.1:18700/v1/rank \
  --shopping-fast-path \
  --purchase \
  --profile-dir /private/tmp/rapid-mlx-cua-profile \
  --pause-timeout 600
```
```

The planner and fast-ranker URLs must use literal loopback IPs. Every selected
model, reasoning level, candidate, score, action, state delta, and fallback is
recorded in `trace.json`.

Artifacts are written under `/private/tmp/rapid-mlx-gui-verifier-poc-runs/`.
Screenshots may contain private browser data; use a clean temporary browser
profile and do not commit run artifacts.
