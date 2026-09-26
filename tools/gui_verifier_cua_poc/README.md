# GUI verifier Computer Use POC

This experiment separates three possible failure sources in a browser task:

1. **Planner:** an OpenAI-compatible vision model chooses the next semantic
   action and proposes click coordinates.
2. **Grounding verifier:** GUI-Actor-Verifier-2B scores the proposed coordinates
   against the current screenshot and the planner's step instruction.
3. **Reflection:** the planner compares screenshots before and after execution
   and decides whether the step worked.

The runner records every candidate, verifier score, DOM element under the point,
selected action, page transition, and reflection. It uses visible Google Chrome
through Playwright, but click execution is coordinate based. DOM information is
supplied only as compact page context and for post-run attribution.

The default shopping task stops at a product detail page. A hard guard rejects
cart, checkout, purchase, account, credential, and payment actions.

## Run

Install the optional vision runtime and browser driver into the active Rapid
development environment. The runner uses the installed system Chrome and does
not download a Playwright browser bundle.

```bash
uv pip install -e '.[vision]' playwright
```

Start a Rapid vision server with the cached Qwen planner:

```bash
rapid-mlx serve qwen3.5-9b-4bit \
  --host 127.0.0.1 --port 18730 --mllm --no-thinking
```

Then run the POC:

```bash
python tools/gui_verifier_cua_poc/run_poc.py \
  --planner-url http://127.0.0.1:18730/v1/chat/completions \
  --start-url https://www.amazon.com/ \
  --goal '在 Amazon 上找到评价最好且评价数量足够可信的手电筒，比较前几个结果，停在推荐商品详情页。不要加入购物车或购买。'
```

For servers that expose OpenAI-compatible reasoning levels, add
`--reasoning-effort low`, `medium`, `high`, or `max`. The selected value is
recorded in `trace.json`.

Artifacts are written under `/private/tmp/rapid-mlx-gui-verifier-poc-runs/`.
Screenshots may contain private browser data; use a clean temporary browser
profile and do not commit run artifacts.
