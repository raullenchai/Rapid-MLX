# General Computer Use MVP

This developer harness connects Meta's open-source `metacua` macOS agent loop
to a local Rapid-MLX server running Muse Glimmer. It is the first dogfood slice
for a general desktop Computer Use experience: screenshot, reason, act, take a
fresh screenshot, and repeat until the model calls `computer.stop`.

The wrapper keeps the first experiment bounded:

- the model endpoint must be an HTTP loopback IP;
- shell access is disabled;
- sessions stop after 20 actions by default, with a hard limit of 40;
- at most five screenshots remain in model context;
- Meta's normalized 0-1000 Computer Use action contract is used;
- planner output is capped at 768 tokens and high reasoning effort by default;
- Meta's dotted tool names and nested screenshots are translated to Rapid's
  OpenAI-compatible Responses wire format.

`metacua` drives the real foreground macOS session and therefore needs Screen
Recording and Accessibility permission. Use a benign test task and watch the
session. Stop it with Control-C if it leaves the intended task.

## Run

Start Rapid-MLX with a per-run bearer:

```bash
export RAPID_MLX_API_KEY="$(openssl rand -hex 24)"
rapid-mlx serve muse-glimmer-30b-4bit \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$RAPID_MLX_API_KEY"
```

In another terminal:

```bash
cd tools/general_cue_mvp
uv run rapid-general-cue \
  --goal "Open Calculator and compute 17 times 23"
```

## Laya shadow scoring

Run the small decision service on CPU:

```bash
export RAPID_MLX_SYSTEM_ONE_API_KEY="$(openssl rand -hex 24)"
rapid-mlx system-one convaiinnovations/laya \
  --device cpu \
  --port 8700 \
  --api-key "$RAPID_MLX_SYSTEM_ONE_API_KEY"
```

Then score each Muse action without changing it:

```bash
uv run rapid-general-cue \
  --goal "Open Calculator and compute 17 times 23" \
  --jev-mode shadow
```

`--jev-mode guard` replaces an action below
`--jev-execute-threshold` with a screenshot-only re-observation. Keep this mode
for experiments: the released Laya checkpoint has not been trained or
qualified as a Computer Use safety policy.

Run the checked-in next-action probe against the decision server:

```bash
uv run python eval_jev_candidates.py
```

The trace records the decision under `response._rapid_jev`. Screenshot bytes
are not sent to System One; it sees the goal, recent text history, Muse
reasoning, and proposed action.

The pinned upstream revision is Meta Model Cookbook commit
`c5a9882b4c7f3900e9c93122249d37317c8abff4`, licensed MIT. Its trace viewer can
render the session records stored under `~/.metacua/traces/`.
