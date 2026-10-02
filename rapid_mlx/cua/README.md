# rapid_mlx.cua — Computer-Use Agent (experimental)

Experimental native macOS Accessibility computer-use agent. It is opt-in and
may stop early, misread an interface, or require the user to finish a task.
Keep consequential work supervised and review every approval prompt.

The current pipeline provides:

- **Execution layer** (`rapid_mlx.computer_use`): model-agnostic AX snapshot +
  typed actions (click / set-value with read-back verification / press / hotkey
  / scroll), typed errors with recovery hints. Works for browsers AND native
  apps.
- **Fast thinking — always local**: laya (`convaiinnovations/laya`) routes each
  action outcome (success / no_effect / wrong_effect) in ~100 ms, and a
  fixation detector catches small-model loops and injects a recovery hint into
  the next planning prompt.
- **Slow thinking — user's choice**: any loopback OpenAI-compatible endpoint.
  Presets: `local-27b`, `local-9b` (built-in, on-device); cloud brains are added from the app settings or a custom URL with
  `--planner-model`. Config lives in `~/.rapid-mlx/cua-config.json`.
- **Consent gates** (hard, not configurable): no credentials / card fields,
  no commerce fills; optional sign-in human gate via file sentinel
  (`--human-login`); domain guard (`--allowed-domain`, needs Automation TCC
  for reliable URL reads).

## Usage

```bash
# list brains
rapid-mlx cua planners

# run with the on-device 9B (fastest, short flows)
rapid-mlx cua run --app "Google Chrome" \
  --planner local-9b \
  --open-url "https://www.wikipedia.org/" \
  --allowed-domain wikipedia.org \
  --goal "在维基百科搜索 Alan Turing 并打开他的词条页面" \
  --max-steps 8

# run with the cloud brain (vision + long-horizon strength)
rapid-mlx cua run --app "Google Chrome" --planner cloud-glm --goal "..."

# custom brain
rapid-mlx cua run --app "Google Chrome" \
  --planner http://127.0.0.1:9999/v1/chat/completions --planner-model my-model
```

Traces land in `~/.rapid-mlx/cua-runs/<timestamp>/trace.json`.

The `/v1/cua/*` HTTP API controls the local computer and requires the server
to start with an API key. Without one, these routes return HTTP 503. Send the
key as `Authorization: Bearer <key>`; the Desktop-managed server supplies its
own per-launch bearer. The standalone `rapid-mlx cua` CLI does not use this
HTTP API.

### Custom GUI clients

Authenticated clients can implement the complete supervised task journey:

```bash
# Discover host readiness and target windows.
curl -H "Authorization: Bearer $RAPID_API_KEY" http://127.0.0.1:8000/v1/cua/capabilities
curl -H "Authorization: Bearer $RAPID_API_KEY" http://127.0.0.1:8000/v1/cua/permissions
curl -H "Authorization: Bearer $RAPID_API_KEY" http://127.0.0.1:8000/v1/cua/apps
curl -H "Authorization: Bearer $RAPID_API_KEY" \
  http://127.0.0.1:8000/v1/cua/apps/Google%20Chrome/windows

# Create a high-level run. Raw click and typing operations are not exposed.
curl -X POST -H "Authorization: Bearer $RAPID_API_KEY" \
  -H 'Content-Type: application/json' http://127.0.0.1:8000/v1/cua/runs \
  -d '{"app":"Google Chrome","goal":"Open the Alan Turing article","planner":"local-9b"}'

# Poll the returned run id. Use events_after_seq as the next `after` cursor.
curl -H "Authorization: Bearer $RAPID_API_KEY" \
  'http://127.0.0.1:8000/v1/cua/runs/RUN_ID/events?after=0'

# When pending_gate is present, echo its gate_id to approve or deny it.
curl -X POST -H "Authorization: Bearer $RAPID_API_KEY" \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8000/v1/cua/runs/RUN_ID/approval \
  -d '{"gate_id":"GATE_ID_FROM_PENDING_GATE","approved":true}'

# Cancellation is idempotent for an existing run.
curl -X POST -H "Authorization: Bearer $RAPID_API_KEY" \
  http://127.0.0.1:8000/v1/cua/runs/RUN_ID/cancel
```

Run responses contain a typed event envelope (`kind`, `seq`, `ts`) with
event-specific fields such as `action`, `target`, `target_label`, `outcome`,
and `tree_changed`, terminal status and summary, and the current `pending_gate`.
Local trace paths are intentionally omitted from HTTP responses. Fresh window
observations are available through the separately permissioned observation API;
screenshots remain disabled unless the server and request both opt in.

Every `gate`, `gate_detail`, and `gate_resolved` event for one decision carries
the same `gate_id` as `pending_gate`. New clients should always send that ID in
their decision request.

Window discovery returns an opaque `window_id` such as `cg:123`, resolved
against the target process ID. Run creation can bind the run to that exact
window; clients should rediscover windows before retrying a stopped run.
Clients should inspect the versioned capability response instead of inferring
support from route presence. Observation capability flags reflect the host
platform, current Accessibility and Screen Recording grants, and screenshot
server policy, while `approval_gate_id` remains true.

Runs and events are retained only in the server process (up to 100 recent
runs). A server restart clears them, so an old `run_id` can return HTTP 404;
clients should treat that as an expired session and start a new run.

Native clients are not subject to browser CORS checks. For a browser GUI on a
different origin, start the server with that exact origin in `--cors-origins`
or `RAPID_MLX_CORS_ALLOW_ORIGINS`; do not use a wildcard for a computer-control
server.

## SDK

```python
from rapid_mlx.cua.config import resolve_planner, CUAConfig
from rapid_mlx.cua.loop import run
from rapid_mlx.cua.planner import Planner

config = CUAConfig(planner=resolve_planner("local-9b"))
trace = await run(config, "Google Chrome", goal="...", planner=my_planner)
```

## Qualification matrix (measured 2026-09-26/27)

| brain | planner-only flows | short browser flows | long-horizon browser |
|---|---|---|---|
| GLM-5.3-Flash (cloud) | ✅ | ✅ | ✅ |
| Qwen3.8-27B local | ✅ | ✅ | ⚠️ premature done |
| Qwen3.5-9B local | ✅ | ✅ | ❌ fixation (mitigated by the loop's intervention gate, still weakest) |

Known gaps: browser URL verification needs Automation permission (AppleScript
is otherwise rejected with -1743); local planners can stop prematurely or loop
on longer tasks; changing or obscured windows can stop a run; and the validator
can reject hallucinated elements but cannot prove that a planner reported every
relevant item. Shopping workflows are unsupported. Credentials, payment
details, and commerce actions remain hard blocked.
