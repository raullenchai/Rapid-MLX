# rapid_mlx.cua — Computer-Use Agent (productized)

Native macOS Accessibility computer-use agent. Replaces the GUI-verifier POC
loop (`tools/gui_verifier_cua_poc/`) with the product pipeline:

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

Known gaps: URL guard needs Automation TCC (AppleScript otherwise rejected
with -1743); validator rejects hallucinated elements but not omitted files in
planner-only flows; shopping pre-ranking is DOM-specific and not yet ported.
