# Vector handoff: model-free CUA server

- Owner: Vector
- Branch/worktree: `vector/cua-model-free-server`, `/private/tmp/harbor-desk-cua-model-free-server`
- Base: server CUA stack `45b92075be7ca0a7a738309c6cfa11494b165bdc`
- Goal: provide an authenticated app-owned CUA control plane without model resolution, download, or load.
- Contract: `rapid-mlx serve --cua-only`; health readiness returns `ready=true`, `model=null`, `model_loaded=false`; clients then authenticate `/v1/cua/capabilities`.
- Scope: lightweight FastAPI host for health and CUA routes, CLI dispatch and validation, tests, server guide. No GUI, inference routes, low-level action API, grounding changes, or model behavior.
- Reference check: existing Rapid lazy audio/model-free startup paths and primary serving precedent readiness semantics were reviewed. The adopted design separates process readiness from model readiness and uses a dedicated lightweight app because importing the inference server initializes engine modules.
- Coordination: Pixel requires exact argv `serve --cua-only --host 127.0.0.1 --port N --cors-origins http://127.0.0.1 http://localhost`, existing watchdog/API-key environment variables, then health and authenticated capability probes.
- Verification: focused CLI tests, CUA server suite, Ruff, diff check, and a real Python 3.12 subprocess smoke covering readiness, unauthenticated 401, authenticated capabilities, and no model startup message.
- Follow-ups: a separate backend PR must correct system overlay handling in topmost-window occlusion checks. A later grounding PR should include unlabeled editable controls and address bounded AX tree truncation.
- FYI: team messaging outside the active task channel was unavailable; equivalent scope/status was recorded here and coordinated directly with Atlas and Pixel.
