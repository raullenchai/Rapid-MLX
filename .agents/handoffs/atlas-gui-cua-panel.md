# Handoff — Atlas: GUI CUA panel (PR #3807)

**Branch**: `atlas/gui-cua-panel`
**Dependency**: #3805 server CUA API; restack onto its merged `main` commit before queueing.

## Product behavior

The experimental Computer Use page can start a goal-driven CUA run through the app-owned loopback server, poll numbered progress events, approve the optional sign-in pause, stop the run, and show its result. The client accepts only `127.0.0.1`, a valid port, and a nonempty bearer token.

## Review fixes

- Recreate panel state when the app-owned server rotates its bearer token, preventing stale credentials after a restart.
- Keep an active run controllable when approve/stop requests fail; the error remains visible and the user can retry instead of losing the Stop button.
- Decode and display server terminal errors and ignore empty summaries when choosing a useful failure message.
- Reject non-HTTP responses and extract FastAPI `detail` strings for readable errors.
- Keep service-side approval, cancellation, empty-AX, and planner-error fixes in dependency #3805 instead of duplicating them in this GUI PR.

## Verification

- `swift build` passes on the restacked GUI tip.
- The added CUA source compiles under Swift 6. The local full `RapidTests` target is currently blocked before execution by pre-existing main-branch actor-isolation errors in unrelated Community Benchmark tests; GitHub CI remains the merge authority.
- #3805 targeted Python verification: 108 passed, 100% changed-line coverage, Ruff clean.

## Earlier Studio dogfood

The original stacked panel completed multi-step Chrome tasks with the configured GLM slow planner and local fast ranker, including a successful rerun after the empty-AX-tree failure was fixed. Traces were written under `~/.rapid-mlx/cua-runs/`.
