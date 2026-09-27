# Handoff — Atlas: GUI CUA panel (PR #3807)

**Branch**: `atlas/gui-cua-panel`
**Dependency**: #3805 server CUA API, now merged into `main`.

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
- The added CUA source compiles under Swift 6. GitHub's full macOS build and Swift test job passes on the restacked branch.
- Tool, loop, and server regression verification: 113 tests pass; Ruff and changed-line coverage pass.

## Earlier Studio dogfood

The original stacked panel completed multi-step Chrome tasks with the configured GLM slow planner and local fast ranker, including a successful rerun after the empty-AX-tree failure was fixed. Traces were written under `~/.rapid-mlx/cua-runs/`.


## Round 3 (2026-09-27 afternoon): tool-layer hardening from live dogfood

Four production failure modes found and fixed on this branch (commits through `2b3c88dc`):

1. Snapshot index drift → stale-element honest results (backend.py) + `_execute` ComputerUseError containment (loop.py).
2. Wedged AX services (no-timeout macOS AX calls froze runs 8+ min) → 20 s daemon-thread watchdog in `backend._collect_with_timeout`, routed into the honest-stop path.
3. pyobjc NSArray is never `isinstance(list)` — `_walk`/`collect` silently emptied all snapshots → shared `_as_list` normalization (ax_driver.py).
4. App resolution: exact-name preference beats system XPC helpers (`ThemeWidgetControlViewService (Rapid)`); AXManualAccessibility poke restricted to Chrome-family (it corrupts AppKit/SwiftUI trees mid-rebuild).

**E2E proof**: pizza form via GUI panel — 5 steps, Run completed, and `AXValue` readback verified. The local fast ranker returned a verdict for each step in 0.17–0.21 seconds.

**Unresolved**:
- Dogfood matrix T4 (form via panel with openURL binding), T5 (commerce consent gate), and T3 (Finder dry-run) are still pending. AX setValue does not sync every SwiftUI TextField binding; those controls need a focus/commit fix or real-keyboard typing fallback.
- Chrome AX wedges repeatedly under load (watchdog now degrades honestly; self-heal via AXManualAccessibility reset is a candidate tool-layer PR).
