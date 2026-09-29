# Harbor handoff: local computer-use action contract

- **Owner:** Harbor, coordinated by Atlas
- **Branch:** `harbor/cua-local-action-contract`
- **Worktree:** `/private/tmp/harbor-desk-cua-contract`
- **Base:** `harbor/cua-auth-gate` at `94353c6b43b4c019e2a2253bf770583c58f29dcf` (#3824)
- **Scope:** Python `rapid_mlx.computer_use`, focused tests, and this handoff

## Intention and boundaries

Bind local macOS observations and actions to a stable process ID plus actual
`CGWindowID`, reject stale or drifted targets before input, keep element
indexes snapshot-scoped, and report whether an action was attempted separately
from whether its outcome was verified. The planner remains model and endpoint
agnostic; all actuation stays on the user's Mac.

Remote VMs, remote browser execution, planner selection, Swift UI, new action
types, and changes to the higher-level CUA loop are outside this change.

## Reference check

Private implementation research reviewed the existing Rapid Swift exact-window
actuator and window-identity code first. The adopted pattern is fail-closed
validation of process, exact window identifier, frame, focus, and occlusion at
the last input boundary. A public computer-use CLI precedent was also reviewed;
its stable window selector and snapshot-act-snapshot workflow were adapted to
the existing Rapid CLI. No external implementation code or branded asset was
copied. Existing browser/server precedents were not applicable because this
change is a local macOS Accessibility boundary.

## Verified facts

- `list-windows` exposes the opaque `cg:<n>` `window_id`, filters by owner PID,
  and performs discovery without activating the app. Plain numeric selectors
  remain accepted at the backend boundary for compatibility.
- `get-app-state --window-id` maps the CG window to exactly one AX window by
  frame; zero or ambiguous matches fail closed. AX collection remains limited
  to that one window.
- Actions revalidate age, PID, ID, frame, point containment and topmost window.
  Keyboard input additionally requires the selected frame to be the focused AX
  window.
- CLI actions opt into one fresh post-action state. The higher-level CUA loop
  does not, because it already performs one post-action observation.
- Synthetic events return `attempted: true`, `verified: null` unless an exact
  AX value readback proves the requested text.
- Domain guards read the active-tab URL only through the browser application
  API. Page-controlled Accessibility values are never accepted as origin
  evidence; unavailable Automation permission produces an empty URL and the
  configured domain guard fails closed. The selected observation window must
  also remain the app's frontmost, focused window before that URL can authorize
  an action, preventing one browser window from authorizing another.

## Verification and limitations

Focused unit tests cover window reorder, stale observation, AX-to-CG exact
window mapping, coordinate escape, and unverified synthetic input. The changed
Python lines have 100% combined focused-test coverage, and the pinned mypy
budget reports no growth. The current
automation host reports Accessibility and Screen Recording permissions false
and exposes no running GUI apps, so a real TextEdit observe/action smoke could
not run. Frame-coordinate agreement and Dock-overlay behavior therefore remain
live-smoke items; both paths fail closed rather than send input when uncertain.

The required team start FYI could not be sent because the role mailbox returned
`terminal_not_found`; this handoff records the same scope and plan per AGENTS.md.

## Next action

Atlas should review the scoped diff and test evidence. Before merge, run one
permissioned local smoke on a disposable TextEdit document: list windows,
observe by `window_id`, set a text value, confirm exact readback and fresh state,
then cover the point with another window and confirm the click is refused.
