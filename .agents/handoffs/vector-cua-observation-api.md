# Vector handoff: CUA visual observation API

- Receiving role: Atlas
- Owner: Vector
- Host: Studio
- Branch: `vector/cua-observation-api`
- Worktree: `/private/tmp/harbor-desk-cua-observation-api`
- Base: selected-window branch `2feecb9c085a46c699bd75125a93141cc18b21eb`
- PR: intentionally not opened or pushed; waiting for stack linearization and manager review

## Intention and boundary

Add one authenticated, rate-limited, read-only API operation that returns a
fresh AX observation for an exact app PID and opaque window ID. Screenshots are
off by default and require both server policy and request opt-in. No action,
focus, activation, click, typing, or native GUI work is included.

## Verified facts

- The existing `get_app_state` action path activates the app by default.
  Observation now passes an explicit `activate=False` boundary while preserving
  the existing default for action callers.
- Every observation uses `use_cache=False`, resolves the app via `pid:<pid>`,
  and supplies the requested window ID to the backend.
- Accessibility TCC must be granted. Screen Recording TCC is additionally
  required for PNG output.
- PNG exposure requires `RAPID_MLX_CUA_EXPOSE_SCREENSHOTS=1` and
  `screenshot=true`; raw PNGs above 4 MiB are rejected before base64 encoding.
- The response omits backend `tree_text` and raw `screenshot_png`; observation
  responses use `Cache-Control: no-store` and `Pragma: no-cache`.
- Capability flags reflect platform, current TCC readiness, and screenshot
  server policy. Public server and CUA docs describe the exact request,
  response, privacy headers, opt-in, TCC, and payload-limit contract.

## Reference check

Primary serving engines, MLX-native implementations, and desktop clients were
reviewed for an existing observation contract. No serving implementation
provided the required host AX/window API. A current macOS desktop capture flow
did provide useful precedent for preflighting Screen Recording permission and
avoiding window activation; those two patterns were adapted without copying
source or assets. Detailed project names and research notes remain outside the
public repository.

## Verification

- Combined computer-use, CUA, server, and observation suites: 186 passed,
  2 real-server import tests deselected because importing MLX aborts in this
  isolated test environment.
- Ruff check and format check pass for all touched Python files
- `git diff --check` passes

## Risks and next action

- The public request/response shape is a compatibility decision. Atlas should
  confirm endpoint naming and whether the 4 MiB raw PNG limit should become a
  normal server configuration field before the stack is published.
- The server opt-in is intentionally environment-only for this slice. A CLI or
  Desktop setting would expand scope and should be assigned separately.
- Next action: Atlas reviews the local commit, chooses the linear stack base,
  and cherry-picks/rebases it before any push or PR creation.
