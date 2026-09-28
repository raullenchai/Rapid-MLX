# CUA Desktop window selection

- **Owner:** Pixel, with Vector contract coordination
- **Branch/worktree:** `pixel/cua-window-selection` at
  `/private/tmp/harbor-desk-cua-window-selection`
- **Base content:** GUI task flow head `82532b23f`
- **Status:** implementation complete locally; do not push before manager review

## Intention and scope

The native task panel discovers running processes and their windows, requires an
explicit PID plus opaque window identity, and binds that pair into run creation.
The change is limited to the Desktop client, view model, SwiftUI panel, user
documentation, and tests. It does not modify the server, computer-use backend,
sidecar packaging, observation API, or low-level action API.

## Verified behavior

- App discovery decodes the server's `bundle_id`; process labels include the app
  name and PID.
- Window discovery addresses the selected process as URL-encoded `pid:N`.
  Window labels include title, one-based index, and dimensions without exposing
  the opaque identity.
- Start remains disabled until both a discovered PID and a currently discovered
  window are selected. Run creation sends `app: pid:N` and `window_id`; it never
  falls back to the legacy free-text app name.
- Refresh revalidates both identities. Missing processes or windows clear the
  selection. A generation token prevents older concurrent responses from
  replacing newer discovery state.
- Typed stale-window failures during creation or execution clear the window and
  tell the user to refresh and choose again. Discovery errors, permission
  failures, loading, and empty results remain visible.
- Older sidecar run views remain decodable when `window_id` is absent. A sidecar
  without discovery routes produces an update/restart message and cannot start
  an unbound run.
- The page subtitle states that actions execute on this Mac and that the user
  chooses the planning endpoint.

## Private reference check

Reviewed the existing native Desktop picker, refresh, permission, empty-state,
and accessibility patterns first. Also checked the established target-selection
flows in Orca and the reference desktop assistants named by the engineering
handbook. Adopted explicit hierarchical process/window selection, stable
secondary identity, refresh-in-place, and fail-closed stale selection. No source
or branded asset was copied.

## Coordination

The server contract comes from selected-window run PR #3832. Orca role mailbox
tools were not available in this session; task start and progress were relayed
to Atlas through the active agent channel. Send the standard Pixel, Vector,
Harbor, Echo, and firefighter completion FYIs when messaging is available.

## Verification

- `swift test --disable-sandbox --filter CUA`
- 36 tests passed in the feature worktree and again in a temporary combined
  server-selection/observation plus GUI tree.
- `swift build --disable-sandbox`
- Combined server contract: `python3.12 -m pytest -q tests/test_cua_server.py
  -k 'not real_server_mounts_cua_router and not
  real_server_lifespan_closes_cua_service'` — 30 passed, 2 deselected. The two
  excluded tests hit the known host MLX import abort before exercising their
  assertions.

## Remaining action

Manager reviews the local commit and diff, then chooses the final stacked base
and publication timing. Do not push or open the Draft PR before that review.
