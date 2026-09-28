# Vector handoff: selected-window CUA runs

- Owner: Vector
- Branch: `vector/cua-selected-window-runs`
- Worktree: `/private/tmp/harbor-desk-cua-selected-window-runs`
- Base: `7acf2e9b455d08864ce3fd7f88dddffdd933334e`
- Receiving role: Atlas for integration sequencing

## Scope and verified behavior

`POST /v1/cua/runs` accepts an optional opaque `window_id`. The service performs
a read-only lookup, binds it to the PID resolved from `app`, freezes the
canonical ID, and returns it from create/list/view surfaces. Selected runs pass
that ID to every state observation and trusted URL read. They revalidate the
window after planning and after approval, and stop with a typed window error if
the window closes, is replaced, or moves. `open_url` is rejected when a window
is selected. Runs without `window_id` retain their existing call shape.

Capability `features.window_selection` is now true. Visual observation and a
raw action API remain outside this branch.

## Reference check

Rapid-MLX's existing backend window identity, PID filtering, snapshot drift,
and trusted URL mechanisms were reused. The required primary serving projects
and MLX-native serving projects were searched for an analogous desktop window
selection contract; none exposes this host GUI concept, so no serving pattern
applied. This note is private and must not be copied into external PR text.

## Verification

- `ruff check` on all changed Python files: passed.
- `pytest tests/test_cua_server.py -q -k 'not real_server'`: 30 passed,
  2 deselected.
- `pytest tests/test_cua.py tests/test_computer_use.py -q`: 145 passed.
- The two real-server import tests abort in the local native MLX extension
  during module import; all server tests before that point and the isolated
  selected-window contract tests pass.
- `git diff --check`: passed.

## Coordination and remaining action

The Orca agent mailbox was not available in this session, so start/completion
FYIs could not be delivered. Atlas should cherry-pick the local commit onto the
final linearized base, resolve any API model overlap, rerun the focused tests,
then start the normal review loop. No push or PR was created.

Main compatibility risk: a selected window intentionally fails closed when its
geometry changes during a planning or approval pause, so users must reselect a
moved window before retrying.
