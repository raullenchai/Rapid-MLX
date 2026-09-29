# Vector handoff: approval resume restores exact target focus

- **Owner:** Vector, coordinated by Atlas
- **Branch:** `vector/cua-approval-resume-focus`
- **Base:** signed integration `39f964a8cbae588d3365c25008b469f074d5aaad`
- **Reproduction:** Mini run `9877fa2826d1`, Finder PID `93136`, window `cg:11409`
- **Scope:** exact selected-window focus restoration and focused backend tests

## Finding

The approval sheet correctly moved focus to Rapid. Finder rename execution then
called `raise_selected_window`, which verified the approved PID, CGWindowID,
frame, and unique AX window but used `AXRaise` without activating Finder.
`AXRaise` does not guarantee that the owning process becomes frontmost, so the
final PID focus check correctly failed with `target_drift` before any write.
The file and sentinel remained unchanged.

## Change

After the existing exact identity and unique-window checks,
`raise_selected_window` activates only the already-bound PID. Both before and
after that focus-changing boundary it validates PID, bundle ID, name, process
start time, CGWindowID, frame, and the unique AX window before `AXRaise`. The
existing final frontmost PID and focused AX-window validation remains the last
gate. Any drift fails closed; the approved action is never converted into a
generic retry or rebound to a different Finder selection.

This follows the established observe, exact actuation, fresh observation pattern
reviewed in Orca while retaining Rapid's stronger PID, CGWindowID, and opaque
Finder file-reference binding. No proprietary code was copied.

## Verification

- Deterministic regression starts with Rapid frontmost and asserts exact Finder
  activation occurs before the final focus check.
- Failure cases cover identity drift during activation and AX-window ambiguity
  introduced by the focus transition.
- Process-start drift before activation cannot activate a reused PID; drift
  observed after activation cannot reach `AXRaise`.
- `tests/test_computer_use.py tests/test_cua.py`: 404 passed before the two
  added fail-closed cases; focused selection now passes 10 tests.
- Ruff format/check: passed.

## Remaining live check

Harbor should integrate this commit onto the signed stack and rerun the disposable
Finder rename fixture. Confirm the approval card leaves Rapid frontmost, Approve
restores PID/window `93136`/`cg:11409` (or the fresh fixture equivalents), and
only the approved exact file reference changes. Deny must keep both Before and
the sentinel unchanged. Pixel should verify the approval card and result states.
