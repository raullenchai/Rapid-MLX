# Vector handoff: approval resume restores exact target focus

- **Owner:** Vector, coordinated by Atlas
- **Branch:** `vector/cua-approval-resume-focus`
- **Base:** signed integration `2f443ced1b211b2ed36003cb2eb23420c8088ec2`
- **Reproduction:** Mini runs `9877fa2826d1`, `2f4fa7d7566a`, and
  `ca67fc70bfcc`; Finder PID `93136`, window `cg:11409`
- **Scope:** exact selected-window focus restoration and focused backend tests

## Finding

The approval sheet correctly moved focus to Rapid. Finder rename execution then
called `raise_selected_window`, which verified the approved PID, CGWindowID,
frame, and unique AX window but used `AXRaise` without activating Finder.
`AXRaise` does not guarantee that the owning process becomes frontmost, so the
final PID focus check correctly failed with `target_drift` before any write.
The file and sentinel remained unchanged.

Signed follow-up run `2f4fa7d7566a` restored Finder focus but showed a second
safe refusal: opening Rapid's approval UI closed Finder's transient rename
editor. The stale element index correctly failed with `target_drift`; the file
and sentinel again remained unchanged.

## Change

After the existing exact identity and unique-window checks,
`raise_selected_window` activates only the already-bound PID. Both before and
after that focus-changing boundary it validates PID, bundle ID, name, process
start time, CGWindowID, frame, and the unique AX window before `AXRaise`. The
existing final frontmost PID and focused AX-window validation remains the last
gate. Any drift fails closed; the approved action is never converted into a
generic retry or rebound to a different Finder selection.

When the approval focus transition closes the unchanged Finder inline editor,
the backend re-enters Rename only after proving the retained opaque file
reference still resolves to the original path and its retained AX row is the
sole selected row in the same outline. It invokes Finder's unique native Rename
command, reacquires an editor for that same row, path, and reference, and only
then performs the adjacent approved `AXSetValue`. It does not click, reuse the
stale serialized index as authority, or transfer approval to another selection.
Because native menu resolution can itself change Finder UI state, exact window
and focus plus sole-row, reference, and path checks run again immediately before
`AXPress`; exact window and focus are checked again after editor reacquisition
and adjacent to `AXSetValue`.

Live inspection after the signed `ca67fc70bfcc` refusal confirmed Finder's
replacement editor is a focused `AXTextField` directly under `AXApplication`,
with no `AXURL` and the unchanged original basename. That detached shape is now
accepted only when it is still focused, its normalized value is the original
basename, and the cached row, opaque reference, original path, sole selection,
window, and process identity all remain exact.

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
- Approval-closed editor recovery is covered for the same bound row, reference,
  and path; a different selected row fails before the native Rename command.
- Selection or window drift during menu resolution cannot reach `AXPress`.
- Detached-editor tests cover the observed app-root shape plus wrong value,
  multiple selection, and opaque-reference drift.
- Ruff format/check: passed.

## Remaining live check

Harbor should integrate this commit onto the signed stack and rerun the disposable
Finder rename fixture. Confirm the approval card leaves Rapid frontmost, Approve
restores PID/window `93136`/`cg:11409` (or the fresh fixture equivalents), and
only the approved exact file reference changes. Deny must keep both Before and
the sentinel unchanged. Pixel should verify the approval card and result states.
