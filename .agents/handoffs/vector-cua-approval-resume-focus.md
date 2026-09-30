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

Cross-app signed run `f0d245996bc3` then exposed the same approval-focus
boundary for TextEdit Save: the approval card changed AX focus/tree state, and
the loop rejected before the backend could restore the exact document window.
For every non-Finder approved action with a bound window, the loop now raises
only that exact PID/window before fresh observation. It freezes the active
target ID and compares PID, bundle ID, name, process start time, window identity,
frame, and domain after approval. Save alone ignores focus-induced AX tree
differences only after its exact native document/menu identity is re-inspected
and matches the pre-approval binding. Other approved actions retain target and
tree equality checks.

The same run also proved that a successful TextEdit `fill` can reach disk later
through native autosave even when the subsequent Save approval is refused. All
`fill` actions now require `external_commit` approval before any AX write,
because persistence semantics cannot be inferred safely for an arbitrary
editable surface. Existing TextEdit files receive an additional exact binding
from the focused window's local `AXDocument` URL plus PID, process start time,
and CGWindowID. That identity is inspected before approval and again after exact
window restoration, adjacent to `AXSetValue`. Unsupported or ambiguous document
identity fails closed. Denial dispatches no write, so no delayed autosave can be
scheduled. Finder rename keeps its existing opaque-reference approval and does
not receive a second approval.

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
- TextEdit Save regression covers Rapid taking focus, changed AX tree shape,
  exact-window restoration, unchanged document binding, then Save dispatch;
  changed menu/document identity still blocks with zero dispatch.
- TextEdit autosave regressions cover denial with no scheduled write, successful
  exact document revalidation after approval, and fail-closed initial inspect,
  window restore, and post-approval document drift paths.
- Combined backend/CUA tests: 437 passed.
- Changed Python line coverage against signed base `2f443ced`: 74/74, 100%.
- Mypy error budget: no growth (811 grandfathered errors across 153 files).
- Ruff format/check: passed.

## Remaining live check

Harbor should integrate this commit onto the signed stack and rerun the disposable
Finder rename fixture. Confirm the approval card leaves Rapid frontmost, Approve
restores PID/window `93136`/`cg:11409` (or the fresh fixture equivalents), and
only the approved exact file reference changes. Deny must keep both Before and
the sentinel unchanged. For TextEdit, wait beyond the observed autosave interval
after Deny and confirm the existing file remains unchanged; Approve must update
only the exact bound document. Pixel should verify the approval card and result
states.

## Signed d4ac Finder focus follow-up

Signed run `385bb5bf81f6` remained fail closed but showed that activating Finder
and sending `AXRaise` does not reliably make the already-bound CG window its AX
focused window. The approved fill therefore stopped before `AXSetValue`; the
Before fixture and sentinel remained unchanged.

Finder rename recovery now requests `AXMain=true` on the uniquely matched AX
window after the existing PID, process-start, bundle, CGWindowID, frame, opaque
file-reference, and path checks, then performs `AXRaise` and the existing fresh
focused-window validation. A rejected focus request fails closed before
`AXRaise`. General window restoration keeps its prior behavior; the stronger
focus mutation is opt-in only for the Finder rename write and commit paths.

Verification on exact signed base `d4ac4107`: 441 backend/CUA tests passed;
changed-line coverage 8/8 (100%); Ruff and the mypy error budget passed. Harbor
must independently review, integrate onto the current test-only head, and rerun
signed Finder Deny and Approve fixtures before release.

Signed follow-up `7180688b19d7` proved `AXMain=true` succeeded but Finder reports
`AXFocusedWindow=nil` during inline rename. A read-only live dump bound
`cg:11657` and `AXMainWindow` to the same `(567,213,920,436)` frame while the
focused app-root rename text field appeared as a contained transient entry in
both Finder `AXWindows` and CG (`cg:11673`). Finder rename validation now accepts
that exact live shape only when the app is active, the exact main window frame
matches, and the focused app-root text field is contained and listed. The
existing cached sole-row, opaque reference, path, and normalized editor-value
checks still run before menu dispatch and adjacent to `AXSetValue`; unrelated
app-root fields therefore cannot receive the approved write. Generic action
validation remains unchanged.
