# CUA GUI lifecycle safety

- **Owner:** Vector implementation with Pixel integration ownership
- **Branch/worktree:** `harbor/cua-gui-lifecycle-safety` at
  `/private/tmp/harbor-desk-cua-gui-release-ready`
- **Base snapshot:** final GUI selected-window commit `eb1ae53a`
- **Status:** implemented and locally verified; not pushed

## Intention and scope

Prevent delayed create, approval, and polling responses from reviving a stopped
Computer Use task. Changes are limited to `CUAViewModel` lifecycle coordination
and focused Swift tests. Server behavior, discovery errors, window selection,
and other GUI surfaces remain outside this patch.

## Result

- Every start receives a lifecycle generation. Poll and approval responses must
  still match both that generation and the captured run ID before changing UI.
- Stop during a pending create keeps the panel in a visible stopping state,
  waits for the create result, and sends server cancellation for the returned
  run ID before returning to idle.
- If that cleanup cancellation fails, the returned run remains attached to the
  view model, polling resumes, and the Stop control stays available with an
  actionable retry error.
- Active-run cancellation invalidates old approval and poll responses before
  awaiting the server. If cancellation fails, polling resumes at the latest
  displayed event sequence without duplicating the visible trace.
- A single published stopping latch coalesces repeated Stop presses, disables
  approval, and changes both Stop buttons to `Stopping…` while cancellation is
  in flight. Failure releases the latch and restores retry controls.
- The defensive superseded-create cleanup cannot be reached through the UI:
  `.starting` disables another Start, while Stop records intent without changing
  the generation. It remains as best-effort containment for programmatic misuse
  and intentionally cannot overwrite a newer lifecycle owner's UI with its
  cleanup failure.

## Verification

- `swift test --disable-sandbox --package-path apps/rapid-mac --filter CUA`
  — 47 tests in 5 suites passed.
- `swift build --disable-sandbox --package-path apps/rapid-mac` passed.
- The combined server contract suite on the selected-window and observation API
  stack passed: 174 tests passed and 2 hardware-bound tests were deselected.
- New regressions cover delayed create cancellation, failed create cleanup,
  approval-versus-Stop ordering, and a non-cooperative late poll response.
- Repeated Stop regression proves two presses during a delayed cancellation
  produce one server request and one terminal idle transition.
- `git diff --check` passed.

## Integration

This patch is replayed directly after the final selected-window GUI commit. It
depends on the explicit PID/window start contract in `eb1ae53a` but does not
alter it.

Team messaging was unavailable in this implementation environment; this handoff
records the start/completion FYI for the other roles.
