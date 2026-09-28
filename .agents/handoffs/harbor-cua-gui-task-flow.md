# Harbor handoff: CUA delegated-task GUI flow

- Owner: Harbor implementing a Pixel-owned Swift UI slice under Atlas scope
- Branch/worktree: `harbor/cua-gui-task-flow`, `/private/tmp/harbor-desk-cua-gui`
- Base dependency: PR #3826 head `2f086d9d`
- FYI: role mailbox unavailable; start/completion context is recorded here.
- Role files requested by the task were absent from this branch. Current
  session ownership guidance and the repository `AGENTS.md` were followed.

## Intention and scope

Make the existing local CUA task panel understandable and controllable during
permission setup, execution, outcome verification, and approval pauses. This is
a Swift-only presentation change using the existing run/events/approve/cancel
API. It does not change the CUA backend, create a remote execution path, add
assets, or claim complete delegated-workflow parity.

## Behavior

- Missing Screen Recording and Accessibility grants appear as a readiness card
  with direct System Settings links and an explicit refresh action. Start stays
  disabled until both grants are visible to the app.
- A running task shows the current structured plan step, action, target, bounded
  step progress, verifier outcome, and a persistent Stop button.
- Approval pauses show app plus optional structured action and target fields.
  Optional `gate_id` is decoded and retained in pending state for the upcoming
  server contract without changing today's approval request shape.
  Legacy events without those optional fields remain usable and no prose is
  parsed to guess missing details. Approve and Stop are separate actions.
- Verifier results use honest labels: Verified, No effect observed, Unexpected
  result, Could not verify, or Verification unavailable. A planner's final
  summary is labeled only as task-ended rather than as independently verified.

## Private reference check

The existing Rapid Mac permission UI, setup rails, compact status cards, and
native action hierarchy were used as the primary interaction precedent. The
existing CUA panel/event contract was reused rather than adding a parallel
workflow model.

## Verification

- Focused CUA and permission Swift suites: 24 tests passed across panel, client,
  brain, decode, and TCC readiness suites; the complete Rapid target compiled.
- Existing Mac automation permission model remains the single source of truth.
- `git diff --check` passed.

## Risk and follow-up

Current server gate events normally provide only a reason. The UI is ready for
optional app/action/target fields without requiring them; richer approval
details remain a separate backend decision. macOS may require app restart after
Screen Recording changes; the UI reports only the TCC state returned by the OS.
