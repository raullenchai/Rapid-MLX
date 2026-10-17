# Merge on green, phase 1

Owner: release engineering. Status: implemented behind `MERGE_ON_GREEN_ENABLED` (off by default).

## Context

The serial integration queue repeats full validation for each rebased candidate.
With one candidate validated at a time, queue latency grows even when PR heads
are independently green. Phase 1 introduces an opt-in merger while preserving
all queue configuration, candidate identity, admission, qualification and tree
evidence. This change itself can therefore pass through the existing queue.

## Decision and boundaries

Once enabled, merge authorized PR heads directly, one squash merge at a time,
without requiring an up-to-date branch. Preserve both exclusive ready labels,
the exact-head `merge-ready-head` commit status, trusted Actions check identity,
and every existing queue admission condition. Optional type checking must pass
or be skipped. Version-bump PRs remain human merges. Use PR-number order rather
than extra timeline API calls. Re-fetch live authorization and match the head
SHA in the merge request; unknown mergeability waits briefly and then skips.

Only default-branch code executes in the privileged merger and main guard.
`RELEASE_PAT` is required for merges and revert branch/PR writes so the resulting
push and PR workflows run. There is no fallback write token. Read failures
cannot authorize a merge. The merger emits per-PR decisions in its step summary.

## Risk and mitigations

Two separately green PRs can conflict semantically after squash merges even
without a textual conflict. Phase 1 accepts this risk to remove serial queue
validation, with these safeguards:

- Ordinary PR heads regain the path-aware Linux matrix, Apple Silicon unit
  tests and 100% changed-line coverage. Reduced source routes remain in the
  tree for cutover compatibility but ordinary PRs cannot select them.
- Main pushes retain full selected-lane validation. Nightly main runs exercise
  the complete Linux, Apple Silicon, L1 smoke and GUI golden-flow gates;
  separate concurrency prevents pushes from cancelling nightly runs.
- Exact-release-SHA CI, desktop candidate and Tier-1 release gates stay intact.
- Main failures rerun failed jobs once, except setup-only infrastructure
  failures. A second failure proposes a revert PR only for the direct child of
  a successful ancestor run. Ambiguous ranges, protected bump/revert commits,
  missing original PRs and revert conflicts go to one `main-red` issue.
  Revert PRs are never automatically authorized for merge. The original author
  receives one marked comment. Scheduled failures use one `nightly-failure`
  issue, closed only when both workflows' latest nightly runs are green.
- Release detection freezes merges while a releasing or undetermined
  auto-release run is active, including manually dispatched releases. A bump
  at main HEAD freezes merges until its exact-SHA push release run completes,
  covering the run-creation race. API failures freeze merges as well.

## Operations and rollback

Provision `RELEASE_PAT` before enabling. It needs merge and revert branch/PR
write access. Land phase 1 through the existing queue. The operator then opens
and labels the phase-2 removal PR, deleting the queue configuration and
queue-only workflows. Set `MERGE_ON_GREEN_ENABLED=true` only while that removal
PR is the only PR carrying a ready label. The merger lands it, leaving the queue
without configuration; the repository admin then uninstalls the queue app.
During the brief overlap, both paths can only merge the same exact head using
head-SHA-matched squash; the loser observes a closed PR and does nothing.
The switch is permanent and defaults to disabled. A dispatched dry run can
restrict evaluation to one PR without merging.

```bash
gh variable set MERGE_ON_GREEN_ENABLED --body true
gh variable set MERGE_ON_GREEN_ENABLED --body false
gh variable set RELEASE_FREEZE --body true
gh variable delete RELEASE_FREEZE
```

Use the manual freeze before release-sensitive human operations; the automatic
freeze supplements it. Removing the variable unfreezes only when no automatic
freeze remains. Inspect step summaries, release runs and tracking issues before
resuming. Neither switch cancels an already submitted merge request.

Rollback in phase 1: disable `MERGE_ON_GREEN_ENABLED`, let any active merger
finish, and resume the existing queue app. Queue configuration remains in this
tree. After phase 2: disable the merger, revert the phase-2 removal PR and
re-enable the queue app. No branch-protection settings change in either phase.

## Verification

Fake-transport tests cover authorization conditions, latest trusted evidence,
freeze races and API failures, serialized merges, reruns, attribution, revert
idempotence and nightly issue recovery. Workflow tests pin trusted checkouts,
permissions, opt-in behavior, PAT use and independent nightly concurrency.
Both automation scripts require 100% changed-line coverage in the Linux matrix.
