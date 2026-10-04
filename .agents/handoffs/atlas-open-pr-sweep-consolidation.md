# Atlas handoff: reviewed open-PR sweep consolidation

- Owner: Atlas, with Harbor owning CI and direct merge operation after the PR
  opens.
- Branch/worktree: `harbor/open-pr-sweep-consolidation` in
  `/private/tmp/rapid-mlx-open-pr-sweep-consolidation`.
- Intention: drain nine independently reviewed internal PRs through one CI
  candidate and one direct merge commit while preserving every exact reviewed
  head in history.
- Scope: PRs #3972, #3290, #3987, #3950, #3569, #3962, #3963, #3587, and
  #3973. The community-authored warning-policy PR #3988 remains separate so its
  contribution record is preserved.
- Non-goals: no CUA implementation changes, no new feature work, no release,
  and no behavior beyond the source PR contracts.

## Verified facts

- Every source PR exact head is an ancestor of the consolidation head.
- The nine source diffs have zero overlapping paths and merged without a
  conflict. The resulting source union changes 232 paths; this handoff is the
  consolidation adds this handoff and the singleton queue contract correction
  required by the already-merged #4055 configuration.
- `git diff --check` and Ruff check/format pass for all 38 changed Python files.
- Focused Python verification passes: 84 GLM capture/contract tests; 3,729
  telemetry, memory-gate, alias, and LTX tests; and 340 drafter/runtime/CI tests
  with one expected platform skip under the repository-pinned `mlx-vlm 0.7.2`.
- The model-unload Swift suites pass 13 tests across two suites.
- A comparison run with a stale host-level `mlx-vlm 0.7.1` produced the
  expected vendored-source parity failure. Re-running with the repository pin
  passed, so this is host dependency drift rather than an integration defect.
- The only CUA-named paths are two stale research documents deleted by #3963;
  no CUA implementation is added or changed.

## Final merge and next action

Do not authorize or enqueue PR #4057 through Mergify. Both configured queue
rules squash their candidates, which would replace the consolidated commits and
break the contract that every source PR exact head remain reachable from
`main` so GitHub can mark those PRs as indirectly merged. After head-bound CI
and independent review pass, Atlas or Harbor must bypass Mergify and merge PR
#4057 directly with GitHub's **merge commit** method (never squash or rebase),
guarded to the reviewed exact head. Then fetch `main`, verify that all nine
source heads are ancestors, and confirm each source PR reports `merged=true`
before closing anything manually. Leave the later feature cohort deferred.

The dedicated role-messaging channel was unavailable in this shell. This file
records the required start/completion FYI for Pixel, Vector, Harbor, Echo, and
ds0731 until that channel is restored.
