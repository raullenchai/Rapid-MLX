# Path-aware PR gates and merge queue

Rapid-MLX uses two validation levels so concurrent pull requests do not each
pay the full release-grade macOS cost.

## PR gate

`scripts/classify_ci_changes.py` assigns changed paths to the engine and desktop
lanes. The policy fails closed: an empty diff, workflow change, or unknown
product area selects all applicable lanes.

- Engine changes run the Linux test matrix, the Apple Silicon test suite, and
  one representative L1 model (`qwen3.5-4b-4bit`).
- Desktop changes run the Swift build/test and the inexpensive GUI harness
  contracts. They do not run engine model smokes.
- Documentation-only changes run universal repository guards and stable
  aggregate jobs, without allocating an engine, type-check, MLX-bound, model,
  or macOS runner. Universal guards retain workflow-expression, immutable
  Action-pin, and architecture-SSOT checks.

Engine-only contracts are admitted by the same fail-closed classifier as the
engine test lanes. They include CLI/config fidelity, release and installer
offline tests, and the parser microbenchmark. Desktop-only and
documentation-only changes do not pay for those engine-specific dependencies
and commands. Whole-repository Ruff and engine/Desktop version synchronization
remain universal because Desktop support code includes Python and either side
of the shared version contract may change.

The required checks are the stable aggregate jobs `tests` and `desktop-tests`.
They must not be renamed or hidden behind workflow-level path filters without a
matching branch-protection migration. `tests` includes lint, type-check job
health, the MLX dependency-bound guard on pull requests, and all selected engine
test lanes; `desktop-tests` includes every selected Desktop lane.

### Type-error budget

Engine changes run a shrink-only mypy debt ratchet. The checked-in
`config/mypy-error-baseline.txt` records the current error count for each dirty
file under the fully pinned Python 3.11 environment in
`config/mypy-requirements.txt`. A new dirty file or an increase in any file's
count blocks `tests`. When fixes reduce a count or clean a file completely, CI
also blocks until the baseline is tightened with:

```bash
python scripts/check_mypy_error_budget.py --update
```

`--update` refuses growth and new dirty files, so it cannot be used as a casual
bypass. The budget intentionally does not claim semantic identity for individual
diagnostics: replacing one error with another while a dirty file's total stays
flat is outside this first ratchet. This keeps the gate deterministic despite
moving line numbers and messages while preventing debt from spreading or
growing. As dirty files are repaired and removed from the baseline, they can
never become dirty again without failing CI.

### Changed-lines coverage

Engine pull requests enforce 100% coverage for executable lines newly added or
modified under `vllm_mlx/`. The Python 3.11 Linux unit-test leg already produces
`coverage.xml`; `diff-cover` compares that report with the pull request's
immutable base SHA and blocks the stable `tests` aggregate when a measurable
changed line was not exercised. Comments, blank lines, deletions, tests, docs,
and unchanged production lines do not enter the score.

This is a new-debt ratchet, not a whole-repository percentage target. Existing
uncovered code remains grandfathered until a pull request changes its executable
lines, so ordinary feature and bug-fix work is not required to repair unrelated
historical coverage debt. A production change that cannot run on the Linux lane
must expose its behavior through a Linux-testable boundary or extend the coverage
gate to consume trustworthy evidence from the relevant required lane; lowering
the threshold is not the normal escape hatch.

## Merge gate

Adding the `full-ci` label upgrades the lanes selected by the pull request's
actual diff. Apply it only when the PR is ready to merge; removing it returns
subsequent commits to the path-aware PR gate.

- Engine changes expand to the full five-model L1 matrix.
- Desktop changes build the release GUI once, then run every journey group
  mapped to the changed controls and product sources in
  `Tests/GUIGoldenFlows/journeys.yaml`.
- Cross-cutting or unknown changes expand both lanes.
- Documentation-only changes require neither product lane and do not need the
  label.

GUI routing expands a changed source to its complete journey group, so sibling
flows around the same user workflow remain covered. It fails closed: empty or
invalid diffs, new unmapped Desktop paths, shared UI components, packaging
inputs, the harness, its manifest, and the CI workflow select every PR journey.
Broad mixed-responsibility directories such as `Sources/Rapid/UI/` never grant
narrow ownership to a new file; that requires an explicit file or cohesive
domain-directory mapping in the manifest.
Each named workflow step remains visible but an unselected journey exits before
preflight, app launch, or artifact creation. The final verdict requires exactly
the number of result records selected by the classifier, so a selected journey
cannot silently disappear.

The label never changes lane classification. This prevents an engine-only PR
from allocating the full Desktop gate, or a Desktop-only PR from allocating
the full model gate. The selected product aggregate intentionally remains
non-successful until the label is present, so branch protection cannot bypass
its merge gate.

The workflows also subscribe to GitHub's `merge_group` event. After the
repository becomes eligible for GitHub merge queues, every queue candidate will
automatically receive the same full coverage against its synthetic candidate
commit. This validates the combined state that will actually reach `main`,
rather than repeatedly validating each PR against an obsolete base.

Pushes to `main` retain the full engine coverage as a post-merge signal.

### Desktop GUI artifact provenance

The full Desktop gate builds one release-configured app with
`SKIP_SIDECAR=1`, packages it, and uploads it under an artifact name containing
the exact candidate SHA. Selected manifest journey groups run as independent
matrix shards and reuse that artifact; GUI jobs do not rebuild the app. Before
extraction, they verify a versioned manifest that binds the SHA, build mode,
sidecar mode, archive filename, and SHA-256 digest; after extraction they
verify the macOS code-signing seal. Missing, stale, malformed, or modified
artifacts fail closed.

The classifier emits the matrix from the same journey SSOT used for routing.
Each selected flow appears in exactly one group shard. Matrix fail-fast is
disabled so one failure cannot cancel evidence from sibling groups, while the
stable `desktop-tests` facade remains red unless every selected shard passes.
Hosted-runner isolation gives every shard a separate HOME, defaults database,
ports, app processes, and result directory. Failure artifact names include the
group so concurrent uploads cannot overwrite one another.

This artifact is test-only and retained for one day. It is not signed for
distribution, notarized, published, or eligible for release promotion. Release
workflows continue to build their own Developer-ID-signed artifact with the
bundled sidecar and release credentials.

## Repository configuration

GitHub currently offers merge queues only to public repositories owned by an
organization, or private repositories owned by an Enterprise Cloud
organization. Rapid-MLX is a public repository owned by a personal account, so
the queue cannot be enabled until ownership moves to an organization.

After that eligibility change, and after the workflows containing
`merge_group` support are present on `main`:

1. Require `tests` and `desktop-tests` for `main`.
2. Require branches to be up to date through the merge queue, rather than
   asking authors or agents to rebase every open PR after each merge.
3. Use squash as the merge method and start with a small queue batch. Increase
   batching only after observing queue latency and failure isolation.

Do not enable the queue before the workflow trigger reaches `main`: otherwise
GitHub creates a merge-group commit whose required checks never start.

## Rollback

Disable the merge queue first, restore `full-ci` label-based merging, and leave
the `merge_group` triggers in place. The triggers are harmless while the queue
is disabled. If path classification is suspect, make its policy select both
lanes for every PR; this restores the previous validation coverage without
renaming required checks. For GUI routing specifically, removing the
`GUI_FLOWS` job environment or making `scripts/select_gui_flows.py` return the
full manifest roster restores the previous all-journey behavior.
If GUI matrix execution is suspect, remove the matrix strategy, restore
`GUI_FLOWS` and `EXPECTED_FLOW_COUNT` to the classifier's whole-selection
outputs, and remove group suffixes from evidence artifact names. This restores
one serial consumer without changing which journeys are selected.
If GUI artifact reuse is suspect, restore the build step inside
`gui-golden-flows` and remove `gui-app-build` from its dependencies. This costs
additional macOS build time but preserves the same release-shaped UI coverage.
If the mypy budget gate is operationally broken, restore the prior advisory
direct mypy command with `continue-on-error: true` while repairing the script.
Do not increase counts or add files to the baseline merely to make a PR green.
If changed-lines coverage is operationally broken, remove only the
`Enforce changed-lines coverage` step while repairing its checkout or tooling;
keep the existing advisory measurement and coverage XML upload as diagnostic
evidence. Do not lower `--fail-under` or exclude changed production lines merely
to make a pull request green.
