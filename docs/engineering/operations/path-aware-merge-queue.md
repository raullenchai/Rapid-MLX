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

Engine classification includes the serving packages and their tests, scripts,
examples, evaluations, benchmark inputs/results, regression harness, and
engine-only type-check configuration. These paths must not allocate Desktop or
GUI runners merely because they live outside the primary Python package.

Engine-only contracts are admitted by the same fail-closed classifier as the
engine test lanes. They include CLI/config fidelity, release and installer
offline tests, and the parser microbenchmark. Desktop-only and
documentation-only changes do not pay for those engine-specific dependencies
and commands. Whole-repository Ruff and engine/Desktop version synchronization
remain universal because Desktop support code includes Python and either side
of the shared version contract may change.

The strict required checks are the stable aggregate jobs `tests`,
`desktop-tests`, and `version-bump-guard`. They must not be renamed or hidden
behind workflow-level path filters without a matching branch-protection
migration. `tests` includes lint, type-check job health, the MLX
dependency-bound guard on pull requests, and all selected engine test lanes;
`desktop-tests` includes every selected Desktop lane. `version-bump-guard`
runs for every pull request, passing quickly when the version is unchanged.

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

Ordinary pull requests run the path-aware PR gate and leave the required
`tests`, `desktop-tests`, and `version-bump-guard` contexts satisfiable. A pull
request becomes an integration candidate only when a maintainer applies the
`merge-ready` label after review and PR validation have converged. The managed
queue collects up to four such pull requests, waits at most 15 minutes from the
first entry, and validates their combined tree once.

The queue creates an internal pull request from a branch whose name is exactly
`mergify/merge-queue/<10 lowercase hex characters>`. The engine and Desktop
workflows treat only that exact same-repository shape as a promoted head. A fork
using the same branch name remains on the ordinary PR path.

The optional `full-ci` label provides an exact-head rehearsal before queueing.
Both that label and a queue batch upgrade the lanes selected by the actual diff:

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

Promotion never changes lane classification. This prevents an engine-only
batch from allocating the full Desktop gate, or a Desktop-only batch from
allocating model runners. A mixed or fail-closed batch selects both.

The queue contract lives in `.mergify.yml`:

- serial mode with one batch in flight, so speculative checks cannot multiply
  scarce macOS capacity;
- up to four pull requests per batch and a 15-minute maximum fill wait;
- no blind CI retry and no skipped intermediate failures;
- at most two batch-split attempts to isolate a failing member;
- a 90-minute check timeout, covering normal hosted macOS queue delay;
- the three GitHub Actions required checks must pass both before queue entry and
  again on the combined temporary pull request;
- successful members are squash-merged individually, preserving one commit per
  pull request and GitHub's `Fixes #N` issue closure behavior.

Release bump and version-correction pull requests are excluded by title and
labels. They continue through the separately authorized release transaction;
they must never be combined with ordinary changes.

All three required workflows retain `merge_group` support so an eventual
organization transfer can use the native queue without another trigger
migration. The managed queue itself uses ordinary `pull_request` events for its
temporary batch pull requests.

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

GitHub's native queue is unavailable while this public repository belongs to a
personal account. The checked-in managed-queue policy is therefore the
authoritative integration configuration.

Production activation is an owner operation and must happen in this order:

1. Land `.mergify.yml`, the workflow recognition logic, and their contract
   tests before installing or enabling the app.
2. Install the GitHub App for this repository only. Do not grant it access to
   unrelated repositories. Confirm its configuration check validates the
   default-branch policy.
3. Create the `merge-ready` label. Applying it is the explicit authorization to
   enter the queue; removing it dequeues the pull request.
   Fork pull requests are deliberately ineligible because composing fork code
   onto an internal queue branch changes GitHub's token and secret boundary.
   After review, bring an accepted external change onto a same-repository
   maintainer branch before authorizing it for the batch queue.
4. In the existing `main` protection, retain required contexts `tests`,
   `desktop-tests`, and `version-bump-guard`, required conversation resolution,
   administrator enforcement, and linear history. Disable only **Require
   branches to be up to date before merging**: temporary batch validation is
   incompatible with that strict flag because the combined branch, rather than
   every original head, is the artifact tested by CI.
5. Keep manual merges limited to the documented version-bump and
   human-authorized hotfix paths. Normal pull requests enter through the
   `merge-ready` label and are merged by the queue.
6. Rehearse with two harmless pull requests that are individually green. Apply
   `merge-ready` to both within the fill window and verify one temporary batch
   PR contains both exact heads, runs each affected full lane once, reports all
   three required checks, and squash-merges both originals in order.
7. Confirm a fork branch named like a queue branch does not receive promoted
   lanes. Then remove `merge-ready` from a queued test PR and verify it leaves
   the queue without merging.

Do not enable batching while strict up-to-date protection remains on, and do
not weaken or remove any required context to make a batch move. A missing,
cancelled, or failed aggregate is a queue failure.

## Rollback

Pause the managed queue and remove `merge-ready` from every queued pull request
first. Wait for the active batch to stop, restore strict up-to-date protection,
and only then disable or uninstall the app. Keep the batch-head and
`merge_group` workflow triggers in place; they are inert without a queue and
make rollback recoverable without weakening CI. If path classification is
suspect, make its policy select both lanes for every PR; this restores the
previous validation coverage without renaming required checks. For GUI routing
specifically, removing the
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
