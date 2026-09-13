# Identical-tree CI evidence

Mergify validates a synthetic candidate commit before writing its contents to
`main`. The candidate commit and the final squash commit have different commit
SHAs, but their Git tree SHAs are identical when no content changed. Rapid-MLX
uses that immutable content identity to avoid running the same full engine and
Desktop validation twice.

## Safety contract

The candidate remains the only place where expensive validation is promoted.
An internal `mergify/merge-queue/<10 lowercase hex>` head must run the complete
six-group GUI inventory plus the Desktop build and unit tests. The queue still
refuses to merge when any of those checks fails.

The `Queue tree attestation` workflow starts after either engine or Mac CI
finishes. Each completion emits an independent `ci` or `mac` proof only when
the latest non-cancelled run of that workflow is complete and successful.
Keeping the scopes separate means a Desktop-only candidate can reuse its Mac
proof even though it never ran engine jobs, and vice versa.
GitHub loads a `workflow_run` workflow from the default branch. The producer
pins checkout and API comparisons to the immutable default-branch `github.sha`
captured by that run, and never executes candidate code. It:

1. verifies the candidate repository, exact branch shape, bot-owned synthetic
   pull request, base branch, source SHA, and source workflow identities;
2. rejects a candidate that changes the controls for the scope being attested:
   the attestation workflow and verifier plus the engine workflow/classifier or
   the Mac workflow/GUI router/journey manifest;
3. requires the latest non-cancelled source workflow to succeed, including
   every expected Python/Apple/L1 job for `ci`, or every Desktop build and GUI
   matrix group for `mac`;
4. records source run attempts and job identities in a scope-named seven-day
   artifact; and
5. adds a `queue-tree-evidence/ci` or `queue-tree-evidence/mac` commit status
   whose target is the trusted attestation run.

On a `main` push, both `CI` and `rapid-mac CI` compare Git tree SHAs. Each
authenticates the status target as a successful run of the exact attestation
workflow, downloads the artifact from that run, and re-queries every source
run, attempt, and job. Only then do the required `tests` and `desktop-tests`
facades reuse the result while asserting that no expensive job was allocated.

Every missing, expired, stale, red, cancelled, malformed, or non-identical
record is a cache miss. The workflow runs the existing full Mac gate instead.
No evidence error can manufacture a green result, and evidence is never reused
across different file contents. Downstream jobs use `always()` at the evidence
dependency boundary, so even an evidence-job timeout or runner outage falls
back to the full gate instead of leaving Desktop CI pending.

## Operator evidence

For the first candidate after this mechanism lands:

1. Open its `Queue tree attestation` run and confirm the artifact and
   corresponding `queue-tree-evidence/ci` or `queue-tree-evidence/mac` status
   name the exact candidate SHA.
2. After Mergify lands the candidate, compare:

   ```bash
   git show -s --format=%T <candidate-sha>
   git show -s --format=%T <main-sha>
   ```

   The tree SHAs must match.
3. Open the final main-push `CI` run. `queue-tree-evidence`, `changes`, and
   `tests` must pass; lint, engine contracts, Python matrices, Apple Silicon,
   coverage, and L1 smoke must be skipped.
4. Open the final main-push `rapid-mac CI` run. `queue-tree-evidence` and the
   cheap Ubuntu contracts must pass; `build`, `gui-app-build`, and every
   `gui-golden-flows` child must be skipped; `desktop-tests` must pass.
5. Confirm no Manzanita or GitHub-hosted macOS runner was allocated by either
   main-push run.

If any assertion differs, treat reuse as disabled and inspect the attestation
and source URLs printed by `desktop-tests`. Do not manually manufacture the
status or artifact. The safe rollback is to revert the reuse conditions; main
then returns to the previous full-gate behavior.

## Cost and latency

Before reuse, a queue batch commonly spent about 19 minutes when runner
capacity was idle and 37–38 minutes under contention, followed by a duplicated
engine and Desktop validation on `main`. Reuse removes those second
allocations. The target steady state is one release-grade queue candidate
followed by roughly two minutes of evidence facades on `main`, saving about
18–25 minutes of wall time plus the duplicated Linux, Apple Silicon, L1-smoke,
and GUI runner minutes per batch. Actual usage must be read from Actions
billing and the runner provider separately; partner-runner capacity is not
inferred from GitHub-hosted usage totals.

As a pre-change reference, main runs `34762671338` and `34762671350` on
2026-09-13 validated one tree twice. Engine CI took 18 minutes 32 seconds of
wall time: the three Linux matrices consumed about 52 runner-minutes, Apple
Silicon 9 minutes 41 seconds, and five L1 smokes about 9 minutes 28 seconds.
Desktop CI took 22 minutes 38 seconds of wall time, including about 26
Manzanita GUI slot-minutes and a 4 minute 39 second hosted app build. These are
workflow timings, not an invoice: platform billing multipliers and partner
runner charges must still be read from their respective usage exports.
