# Identical-tree CI evidence

Mergify validates a synthetic candidate commit before writing its contents to
`main`. The candidate commit and the final squash commit have different commit
SHAs, but their Git tree SHAs are identical when no content changed. Rapid-MLX
uses that immutable content identity to avoid running the same full Desktop GUI
matrix twice.

## Safety contract

The candidate remains the only place where expensive validation is promoted.
An internal `mergify/merge-queue/<10 lowercase hex>` head must run the complete
six-group GUI inventory plus the Desktop build and unit tests. The queue still
refuses to merge when any of those checks fails.

The `Queue tree attestation` workflow starts only after Mac CI finishes.
GitHub loads a `workflow_run` workflow from the default branch. The producer
pins checkout and API comparisons to the immutable default-branch `github.sha`
captured by that run, and never executes candidate code. It:

1. verifies the candidate repository, exact branch shape, bot-owned synthetic
   pull request, base branch, source SHA, and source workflow identities;
2. rejects a candidate that changes the attestation workflow, verifier, Mac
   workflow, GUI router, or journey-manifest controls;
3. requires the latest non-cancelled Mac workflow run to succeed, including
   every expected GUI matrix group;
4. records source run attempts and job identities in a seven-day artifact; and
5. adds a commit status whose target is the trusted attestation run.

On a `main` push, `rapid-mac CI` compares Git tree SHAs. It authenticates the
status target as a successful run of the exact attestation workflow, downloads
the artifact from that run, and re-queries every source run, attempt, and job.
Only then does `desktop-tests` reuse the result while asserting that no Mac job
was allocated.

Every missing, expired, stale, red, cancelled, malformed, or non-identical
record is a cache miss. The workflow runs the existing full Mac gate instead.
No evidence error can manufacture a green result, and evidence is never reused
across different file contents. Downstream jobs use `always()` at the evidence
dependency boundary, so even an evidence-job timeout or runner outage falls
back to the full gate instead of leaving Desktop CI pending.

## Operator evidence

For the first candidate after this mechanism lands:

1. Open its `Queue tree attestation` run and confirm the artifact and
   `queue-tree-evidence/mac` status name the exact candidate SHA.
2. After Mergify lands the candidate, compare:

   ```bash
   git show -s --format=%T <candidate-sha>
   git show -s --format=%T <main-sha>
   ```

   The tree SHAs must match.
3. Open the final main-push `rapid-mac CI` run. `queue-tree-evidence` and the
   cheap Ubuntu contracts must pass; `build`, `gui-app-build`, and every
   `gui-golden-flows` child must be skipped; `desktop-tests` must pass.
4. Confirm no Manzanita or GitHub-hosted macOS runner was allocated by that
   main-push run.

If any assertion differs, treat reuse as disabled and inspect the attestation
and source URLs printed by `desktop-tests`. Do not manually manufacture the
status or artifact. The safe rollback is to revert the reuse conditions; main
then returns to the previous full-gate behavior.

## Cost and latency

Before reuse, a Mac batch commonly spent about 19 minutes when runner capacity
was idle and 37–38 minutes under contention, followed by another 25–27 minute
full Desktop run on `main`. Reuse removes that second Mac allocation. The target
steady state is an 18–22 minute Mac batch followed by a roughly two-minute
evidence facade on `main`, saving about 18–20 minutes of wall time and 20–25
Mac slot-minutes per batch. Actual usage must be read from Actions billing and
the runner provider separately; partner-runner capacity is not inferred from
GitHub-hosted usage totals.
