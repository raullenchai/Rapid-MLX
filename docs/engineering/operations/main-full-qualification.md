# Full main qualification prerequisite

`scripts.ci_main_qualification.qualify_main(client, base)` is a read-only
prerequisite for a future candidate policy. It is not a merge check or a routing
switch. No workflow currently invokes it. A positive result always contains
`authorizes_reduced_ci: false`.

The base must still be current `main`. The newest exact-base push CI run,
including cancelled and failed runs, must have completed successfully. A newer
failed attempt cannot be replaced with an older green attempt. The checker
rechecks the tip and run identity before returning.

Full qualification accepts either:

- Actual execution: every CPU9/model5 identity passed exactly once, together
  with the main static, Apple and Linux coverage checks.
- Authenticated full reuse: the main reuse gate succeeded and its full lanes
  were skipped consistently. The checker independently locates the latest
  identical-tree candidate, downloads one bounded artifact from its trusted
  attestation, and runs the existing complete-evidence consumer again. It
  validates the tree, controller blobs, source run/attempt and recorded/live
  job identities. It checks for attestation revocation during validation.

A generic skipped matrix job is normal on a reuse route; it is never counted
as executed full coverage. Partial execution, missing/duplicate/wrong matrix
names, expired artifacts, source or reduced evidence namespaces, API errors,
newer cancellations and base races fail qualification. The archive is read
without extracting or executing files.

This checker does not change ordinary full validation. Full repair candidates
can continue validating when main is red. It does not qualify mapped candidate
execution, change branch protection, or activate a canary. Those require a
separate reviewed producer/consumer contract and hosted execution evidence.
The future landing policy must also force full main execution after any
reduced candidate; this prerequisite alone does not implement that backstop.

For diagnostics, call the function with `GitHubClient` and inspect `qualified`
and `reason`. Do not create a green anchor by selecting an older run or
replaying a cancelled run without diagnosing the cancellation.
