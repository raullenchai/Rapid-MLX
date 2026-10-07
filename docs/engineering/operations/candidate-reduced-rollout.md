# Controlled mapped candidate rollout

This rollout reuses the existing source allowlist for the **entire combined
candidate diff**. Unknown, mixed, shared, inference, dependency, security and CI
control changes retain complete validation. Desktop gates are unchanged.

## Deployment and activation

1. Deploy the trusted qualification and admission workflows first. Verify a real
   current full candidate artifact, index and admission before adding the
   `@github-actions/candidate-admission/ci` condition to both queue merge rules.
   Source queue conditions do not require this candidate-only status.
2. Land this controller through ordinary full candidate validation. Its own
   controller diff is ineligible for reduced execution. Verify the exact merged
   controller, required job identities and main result.
3. Set `RAPID_MLX_CANDIDATE_CANARY=true` with the authorized operator CLI. This is
   only a startup opt-in. `GITHUB_TOKEN` cannot request Variables API permissions.
4. Dispatch `candidate-admission.yml` on `main` with `rollback=false`. Live
   selectors/qualifiers require the latest dispatch to have succeeded, with one
   successful `activate` job and skipped `rollback`. Pending, failed, cancelled,
   missing, foreign-branch, bot-triggered or control-mismatched generations are
   disabled. Do not search backward for a successful activation. Relevant control
   changes invalidate the generation and require an explicit new activation.
5. Verify hosted mapped tests, retained base collection, no skipped/deselected
   nodes, mandatory 100% changed-line coverage, exact attempt artifact, trusted
   producer and fresh admission. A test-only fixture is not proof of nonempty
   production coverage or user-visible speedup. Record source feedback, queue
   delay and candidate execution separately.

Reduced candidates keep static checks and execute complete mapped suites with
coverage. CPU/Apple/model full lanes must be explicitly skipped; unexpected
execution, failed coverage, missing input or stale attempts fail qualification.
Records bind the activation run/attempt so two successful activations cannot
silently replace each other. Mapped artifacts never enter full-evidence reuse.

Full repair candidates do not depend on an activation or green main. Trusted
actual-combined docs/Desktop policy exemptions have a separate
`engine-not-required` kind with successful universal checks and the correct
merge lane; they are neither mapped execution nor full Engine proof.

## Full main and red main

While the startup opt-in is true, main bypasses candidate reuse and executes
full CPU/model/Apple/coverage validation. After disabling, the existing discovery
barrier still rejects mapped trees without complete full evidence. Each next
mapped candidate needs its exact base's latest successful full qualification.
Pending, red or unavailable main forces an ordinary full repair candidate.

## Rollback

Dispatch `candidate-admission.yml` on `main` with `rollback=true`. Its latest
pending generation already prevents new mapped selection/qualification. The
trusted rollback job enumerates own open queue candidates and writes failure
using the same GitHub Actions admission identity when their proof is not full
or a valid policy exemption. A superseding activation aborts rollback rather than
claiming success. Preserve full repair work; do not dequeue/reset/bypass checks.
Then set the repository startup variable false using the authorized operator
CLI. No privileged variable token is installed in a PR or workflow.

Admission is revalidated immediately before publishing and after the status POST;
a changed live proof repairs its status to failure. Publication/revocation errors
must be investigated, not reported as successful rollback. GitHub's APIs and the
merge provider are not an atomic transaction: rollback cannot undo an already
completed merge, and visibility/status propagation can leave a bounded race.
Confirm actual required gate failures for inflight candidates and a subsequent
normal full candidate before declaring rollback complete. No hosted pilot,
activation or timing improvement is established by the local contract suite.
