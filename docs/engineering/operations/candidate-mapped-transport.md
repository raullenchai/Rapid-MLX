# Mapped candidate input transport

`inspect_mapped_transport()` is a dormant, read-only inspection library.
No current workflow calls it, creates its artifact or selects a mapped candidate
job. It grants no merge or routing permission. Independent final-head review,
actual execution wiring and hosted proof remain mandatory before rollout.

The latest successful own-repository queue CI run must have a successful
`candidate-canary-unit` job and an executed `Upload mapped execution proof`
step. A single nonexpired artifact is named
`candidate-mapped-input-<candidate-sha>-<run-id>-<attempt>`, bound by authenticated
artifact run metadata, and contains only `candidate-mapped-input.json`.
The JSON follows `rapid-mlx/candidate-mapped-input/v1`, with candidate/base/run/
attempt/test-selection/tested-SHA and baseline/execution manifests. Archives
are bounded and read without extracting or executing contents.

Existing qualification checks are repeated: actual combined-path allowlist,
controller pinning, mandatory static/mapped jobs, distinct successful execution
and coverage steps, base collection retention, successful setup/call/teardown
for every node, tested tree equality, current full main qualification and final
candidate/main state checks. This transport helper is additionally pinned to
the trusted caller's version. Source/advisory/full-evidence artifacts cannot be
substituted for mapped input. Newer runs, cancellation, missing uploads,
coverage failures, stale base/tree, malformed identities and skips reject.

Inspection output always has `authorizes_merge=false` and
`authorizes_reduced_ci=false`, including its nested qualification. Missing
mapped execution on current full CI is a rejection, not a mocked hosted pass.
Only local fixture transport was exercised; no production changed-line pilot
or performance gain is established here.

Future wiring must supply real manifests and mandatory coverage from reviewed,
pinned job steps; authenticate transport again in the trusted producer and
required consumer; preserve full/main namespaces, forced full main after a
reduced landing, red-main admission and in-flight rollback. Full candidates
and full repair admission retain their independent existing path.
