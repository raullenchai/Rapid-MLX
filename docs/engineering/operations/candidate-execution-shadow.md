# Candidate mapped execution shadow

This additive pilot runs only when `RAPID_MLX_CANDIDATE_SHADOW_EXECUTION=true`.
It leaves all full candidate checks required. The variable is initially unset;
landing this code does not authorize activation or reduced candidate routing.

The classifier selects only a current own-repository Mergify candidate whose
actual combined diff is completely allowlisted, whose controller blobs match
its first-parent base, and whose current main base has authenticated full
qualification. Missing, stale, unknown, critical, mixed or red-main evidence
keeps ordinary full routing and disables the shadow.

Selected execution checks out the exact candidate head, collects the same
mapped suites on its actual base, then runs those suites without marker filters.
The recorder rejects removed tests, empty collections, deselection, skips,
xfails and incomplete setup/call/teardown. A separate required step enforces
100% changed-line coverage. The packer verifies the checkout and tracked-file
cleanliness, JUnit and coverage presence, and binds the input to SHA/run/attempt.
The distinct artifact is `candidate-mapped-input-SHA-RUN-ATTEMPT` and contains
only `candidate-mapped-input.json`. It cannot replace full evidence.

When selected, any shadow failure or cancellation fails the stable `tests`
aggregate; full checks must still succeed independently. Default-off execution
must remain skipped. The full-only qualification producer remains unchanged;
this pilot grants no merge authority or reduced-CI permission.

Before activation, require independent exact-head review and genuine full
candidate qualification, then verify an actual eligible hosted pilot, retained
base nodes, nonempty production changed-line coverage, artifact transport and
negative controls. Local tests and test-only diffs do not prove that pilot.
Rollback clears the variable to disable added execution; full checks continue.
