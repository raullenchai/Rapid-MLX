# Candidate qualification rollout

The trusted `workflow_run` producer uses immutable default-branch code. It
revalidates the latest exact candidate CI attempt, trusted open queue PR,
first-parent base and candidate tree. It never checks out or executes candidate
code. Its artifact and status are in the separate `candidate-qualification/ci`
namespace. They are not complete-evidence inputs for main reuse.

## First producer slice

Only complete candidates are published. A full repair candidate does not need
a green main anchor. Existing Mergify rules do not consume this optional status.
No reduced routing, queue enrollment or repository variable is activated here.

The library additionally defines a default-off mapped qualification contract:
exact combined-path allowlist, pinned controllers, mandatory static and mapped
jobs, successful separate execution/coverage steps, base collection retention,
no skipped tests, tested tree equal to the queue tree, latest full current main,
and repeated candidate identity checks. Source/advisory artifacts are rejected.
This library capability is not an enabled hosted mapped producer: CLI and
workflow intentionally cannot activate it.

## Remaining rollout gates

1. Merge the mapped-execution and main-qualification dependencies, then independently
   review this producer at its final rebased head and observe real full output.
2. Add authenticated artifact consumption in queue qualification before enabling
   any reduced execution. Status success alone is an index, not artifact validation.
3. Wire mapped candidate execution and its distinct input artifact, preserving
   100% changed executable-line coverage and the exact reviewed controller set.
4. Prove reduced landing forces ordinary full main and cannot form a reduced
   qualification chain while main is pending/red/cancelled.
5. Prove rollback for already-qualified in-flight candidates, including supported
   status revocation/dequeue and full reroute; then enable the default-off canary.

Full/main/release coverage, scopes, queue policy and existing full-evidence
consumers are unchanged by the first producer slice. Do not describe it as
completed candidate acceleration or complete coverage of every repository test.
