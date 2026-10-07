# Full candidate qualification consumer

`scripts.ci_candidate_consumer.consume_full()` is a read-only, unused-by-queue
transport verifier. It does not authorize merging or reduced CI. It is a
prerequisite for later queue wiring, not an enabled candidate fast lane.

The newest `candidate-qualification/ci` commit status must be successful,
created by the workflow bot, and point to the repository's exact trusted
`workflow_run` producer. The current producer attempt must have successfully
executed its qualification upload and index steps. The bounded artifact must
belong to that run, have the exact SHA-bound name, and contain only
`candidate-qualification.json`. Publication timestamps must not predate the
current producer attempt. No archive content is extracted or executed.

Transport alone does not establish authority. The consumer accepts only the
full qualification schema and recomputes its complete output from live APIs:
latest source run/attempt, exact CPU9/model5/static/Apple/coverage jobs,
open own-repository queue PR, creator, ref, parent, base and tree. The artifact
must equal that recomputation. Its base must still be current main. Producer,
status, main and candidate are read again to reject observed changes or
revocation. A full repair candidate may qualify while main is red; this does
not grant mapped admission.

Missing, expired, malformed, stale, scoped, partial or revoked proof returns
`verified=false`. All results retain `authorizes_merge=false` and
`authorizes_reduced_ci=false`. Full proof namespaces used for main reuse,
existing checks, Mergify rules, repository variables and release gates are
unchanged. Source and advisory artifacts are not accepted. Mapped transport,
nonempty production coverage proof, queue consumption, reduced landing's full
main barrier and rollback remain separate rollout obligations.

Local fixtures validate transport and mutable-state rejection. They do not
prove a hosted producer deployment or an atomic GitHub merge transaction.
After dependencies land, rebase and independently review the exact final head;
then observe actual hosted full producer/consumer proof before any queue wiring.
