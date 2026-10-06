# Trusted candidate admission deployment

`Candidate admission` consumes only the successful trusted qualification
workflow. It reads one bounded exact producer artifact identity, requires the
candidate's newest qualification index to bind that same producer run, and
calls the full consumer twice to revalidate the archive, actual full CI jobs,
current attempt, open candidate, first-parent base/tree and current main.
The producer and index are reread before returning a verified candidate.
Candidate artifacts are parsed as data and never executed; checkout uses the
trusted workflow SHA with credentials disabled.

Verified full results upload separate admission evidence before publishing
`candidate-admission/ci`. Missing, stale, malformed, superseded, mapped or
unavailable qualification cannot publish success. This workflow is an observer
until queue enrollment is independently reviewed and deployed; current queue
conditions and required checks are unchanged. No result authorizes merging or
reduced CI. Absence of a success index is not a successful admission.

Deploy and validate genuine hosted producer/consumer output before enrolling
this context in merge conditions. Enrolling a context in its own bootstrap
candidate before its default-branch workflow exists can deadlock that candidate.
A later mandatory gate must define current-attempt validity/revocation and
inflight rollback; an old advisory success must not be reused as new authority.
Full repair candidates continue to use the existing complete validation policy;
this observer grants no red-main exception for reduced candidates.

This slice does not enable mapped routing, change full-main reuse/backstop,
activate variables, or modify Mergify/protection. Local fixtures establish the
consumer contract, not hosted behavior or speedup. During observer rollout,
rollback may disable the workflow without changing existing full checks.
