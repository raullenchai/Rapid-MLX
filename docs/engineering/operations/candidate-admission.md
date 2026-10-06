# Trusted candidate admission deployment

`Candidate admission` starts on successful own queue CI completion in parallel
with qualification, and retains successful qualification notifications. It reads one bounded exact producer artifact identity, requires the
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

## Observer bootstrap timing

Before enrollment, normal merging may advance main before a chained observer
finishes. The CI completion trigger starts trusted checkout alongside the producer,
then waits up to 45 seconds for its owned completed qualification index. This wait
never replaces full consumption: both archive validations and final producer/index
reads still run, and their source run/attempt must match the triggering CI exactly.
Missing or pending producer, API failure, wrong trigger, source retry, closed
candidate or changed main cannot publish positive admission. The later producer
notification remains available; neither trigger can use historical proof as live
merge authority. No queue hold, check reset, replay or gate waiver is introduced.
A slow runner can still miss the live candidate window; inspect actual upload and
index outcomes rather than treating a green observer aggregate as positive proof.

## Connection reuse in the observer

The hosted observer opts into one authenticated HTTPS connection per CLI for its
JSON reads. Every run, job, status, open-candidate and main check still makes a
fresh GET; responses and authorization are never cached. Pagination keeps the
original own-repository endpoint and filters, follows only authenticated API
page numbers, and rejects redirects, foreign links and malformed responses.
Artifact downloads retain the bounded existing ZIP transport and validation.
Network or HTTP failures reject the proof without retrying through another client.

Producer readiness can be awaited within the existing deadline. Once the producer
is completed, a rejected actual proof is terminal; stale or closed proof does not
become a readiness retry. Connection reuse reduces subprocess/TLS overhead; it does
not guarantee a hosted window, positive admission or a general CI speedup.
