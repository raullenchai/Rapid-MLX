# Trusted candidate admission

The trusted default-branch workflow consumes the current qualification artifact
as data, revalidates actual source jobs, open queue candidate, exact run/attempt,
first-parent base/tree, current main and controller identities, and rereads the
producer/index after the second consumer. No candidate code is executed with
publication credentials.

Successful own queue CI also starts admission in parallel with qualification; failed qualification notifications remain available for revocation.

Full, mapped and explicit Engine policy exemptions have separate scope kinds.
Only full proof can qualify complete-evidence reuse. Mapped proof additionally
requires live authenticated activation generation, mapped execution/coverage
and a currently full-qualified base. Admission evidence uploads before the
commit-bound status is published; publication revalidates again and repairs a
changed proof after posting. Missing/stale/revoked proof is never a green gate.

Both queue merge rules require the GitHub Actions-owned admission context.
Deploy observer workflows and verify a genuine current full admission before
publishing enrollment changes; no candidate-only gate is added to source queue
admission. The default-off rollout and recovery procedure are documented in
[candidate-reduced-rollout.md](candidate-reduced-rollout.md).

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
