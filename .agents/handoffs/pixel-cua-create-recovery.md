# CUA create-response recovery

- **Owner:** Pixel, coordinated with Vector's idempotent create contract
- **Branch:** `pixel/cua-create-recovery`
- **Base:** `harbor/cua-gui-lifecycle-safety` at `ce2497a4`
- **Scope:** Desktop CUA client, lifecycle state, recovery UI, tests, release note

## Contract and behavior

Desktop requires `features.idempotent_run_create` before sending a create
request. Each Start owns one UUID `client_request_id`; all recovery attempts
reuse the exact request body. A normal acknowledgement must echo both that ID
and the selected opaque window ID.

Transport failures, server 5xx responses, and response decoding failures are
ambiguous because actuation may already have started. Desktop retries the same
idempotent request, falls back to `GET /v1/cua/runs/by-request/{id}`, and cancels
any recovered run. It does not silently continue automation after an ambiguous
acknowledgement. If recovery remains unreachable, Start stays locked and the
Stop control becomes Retry Recovery. A recovered run whose cancellation fails
enters the existing quarantine state with Stop available and approval disabled.
Known typed 4xx responses remain definitive and do not enter recovery.

## Private reference check

Reviewed the repository's bounded retry rule for draft transfer and the request
identity and duplicate isolation patterns in the primary serving precedents.
Those systems use caller request IDs for ownership and reject conflicting reuse;
they do not by themselves make side-effecting desktop actuation recoverable.
This implementation therefore combines matching-payload idempotency with an
explicit lookup and defaults to cancelling any run recovered after an ambiguous
create response.

## Coordination

Vector confirmed the server schema: body and acknowledgement
`client_request_id`, `features.idempotent_run_create`, and authenticated lookup
`GET /v1/cua/runs/by-request/{id}` returning the create acknowledgement shape.
The server retains request identities with their retained in-process runs.
