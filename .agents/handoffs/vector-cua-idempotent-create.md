# CUA idempotent run creation

- **Owner:** Vector
- **Branch/worktree:** `vector/cua-idempotent-create` at
  `/private/tmp/harbor-desk-cua-idempotent-create`
- **Base:** `harbor/cua-sidecar-pyobjc` at `9bf8177a`
- **Status:** implemented and locally verified; awaiting manager review before push

## Intention and scope

Make a high-level CUA run recoverable when its accepted HTTP create response is
lost. The server accepts an optional client request identity, deduplicates the
normalized create request, exposes authenticated lookup by that identity, and
serializes the single-computer create transaction through window validation and
commit. This change does not add low-level action routes, persistent storage, or
GUI behavior.

## Contract

- `POST /v1/cua/runs` accepts and echoes optional `client_request_id` (1–128
  characters, one URL path segment with no `/`). Legacy requests remain valid
  and receive `null` in that field.
- Same identity and normalized payload returns the original run with HTTP 202;
  no second validation or background task starts.
- Same identity and different payload returns typed HTTP 409 code
  `request_identity_conflict`.
- `GET /v1/cua/runs/by-request/{id}` returns the create response shape, or typed
  HTTP 404 code `request_identity_not_found`.
- `features.idempotent_run_create` advertises support.
- Identity mappings are pruned with their retained run and remain process-local;
  expired identities provide no continuing idempotency guarantee.

## Correctness notes

The async create lock covers the active-run check, awaited selected-window
validation, registry commit, and task creation. Cancellation during validation
leaves no run, mapping, or task. Once commit completes, a lost or cancelled
response handler cannot remove the run identity, so a client can recover and
cancel the task. Shutdown is checked again after awaited validation.

## Reference check

Existing Rapid run retention, typed error, authentication, and create-response
patterns were reused. The two primary serving precedents were checked: one
supports caller request IDs for correlation and cancellation, while neither
documents payload-bound idempotent task creation plus recovery lookup. MLX-native
serving code did not provide a closer retained-task precedent. The implementation
therefore adapts Rapid's existing in-process run registry and preserves its
retention boundary rather than introducing a second durable store.

## Verification

- `python3.12 -m pytest -q tests/test_computer_use.py tests/test_cua.py
  tests/test_cua_server.py -k 'not real_server'` — 182 passed, 2 deselected.
- Focused regressions cover same-ID replay, payload conflict, authenticated and
  typed lookup, concurrent same/different IDs with delayed validation, commit
  cancellation boundaries, one-task creation, and mapping retention/pruning.
- Ruff check and format checks pass for changed Python files.
- `git diff --check` passes.

## Coordination

Pixel's GUI owner received the exact capability, request, replay, lookup, and
typed error schema before implementation. Agent FYI messaging outside the
active task channel was unavailable; this handoff records the start and
completion FYI for Atlas, Pixel, Harbor, Echo, and ds0731.
