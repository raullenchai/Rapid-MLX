# CUA server API handoff

- Receiving role: Atlas (architecture/public API) and Vector (server ownership)
- Owner: Harbor implementation agent
- Branch: `harbor/cua-server-api`
- Worktree: `/private/tmp/harbor-desk-cua-server-api`
- Base: PR #3824 head `94353c6b43b4c019e2a2253bf770583c58f29dcf`

## Intention and scope

Give authenticated third-party GUI clients enough server contract to discover
the local CUA host, create and supervise high-level runs, render typed events
and approval gates, deny a gate, cancel, and avoid receiving host-private trace
paths. Raw click/type endpoints and screenshots are outside this change.

## Verified facts

- Every new endpoint inherits the fail-closed bearer and rate-limit router
  dependencies from PR #3824.
- Discovery covers protocol capabilities, macOS permissions, regular apps, and
  on-screen windows. Backend work runs off the async event loop and imports are
  lazy, preserving Linux server import safety. Window discovery no longer
  activates or mutates the target application.
- Run events expose a typed `kind`/`seq`/`ts` envelope with forward-compatible
  event fields. `pending_gate` carries reason and expiry. The existing approval
  route preserves empty-body approval for the existing Swift client. New clients
  echo the stable `gate_id` with an explicit decision; stale IDs fail with 409,
  so a delayed approval cannot resolve a later gate in the same run. Decisions
  are one-shot under the run lock: identical retries are idempotent and a
  conflicting second decision fails with 409 without changing the first.
- Window discovery returns the integer Core Graphics `window_id` that stays
  stable for the lifetime of that on-screen window, alongside its current list
  index. Windows are matched to the resolved process PID rather than owner name.
- `run_dir` is removed from run responses and stripped from public start events.
- Focused verification: `96 passed, 2 deselected` in `tests/test_cua_server.py`
  and `tests/test_cua.py`
  under Python 3.12. The two real-server import tests were excluded because the
  local MLX/tokenizer import aborts this host Python process; this is unrelated
  to the route tests. Ruff and `git diff --check` pass.

## Reference-first notes (private)

- Rapid-MLX's existing agent route convention was the closest precedent:
  asynchronous acceptance, stable IDs, numbered polling cursors, explicit
  cancellation, and typed lifecycle state. The CUA surface retains it.
- vLLM and SGLang serving precedents were checked first. Their relevant public
  pattern is resource creation plus ID-scoped polling/cancellation; neither
  provides computer-host discovery or interactive action approval semantics.
- MLX-native serving code does not provide a closer CUA contract.
- Open WebUI's task supervision patterns were checked for UI-independent task
  history, stop controls, and visible Allow/Deny gates. The adopted part is a
  server-owned pending gate that any authenticated GUI can render and resolve.
  The unattended full-approval pattern was rejected because Rapid-MLX requires
  explicit sensitive-step supervision.
- Jan, Cherry Studio, and LM Studio were checked conceptually for local server
  discovery/OpenAI compatibility; none offers a closer native macOS CUA host
  permission/window contract than the existing model-free backend.

## Risks and next actions

- Screenshots/observations remain deferred: an authenticated observation can
  include private pixels, accessibility labels, URLs, and entered text. Atlas
  should first define redaction, per-app consent, maximum payload, rate limit,
  freshness/cache, and remote-bind policy. Vector can then implement a scoped
  `GET /v1/cua/apps/{app}/windows/{window_id}/observation` whose screenshot is
  opt-in and whose structured tree is redacted. Structured plan/result events
  (`action`, target label/index, outcome, tree change, URL transition) remain
  sufficient for the first custom GUI API.
- Run storage remains in-process and is lost on server restart; persistence or
  resumable runs require a separate architecture decision.
- Run creation still targets the app's front window; it cannot bind a discovered
  non-front `window_id`. Atlas should define whether `window_id` is immutable
  for a run and how disappearance/replacement fails. Vector can then thread the
  selected ID through snapshots, screenshots, and every semantic action without
  exposing raw action endpoints.
- Team FYI transport was unavailable to this worker. Atlas should send the
  required start/completion FYIs when preparing the PR.
- Next action: Atlas reviews the intention contract and diff, then decides
  whether this is opened as a standalone PR stacked on #3824.
