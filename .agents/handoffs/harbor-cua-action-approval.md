# Harbor handoff — scoped CUA action approval

## Task contract and FYI

- Owner: Harbor, with backend review requested from Vector and product scope held by Atlas.
- Branch/worktree: `harbor/cua-action-approval` in `/private/tmp/harbor-desk-cua-approval`.
- Goal: pause local CUA before an externally committing or destructive action,
  describe the exact app/action/target, and revalidate fresh UI state after approval.
- Scope: `rapid_mlx/cua/gates.py`, `loop.py`, `service.py`, and focused CUA tests.
- Non-goals: computer-use backend changes, planner/config changes, Desktop CUA UI,
  remote browser/VM work, credentials, payments, and commerce. Existing hard stops remain.
- Verification: focused CUA loop suite, approval service tests, diff/format checks.
- Agent messaging was unavailable for the requested Atlas route (`terminal_not_found`),
  so this handoff records the required start FYI for Atlas, Pixel, Vector, Echo, and
  ds0731. Work continued because the FYI is non-blocking.

## Private reference check

- Rapid-MLX already had a one-shot run approval endpoint and `awaiting_approval`
  state for sign-in. The implementation reuses that surface instead of adding API.
- Open WebUI's current tool approval flow binds resolution to a pending call id,
  rejects duplicate/stale resolutions, and represents denial as a terminal tool result.
- Cherry Studio's approval flow pauses execution at the tool wrapper, makes the main
  process authoritative, and resumes only the held invocation after a specific decision.
- Jan's documented permission QA requires showing tool parameters, supports Allow Once,
  and verifies that denial prevents execution.
- LM Studio documents tool integrations but did not expose an equivalent action-level
  confirmation contract in the reviewed public material.
- Adopted pattern: one-shot, execution-time approval with explicit proposed parameters;
  stale or late approvals fail closed. Rapid additionally re-observes macOS AX state
  because GUI indexes, geometry, process, window, URL, and surrounding state can drift.

## Current state

- Consequential click/press controls are classified by a deterministic multilingual
  policy. Search submission is exempt only from the observed target label, never model
  prose. Draft/fill and read-only navigation remain uninterrupted.
- Approval reason carries `external_commit`, action, target, proposed instruction, and app.
- Credential/payment and cart/checkout hard stops run before approval and remain unchanged.
- Each service gate allocates a fresh event before exposing `awaiting_approval`; fast,
  late, timed-out, and replayed approvals cannot authorize another action.
- After approval, the loop fetches a no-cache snapshot and checks app PID/bundle/name,
  window index/optional stable window ID, target index/role/label/actions/geometry,
  tree signature, URL, domain, and hard consents. Missing required identity or any drift
  stops before input.
- CLI approval uses a reason-derived one-shot marker and consumes it after use.

## Verification evidence

- `python3.12 -m pytest tests/test_cua.py -q`: 85 passed.
- `python3.12 -m pytest tests/test_cua_server.py -q -k approval`: 4 passed.
- A combined full CUA/server run reached all CUA tests and most server tests, then the
  existing AX watchdog test's intentionally wedged daemon threads aborted during an
  unrelated engine import. Running the suites separately avoids that host-runtime issue.

## Follow-up

- This draft depends on the pending local action-contract/snapshot PR. Current `main`
  does not expose a stable CGWindowID, so PID + window index + target geometry cannot
  distinguish an identical replacement window. This code compares `window_id` as soon
  as the dependency supplies it; do not mark merge-ready until the dependency lands and
  the branch is rebased/tested.
- User-facing claims must say "recognized labeled consequential controls." Unlabeled
  controls and synonyms outside the deterministic multilingual policy are not reliably
  classified until the action contract provides stronger semantics.
- Pixel should render gate event `action` and `target` in the existing approval UI. The
  backend event is backward compatible, but the present panel was intentionally untouched.
- Draft PR: https://github.com/raullenchai/Rapid-MLX/pull/3825. Atlas independently
  reviewed the classifier, gate/service races, and positive/negative tests. Keep Draft
  until the stable-window dependency lands.
- The required automated reviewer on `spark2` could not run because that host's Codex
  refresh token is expired (HTTP 401). Retry the review loop after host authentication is
  restored; no review verdict was produced.
- Completion FYI for Atlas, Pixel, Vector, Echo, and ds0731: implementation and focused
  tests are complete; affected subsystem is local CUA approval only; known rollout risk is
  the missing stable window ID on current `main`; Pixel owns the separate display follow-up.
