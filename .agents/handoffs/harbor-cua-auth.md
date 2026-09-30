# Handoff — Harbor: CUA HTTP authentication

## PR-start FYI (agent messaging unavailable)

- Intended recipients: Atlas, Pixel, Vector, Echo.
- Owner/host: Harbor on Studio.
- Branch/worktree: `harbor/cua-auth-gate` in `/private/tmp/harbor-desk-cua-dogfood`, based on `6be91bab`.
- Intention: require an explicitly configured server bearer for every `/v1/cua/*` route, including planner configuration and run approval/cancellation.
- Scope: CUA router dependency, focused server regression coverage, CUA API documentation. No changes to generic inference authentication, Desktop UX, planner semantics, or action consent policy.
- Verification: route tests with and without configured credentials, Ruff, diff check, and Desktop client contract inspection.
- Coordination: Vector owns the server route and Pixel owns the Desktop client. The Desktop-managed server already supplies a per-launch bearer; neither client contract nor UI changes are planned.

## Current state

The route dependency now fails closed when the server has no usable API key, including an empty key; configured servers reuse the existing bearer validator. The independent review found the empty-key case and it is covered by the regression test. Two full-server import tests were excluded because this sandbox lacks Metal.

## PR-complete FYI (agent messaging unavailable)

- Intended recipients: Atlas, Pixel, Vector, Echo.
- PR: https://github.com/raullenchai/Rapid-MLX/pull/3824 (draft; review and CI in progress).
- Outcome: `/v1/cua/*` requires a configured server bearer. No Desktop protocol or standalone CLI change.
- Affected files: CUA router, route tests, CUA README, this handoff.
- Verification: 132 focused Python tests passed; Ruff check and format passed; no mypy errors in the changed router. The full local mypy baseline differs from CI on this Mac and is being checked by GitHub CI.
- Known risks: a standalone server with no API key will now return HTTP 503 for CUA requests. Configure a key to enable that API. No deployment or release was performed.
- Review blocker: the prescribed reviewer on spark2 could not start because its Codex refresh token is expired. A separate independent reviewer in this session completed a read-only round; the spark2 review remains pending reauthentication.
- Next owner/action: Vector should review the CUA route contract before merge; Pixel should confirm the Desktop bearer path on a physical Mac when GUI permissions are available. Atlas decides sequencing of later consent and durable-task work.
