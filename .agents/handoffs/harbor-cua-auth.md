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

The route dependency now fails closed when the server has no API key; configured servers reuse the existing bearer validator. The 126 focused CUA/computer-use tests and Ruff pass in the isolated checkout. Two full-server import tests were excluded because this sandbox lacks Metal. A review and PR handoff remain.
