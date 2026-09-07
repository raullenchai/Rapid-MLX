# Atlas handoff: lazy service model qualification

- **Owner:** Atlas
- **Branch:** `atlas/service-qualify-lazy`
- **PR:** #3222
- **Base/dependency:** `atlas/319-lazy-idle`, PR #3214
- **Host:** local Apple silicon Mac
- **Scope:** require configured-primary activation before service install,
  configuration apply, or runtime upgrade commits

## Verified facts

- Lazy `standby` is intentionally endpoint-ready and does not load weights.
- `POST /v1/models/activate` is an auth-gated, model-selector-free operation
  that loads only the configured primary through `PrimaryModelLifecycle`.
- Install, apply, and upgrade now require `/readyz` followed by activation with
  `state=ready` and `model_loaded=true`; existing rollback paths remain intact.
- Rollback checks only endpoint readiness so restoration remains compatible
  with an older runtime that does not expose the activation operation.
- The root-run client reads a credential through `O_NOFOLLOW`, validates the
  opened inode's type, mode, and service-account uid, and sends it only in an
  in-memory loopback Authorization header.
- Product docs, API reference, architecture decision, website handoff, and
  release-notes source copy are recorded in the PR description and docs diff.

## Verification

- Diff-aware targeted suite: 548 passed.
- Focused service/route/lifecycle suite: 345 passed.
- Patch coverage: 85/85 executable changed lines, 100%.
- Ruff lint/format and diff check: pass.
- Independent Codex review: no blocking findings.
- Full unit: 23,069 passed; three environment-only failures from missing
  optional `mflux` and the running Desktop service owning hard-coded port 8000.
- Real `Qwen3-0.6B-4bit` dogfood: authenticated activation moved one lazy PID
  from 48,816 KiB standby to 795,728 KiB ready; five-second idle unload returned
  it to 52,864 KiB standby, then the test server shut down cleanly.

## Remaining work / next action

1. Wait for #3214 to merge from the managed mac queue.
2. Retarget/rebase #3222 onto `main` without changing the service behavior.
3. Re-run the final CI/Apple Silicon service qualification on the rebased head.
4. Only after those gates pass, add the repository's merge-ready/mac-queue
   labels to #3222.

## Risks and rollback

- Model activation can add up to one cold-load interval to maintenance actions;
  the client timeout is bounded at ten minutes.
- Qualification verifies load/lifecycle readiness, not semantic answer quality.
- Reverting #3222 restores endpoint-only gating; no persisted schema migration
  or user data rollback is required.
