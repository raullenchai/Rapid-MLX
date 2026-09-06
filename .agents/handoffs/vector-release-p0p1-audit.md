# Vector -> Atlas: v0.13.4-to-main release audit

- Owner/host: Vector, Studio
- Audit branch: `audit/release-p0p1-20260906`
- Audit range: `v0.13.4..901611faa` (64 first-parent merged PRs)
- Durable report: `docs/engineering/operations/2026-09-06-v0.13.4-to-main-release-audit.md`

## Verified facts

- P0 credential-directory symlink escalation is fixed in #3179; 195 focused
  tests, 100% statement coverage for both changed production modules, and local
  lint/diff checks pass. Hosted exact-head checks remain the merge gate.
- P1 unauthenticated non-loopback persistent bind is fixed in #3178; 195
  focused tests and local lint/diff checks pass. Hosted exact-head checks remain
  the merge gate.
- P1 Desktop upgrade port compatibility is fixed in #3177; local focused tests
  and all hosted exact-head checks pass, and the PR is queued.
- Image qualification #3173 and Qwen3.5 4B MTP default-off #3144 are queued.
- Ten image aliases and Desktop release-mode image journeys have real-weight
  dogfood receipts. The final built artifact still needs the documented rerun.
- The M3 G0 gate passed four families and hit two honest capacity-skips. The
  Qwen3.5 35B cache is on a nearly-full, pathologically slow external ExFAT
  volume; this is a host-storage blocker, not evidence of a model regression.

## Risks and next action

Atlas should hold release integration until #3178/#3179 merge and arrange a
healthy cache volume for a complete `make release-check-m3` rerun. Harbor should
rerun the exact source/artifact image dogfood receipt on the final signed and
notarized candidate. Do not delete shared model caches without explicit human
authorization and a recovery plan.
