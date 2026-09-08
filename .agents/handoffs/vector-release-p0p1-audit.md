# Vector -> Atlas: v0.13.4-to-main release audit

- Owner/host: Vector, Studio
- Audit branch: `vector/release-delta-audit-20260906`
- Audit range: `v0.13.4..18d488664` (103 first-parent merged PRs)
- Durable report: `docs/engineering/operations/2026-09-06-v0.13.4-to-main-release-audit.md`

## Verified facts

- P0 credential-directory symlink escalation is fixed on main by #3179.
- P1 unauthenticated non-loopback persistent bind is fixed on main by #3178.
- P1 Desktop upgrade port compatibility is fixed on main by #3177.
- Image qualification #3173 and Qwen3.5 4B MTP default-off #3144 are on main.
- #3169, #3150, #3158, and #3110 landed together after their full combined
  Python, Apple Silicon, Desktop, GUI, L1-smoke, coverage, and contract gates
  passed. Their production diffs were included in the final audit; no new
  P0/P1 finding survived review.
- Ten image aliases and Desktop release-mode image journeys have real-weight
  dogfood receipts. The final built artifact still needs the documented rerun.
- The M3 G0 gate passed four families and hit two honest capacity-skips. The
  Qwen3.5 35B cache is on a nearly-full, pathologically slow external ExFAT
  volume; this is a host-storage blocker, not evidence of a model regression.
- The 12-merge delta after the original cutoff was rereviewed through #3193.
  The executable draft flow, dormant local visual grounder, and signed Dock
  backdrop exception preserve their opt-in, observation, approval, bounded
  actuation, and fail-closed boundaries. #3193 is test-only and keeps its live
  Calculator action behind three explicit operator values. No new P0/P1
  finding survived.
- Harbor reviewed the final 23 merges through #3231. The two resulting P1
  findings are fixed on main: #3242 binds CUA visual recovery to an exact
  per-launch Desktop server session (`430974e56`), and #3244 removes persistent
  credential transmission from root-run service qualification (`463ecd39e`).
  Their exact-head and combined queue candidates passed; both issues closed on
  merge.
- The exact `18d488664` Desktop package passed 3,610 tests across 311 suites.
- Combined-main validation passes 204 service tests, Python 3.12 compileall,
  release-range diff checks, and 3,542 Swift tests across 306 suites at the
  exact #3193 head.

## Risks and next action

The reviewed source range is P0/P1-clean at `463ecd39e`. Atlas should arrange
a healthy cache volume for a complete `make release-check-m3` rerun. Harbor
should rerun the exact source/artifact image dogfood receipt on the final
signed and notarized candidate. Do not delete shared model caches without
explicit human authorization and a recovery plan.
