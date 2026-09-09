# Atlas handoff: lifecycle-aware service observability

- **Owner:** Atlas
- **Branch:** `atlas/service-lifecycle-observability`
- **Base/dependency:** stacked on `atlas/service-qualify-lazy` at `a314456a`
  while PR #3222 is in the mac-batch merge queue
- **Goal:** make an always-on endpoint's availability, primary-model residency,
  and recent lifecycle behavior visible without issuing inference traffic

## Implemented contract

- `PrimaryModelLifecycle.snapshot()` retains process-local load attempts,
  failures, latest completed load duration, successful unload counts, and last
  unload reason. Error output remains the sanitized exception type.
- `rapid-mlx service status` reads the existing public `/health` response and
  renders lifecycle state, residency, idle policy, load/unload summary, and last
  error. `--json` exposes stable flattened keys.
- A running older server or malformed/unavailable `/health` detail response is
  non-fatal: existing launchd, liveness, readiness, and exit-code behavior is
  unchanged and lifecycle keys are `null`.
- `/metrics` exposes primary residency, a fixed one-hot lifecycle state family,
  load attempt/failure counters, latest load duration, and successful idle
  unloads. Lifecycle metrics render before engine-dependent statistics, so
  standby and partial-engine scrapes remain useful.
- User and server guides, CLI reference, and the Always-on service decision
  record document the contract for later website/release-note reuse.

## Deliberate non-goals

- No current/candidate workers, multi-model concurrency, or GUI state machine.
- No manual or memory-pressure unload behavior. The unload reason label is
  currently the one implemented policy, `idle`; future policies can add their
  own bounded reason values when their behavior lands.
- Metrics are process-local and reset on restart, matching Prometheus counter
  semantics for other Rapid-MLX runtime metrics.

## Verification

- `ruff check` and `ruff format --check` on all touched Python files
- 290 targeted tests passed on Python 3.12 after error/fallback coverage
- changed-lines coverage: 77/77 lines, 100%
- Python 3.11 pinned mypy budget passed: 701 grandfathered errors, zero growth
- Apple Silicon dogfood with cached `mlx-community/Qwen3-0.6B-4bit`, API auth,
  lazy load, and a 3-second idle timeout passed. The same PID stayed ready from
  cold standby through activation (2.163s load) and back to standby; load and
  idle-unload metrics advanced from 0 to 1 as expected.

## Remaining action

Commit, push, open a stacked PR with a detailed docs/release-note-ready
description, and rebase onto `main` after #3222 merges.
