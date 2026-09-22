# Atlas handoff: telemetry v2 inference emitters

- Receiving owner: Atlas
- Branch: `feat/telemetry-v2-emit-inference`
- PR: manager-owned; not opened by this task
- Host: Studio

## Verified facts

- The six existing v1 generation-terminal sites each have exactly one adjacent
  unsampled v2 inference call. Streaming and non-streaming chat paths execute
  the v2 call exactly once.
- Successful inference uses `track.emit_active_day()`; failed outcomes do not
  attempt the claim. Bucket properties use the `store.record()` crossing.
- All capability sites named by the registry emit a required closed
  `model_type`; unresolved models use `other`.
- A 100-item `capability_rejected` loop admits 30 items under the sender's
  existing per-event burst cap.
- The isolated SQLite-record plus registry/envelope-track benchmark measured
  0.5933 ms p50 across 500 pessimistic first-bucket samples on the M3 Ultra.
- Focused tests pass with clean environment variables, hostile exported kill
  switches, and `mlx` made unimportable. Changed production lines have 100%
  combined coverage. Ruff and the pinned Python 3.11 mypy budget pass.

## Remaining integration action

Rebase this branch after the T7 model-emitter branch lands, resolve any overlap
in `cli.py` / `server.py`, then run the required `stress_e2e_bench` without
`PR_VALIDATE_NO_STRESS=1`. The task explicitly reserves PR creation and that
stress-gate action for the manager.
