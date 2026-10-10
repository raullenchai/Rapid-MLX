# GUI journey cost evidence

The AX harness writes an overall `result.json` and one
`journeys/<flow>.json` for each journey it starts. This applies both to named
`--flow` invocations and to `--flow all`. Existing CI artifact uploads retain
these files with the accessibility snapshots and app/sidecar logs.

```sh
RAPID_GUI_GOLDEN_OUT=/private/tmp/rapid-gui-cost/run-1 \
  apps/rapid-mac/scripts/gui-golden-flows.sh --flow settings-persistence
jq . /private/tmp/rapid-gui-cost/run-1/journeys/settings-persistence.json
```

Use a fresh output directory for each invocation. A journey record contains:

| Field | Meaning |
| --- | --- |
| `flow` | Named journey, even when the invocation selects `all` |
| `started_at` | UTC wall-clock start, for correlating logs |
| `duration_seconds` | Elapsed journey time from the monotonic system clock |
| `launch_duration_seconds` | Sum of app spawn → accessible main-window waits, including relaunches and interrupted startup |
| `execution_duration_seconds` | Remaining journey time, including fixture setup, assertions, and cleanup |
| `launch_count` | Number of attempted app launches, including relaunches |
| `status` | `pass`, `fail`, or `cancelled` (exit 130/143) |
| `exit_code` | Journey failure/cancellation exit code, or 0 for success |
| `artifact_path` | Invocation's artifact directory, containing all personas used by this journey |

Timing values have millisecond resolution and include harness overhead. Launch
time measures main-window accessibility, not model readiness or inference.
Execution time is not a pure interaction benchmark. The overall legacy result
continues to include shared tool preflight and cleanup; per-journey time starts
after shared preflight. A preflight failure produces only the overall result.

On the first failed assertion, the harness stops and retains earlier journey
records. The active journey receives failure evidence after cleanup; an
interrupted startup stops its launch timer before cleanup. Journeys that never
start and flows excluded by source routing produce no record. SIGKILL and host
loss cannot run the EXIT handler and may leave the active journey without a
record; missing evidence must never be interpreted as a pass.

This is the measurement foundation for issue #2254. Rolling P50/P95, failure
rates and flake classification need historical run/attempt data and are separate
work. Do not infer a feedback-time reduction or a flaky test from one run.
