# Atlas handoff: telemetry v2 inference emitters

- Receiving owner: Atlas
- Branch: `feat/telemetry-v2-emit-inference`
- PR: manager-owned; not opened by this task
- Host: Studio

## Verified facts

- The public completion emitter gates on official build plus live consent
  before touching local state, then submits SQLite, capture, and active-day
  work to the default executor without awaiting it on the response path.
- The six existing v1 generation-terminal sites emit `failed` on generation
  exceptions and `ok` only at their success boundary. Client disconnects emit
  neither result. Chat non-streaming emits only after response serialization.
- Responses, embeddings, audio transcriptions, and image generations now emit
  at their real completion boundaries; Responses streaming uses its own loop
  and does not double count through chat.
- Successful inference uses `track.emit_active_day()`; failed outcomes do not
  attempt the claim. Bucket properties use the exact `store.record()` crossing,
  including `observed_existing` for a pre-existing unclaimed bucket.
- All capability sites named by the registry emit a required closed
  `model_type`; unresolved models use `other`, and unknown capability values
  are dropped at the emitter boundary.
- A 100-item `capability_rejected` loop admits 30 items under the sender's
  existing per-event burst cap.
- The real worker body, including `emit_active_day`, measured 0.8443 ms p50 and
  1.2525 ms p95 over 500 samples. The public on-loop gate plus executor submit
  measured 3.62 us p50 and 18.04 us p95 over 1,000 samples.
- Worst-case cardinality is 336 keys/model. `MAX_KEYS=12_000` holds all
  endpoint/caller/result combinations for 35 complete models; the 36th is the
  first fully saturated model that can exhaust new keys.
- The exact telemetry/route/request selector passes 4,909 tests. Changed tests
  pass three consecutive 294-test runs under `USER=rc` and CI markers. With
  `mlx` genuinely unimportable, 132 tests pass and changed production lines
  have 100% coverage (70/70), with no coverage pragmas. Ruff and the pinned
  mypy budget pass with no growth (738 errors across 144 grandfathered files).

## Remaining integration action

Rebase this branch after the T7 model-emitter branch lands, resolve any overlap
in `cli.py` / `server.py`, then run the required `stress_e2e_bench` without
`PR_VALIDATE_NO_STRESS=1`. The task explicitly reserves PR creation and that
stress-gate action for the manager.
