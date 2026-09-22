# Atlas handoff: telemetry v2 inference emitters

- Receiving owner: Atlas
- Branch: `feat/telemetry-v2-emit-inference`
- PR: #3665
- Host: Studio

## Verified facts

- The public completion emitter gates on official build plus live consent
  before touching local state, then submits SQLite, capture, and active-day
  work to one lazy daemon thread through a 64-slot queue. Overflow drops
  immediately, interpreter exit drops queued work without joining a blocked
  worker, and a forked child receives fresh queue/thread state. The lane never
  occupies the asyncio default executor.
- The six existing v1 generation-terminal sites emit `failed` on generation
  exceptions and `ok` only at their success boundary. Client disconnects emit
  neither result. Chat non-streaming emits only after response serialization.
- Responses, embeddings, audio transcriptions, and image generations now emit
  at their real completion boundaries; Responses streaming uses its own loop
  and does not double count through chat. Guided chat success/strict failure
  and every Responses streaming failure terminal are covered.
- Successful inference uses `track.emit_active_day()`; failed outcomes do not
  attempt the claim. Bucket properties use the exact `store.record()` crossing,
  including `observed_existing` for a pre-existing unclaimed bucket.
- All capability sites named by the registry emit a required closed
  `model_type`; unresolved models use `other`, and unknown capability values
  are dropped at the emitter boundary.
- A 100-item `capability_rejected` loop admits 30 items under the sender's
  existing per-event burst cap.
- The real worker body, including `emit_active_day`, measured 0.5171 ms p50 and
  0.6275 ms p95 over 500 samples. Public submission with the real worker
  completing every item measured 4.29 us p50 and 5.50 us p95 over 1,000
  samples.
- Worst-case cardinality is 416 keys/model after adding the five first-party
  caller labels. `MAX_KEYS=12_000` holds all endpoint/caller/result combinations
  for 28 complete models; the 29th is the first fully saturated model that can
  exhaust new keys.
- The exact telemetry/route/request selector passes 5,879 tests (25 skipped,
  6 xfailed, 1 xpassed). Changed tests pass three consecutive 577-test runs
  under `USER=rc`; the opt-out/CI-marker run also passes all 577. With `mlx`
  genuinely unimportable and the platform faked to Apple Silicon, the focused
  regression tests pass and the CI-style coverage union reports 100% changed
  line coverage (282/282), with no coverage pragmas. Latest Ruff and the pinned
  Python 3.11 no-MLX mypy budget pass with no growth (738 errors across 144
  grandfathered files).

## Review round 3

- Added all five `RAPID_CLIENT_LABELS` to the registry and made the emitter
  collapse a future registry mismatch to `other` before consuming a crossing.
- Guided client cancellation remains uncounted; a lifecycle-owned model
  replacement records one `failed` result.
- Added direct wire, process-exit, fork, Responses-latch, and named capability
  rejection coverage.
- Round-3 verification: changed tests passed 237/237 under `USER=rc` and
  237/237 with CI/opt-out variables; the focused no-MLX run passed 210 with 27
  expected skips. The saved full-lane coverage plus round-3 focused coverage
  reports 100% changed-line coverage versus `origin/main` (315/315). Swift
  build, Ruff, the pinned mypy budget (738 errors across 144 grandfathered
  files), and all 14 no-out-of-band routing tests pass.
