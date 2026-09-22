# Atlas handoff: telemetry v2 inference emitters

- Receiving owner: Atlas
- Branch: `feat/telemetry-v2-emit-inference`
- PR: #3665
- Host: Studio

## Verified facts

- The public completion emitter gates on official build plus live consent
  before touching local state, then submits SQLite, capture, and active-day
  work to a dedicated single-worker executor. Admission is capped at 64
  running/queued items; overflow drops immediately and never occupies the
  asyncio default executor.
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
- Worst-case cardinality is 336 keys/model. `MAX_KEYS=12_000` holds all
  endpoint/caller/result combinations for 35 complete models; the 36th is the
  first fully saturated model that can exhaust new keys.
- The exact telemetry/route/request selector passes 5,879 tests (25 skipped,
  6 xfailed, 1 xpassed). Changed tests pass three consecutive 577-test runs
  under `USER=rc`; the opt-out/CI-marker run also passes all 577. With `mlx`
  genuinely unimportable and the platform faked to Apple Silicon, the focused
  regression tests pass and the CI-style coverage union reports 100% changed
  line coverage (282/282), with no coverage pragmas. Latest Ruff and the pinned
  Python 3.11 no-MLX mypy budget pass with no growth (738 errors across 144
  grandfathered files).

## Remaining integration action

No code handoff remains for review round 2. The branch was rebased onto
`origin/main` after #3663 merged; PR #3665 only needs CI/reviewer confirmation.
