# Community Benchmark read-API seam and the unknown-count rule

- **Status:** Client seam implemented and bound to the shipped public atomic
  endpoints; some aggregates remain unavailable (see "What the feed cannot
  answer")
- **Date:** 2026-09-15
- **Owner:** Pixel (Desktop UI), escalation to Atlas for the service contract
- **Consumers:** Rapid Desktop
- **Client seam:**
  [`CommunityBenchmarkDirectory`](../../../apps/rapid-mac/Sources/Rapid/UI/CommunityBenchmark/CommunityBenchmarkDirectory.swift)
- **Production adapter:**
  [`CommunityBenchmarkAPIDirectory`](../../../apps/rapid-mac/Sources/Rapid/UI/CommunityBenchmark/CommunityBenchmarkAPIDirectory.swift)
- **Shared identity mapping:**
  [`CommunityContributorAvatar`](../../../apps/rapid-mac/Sources/Rapid/UI/CommunityBenchmark/CommunityContributorAvatar.swift)
- **Branch rule:**
  [`CommunityContributionBranch`](../../../apps/rapid-mac/Sources/Rapid/UI/CommunityBenchmark/CommunityBenchmarkBranch.swift)
- **Cell projections:**
  [`CommunityTableProjection`](../../../apps/rapid-mac/Sources/Rapid/UI/CommunityBenchmark/CommunityTableProjection.swift),
  [`CommunityCoverageProjection`](../../../apps/rapid-mac/Sources/Rapid/UI/CommunityBenchmark/CommunityCoverageProjection.swift)
- **Publish bookkeeping:**
  [`CommunityPublicationState`](../../../apps/rapid-mac/Sources/Rapid/UI/CommunityBenchmark/CommunityPublicationState.swift)
- **Parent decision:**
  [Community benchmark wire contract v1](2026-08-31-community-benchmark-wire-contract.md)

## Correction (2026-09-15)

An earlier revision of this document asserted that no community read API
existed. **That was wrong.** rapidmlx.com ships three public routes, and a
second claim in the same revision — that contributor avatars are unavailable
and the app therefore draws a monogram — was wrong for the same reason. Both
are corrected below; the monogram fallback no longer exists in the code.

## Endpoints Desktop calls

| Route | Shape | Completeness | Called by |
| --- | --- | --- | --- |
| `GET /api/benchmarks/atomic/public` | `{schema_version, beta, ranking_status, summary[], runs[]}` | **Bounded** — newest 50 runs; `summary[]` is aggregated over exactly those | `observations(for:viewerSlug:)`, `table(macProfile:workload:metric:viewerSlug:)`, `coverageGaps(macProfile:)`, `pulse()` — one route, four questions |
| `GET /api/benchmarks/atomic/contributions?contributor=&limit=&cursor=` | `{schema_version, contributor, runs[], cursor, complete}` | Cursor-paginated over everything | `contributions(forSlug:)`, following `cursor` until `complete == true` (ceiling: 40 pages, then `.incompleteAggregate`) |
| `POST /api/benchmarks` | publish | — | Not called by the app directly: `rapid-mlx benchmark share` uploads and returns the receipt |

`GET /api/benchmarks/atomic/contributors/<slug>` exists and is equivalent to the
`contributions` route pre-filtered. Desktop uses the query-parameter form
instead, so one code path covers both, and the slug it filters by is the
server-issued one from the receipt.

The CLI has `catalog`, `plan`, `run`, `results`, `inspect` and `share`. It has
**no** read subcommand, which is why the community reads go to the site
directly rather than through the binary.

What follows is unchanged in principle — the unknown-count rule still governs —
but the reason a count can be unknown is now "the feed is bounded", not "there
is no endpoint".

### Model identity

A repo id is not a model. `proto/community-benchmark/v1` identifies the primary
component by `source.repo_id` plus optional `subfolder` and `resolved_revision`,
by the artifact's `quantization` facts, and by `identity_strength`.
`run_builder.unresolved_model_identity` fills all of it from the local cache, so
a `4bit/` subfolder or a different snapshot is a different model locally.
`CommunityModelIdentity` mirrors that and is carried through local decoding, the
scope, summary decoding, exact matching, both projections and publication
floors. Cells that differ on any facet are never merged; where the data cannot
single out one population, the count is reported (each run is in exactly one
group, so it stays exact) and the **median is withheld**.

**What the live service actually does**, verified against
`landing/src/index.js` at `519fd61`:

| | Contract | Deployed worker |
| --- | --- | --- |
| `source.subfolder` | optional | **rejected** — `atomicExactObject(source, ["kind","repo_id"], …)` |
| `source.resolved_revision` | optional | **rejected**, same allowlist |
| `quantization` | facts | must be exactly `{kind: "unknown", base_dtype: "unknown"}` |
| `identity_strength` | enum | must be `"unresolved"` |
| `summary[]` group key | — | `[task_type, repo_id, chip, memory_gib, protocol.id, protocol.version, case_id, canonical(execution)]` — **repo id only** |
| `summary[].model` | — | `{repo_id, identity_strength}` |

So the worker does **not** group by subfolder, revision or quantization: it
refuses to accept runs that carry them, which makes each published cell
homogeneous by construction. A local record *does* carry them, and a run whose
cache resolved a revision or real quantization facts cannot currently be
uploaded at all — the worker rejects the payload, because `preview_run` sends
the record verbatim without stripping. That is a service-side limitation, not a
client one.

Matching is therefore asymmetric on purpose: a facet discriminates only when
**both** sides state it. Requiring the feed to confirm a revision it never
records would withhold every comparison on the live service; treating a facet
the feed *does* state as optional would merge variants. When more than one
published cell is compatible with a run's identity, the count is summed and the
median withheld.

### Fields read from `summary[]`

`task_type`, `model.repo_id`, `machine.{chip,memory_gib}`, `protocol.{id,version}`,
`case_id`, `execution` (the allowlisted projection), `metric.{name,median,unit}`,
`samples`, and `contributors[]`. The last one answers "is one of my runs already
in this aggregate?" from public data, so the **INCLUDES YOURS** badge survives a
relaunch, a reinstall, and a failed local receipt write. It is matched on the
canonical slug only.

Because the worker's group key includes `case_id` and `execution`, one model on
one Mac routinely has several cells. Two projections narrow them, and neither
ever picks a cell arbitrarily:

- `CommunityTableProjection` → at most **one row per model**: filter to the
  column's metric, keep the newest protocol version, keep the canonical case,
  and if execution variants remain, sum the counts and **withhold** the median.
- `CommunityCoverageProjection` → aggregate **before** applying the
  under-represented threshold. Thresholding individual cells and keeping a
  survivor reported a model with 4 + 5 samples as "Only 4 published results",
  a number belonging to no population.

## Context

The Community Benchmark redesign asks the Desktop client to answer questions
whose exactness the public feed cannot always guarantee:

- how many public results exist for this model, on this workload, on a Mac like
  mine (the observation count, median, and observed range),
- which models on this Mac profile still have no coverage,
- how many contributors, runs, and models the community has in total.

The decisive asymmetry: **a bounded feed can prove that results exist, but
never that none do.** A scope missing from `summary[]` might simply have fifty
newer runs in front of it. Treating that absence as zero is the dangerous
move — zero is not "unknown", it is the factual claim *nobody has published
this*, and it is exactly the input that turns on first-reference language.

## Decision

The client depends on a protocol, not an endpoint.

- `CommunityBenchmarkDirectory` is the seam. Four queries, all scoped by
  `CommunityBenchmarkScope` = **model alias + benchmark protocol/workload + Mac
  profile**, because a comparison or a "first" claim is only meaningful for one
  model, measured one way, on one class of machine. When a finished run is on
  screen the scope additionally carries that run's `comparison` identity —
  case, metric and execution configuration — because a comparison is only
  answerable against results produced the same way. The two read queries also
  take a `viewerSlug`, which is used for nothing except "is one of mine in
  here?".
- Every query returns `CommunityDataState<Value>`: `.loading`,
  `.unavailable(reason)`, or `.ready(value)`. The type makes
  "unknown means zero" inexpressible.
- `CommunityBenchmarkAPIDirectory` binds the seam to the three public routes.
  A scope **present** in `summary[]` yields a positive count flagged
  `isBounded`, which the UI renders as "at least N"; a scope **absent** yields
  `.unavailable(.boundedFeed)`, never zero.
  `UnavailableCommunityBenchmarkDirectory` remains the offline/unconfigured
  fallback. Running, inspecting, and publishing are unaffected either way.
- `CommunityContributionBranch.select(from:)` is the single place the
  contribution story is chosen, and it has three outcomes, not two:
  - `observationCount == 0` **and not bounded** → `.firstReference` (the only
    state permitted to use first-reference language). A bounded zero is
    incoherent and is rejected,
  - `observationCount < 0` → `.unknown(.unavailable(.invalidCount))`; a
    malformed count is a broken answer, not an empty one, and must never be
    clamped to zero,
  - `observationCount > 0` → `.strengthen` (the only state permitted to render a
    median, observed range, or difference-from-median),
  - loading / offline / failed / not configured → `.unknown`, which selects
    neither branch and withholds every comparison.

After a successful publication the displayed count is advanced by
`CommunityObservationSummary.incrementedAfterPublishing()`, which increments the
count and **drops** the median and range: the client holds no sample population,
so any locally derived median would be fabricated. A duplicate submission
(`already_exists`) does not move the count.

`CommunityPublicationState` owns that bookkeeping, and three rules live there:

- The publish **context** — run id, scope, pre-publish count, branch — is frozen
  before the first `await`. The upload is a CLI subprocess taking seconds, and
  Run again is one click away; a receipt is applied only to the scope it is
  about, and only painted when that scope is still on screen. That scope is
  built **entirely from the record** (`CommunityBenchmarkResult.communityScope`),
  so publishing a stored result from My Results is attributed to its own model,
  protocol version and Mac rather than to whatever the Run tab is showing.
- The server-issued pseudonym is adopted whether or not the CLI managed to save
  the receipt locally. It is the slug the contributions query is made with.
- `/atomic/public` is edge-cached for 30 s
  (`ATOMIC_BENCH_PUBLIC_CACHE_SECONDS`), so the refresh fired straight after
  publishing usually reads a body that predates the submission. The confirmed
  count is a **floor** for that scope: a lower read is treated as stale and the
  floor is kept, a read at or above it retires the floor, and a re-read is
  scheduled for after the cache can have expired. Floors are held **per scope**
  and are not cleared by navigation — a floor is a fact about a scope, and
  looking at another model is not evidence against it.
- That re-read covers every projection the cached body feeds —
  `publicFeedBackedReads` = observations, table, coverage, pulse — so a first
  contribution gains **INCLUDES YOURS** in the table without a relaunch or a
  workload toggle. `contributorTotals` is excluded: it comes from the
  cursor-paginated contributions route, which this cache does not serve.

## What the feed cannot answer

These are real gaps, and each one is preserved as an explicit unavailable state
rather than guessed:

| Missing field | Consequence today |
| --- | --- |
| An **exact** observation count for one model + workload + Mac profile | `summary[].samples` is a floor over the newest 50 runs. The UI says "at least N"; a scope absent from the feed is `unavailable`, so the first-reference branch can never fire from the live service. |
| `observed_minimum` / `observed_maximum` per summary cell | `summary[].metric` carries `median` and `best` only. `best` is one end of the range, so the observed range is omitted rather than half-drawn. |
| Models with **zero** observations on a Mac profile | Coverage gaps can only report under-represented pairings that appear in the feed. No "FIRST RESULT NEEDED" mission is derived from the live service, and an empty gap list is worded as "nothing in the recent published results looks thin" — never as "every catalogue model is covered", which is the one claim an empty list cannot support. |
| Exact global contributor / run / model totals | Pulse totals are floors and are rendered "at least N". |

A per-contributor avatar asset is **not** a gap: the portrait is derived from
the slug with the website's own mapping, so no new field is required. See
"Cross-platform identity".

A single server-side addition would close all four: an aggregate endpoint
returning, per (model, workload, protocol, Mac profile), the **complete**
observation count, median, observed minimum and maximum — plus the set of
catalogued models with a count of zero for a given profile.

Exact per-contributor totals ARE available: the contributions route paginates,
and the adapter follows `cursor` until `complete == true`, which is what the My
Results identity band reports instead of `receipts.count`.

## Cross-platform identity

The contributor identity is shared with rapidmlx.com and ported, not designed:

- `slug` comes from the API; the fallback is exactly `name + "-" + tag`, which
  is what `normalize()` in `landing/public/community-identity.js` composes.
- The portrait is `AVATAR_ASSETS[hash32(salt + "|" + slug) % 18]` with salt
  `rapid-mlx/leaderboard/avatar/v12`, FNV-1a over **UTF-16 code units** plus the
  MurmurHash3 avalanche finalizer — a byte-for-byte port of
  `deriveAvatar` in `landing/public/leaderboard-data.js`.
- The 18 plate PNG/WebP files are vendored unmodified; `scripts/build.sh` fails
  the build if any is missing rather than shipping an app whose avatars all
  collapse to the fallback.
- Pinned vectors (verified by running the website's own `deriveAvatar`):
  `swift-otter-4417` → 12, `modest-slate-wombat-545` → 02,
  `sleepy-alpine-okapi-e22` → 09.
- There is **no monogram fallback**. An earlier revision of this document said
  the receipt returns "an alias and nothing else" so the identity row draws a
  monogram; that was wrong on both counts. The receipt carries `{name, tag,
  slug, url}`, and the plate is a deterministic function of the slug, so the
  same contributor shows the same face in the app and on the leaderboard.
  Nothing is invented locally: with no slug, no portrait is drawn at all.

Changing the salt or the asset list re-deals every contributor, so it must
happen on the website first and be mirrored here in the same change.

## Consequences

- Against the live service the Community tab renders real aggregates, the Result
  screen compares against the cell produced with the same case, metric and
  execution configuration, and the Ready screen invites a run without claiming
  coverage either way. No screen implies the user would be first unless an
  exact, unbounded zero says so — which the public feed cannot produce.
- Offline or not configured, `UnavailableCommunityBenchmarkDirectory` reports
  every query unavailable and every community claim is withheld. Running,
  inspecting and publishing are unaffected.
- The model picker's "Needed by the community" group falls back to the benchmark
  catalogue's own `focus` flag, and its rows carry **no** coverage sentence,
  rather than asserting "no results yet" from missing data.
- A publication's context is frozen when Publish is pressed, so a receipt is
  applied only to the run it is about even if the user starts another
  measurement while the upload is in flight.

## Related client gaps

The benchmark emits no byte-level download phase — the download happens inside a
single `benchmark run` and reports nothing. The picker and Running screens
therefore show an indeterminate "Preparing model" state with no byte count, rate,
percentage, or ETA, and "Choose model" stays enabled for a model that merely has
not been downloaded. Exact progress, an independent cancel-download action, and a
disabled confirmation button are all blocked on new client/CLI events.
