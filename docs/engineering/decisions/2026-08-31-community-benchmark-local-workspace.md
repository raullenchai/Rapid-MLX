# Community Benchmark is a local model workspace before it is a campaign

Date: 2026-08-31
Status: accepted for internal beta

## Decision

Rapid-MLX exposes Community Benchmark as one model-first workspace in CLI and
Desktop. The user chooses a model; the model catalog's atomic task and
operation capabilities select a registered workload. Image, video, and LLM are
metadata on the selected model, not separate navigation tabs.

The initial commands are:

```text
rapid-mlx benchmark catalog [--memory-gib N] [--json]
rapid-mlx benchmark plan MODEL [--json]
rapid-mlx benchmark run MODEL [--json]
rapid-mlx benchmark results [--limit N] [--json]
rapid-mlx benchmark inspect RUN_ID [--json]
rapid-mlx benchmark share RUN_ID
```

`rapid-mlx bench` remains available as the legacy freeform/submission surface
during migration. New product code must use `benchmark` and its v1 atomic run
records rather than parse legacy submission JSON.

## Boundaries

- A run is private by default. The client writes a schema-valid JSON record to
  `~/.rapid-mlx/benchmarks/runs/` with directory mode `0700` and file mode
  `0600`. Running a benchmark never uploads it. A separate share action shows
  the complete wire payload and destination, including the candidate random
  resettable install ID used for abuse control, and requires explicit consent.
  Previewing writes nothing. The subsequent upload is pinned to that candidate
  and the preview's canonical payload digest, exact serialized HTTP-body
  digest, and destination. Desktop displays that exact body string without
  parsing and reformatting it. Upload aborts if another process established a
  different install ID, the archived run or serialization changed, or endpoint
  configuration changed while the consent sheet was open; it never silently
  sends bytes or sends to a destination different from what was approved.
  The JSON has no IP-address field. The HTTPS service necessarily observes the
  source IP and uses it for short-lived request limiting, but the application
  does not retain it in the benchmark record.
- Desktop stops its active inference server before benchmarking so two large
  model processes cannot compete for unified memory. An owner-scoped lifecycle
  reservation remains active until cancellation has terminated and reaped the
  benchmark CLI's entire process group, including its child server. Desktop's
  hidden supervisor flag keeps that server in the owned group; navigation away
  cancels the run, and a replacement view waits for the prior lease to drain.
  A cancelled queued view is removed immediately rather than acquiring later.
  Foreground teardown is bounded after SIGKILL. If a process group remains,
  the foreground lease is atomically replaced by a ServerManager quarantine
  lease; UI cancellation can finish, but model launches remain excluded until
  a background monitor confirms the process group is gone.
  The same quarantine rule applies before a run: if the embedded inference
  server survives its TERM/KILL grace, Desktop aborts the benchmark and keeps
  launches excluded until that prior server group is confirmed gone.
- A completed run always contains its atomic `MachineObservation`. If the
  privacy-allowlisted machine probe itself fails, the local failed attempt uses
  `machine_probe_failed` and omits `machine` rather than fabricating identity;
  this exception is permitted only for non-completed outcomes.
- Model aliases are presentation/routing metadata, not model identity. The v1
  local run stores an unresolved `ModelIdentity` until the registry can pin a
  revision and artifact manifest. Such a result is useful local evidence but
  is explicitly not marked formally comparable.
- The installed package carries exact copies of the benchmark schema,
  protocols, and datasets. Tests compare them byte-for-byte with `proto/`.
- Only real registered workload compatibility is shown. The video v1 workload
  is currently Wan-only and locks `RAPID_MLX_WAN_STEPS=20` in the child
  process. LTX and CogVideoX need separate registered protocols. Image models
  must advertise `text_to_image`; VLM and audio remain excluded until they
  have their own registered workload.
- Registered language rows bind observed prompt/output token counts exactly to
  the case targets. A tokenizer round-trip that turns `pp512` into 510 tokens
  is archived as a failed attempt, never admitted as comparable `pp512` data.
  The atomic runner uses the dataset's xorshift32 golden algorithm and passes
  its IDs directly to the engine. The registered eligible-ID list explicitly
  excludes `tokenizer.all_special_ids`; a tokenizer that cannot provide that
  evidence cannot claim the v2 registered protocol. The original dataset and
  protocol v1 files remain immutable and accepted by archive validation, so
  earlier local rows stay visible; new runs select v2. Legacy `rapid-mlx bench`
  keeps its historical decoded-text generator until its separate submission
  format is migrated.
- Peak memory uses the status endpoint's decimal-GB definition converted to
  MiB (`GB × 1e9 / 2^20`). Missing metrics remain JSON `null`; zero is never
  used as a stand-in for unavailable evidence.
- Completed video jobs expose their generated `size`, `frames`, and `fps` as
  response metadata. The registered runner rejects a result whose metadata
  differs from the protocol workload, then downloads the MP4 into a private
  temporary file and probes its actual dimensions, frame count, and frame
  rate. Missing, empty, corrupt, oversized, deadline-exceeding, or shape-drifted
  artifacts fail closed. Every non-active job status is terminal instead of
  waiting until timeout.
- Image results are decoded before measurement so their actual dimensions and
  count must match the registered workload. Observed zero token counts remain
  zero and fail comparability validation instead of being replaced by targets.
- Catalog, planning, archive, and inspect failures remain structured CLI
  errors. The CLI states whether a privacy-safe failed run was actually saved;
  it never claims local persistence after a disk or permission error.
- Execution fields are evidence, not defaults: unobserved diffusion, temporal
  chunking, KV-cache, context-length, and offload details remain unknown/null
  or omitted. A later runtime-observation layer can strengthen them without
  changing the atomic shape or fabricating present-day configuration.

## Product flow

```text
atomic catalog -> compatible model picker -> registered protocol preview
               -> local executor -> BenchmarkRun validation -> private archive
```

The Desktop page is deliberately a single column: model picker, automatically
derived protocol and download state, one local run/stop action, then recent
local results. There is no home-page campaign card and no button repeated on
every model card in the internal beta.

If CLI metadata loading is unavailable, Desktop's legacy fallback is
conservative: chat remains usable, while image/video rows without atomic
capability evidence are hidden. Atomic video rows still require a registered
Wan alias, matching the CLI planner rather than advertising LTX/CogVideoX.

## Follow-up

The internal-beta share action revalidates a run, POSTs it to the separate
atomic ingestion endpoint, and saves a schema-valid acceptance receipt under
`~/.rapid-mlx/benchmarks/receipts/`. Retries are idempotent by run ID and exact
run digest. The client verifies that the returned receipt digest identifies the
exact uploaded payload before marking a result shared. Atomic rows remain
outside the legacy public aggregation path.

The receipt's server-assigned contributor `name` and `tag` close the feedback
loop after consent: CLI prints the public identity and contributor-page URL;
Desktop presents the identity immediately after upload and keeps the page link
on the local result. Clients construct the website route from those two atomic
fields rather than adding a presentation URL to the strict receipt schema.

Model-identity resolution, public aggregation, campaign prompts, rewards, and
leaderboard growth mechanics remain later work. The ingestion boundary accepts
the current unresolved identities deliberately; it never upgrades them to
formally comparable evidence.
