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
rapid-mlx benchmark results [--json]
rapid-mlx benchmark inspect RUN_ID [--json]
```

`rapid-mlx bench` remains available as the legacy freeform/submission surface
during migration. New product code must use `benchmark` and its v1 atomic run
records rather than parse legacy submission JSON.

## Boundaries

- A run is private by default. The client writes a schema-valid JSON record to
  `~/.rapid-mlx/benchmarks/runs/` with directory mode `0700` and file mode
  `0600`. No network upload or share control is part of this execution path.
- Desktop stops its active inference server before benchmarking so two large
  model processes cannot compete for unified memory. An owner-scoped lifecycle
  reservation remains active until cancellation has terminated and reaped the
  benchmark CLI's entire process group, including its child server. Desktop's
  hidden supervisor flag keeps that server in the owned group; navigation away
  cancels the run, and a replacement view waits for the prior lease to drain.
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
  its IDs directly to the engine; legacy `rapid-mlx bench` keeps its historical
  decoded-text generator until that separate submission format is migrated.

## Product flow

```text
atomic catalog -> compatible model picker -> registered protocol preview
               -> local executor -> BenchmarkRun validation -> private archive
```

The Desktop page is deliberately a single column: model picker, automatically
derived protocol and download state, one local run/stop action, then recent
local results. There is no home-page campaign card and no button repeated on
every model card in the internal beta.

## Follow-up

Sharing is a separate post-result capability. It must show the exact privacy
allowlist, resolve model identity where possible, revalidate the record, and
require explicit user action. Server ingestion, public aggregation, campaign
prompts, rewards, and leaderboard growth mechanics do not belong in this PR.
