# Migrate `aliases.json` onto atomic model contracts

Status: Phase 1 shadow materialization implemented; legacy runtime resolution
remains authoritative.

Owner: Atlas. Vector owns benchmark evidence thresholds; Pixel/Harbor consume
the catalog through generated artifacts and must not create parallel profiles.

## Goal

Preserve the alias as the friendly product entry point while making it a second
layer built on shared atomic facts:

```text
ModelIdentity ───────┐
MachineObservation ─┼─> benchmark, serve, GUI, website, recommendations
ExecutionConfig ────┘

ModelAlias -> ModelIdentity digest
           -> ExecutionConfig preset digests + machine/workload applicability
           -> presentation and route capabilities
```

Today `vllm_mlx/aliases.json` is useful but mixes mutable names, repository
coordinates, route capabilities, inferred architecture, runtime defaults,
benchmark results, and product recommendations. A benchmark cannot treat that
whole record as identity: changing a display label or recommendation would make
the same weights look like a different model, while a moving Hugging Face
branch can leave the alias unchanged but change the weights.

## Target records

1. A model registry stores schema-valid `ModelIdentity` snapshots. A snapshot
   pins every component's resolved revision and manifest. Multi-file pipelines
   may contain a primary denoiser/language model, VAE, text/vision encoder,
   projector, adapters, ControlNet, tokenizer/processor, or other component.
2. An execution registry stores complete effective `ExecutionConfig` snapshots.
   These are reusable by serve, Desktop, benchmark, and recommendation code.
3. `ModelAlias` references one model identity and optional execution presets by
   digest. It owns presentation, exposed task routes, parser/template choices,
   and evidence references—not artifact identity. Candidate/promoted presets
   bind an atomic machine-profile digest and workload-protocol digest; memory
   ranges and chip-family matching stay in recommendation policy.
4. Recommendation policy ranks registered execution presets for a model,
   machine class, workload, runtime range, and correctness gate. Raw benchmark
   numbers never write a default directly.

## Legacy field mapping

| Current alias field | Target | Rule |
| --- | --- | --- |
| alias key | `ModelAlias.alias` | Stable product name; excluded from identity. |
| `hf_path`, `subfolder` | registry resolver input | Resolve to component source + immutable revision; do not retain as an alias-owned identity fact. |
| `modality`, `video_modes`, `is_text_only` | `ModelAlias.capabilities` | Normalize to task types/modes; validate against the resolved pipeline. |
| tool/reasoning parser, chat template | alias capabilities | Product routing/rendering metadata. |
| architecture/MoE/hybrid flags | discovered registry metadata | Verify from pinned config; capability gates may reference it but alias text is not authoritative. |
| spec/MTP/DFlash/DDTree/PFlash/TurboQuant flags and draft models | execution preset + compatibility evidence | Materialize a complete config; draft models are their own `ModelIdentity`. |
| prefill, speculative token, cache and sampling defaults | execution preset or workload | Effective runtime optimization goes to execution; request sampling belongs to a workload/product request policy. |
| minimum memory and speedup fields | recommendation evidence | Scope by model, machine, workload, runtime and correctness; never an identity field. |
| experimental/display-only fields | alias presentation/capabilities | No effect on artifact comparison. |

No value is silently guessed during conversion. Missing or conflicting facts
produce an explicit diagnostic and either an unresolved identity or no promoted
preset.

## Compatibility adapter

The adapter accepts both legacy string entries and rich objects through today's
`model_aliases` coercion, then emits a `LegacyAliasMaterialization`:

- normalized alias presentation and capabilities;
- `repo_id`/`subfolder` as resolver inputs;
- after download, the immutable snapshot revision, component inventory,
  manifests, quantization, and resulting `ModelIdentity` digest;
- effective runtime defaults as complete `ExecutionConfig` candidates;
- provenance for each candidate field (`legacy_explicit`, runtime default,
  discovered config, or benchmark-promoted).

A branch/tag or bare repo ID is never `repository_revision`. Before resolution
the adapter emits `identity_strength: unresolved` and no identity digest. Local
models become `local_manifest` only after content manifests are built. Alias
strings and absolute cache paths never enter a digest.

The catalog resolver treats `(registry_model_id, model_identity_digest)` as one
checked reference: the ID must resolve to that exact digest or loading fails
closed. Preset IDs are unique within an alias. The target pipeline kind must be
advertised; additional task types require an explicit registered execution
adapter for that identity.

The Rapid-owned `registry_model_id` is not an HF repo ID and never case-folds a
case-sensitive upstream coordinate. A moving upstream ref creates a new
immutable identity snapshot; publication atomically updates the alias target
only after manifest verification. Existing digests remain addressable until no
shipped alias or rollback window references them. Withdrawn digests are denied
by registry policy, and catalog generation fails if any alias references one.

`default_execution_preset_id` is the only compatibility default and must name
one unique preset (or be null with an empty preset list). Recommendation policy
may automatically select only `promoted` evidence; candidate and legacy status
remain visible but cannot claim a recommendation.

## Staged rollout

### Phase 0 — contracts (this PR)

Add atomic schemas, catalog schema, multimodal examples, contract tests, and
this plan. Keep the current loader, CLI, server, Desktop, and uploads unchanged.

### Phase 1 — shadow materialization

Add an adapter under `model_catalog_mode=shadow`. The legacy resolver stays
authoritative. At resolution/load time, write only structured diagnostics and a
local sidecar cache containing the new records; never upload it implicitly.
Compare legacy and new route, source/subfolder, parser, capability, and effective
default decisions. Redact paths and secrets.

Exit criteria: 100% of checked-in aliases materialize or have an owner/expiry
diagnostic; zero routing/default mismatches across the alias fixture matrix;
all model-runtime digests reproduce across Python, Swift, and TypeScript; and
cold/warm loads reuse the same existing HF cache objects without redownload.

Implemented foundation:

- `vllm_mlx.catalog.legacy` projects the text/VLM/image/video and audio alias
  registries plus RAM-tier recommendations into one deterministic graph;
- `rapid-mlx models --json` retains all legacy buckets and adds the graph under
  `atomic`, including a local deterministic equivalence report;
- Rapid Desktop prefers atomic task/operation capabilities for Chat, Images,
  Video, TTS, and STT placement, then falls back to the legacy parser for an
  older sidecar;
- the content-addressed store, RCJ-1 digest implementation, packaged validators,
  and schema-drift checks are reusable by Server, GUI tooling, and a future
  website build.

The current projection is deliberately unresolved: legacy repo IDs are resolver
inputs, not immutable model identities. It also centralizes existing image and
audio capability inference in the adapter; Phase 2 must replace those bridge
rules with generated explicit alias capabilities before authoritative cutover.
Likewise, Desktop's RAM recommendation reader remains on the legacy file while
the atomic policy is shadow-validated. No automatic model choice changes in
Phase 1.

### Phase 2 — generated catalog, dual read

Check in or deterministically generate model identities, aliases, and candidate
execution presets. Add equivalence tests for every current alias. Consumers may
read the new catalog but fall back to legacy data on missing records. The old
file remains authoritative and release-compatible.

`model_catalog_mode` has four closed values: `legacy`, `shadow`, `dual_read`,
and `authoritative`. In dual-read, only an absent record falls back; digest
mismatch, invalid schema, withdrawn digest, or divergent materialization fails
closed and increments a release-blocking diagnostic. Authoritative mode never
falls back implicitly; operators use the `legacy` kill switch.

Exit criteria: alias enumeration, `resolve_alias`, modality routing, parser and
template selection, download repo/subfolder, feature gates, and defaults are
observably equivalent. Image/video/VLM component discovery has fixture coverage.
Fallback rate must be zero for checked-in aliases over the release-candidate
test matrix before the authoritative flip.

### Phase 3 — atomic registry authoritative

Switch `model_catalog_mode` to `authoritative` so normal model loading resolves
an alias to a registry identity and execution preset. Keep the legacy adapter
available only through the explicit `legacy` kill switch; authoritative reads
never fall back per record. `aliases.json` becomes a generated compatibility
view rather than an independent source of truth.

Cache lookup remains based on the existing case-sensitive repo ID, immutable
revision, and subfolder; manifest/identity digests index metadata but never
rename cached blobs. Alias presentation or recommendation changes must cause
zero model redownloads. New catalog-only aliases must also appear in the
generated compatibility view throughout the rollback window.

### Phase 4 — evidence-driven recommendations

Move memory floors, optimization tiers, and measured speedups into scoped
recommendation evidence. Promotion requires registered workloads, resolved
identity, complete machine profile, supported runtime, enough independent
samples, and correctness thresholds. Deprecate legacy performance fields only
after at least one stable release reads both forms.

## Rollback

Until Phase 4 completes, every release retains the unchanged legacy resolver
and its input. A single documented config switch returns all reads to legacy;
registry generation is additive and no migration rewrites user caches. Server
ingestion continues accepting existing benchmark schema v1-v3 during its own
version window. Rollback never changes a digest's meaning: bad registry records
are withdrawn by reference and replaced with new snapshots.
The kill switch is `model_catalog_mode=legacy`; it is runtime configuration and
is never serialized into an atomic record. Rollback drills must pass for a
missing registry, corrupt record, withdrawn digest, warm cache, and a model
introduced after Phase 3 before each release advances phases.

## Required equivalence and safety tests

- every legacy alias resolves to the same repo/subfolder and visible routes;
- string and object legacy entries materialize deterministically;
- moving revisions remain unresolved until pinned;
- all composed component IDs are unique and exactly one is primary;
- local paths, environment values, tokens, serials, and usernames cannot enter
  an atomic/catalog record;
- a presentation or recommendation edit does not change model identity;
- a component revision, manifest, quantization, role, or subfolder does;
- execution defaults reproduce the effective runtime, not merely requested CLI
  flags;
- shadow mismatch telemetry is local/redacted and cannot alter routing;
- fallback and rollback work with a deliberately corrupt/missing registry.
