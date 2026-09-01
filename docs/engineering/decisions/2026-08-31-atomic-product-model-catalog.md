# Atomic product model catalog and shadow migration

Status: accepted for shadow/dual-read implementation

Date: 2026-08-31

Owner: Atlas, with Pixel responsible for Desktop projections and Vector for
recommendation evidence.

## Decision

Rapid-MLX will represent model support as a content-digested product graph built
on the atomic model, machine, and execution contracts. The graph is shared by
CLI/Server, Desktop, website tooling, recommendations, and community benchmark
work. It is not a benchmark-specific database.

The base graph separates four concerns:

1. `ModelRegistryRecord`: stable Rapid model ID and upstream resolver input,
   later bound to a verified immutable `ModelIdentity`.
2. `ModelAlias`: mutable product entry point with explicit tasks, operations,
   runtime adapter, availability, and execution preset references.
3. `RecommendationPolicy`: machine-scoped, task-scoped ranked aliases with
   reason IDs, limitations, metrics, and evidence state.
4. `CatalogSnapshot`: ordered records and policy references protected by one
   RCJ-1 content digest.

GUI tabs and operation pickers are projections of explicit capabilities. LLM,
VLM, image, video, TTS, STT, and alignment use the same graph; adding a modality
does not justify another registry or parser. A tab is presentation and is not
part of model identity. Recommendation is also a policy over catalog IDs, not a
field copied into each model.

## Compatibility boundary

The first rollout is additive. `rapid-mlx models --json` keeps the existing
`text`, `image`, `video`, and `audio` arrays and adds an `atomic` envelope.
Desktop prefers the envelope and falls back when it is absent. Runtime alias
resolution and the existing RAM default stay authoritative during shadow mode.

Legacy text/media and audio registries are projected read-only. Shared HF
coordinates de-duplicate into one unresolved registry record; no moving branch
is mislabeled as immutable identity. The shadow report verifies alias coverage,
recommendation references, task coverage, and the catalog digest. It contains
no prompts, user paths, machine serials, or uploaded data.

## Consequences

- New model support can declare stable routing once and appear consistently in
  Server, Desktop, website, and benchmark surfaces.
- Model identity, machine observation, execution configuration, product alias,
  and recommendation policy change independently and receive independent
  digests.
- Existing aliases and caches do not migrate or redownload during shadow mode.
- The temporary adapter still translates legacy capability conventions. That
  inference must disappear into generated explicit records before the atomic
  catalog becomes authoritative.
- Memory fit is only one recommendation axis. Later policies can add workload,
  quality, runtime, and evidence gates without changing model identity or UI
  tab taxonomy.

## Rejected alternatives

Extending each legacy alias registry independently would retain conflicting
parsers and force every consumer to repeat modality and recommendation logic.
Making GUI tabs the source of truth would couple product navigation to runtime
identity. Treating an unpinned Hugging Face path as an immutable identity would
make community results appear comparable when the underlying bytes can differ.

## Next cutover gate

Phase 2 may generate the catalog as a checked artifact and dual-read it only
after every checked-in alias has explicit capabilities, resolved records have
verified component manifests, execution presets reproduce actual effective
configuration, and Python/Swift/TypeScript digest golden vectors agree.
