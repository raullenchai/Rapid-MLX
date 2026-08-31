# Model catalog contracts

The catalog is a product layer built on the atomic contracts in
`proto/model-runtime/`. `ModelAlias` is a friendly CLI/GUI entry point, never a
model identity.

An alias references a separately stored immutable `ModelIdentity` snapshot and
zero or more full `ExecutionConfig` snapshots by digest. Capabilities describe
which product routes to expose. Evidence references bind a candidate/promoted
preset to an atomic machine profile and registered workload. Mutable presentation text stays here and is
excluded from every comparison key.

The target and preset records are deliberately references instead of embedded
copies. This prevents `hf_path`, modality, MTP defaults, and memory advice from
drifting into competing sources of truth.

The v1 product graph has four records:

- `ModelRegistryRecord` gives a Rapid-owned stable ID a resolver input and,
  after resolution, an immutable `ModelIdentity` digest. Download size is an
  advisory catalog fact, never identity.
- `ModelAlias` is the mutable product layer: display entry point, task and
  operation capabilities, runtime adapter, surface availability, and execution
  preset references.
- `RecommendationPolicy` ranks aliases for one task and machine dimension. Its
  legacy evidence states are intentionally not promotion states.
- `CatalogSnapshot` publishes an ordered, content-digested graph of registry
  records, aliases, and recommendation-policy digest references.

Product surfaces project from `task_types` and `operation_modes`. A GUI tab is
not an identity field, and consumers must not infer it from an alias or repo
name. One model may consequently appear in more than one operation picker.
Current task types cover LLM, VLM, image, video, TTS, STT, and forced alignment.

`scripts/sync_model_catalog_schemas.py` copies the schemas into the Python
wheel. CI runs it with `--check`; consumers validate the packaged copies so an
installed sidecar does not depend on a source checkout.

Semantic validation resolves `registry_model_id` to exactly the accompanying
identity digest, requires unique preset IDs, and requires the target pipeline
kind among advertised task types. Extra task types are allowed only when a
registered execution adapter proves that the same pipeline supports them.
`registry_model_id` is a Rapid-owned lowercase stable ID, not a normalized
Hugging Face repo ID; case-sensitive upstream coordinates live only in the
referenced `ModelIdentity`. An unresolved target has no digest and remains a
legacy/exploratory entry until atomically repointed to a resolved snapshot.

`default_execution_preset_id` makes compatibility behavior deterministic. It
does not mean “recommended”: automatic machine/workload recommendations may
select only evidence records with `status: promoted`.
