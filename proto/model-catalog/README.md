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
