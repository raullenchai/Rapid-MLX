# Community benchmark wire contract v1

- **Status:** Accepted contract; producer and ingestion migration deferred
- **Date:** 2026-08-31
- **Architecture owner:** Atlas
- **Performance evidence owner:** Vector
- **Consumers:** Rapid Server/CLI, Rapid Desktop, benchmark service, website
- **Atomic contracts:** [`proto/model-runtime/v1/`](../../../proto/model-runtime/v1/)
- **Catalog layer:** [`proto/model-catalog/v1/`](../../../proto/model-catalog/v1/)
- **Benchmark composition:** [`proto/community-benchmark/v1/`](../../../proto/community-benchmark/v1/)
- **Parent decision:**
  [Model management and performance SSOT](2026-08-22-model-management-performance-ssot.md)

## Context

The production community benchmark schema identifies a model primarily by a
mutable Rapid-MLX alias and Hugging Face path, and identifies a machine by a
small hardware tuple. Runtime configuration records sampling and one speculative
decode axis. That was sufficient for the original fixed-model speed board but
cannot safely aggregate arbitrary models, distinguish immutable artifacts, or
compare MTP/KV-cache/prefill configurations for recommendations.

Server, CLI, Desktop, and website also need the same entity meanings. Hiding
machine and execution definitions inside an upload schema would encourage each
consumer to recreate partial types and drift.

## Decision

Rapid-MLX separates reusable atomic facts from product catalog and benchmark
composition:

1. `ModelIdentity` identifies the complete composed pipeline, immutable
   component revisions/manifests, quantization, and provenance strength. The
   same shape covers LLM, VLM, image, and video models.
2. `MachineObservation` separates stable, non-unique hardware profile facts from
   volatile before/after run conditions.
3. `ExecutionConfig` records effective values after defaults and runtime
   resolution, including acceleration and memory/cache choices.
4. `ModelAlias` is a second-layer product entry that references identities and
   evidence-scoped execution presets by digest.
5. `BenchmarkRun` composes the atomic contracts with a workload protocol and raw
   measurement samples.

The first three live under `proto/model-runtime/` because serve, model loading,
GUI, website, recommendations, and future model onboarding all consume them.
Aliases live under `proto/model-catalog/`. Only workload and measurement
semantics live under `proto/community-benchmark/`.

The shared wire format is strict JSON Schema 2020-12. Every object is a positive
allowlist with `additionalProperties: false`. Widening consent or meaning creates
a new protocol directory instead of mutating a shipped version.

Client payloads contain observations, not trust conclusions. Verification,
comparability, deduplication, anomaly classification, aggregates, and
recommendation eligibility are server-generated records outside this upload
contract.

## Identity and comparability

Model alias and display metadata are never grouping keys. An alias remains the
friendly CLI/GUI entry point, but is layered on a separately stored identity
rather than owning `hf_path` and runtime truth. Hugging Face sources
resolve tags or branches to immutable revisions. Local models use a privacy-safe
artifact manifest digest and remain separate from repository artifacts unless a
server-side registry verifies equivalence.

Machine identity means a shared configuration such as chip/core/RAM shape, not a
unique physical device. Volatile available RAM, thermal state, memory pressure,
and power state are captured separately and filter evidence quality.

Execution identity is task-discriminated and records the effective
configuration, not requested flags. Text/VLM include MTP/speculative, KV cache,
prefix cache and prefill behavior. Image/video include attention backend,
compilation, offload, VAE tiling/slicing, denoiser and temporal chunking.
Request choices such as tokens, dimensions, frames, steps and guidance remain
in workload. Runtime versions remain separate from the config digest.

Canonical atomic digest projections are normative in the
[model-runtime README](../../../proto/model-runtime/README.md); benchmark rules
are in the [protocol README](../../../proto/community-benchmark/README.md).

## Industry references

LM Studio's `model.yaml` separates a canonical `publisher/model` from concrete
GGUF and MLX variants and keeps load/inference configuration separate. Rapid-MLX
adopts that separation but additionally pins immutable revisions and manifests
for benchmark identity:
<https://github.com/lmstudio-ai/docs/blob/main/0_app/3_modelyaml/index.md>.

oMLX uploads an explicit allowlist of performance-relevant settings, uses stable
keys for acceleration features, excludes user-authored display/alias fields, and
reduces path-valued fields before upload. Rapid-MLX adopts the allowlist and
privacy posture while using a stronger artifact identity:
<https://github.com/jundot/omlx/blob/main/omlx/admin/benchmark.py>.

## Compatibility and rollout

This decision does not replace `community-benchmarks/schema.json` v1-v3 in the
same PR. Existing submissions, CLI uploads, aggregation, and the board continue
unchanged. Follow-up work will add collectors and an adapter, dual-read or
versioned ingestion, aggregate-v2 cohorting, and consumer types for Python,
Swift, and TypeScript.

The alias migration preserves aliases as a second layer and is specified in
[aliases to atomic contracts](../architecture/model-management/migrations/002-aliases-to-atomic-contracts.md).

## External design review

Three adversarial Kimi K3 review/revision rounds were applied before freezing
the contract:

1. The schema/privacy round found an impossible measurement capacity, incomplete
   KV/prefix-cache identity, and ambiguous incomplete samples. Capacity and cache
   conditionals were fixed; incomplete attempts became structured outcomes.
2. The recommendation round found that a random grouping ID alone could not
   support causal MTP/KV claims and that success-only uploads would bias model-fit
   recommendations. Explicit experiment arms/order/varied fields and closed
   failure outcomes were added, along with cross-language digest golden values.
3. The freeze round challenged unresolved identity, machine grouping, and schema
   evolution. Unresolved models remain intentionally exploratory, while artifact
   manifest bases, machine-profile completeness, the macOS comparison axis, and
   source-runtime revisions were made explicit.

Two suggestions were rejected deliberately: excluding every unresolved model
would undermine open campaign participation, and open-ended `x-*` failure codes
would bypass the versioned privacy allowlist. Neither can enter trusted
recommendation evidence under v1.

After generalizing the contract beyond LLMs, three additional Kimi K3 rounds
reviewed the atomic/catalog split and legacy migration:

1. Model identity became a manifest-pinned component graph and alias preset
   evidence stopped carrying free-form chip/memory selectors.
2. Unresolved aliases, deterministic defaults, moving upstream snapshots,
   cache preservation, dual-read failure rules, measurable phase gates, and the
   explicit legacy kill switch were made normative.
3. The freeze review found no remaining merge-blocking P0/P1 in the layering,
   four-modality coverage, reference integrity, or rollback design.

## Consequences

- Arbitrary compatible models can participate without merging same-name but
  different artifacts.
- Focus Models are registry-pinned artifact sets, not mutable aliases.
- The same evidence can power machine-aware model-fit recommendations and
  workload-aware runtime recommendations.
- Strict versioning makes evolution more explicit, but prevents silent privacy
  and comparability drift.
- JSON Schema cannot express all relational rules, so ingestion needs a semantic
  validator in addition to structural validation.
