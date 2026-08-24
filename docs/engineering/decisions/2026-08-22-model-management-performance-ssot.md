# Model management and performance decision SSOT

- **Status:** Accepted direction; incremental rollout
- **Date:** 2026-08-22
- **Architecture owner:** Atlas
- **Performance evidence owner:** Vector
- **Consumers:** Rapid Desktop, Rapid Server, CLI, benchmark tooling
- **Related evidence:**
  [long-context service prefill](../performance/2026-08-22-long-context-service-prefill.md),
  [Mac mini model matrix](../performance/2026-08-21-mac-mini-model-matrix.md)

## Context

Rapid-MLX already has useful per-model knowledge: stable aliases, model metadata,
compatibility gates, parser and sampling defaults, speculative-decoding tiers,
and benchmark-verified prefill recommendations. That knowledge currently lives
mostly in a sparse `ModelProfile` backed by `aliases.json`, with resolution spread
across several runtime entry points.

This has delivered real performance gains. For example, measured aliases can
select a smaller `prefill_step_size`, and an untouched Desktop launch benefits
because it starts the same CLI/server path. However, the current shape cannot yet
represent the complete decision we need:

> model artifact × quantization × machine class × workload × runtime version

It also does not carry enough provenance for the GUI to explain which values were
selected, why they are safe, or which benchmark evidence supports them. Community
benchmarks will make this riskier unless untrusted observations are separated from
production defaults by validation and promotion gates.

## Decision

Rapid-MLX will evolve toward one model-management and runtime-decision SSOT with
five explicit stages:

1. Immutable model and artifact identity records facts.
2. Validated performance and quality evidence determines eligibility.
3. Product policy produces an ordered, explainable recommendation set.
4. A single runtime resolver produces one `EffectiveRuntimeConfig`.
5. Consented runtime telemetry re-enters the evidence pipeline; it never edits a
   production profile directly.

Selection chooses **which model variant** to run. It does not select a performance
profile. The runtime resolver chooses **how that variant runs** for the current
machine and workload. Explicit user runtime flags take precedence over profile
recommendations, subject to hard compatibility and safety constraints.

## Living architecture

The accepted relationships, current implementation status, migration work, and
GitHub-rendered diagrams live in the
[model-management architecture workspace](../architecture/model-management/README.md).
Its YAML manifest is the machine-readable source of truth. This ADR deliberately
does not duplicate that changing project state.

## Entity contracts

| Entity | Owns | Must not own |
| --- | --- | --- |
| `ModelVariant` / `Artifact` | Stable identity, quantization/modality/format, immutable artifact revision and manifest | Product ranking or mutable performance claims |
| `Capabilities` | What the artifact can do | Quality judgement or defaults |
| `RuntimeCompatibility` | Supported runtime ranges and hard feature/safety constraints | Product preference |
| `VariantQualification` | Versioned quality-gate result for an artifact revision and runtime/eval range | Performance tuning |
| `MachineFingerprint` | Exact benchmark/runtime observation: SoC, OS, physical and available RAM, thermal/power state | Reusable default identity |
| `MachineClass` | Normalized hardware target such as `m3-max-64` | Volatile available RAM or thermal state |
| `WorkloadProfile` | Structured applicability: modality, context bucket, concurrency, latency objective, cache state, resolution/token mix | Free-form benchmark descriptions |
| `PerformanceProfile` | Promoted runtime knobs for artifact/quantization, machine class, workload, and runtime range | Model selection or unvalidated observations |
| `RankedRecommendationSet` | Ordered model candidates with role, reasons, tradeoffs, limitations, policy version | Runtime knob resolution |
| `EffectiveRuntimeConfig` | Final values plus field-level source, source ID, reason, override and fallback trace | New policy decisions in GUI or Server |

Physical RAM is used for model-fit classification. Available RAM is a runtime
safety input and may force a fallback; it must not change the machine's stable
class.

## Delivery model

Implementation is intentionally incremental. The living architecture owns phase,
status, evidence, code/test links, tracking issues, and migration documents. A
component is only `implemented` when its contract, production path, proportional
tests, and operational ownership exist; a schema or document alone does not count.

## Invariants

1. User model-selection overrides and runtime-config overrides are separate.
2. Explicit runtime flags win unless they violate a hard safety or compatibility
   constraint; the resulting adjustment is visible in provenance.
3. Selection returns a `SelectedModelVariant`, not a profile candidate.
4. GUI, Server and CLI consume the same effective config and do not reimplement
   recommendation policy.
5. A benchmark claim is scoped to artifact revision, quantization, machine,
   workload, runtime stack and protocol version.
6. Community submissions and runtime telemetry cannot directly write a production
   performance profile.
7. Fallbacks remain qualified recommendations and preserve the user's intent as
   far as safety permits.
8. Every automatic default has a rollback path.

## Alternatives considered

### Keep adding fields to `aliases.json`

This is cheap and remains suitable during Phase 0, but it mixes facts, quality,
policy and performance evidence. It cannot safely express machine/workload scope
or evidence promotion, so it is a migration source rather than the final schema.

### Let Desktop own model recommendations and tuning

Rejected. Server and CLI would drift, non-technical users would see inconsistent
behavior, and every backend optimization would require a second implementation.
Desktop should render decisions and submit overrides, not make policy.

### Apply community benchmark winners directly

Rejected. Hardware state, thermal throttling, runtime revisions, correctness
failures, duplicated samples and malicious submissions all make an unvalidated
winner unsafe as a product default.

### Build the complete schema in one migration

Rejected. The existing sparse profile is production-critical and broad. The
phased approach gives each extracted entity an equivalence test and rollback path.

## Consequences

- The immediate cost is additional schema/versioning and a migration adapter.
- Runtime behavior becomes explainable and reproducible across GUI, Server and
  CLI.
- Performance defaults can become machine- and workload-specific without turning
  the GUI into a tuning panel.
- Product quality and runtime compatibility can block unsafe recommendations
  independently of raw speed.
- Community benchmarks become useful evidence rather than an unaudited source of
  truth.

## Updating this decision

Each implementation PR that advances this design must update the living manifest,
regenerate its views, and link code/tests or evidence. Change this ADR only when
clarifying the original decision. If an invariant changes or this design is
reversed, create a superseding ADR rather than rewriting history.
