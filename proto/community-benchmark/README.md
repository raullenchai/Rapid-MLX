# Community benchmark protocol

Community benchmark is one consumer of the product-neutral atomic contracts:

- [`../model-runtime/v1/`](../model-runtime/v1/) supplies model, machine, and
  execution objects used by normal product flows as well as benchmarks.
- [`../model-catalog/v1/`](../model-catalog/v1/) defines the alias/recommendation
  layer built on those objects.
- [`v1/benchmark-run.schema.json`](v1/benchmark-run.schema.json) adds only a
  workload, raw measurements, outcome, collector, and run envelope.

The run union supports `text_generation`, `vision_language`,
`image_generation`, and `video_generation`. Image and video are first-class,
not LLM records with optional media fields. Generation request choices such as
resolution, frames, steps, guidance, prompt/output length, and seed live in the
workload. Runtime choices such as MTP, KV cache, attention backend, VAE tiling,
offload, and temporal chunking live in `ExecutionConfig`.

The production `community-benchmarks/schema.json` v1-v3 remains unchanged until
a later rollout PR adds an adapter and switches producers/ingestion.

## Registered protocols and datasets

For `protocol_strength: registered`, ingestion looks up `(protocol_id,
protocol_version)` under `v1/protocols/` and requires exact RCJ-1 equality after
removing no fields. It recomputes `protocol_digest` from the workload excluding
only that field and verifies the dataset digest against `v1/datasets/`. Custom
workloads are exploratory and never enter formal comparison or recommendation
evidence.

Semantic validation also requires:

1. model, machine, execution, workload, and dataset digests are recomputed;
2. model/execution/workload task types agree;
3. component IDs and `(case_id, round_index)` pairs are unique;
4. every measurement belongs to a case and each case has its declared rounds;
5. actual dimensions, frames, image counts, and token counts meet protocol
   tolerances;
6. completed timing phases fit inside total duration within timer tolerance;
7. timestamps are ordered and run duration is bounded;
8. server-side correctness, duplicate, anomaly, identity, and trust checks run
   before aggregation.

Clients upload raw timing and memory observations, never TPS, frames/sec,
rank, `verified`, `comparable`, or `outlier`. Prompt text, output text, image or
video bytes, paths, exception strings, and environment values are outside the
allowlist. Datasets use public IDs/digests; content is not repeated in uploads.

## Comparison and recommendation

The server derives:

```text
comparison cohort = model identity + pipeline/task type + machine profile
                  + OS + runtime stack + execution config + workload protocol

model-fit evidence:
  model × machine profile × workload -> success/OOM/correctness/performance

runtime-profile evidence:
  model × machine profile × workload -> ranked execution configs
```

Only complete machine profiles, resolved model identities, registered
protocols, compatible runtime stacks, and correctness-passing runs support a
promoted recommendation. Failed/OOM runs are censored fit evidence, not
zero-speed samples.
