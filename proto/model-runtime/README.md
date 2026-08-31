# Model runtime atomic contracts

These JSON Schemas are product-neutral building blocks shared by model loading,
CLI, server, Desktop, website, recommendations, telemetry, and benchmarks.
They are not owned by the community benchmark feature.

- `ModelIdentity` identifies the complete loaded pipeline as an ordered set of
  immutable components. It covers LLM, VLM, image, video, speech synthesis,
  and speech recognition pipelines plus optional adapters, vocoders, and
  auxiliary encoders/decoders.
- `MachineObservation` separates stable machine-class identity from the OS and
  volatile run conditions. It never contains a device identifier.
- `ExecutionConfig` records the effective post-resolution configuration. Its
  task-discriminated union covers text, VLM, image, video, speech synthesis,
  and speech recognition execution.

Atomic means each object can be stored, validated, hashed, and reused without a
benchmark run. A consumer composes these objects rather than creating a local
variant. New model support normally adds registry data; new optimization
support normally adds an execution variant. Change this schema only when the
system needs a genuinely new atomic fact.

Once shipped, a version's accepted meaning and privacy boundary are immutable.
Add a new version directory for additive fields because every object uses
`additionalProperties: false`.

## Canonical digests

Digest projections use Rapid Canonical JSON v1 (RCJ-1): sorted ASCII object
keys, NFC strings, UTF-8, shortest required escapes, no insignificant
whitespace, safe integers only, and no JSON floating-point numbers. Fractional
configuration values use named scaled integers.

- Model identity: `{schema_version, pipeline_kind, components}`. Components are sorted by
  `component_id`; IDs are unique and exactly one component has role `primary`.
- Machine identity: `machine.profile` only.
- Execution identity: `{task_type, resources, task}`. Runtime versions are
  separate compatibility/cohort axes.

Hash canonical bytes with SHA-256 and prefix lowercase hexadecimal with
`sha256:`. The examples under `v1/examples/` are normative golden vectors.

`identity_strength: unresolved` supports arbitrary compatible models without
pretending they are formally comparable. It has no identity digest. Registered,
repository-revision, and local-manifest identities require a digest that a
consumer recomputes.

No atomic object admits local paths, hostnames, usernames, serial numbers,
hardware UUIDs, environment variables, prompts, or generated content.
