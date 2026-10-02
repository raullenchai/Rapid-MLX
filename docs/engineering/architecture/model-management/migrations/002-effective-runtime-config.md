# Migration 002: central EffectiveRuntimeConfig

- **Status:** complete
- **Owner:** Atlas
- **Rollback:** preserve existing CLI/server resolver helpers behind an adapter

## Goal

Create one immutable output consumed by CLI, Server, and Desktop. Each resolved
field records `value`, `source`, `source_id`, `reason_code`, and whether it
overrode or adjusted another source.

## Sequence

1. **Complete:** define the internal schema and precedence tests without changing
   behavior. `rapid_mlx.runtime.effective_config` now owns immutable resolved
   fields, typed override/constraint inputs, field-level source IDs and reason
   codes, and the complete override/fallback trace. Unknown, missing, duplicate,
   or incompatible inputs fail closed.
2. **Complete:** wrap existing resolver helpers and compare old/new output. The
   rollback adapter rejects any mismatch and the fixture matrix covers real
   model-profile prefill, machine-memory overrides, reasoning workloads,
   explicit flags, and incompatible KV-cache options without downloading a
   model.
3. **Complete:** route Server and CLI through the central resolver after an
   exact legacy-result parity assertion. Programmatic Server calls use the same
   boundary and conservatively identify non-default inputs as explicit.
4. **Complete:** expose the authenticated, read-only `/v1/runtime/config` DTO
   with a versioned schema and full field traces.
5. **Complete:** decode the DTO in Desktop and present the engine's active
   values and provenance in Settings → Performance. Saved controls continue to
   describe operator intent; the active-value display contains no copied
   compatibility or fallback policy.

Rollback remains explicit: remove the production handoff while retaining the
legacy resolver computations. The parity assertion runs before model I/O, so a
new/old mismatch fails closed and identifies every differing field.

## Exit criteria

- The same launch inputs produce the same config on every surface.
- Every non-global default identifies its source and evidence where applicable.
- Compatibility adjustments and fallbacks are visible.
- Desktop contains no duplicated recommendation policy.
