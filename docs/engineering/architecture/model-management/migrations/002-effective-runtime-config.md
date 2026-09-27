# Migration 002: central EffectiveRuntimeConfig

- **Status:** in progress — stage 1 contract complete
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
3. Route Server and CLI through the central resolver.
4. expose a read-only DTO for Desktop.
5. Add GUI presentation for active optimizations, warnings, and overrides.

Stage 1 deliberately has no production caller. Its rollback is deletion of the
new module and tests; CLI and Server continue to use their existing helpers until
the stage 2 parity fixtures prove equivalent results.

## Exit criteria

- The same launch inputs produce the same config on every surface.
- Every non-global default identifies its source and evidence where applicable.
- Compatibility adjustments and fallbacks are visible.
- Desktop contains no duplicated recommendation policy.
