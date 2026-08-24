# Migration 002: central EffectiveRuntimeConfig

- **Status:** planned
- **Owner:** Atlas
- **Rollback:** preserve existing CLI/server resolver helpers behind an adapter

## Goal

Create one immutable output consumed by CLI, Server, and Desktop. Each resolved
field records `value`, `source`, `source_id`, `reason_code`, and whether it
overrode or adjusted another source.

## Sequence

1. Define the internal schema and precedence tests without changing behavior.
2. Wrap existing resolver helpers and compare old/new output.
3. Route Server and CLI through the central resolver.
4. expose a read-only DTO for Desktop.
5. Add GUI presentation for active optimizations, warnings, and overrides.

## Exit criteria

- The same launch inputs produce the same config on every surface.
- Every non-global default identifies its source and evidence where applicable.
- Compatibility adjustments and fallbacks are visible.
- Desktop contains no duplicated recommendation policy.
