# Migration 001: extract scoped prefill profiles

- **Status:** discovery
- **Owners:** Vector (evidence), Atlas (runtime integration)
- **Rollback:** retain the current `ModelProfile` fields and resolver precedence

## Current state

`recommended_prefill_step_size` is a sparse alias-level field. It is
bench-verified and automatically reaches no-flag Desktop launches, while an
explicit user flag wins. Language prefill and vision admission budgets are
resolved separately.

## Target state

Represent prefill chunk and vision admission budget as the first
`PerformanceProfile` vertical slice, scoped by artifact/quantization, machine
class, workload, and runtime range, with evidence and promotion IDs.

## Safe sequence

1. Introduce an adapter that emits the existing values from current aliases.
2. Add equivalence tests for every verified alias and explicit override.
3. Introduce machine/workload keys without changing fallback behavior.
4. Migrate measured profiles one family at a time.
5. Remove the alias fields only after all consumers use the resolver output.

## Exit criteria

- Existing aliases resolve byte-for-byte equivalent values.
- Explicit user values retain precedence.
- GUI and Server expose the resolved value and provenance.
- Missing or expired evidence falls back to the conservative global default.
