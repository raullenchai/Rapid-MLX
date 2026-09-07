# Vector handoff: GenerationOutput field-order contract

Owner: Vector

Branch: `fix/generation-output-spec-metrics-order`

Worktree: `/private/tmp/vector-generation-output-field-order`

## Intention

Restore the full-unit baseline after request-scoped MTP telemetry appended
`spec_decode_metrics` to `GenerationOutput` but the chronological field-order
test retained the old expected tail.

The runtime/dataclass behavior is already correct. This task changes only the
test expectation and its diagnostic text. It does not alter the API, telemetry,
serialization, field order, or MTP behavior.

The failure was reproduced on exact `origin/main` (`42cf53c31`) before the
change. Planned verification is the focused regression, lint, scoped self
review, PR validation, and hosted CI.

PR-start FYI delivery to the active Atlas, Harbor, and ds0731 worktrees failed
with Orca `invalid_argument`; this handoff preserves the same awareness payload.
