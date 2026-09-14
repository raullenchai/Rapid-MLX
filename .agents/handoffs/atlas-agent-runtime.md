# Atlas handoff — Rapid Agent Runtime

- **Owner:** Atlas
- **Branch:** `atlas/minicpm-agent-runtime-p0`
- **Base:** `origin/main` after MiniCPM harness qualification PR #3425
- **Status:** P0 kernel and architecture contract implemented; server adapter is next

## Verified facts

- Desktop already owns built-in tools, MCP adapters, approval UX, MemoryStore,
  a bounded chat tool loop, and a transient LocalWorkflow ledger that is never
  copied into public events.
- Server already owns model routing, MCP discovery/execution, tool parsing,
  authentication, and Chat/Responses APIs.
- The P0 kernel adds no dependency or process. It stores a complete immutable
  model-profile snapshot, snapshots the exact per-turn tool policy, and emits a
  versioned append-only event stream writable only by the reducer.
- Raw call arguments, tool-result content, goal-bearing ledger context, and final
  model content are transient and never enter that event stream. Request events
  retain call identity and argument names; result/completion events retain safe
  metadata only.
- Call IDs are single-use and approvals match the exact pending ID. External
  calls remain runtime-private and are released to an executor only after a
  positive approval.
- Tool risk is mandatory at the registry boundary. Repeat fingerprints are
  held only inside the live runtime and are never serialized. Weak tracking
  references prevent abandoned runs from being retained in memory.
- Calls are validated against the exact advertised JSON Schema before release.
  One runtime-level lock serializes P0 transitions so concurrent approval or
  result requests cannot release/complete the same call twice.
- Result metadata distinguishes executed calls from denied/loop-blocked calls;
  approval input is a strict Python boolean.
- MiniCPM5-2B defaults are six visible tools, eight tool rounds, one call per
  model turn, and two identical calls before forced final synthesis.
- Focused unit tests, Ruff, and focused mypy pass.

## Architecture boundary

The Python server owns run state and orchestration. Desktop will consume run
events plus an authenticated transient call channel, retain presentation and
its client-local tool executors, and return typed results. Plain Chat/Responses
endpoints remain stateless and compatible.
Do not add a second tool registry, memory implementation, sandbox, or planning
framework to the runtime kernel.

## Next concrete action

Add the authenticated Server adapter as a separate PR: a bounded in-memory run store,
create/get/events/approval/result/cancel endpoints, and a model-turn adapter
over the existing generation path. Do not expose endpoints until a created run
can make progress end to end. Follow with a Desktop client migration behind a
rollback feature flag, then physical 8 GB / 16 GB qualification.

## Risks

- Model generation is still route-owned; extracting a reusable internal
  generation service is preferable to making an in-process HTTP call.
- P0 deliberately makes no crash-durability claim. A process restart terminates
  in-flight runs; durable recovery would require a separate reviewed replay and
  migration design.
- Risk classification must come from the registry snapshot the model saw, not
  from a later client-supplied tool definition.
