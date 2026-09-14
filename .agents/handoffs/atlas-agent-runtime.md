# Atlas handoff — Rapid Agent Runtime

- **Owner:** Atlas
- **Branch:** `atlas/agent-server-adapter`
- **Base:** `origin/main` after merged runtime kernel PR #3439
- **Status:** Server adapter implemented and under validation; Desktop adapter is next

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

After this branch merges, add the Desktop client adapter behind a rollback
feature flag. It should consume `/v1/agent/runs` plus the shared event schema,
reuse the existing approval UI and tool executors, and post typed results back
to the server. Follow with physical 8 GB / 16 GB qualification.

## Server adapter facts

- The server reuses the production non-streaming Chat Completions function
  in-process; there is no loopback HTTP request or second inference stack.
- The store is bounded to 32 runs, retains completed runs for 15 minutes, and
  never evicts active work. Shutdown cancels runs before MCP and engine teardown.
- Server and client execution modes share one reducer. Client mode is the
  narrow handoff Desktop needs; clients cannot submit tool definitions, risk
  labels, or event summaries.
- MCP has no standard risk metadata. Only exact tools in the operator-owned
  `agent_read_only_tools` config are automatic; every other call pauses for
  exact-ID approval and still passes the existing MCP sandbox before execution.
- A custom served name or arbitrary local directory cannot widen the MiniCPM
  profile: selection uses catalog identity or exact loaded architecture
  metadata plus the native parser, while inference uses the public served name.
- Each run pins the MCP manager/executor generation that advertised its tools;
  hot reload cannot redirect an old validated call to a replacement registry.
- Each run binds its concrete model-registry entry (or single engine); an
  unload/replacement fails instead of switching to a new default mid-run.
- Model goals, raw call arguments, results, and final output are kept only in
  the live adapter. The event API carries redacted metadata and monotonic cursors.

## Risks

- Model generation is still route-owned. The adapter calls that function
  directly; a later extraction is warranted only if another internal consumer
  appears.
- P0 deliberately makes no crash-durability claim. A process restart terminates
  in-flight runs; durable recovery would require a separate reviewed replay and
  migration design.
- Risk classification must come from the registry snapshot the model saw, not
  from a later client-supplied tool definition.
