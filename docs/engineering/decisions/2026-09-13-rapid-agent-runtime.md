# Rapid Agent Runtime: one bounded harness for Server and Desktop

- **Status:** Accepted for P0 implementation
- **Owner:** Atlas
- **Date:** 2026-09-13

## Context

MiniCPM5-2B 4-bit is Rapid's leading low-memory local-agent candidate. In the
paired M2 Pro qualification, a bounded harness raised strict completion from
19/36 to 25/36 while enhanced Qwen3.5-4B completed 28/36 at 2.7 times the mean
task wall time. The useful intervention was small: expose fewer tools, retain
host-owned progress, stop repeated actions, and require a final answer.

Rapid already owns most necessary mechanisms. Server has model routing, tool
parsers, MCP discovery/execution, authentication, and OpenAI-compatible APIs.
Desktop has built-in tools, MCP adapters, approvals, MemoryStore, a bounded chat
tool loop, and a metadata-only LocalWorkflow ledger. A second tool ecosystem or
a Desktop-only MiniCPM loop would duplicate these and drift.

The upstream projects are references, not runtime dependencies:

- Hermes Agent: one platform-independent loop, bounded toolsets, prompt layers,
  compression, and SQLite session lineage.
- OpenClaw: host-owned deadlines, durable tasks, approval waits, tool policy,
  and large-catalog tool search.
- OpenHands: action/observation events and a UI that never executes backend
  actions directly.
- OpenCode: client/server sessions and clear read-only versus mutating modes.

## Decision

Rapid will own a small **Agent Runtime** inside the Python server. It is a
stateful orchestration layer above inference, not part of the inference engine.
Plain `/v1/chat/completions` and `/v1/responses` remain stateless and unchanged.

The runtime kernel is a deterministic reducer. It does not call a model, run a
tool, persist secrets, or store hidden reasoning. Adapters drive model requests
and tool execution around it. `AgentRun` is frozen, process-local state owned by
the runtime; it is deliberately not a restore format. Immutable `AgentEvent`
objects are the only GUI/server wire contract. A process restart terminates
in-flight P0 runs instead of reconstructing safety state from client data.

Desktop will eventually submit and observe runs over the server API. Desktop
built-in tools may remain client-executed: the server emits a redacted
`tool.requested` event and delivers the transient call payload over the
authenticated live request channel. For external side effects, the runtime
holds the raw call internally and releases executable output only after an
exact positive approval. Desktop then uses its existing tool registry and
returns a typed result. Headless operation sends the same approved transient
output to the existing Server MCP executor. This keeps policy identical without
forcing macOS-only tools into Python.

### P0 invariants

1. A run is bound to one immutable model profile, recorded completely in
   `run.created`. Only the runtime instance that created a run may advance it.
2. The MiniCPM5-2B profile exposes at most six tools and permits eight tool
   rounds. P0 accepts one tool call per model turn for every profile.
3. Tools not advertised for that exact turn fail closed.
   Registry adapters must classify every tool explicitly; there is no
   permissive default risk. Calls are validated against the exact JSON Schema
   snapshot shown to the model before any executable payload is released. P0
   accepts inline schemas only; references and resolver/network behavior are out
   of scope.
4. External side effects pause for an explicit approval result tied to the
   exact pending call ID; their raw executable call is released only after
   `approved=True`. A call ID may appear only once in a run. Blocked/denied
   results are explicitly marked unexecuted.
5. Repeating the same tool and arguments more than twice disables tools and
   forces final synthesis.
6. Tool-round exhaustion reserves one tools-disabled final synthesis turn.
7. Only the reducer can append events. Every transition emits a versioned,
   monotonically sequenced immutable event.
8. Events contain action metadata and safe summaries, never raw tool payload
   values, model reasoning, credentials, screenshots, or clipboard contents.
   Request events retain only call identity and argument names. Credentials
   must be resolved from opaque references out of band. Raw calls, results,
   final model content, and repeat fingerprints remain transient. Weak run references ensure
   abandoned runs do not pin memory.
9. Host-generated denial and loop-guard observations are returned transiently
   to the adapter, so every model tool call receives a matching tool result.
10. Adapters attach a short host-authored ledger block to the next model request.
    It stays transient because it includes the goal, and is never copied into a
    wire event. It is not appended as a new user instruction; the A/B test showed
    that shape can restart the task.
11. P0 serializes transitions with one runtime lock, preventing concurrent
    approval/result requests from executing or completing the same call twice.
    Per-run locking is deferred until measured contention justifies it.

### Deliberately absent from P0

- multi-agent delegation or swarms;
- channels, cron, skills marketplace, or plugin framework;
- Docker or a second sandbox implementation;
- automatic long-term-memory writes;
- an LLM-based planner on every turn;
- a semantic tool router before a measured need exists.

These omissions are architectural boundaries, not a roadmap promise.

## Integration sequence

1. **Kernel and contract:** model profiles, reducer, event schema, budgets,
   approval pause, repeat guard, and event serialization tests.
2. **Server adapter:** bounded in-memory run store and authenticated
   run/event/result endpoints;
   drive the existing chat generation path and Server MCP executor.
3. **Desktop adapter:** decode the same event schema; execute existing built-in
   tools and approvals; retain the old loop as rollback until parity tests pass.
4. **Qualification:** real GUI tasks on physical 8 GB and 16 GB Macs. Promotion
   of MiniCPM5 to the primary recommendation is a measured catalog change, not
   an architectural default.

## Consequences

The first kernel adds no dependency and no idle process. Model-specific tuning
is data in `AgentProfile`, so Qwen and future compact models reuse the same
runtime. The server becomes the owner of run state, while the GUI stays the
owner of macOS presentation and client-local tool execution.

P0 does not claim crash durability and does not accept serialized runs back from
clients. Durable recovery is a separate future decision that would require an
explicit event-replay state machine, atomic storage, migrations, and corruption
tests. Until Desktop has migrated, its existing tool loop remains the shipping
path.

## References

- <https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/architecture.md>
- <https://github.com/openclaw/openclaw/blob/main/docs/concepts/agent-loop.md>
- <https://docs.openclaw.ai/tools>
- <https://github.com/OpenHands/OpenHands/blob/main/docs/architecture.md>
- <https://github.com/anomalyco/opencode>
- [MiniCPM5 harness qualification](../performance/2026-09-13-minicpm5-small-agent-harness-ab.md)
