# Rapid Agent productization boundary

## Decision

Rapid will maintain a thin Agent Runtime shared by Server and Desktop. The
Python server owns the model loop, immutable run policy, event ordering, call
identity, budgets, and approval state. Desktop owns presentation, the user's
approval interaction, and explicitly selected local executors. Both surfaces
use the same `/v1/agent/runs` contract.

This is not an attempt to clone a general-purpose coding agent or gateway. We
will reuse Rapid's model server, MCP connections, Desktop tools, MemoryStore,
and workflow safety boundaries. New infrastructure is justified only where a
measured end-user task cannot be completed reliably with those pieces.

We will not inspect or derive implementation details from leaked proprietary
artifacts. The design inputs below are public product contracts, official
documentation, and licensed open-source repositories.

## What makes a small model feel capable

A small model should decide the next bounded action, not retain the whole
operating system, task plan, or tool catalog in its context. The harness must:

1. project only the tools relevant to the task, with a six-tool default for
   MiniCPM5-2B;
2. keep transient task progress separate from long-term personal memory;
3. return compact progress state with tool observations so the model continues
   the whole request;
4. distinguish allow, ask, and deny at the exact tool/call boundary;
5. stop repeated searches and calls, then require a final synthesis;
6. make progress, approvals, failures, and cancellation visible to the GUI;
7. preserve a fast direct-chat path for requests that need no tools.

The intended local primitives are deliberately small: search local content,
read user-selected content, search the web, open a source, calculate or run
bounded code, update a task, and write through an approved destination. Skills
compose those primitives; they do not automatically widen authority.

## Public design inputs

- Claude Code exposes a compact set of local primitives, resumable sessions,
  explicit allowed/disallowed tools, MCP, hooks, and bounded turns. Rapid takes
  the interaction and permission pattern, not its coding-specific scope.
  <https://docs.anthropic.com/en/docs/claude-code/cli-usage>
- OpenCode applies `allow` / `ask` / `deny` to built-ins and MCP tools and has
  an explicit doom-loop permission. Rapid keeps risk and repeat handling in
  the server-owned policy snapshot.
  <https://github.com/anomalyco/opencode/blob/dev/packages/web/src/content/docs/agents.mdx>
- OpenHands separates actions, observations, runtime, server session, and an
  event stream consumed by the frontend. Rapid uses the same clean direction
  of travel while retaining its much smaller process-local reducer.
  <https://github.com/All-Hands-AI/OpenHands>
- Hermes groups tools into optional toolsets, bounds persistent memory, reports
  fixed prompt/schema cost, and avoids re-reading/searching loops. Rapid will
  add prompt-budget visibility before it adds a broad skill catalog.
  <https://github.com/NousResearch/hermes-agent>
- OpenClaw filters skills by runtime capability and allowlist, snapshots their
  identity for a session, and loads full instructions only when selected.
  Rapid will use progressive disclosure when skills arrive; installed does not
  mean visible to every model turn.
  <https://github.com/openclaw/openclaw/blob/main/docs/tools/skills.md>

## Delivery slices

### 1. Desktop adapter

- Add an opt-in typed client for create, observe, approve, result, and cancel.
- Reuse the embedded server bearer and active port.
- Treat server cursors and exact call IDs as authoritative.
- Reuse existing approval UI and executor boundaries; do not add a Swift agent
  loop.

### 2. Task-scoped capability projection

- Map user intent to a deterministic, small toolset from the existing registry.
- Add a separately typed transient task ledger and compact completion checks.
- Reuse the MemoryStore substrate only for durable user facts; never persist
  raw tool arguments, results, model reasoning, or temporary plans as memory.
- Load skill instructions on demand and keep them behind the same tool policy.

### 3. Product qualification and recovery

- Run the real GUI path on physical 8 GB and 16 GB Macs.
- Compare MiniCPM5-2B against current 4B choices on task completion, tool-call
  correctness, repeated-call rate, latency, and peak memory.
- Add restart recovery only after a reviewed replay/migration format exists.
- Change onboarding recommendations only when the physical-device gates pass.

## Explicitly deferred

- arbitrary shell or filesystem authority;
- autonomous skill installation;
- subagents and delegation;
- a second memory database or model server;
- background cron and messaging gateways;
- persistence of raw model/tool payloads.

These may become separate products or adapters later. None is required to
prove that a small local cognition core can reliably complete common Desktop
tasks.
