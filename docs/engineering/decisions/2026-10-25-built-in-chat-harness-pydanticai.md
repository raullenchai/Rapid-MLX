# Built-in chat harness: persistent PydanticAI with Rapid-owned state

- **Status:** Accepted architecture; client rollout and heuristic-path retirement gated by the qualification below
- **Owner:** Human owner (decision); Vector (engine/runtime implementation), Pixel (Desktop integration), Harbor (packaging)
- **Date:** 2026-10-25

## Context

Both `rapid-mlx chat` and Desktop Chat need one built-in tool-capable harness.
The existing heuristic agent runtime is not the long-term chat architecture.
Rapid must own conversation branches, tools, citations, permissions, thinking
policy and persistence rather than delegate product semantics to a harness CLI.

The 468-turn local bake-off compared Goose-min, Pi-min, a per-turn PydanticAI
thin runtime and direct calls on Qwen3.5-4B, Qwen3.6-35B-A3B and Qwen3.6-27B.
Pi-min's warm median time to first visible token was 0.30/0.40/0.73 seconds;
per-turn thin was 0.58/0.65/1.58 seconds, including Python startup. Goose had
stronger citation-task results in parts of that sample. Persistence was not
measured, so the bake-off alone did not reject an engine-side Python runtime.

The selected PydanticAI slim/openai license closure was assessed GREEN:
26 distributions, top-level MIT, low incremental packaging burden in Rapid's
existing Python runtime. GREEN still requires notice staging, the recovered
Logfire API MIT notice, certifi MPL exact-source availability and final-artifact
verification. It is not certification of every extra or the full benchmark venv.

## Options considered

| Option | Evidence and disposition |
| --- | --- |
| Persistent PydanticAI slim/openai | Selected: engine-native Python lifecycle, streaming graph/tool events, smallest incremental compliance burden; Rapid supplies product policy/state. |
| Pi-min | Useful latency/reference baseline, but introduces a second runtime/packaging boundary and a larger notice/provenance closure. Not selected for the built-in harness. |
| Goose-min | Useful robustness reference; default CLI adds native LGPL relink/source and broader notice obligations. Goose MVP is not revived. |
| Direct model calls | Lowest orchestration cost, but no complete tool loop, provenance or permission lifecycle. Keep as a regression baseline only. |
| Existing heuristic runtime | Migration/rollback context only; do not port goal-regex intent/permission heuristics into the new chat runtime. |

## Decision

Use **`pydantic-ai-slim[openai]==2.61.0` (MIT)** in a new persistent
`rapid_mlx/chat_runtime/` inside the serve process for **both CLI chat and
Desktop Chat**. Create/reuse Agents and model clients; pass isolated per-turn
dependencies and branch-scoped message history. Use `run_stream_events()` or
`iter()` so text before a tool call cannot prematurely terminate the turn.
Do not add the optional PydanticAI coder/research harness extras by default.

Rapid owns the tool registry/execution adapters, source registry/rendering,
permission decisions, conversation tree, storage, budgets and thinking policy.
Start with OpenAI-compatible loopback through the same server's existing
model/parser/cancellation pipeline. An in-process adapter must prove equivalent
semantics and a measured benefit before replacing that boundary.

Expose a versioned `POST /v1/rapid/chat/turns` SSE contract plus cancel/replay/
permission endpoints. Events cover turn start, visible text, separate reasoning,
tool start/arguments/results, sources, permission requests, usage and one
terminal outcome. A completed turn must contain nonempty final visible text;
reasoning, a preamble or source list alone does not satisfy it. Permit at most
one bounded tools-disabled final-synthesis recovery, then explicit error.
Never convert hidden reasoning into the visible answer to recover emptiness.

Use immutable message parents and branch heads for edit/regenerate, with
linked tool results, attachments and sources in a Rapid-owned versioned store.
Persist authorized conversation context, not secrets or hidden reasoning by
default. Keep PydanticAI history a projection/cache, not the storage authority.
ACP v1 is a separate interoperability adapter; permission is the
`session/request_permission` RPC and tool updates follow v1 create/update rules.

Provide an explicit Fast/Reasoning user control, recording the effective model
setting. Keep Fast as the general chat default; qualify Reasoning for tasks
where correctness warrants its cost. Do not introduce automatic goal-regex
intent routing from this small study. Explicit thinking-on must survive tool
presence, subject to the disclosed server no-thinking kill switch.

## Phase 0 evidence

The uncommitted spike used cached models, the prior tasks ×3, A/B citations,
and five known-answer math problems plus chatter/search off/on ×3. The full
report defines denominators, source provenance limits and cancellation results.

| Model | Warm TTFVT A / B (s) | B inline / inline-or-attached cite success | Math clean correct off / on |
| --- | --- | --- | --- |
| 4B | 0.85 / 0.60 | 9/12 / 12/12 | 8/15 / 15/15 |
| 35B-A3B | 0.86 / 0.69 | 10/12 / 10/12 | 5/15 / 15/15 |
| 27B | 1.67 / 1.38 | 12/12 / 12/12 | 10/15 / 15/15 |

The historical Pi latency target is **not met** on the tested current main.
PR #5267, newer than the bake-off revision, makes Qwen thinking-off parsing
hold unmarked text until a boundary or EOF. A parser probe and same-body direct
HTTP replays confirm late content delivery without PydanticAI. Warm pre-model
orchestration is small; persistence alone cannot fix this serving behavior.
Resolve latency without undoing reasoning/tool-wire privacy protections.

The citation registry improves 4B source-list coverage but does not eliminate
tool omissions or ungrounded model-written links; URL provenance does not
prove factual correctness. Explicit thinking-on improves clean math accuracy
at a model-dependent latency cost. Existing tool default-off originated in
PR #891; separate no-tool chat default-off in PR #895 explains the POC's zero
reasoning even for direct calls. Cancellation is measured through scheduler
batch removal/tool termination, not merely a fast HTTP acknowledgement. All
72 qualified stop probes reached cleanup: maximum generation-stop latency
244 ms and local tool-stop latency 3.4 ms; these small samples do not certify p95.

## Consequences

- One backend policy and event contract replaces parallel CLI/Desktop loops.
  Rapid takes responsibility for durable branches, concurrency, lifecycle,
  permissions, cancellation and citation correctness.
- A persistent Python dependency fits existing distribution machinery, but
  selected notices/source availability and offline startup remain release gates.
- The spike is not production-ready: it has in-memory history, a fixture-only
  tool boundary, no production permission store, and incomplete citation syntax
  handling. Its recovery must become tools-disabled before side effects ship.
- This decision replaces the built-in chat harness direction, not every public
  compatibility endpoint. Retire heuristic paths after CLI/Desktop parity;
  preserve migration/rollback until qualification. No Goose MVP revival.

## Phase plan and acceptance gates

1. **Vector: contracts and runtime.** Add `models/events`, `runtime`,
   `pydantic_adapter`, `model_client`, `thinking`, `sources`, `tools/permissions`,
   `store/history/attachments`, `api` and `acp` under `rapid_mlx/chat_runtime/`.
   Integrate lifespan ownership and authenticated routes; late SSE is #5277.
2. **Vector + Pixel: shared clients.** Move CLI and Desktop Chat to the same
   endpoint/tree behind one rollout flag; support sources, stop, permissions,
   edit/regenerate and restart/replay. No second Desktop orchestration loop.
3. **Harbor: packaging.** Pin the selected dependency closure, stage notices and
   certifi source availability, inspect signed/bundled artifacts and offline start.
4. **Qualification and retirement.** Only then retire built-in heuristic paths.

Release gates on the documented Studio/model/task matrix:

- Latency measured on the 12-task set ×3: 39 turns per model because the follow-up task has two turns, of which the first greeting is cold and 38 are warm; same-revision baseline in the same session. Warm TTFVT p50 ≤ **0.30/0.40/0.73 s**; cold ≤ **0.70/0.66/2.45 s**,
  with same-revision controlled baselines and no privacy bypass; greeting prompt
  ≤ **650 tokens** and warm pre-model orchestration p50 ≤ **20 ms**.
- **100%** completed turns have nonempty visible final answers; **0** reasoning/
  tool-wire leaks, ≤ **1** recovery and **0** duplicated side effects.
- **12/12** grounded inline-or-attached citation tasks per model; **100%**
  displayed-source provenance, **0** invalid markers displayed; separately
  target ≥ **11/12** inline citation success and assess semantic grounding.
- Cancel p95 ≤ **250 ms** to actual decode batch removal/local tool termination,
  with ≥ **30 trials per case/model**, and **0** continuing owned work after
  **1 second**. Cover disconnect, explicit stop, prefill, tool I/O and approvals.
- Reasoning-on ≥ **14/15 clean correct** on the math qualification set; all
  explicit thinking settings reach the template. Expand held-out tasks before Auto.
- **0** cross-session/branch leaks in ≥ **100 concurrent scripted turns**;
  replay/edit/regenerate/restart tests and exact-once approval execution.
- **100%** required selected-dependency notices/source obligations staged;
  authenticated shipping endpoints and offline launch with no model downloads.

## Evidence

The key numbers are summarized inline above. Raw data, scripts and full reports
are in the internal engineering archive on the Studio at
`/Volumes/RTL-2T/scratch-archive/rapid-mlx-chat-harness-2026-10-25/`:

- `bakeoff/report.md`: 468-turn harness bake-off (tasks, proxy logs, summary CSV).
- `bakeoff/license-audit/REPORT.md`: dependency license inventories and staging checklist.
- `phase0/report.md`: Phase 0 spike (latency, citations, thinking, cancellation, event contract).
- `spike-chat-runtime-p0.diff`: uncommitted spike code against `e899b689c`, reference only.

Related: [#5277](https://github.com/raullenchai/Rapid-MLX/issues/5277) tracks the
thinking-off streaming hold that currently blocks the latency gate.
[Rapid agent runtime](2026-09-13-rapid-agent-runtime.md) is superseded for the
built-in chat harness by this record.
