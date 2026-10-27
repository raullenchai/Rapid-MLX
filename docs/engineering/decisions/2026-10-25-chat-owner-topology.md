# Built-in chat owner: two processes and an opt-in CLI

- **Status:** Accepted amendment; experimental CLI implementation
- **Date:** 2026-10-25
- **Owner:** Atlas (integration), Vector (runtime/storage), Pixel (client), Harbor (packaging)
- **Amends:** [Built-in chat harness](2026-10-25-built-in-chat-harness-pydanticai.md)

## Decision

The chat harness runs in a long-lived, model-free profile owner, separate from
inference servers. This supersedes the earlier decision's placement inside an
inference-serving process. The owner contains PydanticAI, authenticated chat
routes, tools, and SQLite storage. It calls an explicitly selected loopback
OpenAI-compatible inference target for each turn. Inference replacement or
shutdown does not remove access to conversation history.

The first CLI client discovers or starts the detached owner. A startup lock
serializes discovery and launch; a lifetime lock prevents concurrent owners.
Renewable client leases and active turns keep it alive. It exits five minutes
after the last client detaches or expires and no turn is active. The owner
never signals an inference server. The CLI retains its existing inference
selection, attachment, startup, and owned-server cleanup behavior for this slice.

The internal entry point is `python -m rapid_mlx.chat_runtime.owner --profile NAME`.
It bypasses the inference CLI's model resolution, downloads, Metal imports, and
parent watchdog. Adding catalog behavior to the existing `serve` startup would
couple these lifecycles again; a future public command can wrap this same entry
point without changing the profile or wire contracts.

Profiles live at `~/Library/Application Support/Rapid/chat/<profile>/`, with
`RAPID_MLX_CHAT_HOME` as an alternate root. The private `run/` directory contains
atomic discovery with versions, store UUID, owner epoch, PID/start identity,
endpoint, capabilities, and a credential filename. The epoch-specific bearer
file and discovery file are mode 0600. The bearer never appears in discovery,
argv, URLs, logs, or stored turns. Stale discovery never authorizes killing a PID.
All catalog and execution routes require this bearer independently of inference
authentication. Incompatible schemas/protocols fail explicitly; no legacy
fallback or automatic downgrade is permitted. Automated upgrade draining is
outside this first compatible-version slice.

## Durable and transient boundaries

SQLite WAL is the canonical authority. A dedicated writer uses FULL durability
and macOS full synchronization, committing before event publication. Prompt,
tool, and terminal transitions are immediate barriers. Immutable parent-linked
messages, ordered typed parts, revisioned branch heads, stable operation IDs,
and request-hashed turn IDs preserve edits and regenerated siblings.

Wire v1 uses a typed envelope and `turn_id:sequence` SSE IDs. Replay is ordered;
disconnection detaches the consumer, while explicit cancellation closes and
awaits the model stream. A turn ends exactly once as completed, cancelled, or
error. Completion requires a nonempty final answer after tool rounds; one
bounded tools-disabled synthesis attempt is allowed. Owner restart records an
error terminal for interrupted turns and never automatically repeats inference.

Reasoning is live-only. Durable sequence placeholders retain replay continuity
without reasoning text. Explicit uploads are extracted to UTF-8 text snapshots
in a private content-addressed store; `read_attachment` accepts only
branch-authorized IDs. Migration 002 records the client-supplied canonical workspace
on the conversation; missing v1 workspaces remain unset until a client supplies
one. The owner never substitutes its own cwd. A conversation's workspace cannot
change after assignment.

`read_file`, `grep`, and `glob` use that workspace with realpath containment and
secrets-path exclusions. Reads pin directory and file descriptors without
following symlinks after resolution. Discovery skips symlinks, ignored paths,
`.git`, dependency folders, and venvs. A disposable worker limits regex/document
processing to ten seconds and is killed on cancellation; search traversal also
has a five-second/10,000-file bound. Output uses the turn's shared read budget
and explicit continuation/incomplete markers. Tool result events and persisted
parts use the existing wire v1/store shapes. Regeneration reruns read-only tools
and sees current workspace files. Explicit attachment uploads bypass path
exclusions only for their stored text snapshot, never for subsequent path access.

PDF and Office extraction reuse the local-search parser and safe-open helpers;
spreadsheet extraction retains sheet names and cell coordinates. No new
runtime dependency is required. Images remain typed unsupported until model
capabilities, image tool parts, and durable history can be handled together.
The action ledger, checkpoints, grants, branch approval freshness, hidden child
runs, and child event journal use one schema for this unreleased harness. Discovery and
storage share schema version 4; existing v4 stores reopen unchanged. Earlier
experimental and future schemas are refused before journal or schema writes.

A single registry selects delegation, attachment, file, action, and web tools per turn.
File reads and `write_file`/`patch` share `path_policy.py`: explicit canonical workspace,
lexical and resolved secrets exclusions, and a non-overridable workspace boundary.
No approval permits a direct file adapter to access a secret or outside path.
Command/Python inspection uses the same secrets list and path resolution, but
remains heuristic rather than an OS sandbox. Dangerous commands require a fresh
approval. Shell and Python effects are not covered by file undo checkpoints.

All execution and durable behavior belongs to the owner API. Clients upload bytes
for attachment extraction, supply launch cwd, submit turns, answer
`permission.request` via `/permission.response`, and call the conversation's
`/undo` route. Operation identity, expiry, scope, idempotency, dispatch fencing,
checkpoints, cancellation, and process cleanup are owner responsibilities.
`tool.started` now includes an optional owner-provided `summary`; `tool.output`
and `tool.result` provide live/replayable progress. Clients render these events
and translate input keys into API choices. Attachment MIME capabilities advertise
the supported document formats. File paths and attachment IDs remain available
in persisted tool calls/results; web citations use the same durable event stream.

Read-only `weather`, `web_search` and `web_fetch` tools run in the owner unless
`RAPID_MLX_CHAT_WEB_TOOLS=0` is set at owner startup. Provider requests use fixed
HTTPS hosts and bounded responses. Weather and search have a 10-second total
deadline; fetch has 20 seconds total, with up to 10 seconds for Keenable before
the direct fallback. Provider credentials come only from the environment, never
the chat database. Tool results and source events are durable; regeneration
and edits run fresh read-only calls without approval. Source events add an
optional `url` field in wire v1; completed message snapshots expose their journaled sources for export.
Search snippets and fetched pages are untrusted reference material. Page fetching
uses Keenable with a direct HTTP(S) fallback: DNS answers and every redirect are
checked against private networks, connections use validated IPs with the original
TLS hostname, and bodies have size and content-type limits. Page text is capped
at 8,000 characters and trimmed to the remaining context with an explicit marker.
The owner-authorized default has no per-domain prompt. No MCP or native client
executor ships in this slice.

## Delegated child runs

`delegate_task(task, context?, tools?)` starts a fresh PydanticAI agent inside the
same owner with the parent's model and inference settings. Its initial history
contains only the delegated task and explicit context; the owner adds workspace,
clock and tool-policy instructions. Requested tool names must be a subset of the
parent's available tools. The registry excludes delegation from children and the
handler also rejects recursive calls. An owner-wide semaphore caps active helpers
at three; parallel calls share the existing inference endpoint and its batching.

Children use the same action manager, conversation grants and path policy.
Child operation call IDs are namespaced by child ID, while deferred model-call
identities remain stable. `permission.request` has an optional `subagent_id`, and
the existing `/permission.response` route handles approvals. A child timeout
settles only its own pending/dispatched operations; parent Stop awaits all
children, including queued ones, before the parent terminal event.

Each child has six model requests, eight tool calls, 12,000 total usage tokens,
and a 120-second deadline including queue/approval waits. The existing four-web-call
and 16,000-byte model-result limits apply separately to each run. One tools-disabled
summary may use the remaining request/token budget; otherwise a bounded partial
evidence summary reports the exhausted budget or failure. Child usage rolls into
parent usage once without spending the parent's independent request budget.

`subruns` and `subrun_events` are hidden tables linked to the parent turn and tool
call, never branch messages. Domain transcript snapshots and original tool events
persist incrementally; reasoning is removed before SQLite writes. Restart retains
partial transcripts, settles unfinished children, and publishes missing completion
events before terminating the parent. It never automatically reruns children.
Wire v1 adds `subagent.started`, `subagent.progress`, and `subagent.finished`, each
with `id`, `call_id`, `title`, `status`, `tool_count`, and activity `summary`.
Statuses are `queued`, `running`, `completed`, `error`, `cancelled`,
`budget_exhausted`, and `timeout`. Child text/tool output stays in the hidden
journal; the parent receives a bounded summary and canonical source citations.

Authenticated `GET /v1/rapid/chat/turns/{turn_id}/subruns` lists child metadata;
`GET /v1/rapid/chat/subruns/{id}` adds task, context, domain messages and hidden
events. Metadata includes start/finish timestamps, status, summary, tool count,
sources and usage. Deleted conversations are inaccessible through these routes.
The CLI renders nested progress and exports child details without inserting them
into the visible conversation. These owner contracts can support later Desktop
drill-down without moving child execution or authorization into a client.

## Rollout and verification

`rapid-mlx chat --harness` (or `RAPID_MLX_CHAT_HARNESS=1`) is opt-in; the existing
chat path remains the default. The CLI supports persistent history, resumption,
editing, regeneration, branch selection, explicit Fast/Reasoning, attachment
reading, Markdown export, Stop, and owner reconnection.

Focused deterministic tests cover schema refusal, SIGKILL recovery, reasoning
exclusion, idempotency/CAS, projection, tool rounds, empty-final recovery,
cancellation, auth/replay, owner races/leases/permissions, and a PTY journey.
The broader model qualification and final packaged-dependency notice gates in
the original decision remain required before a default flip or Desktop rollout.
