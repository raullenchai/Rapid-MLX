# Rapid Agent Runtime

Rapid Agent Runtime is a bounded tool loop shared by headless Server and the
Desktop client. It reuses the model, tool-call parser, admission control, MCP
connections, sandbox, and authentication already owned by `rapid-mlx serve`.
It does not add a second model server, tool ecosystem, database, or daemon.

This first server surface is process-local. Restarting the server cancels all
in-flight runs, and completed runs expire after 15 minutes. Use ordinary
`/v1/chat/completions` or `/v1/responses` when a managed tool loop is not
needed; those endpoints are unchanged.

## Start the server

Agent runs use the MCP servers already configured for `serve`:

```bash
rapid-mlx serve minicpm5-2b-4bit \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$RAPID_API_KEY" \
  --mcp-config ./mcp.json
```

MCP has no standard side-effect annotation. Rapid therefore requires approval
for every MCP tool unless the operator explicitly declares an exact namespaced
tool as read-only:

```json
{
  "mcpServers": {
    "files": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/safe/root"],
      "agent_read_only_tools": [
        "files__read_file",
        "files__list_directory"
      ]
    }
  }
}
```

Do not put a write, delete, shell, messaging, payment, or other consequential
tool in that server's `agent_read_only_tools`. Name patterns are intentionally not trusted:
a connector could call a mutating tool `get_and_delete`, so undeclared tools
always pause for approval. The existing MCP sandbox still runs immediately
before execution and may reject an approved call.

Each run also pins the MCP manager/executor generation that advertised its
tools. If MCP is reloaded while a run is generating or awaiting approval, that
run never redirects an already validated call to a replacement tool with the
same name. The old connection may instead report unavailable; create a new run
to use the reloaded registry.

While a run is `awaiting_approval`, its authenticated run view includes a
Rapid-generated `approval_summary`. It preserves ordinary decision fields such
as recipient, path, command, and amount while recursively redacting
credential-shaped keys. The preview is capped at six nested levels, 32 fields
or list items, 128 characters per key, and 256 characters per string; omitted
material is marked `[truncated]`. Neither that summary nor the original
arguments enter the event stream.

## Create and observe a run

`server` execution asks Rapid to execute approved MCP calls. Omitting
`tool_names` selects a deterministic profile-sized subset, preferring tools
declared read-only. Supplying `tool_names` selects an exact subset from the
connected MCP registry; unknown, unsupported, duplicate, or over-budget names
are rejected.

```bash
curl -sS http://127.0.0.1:8000/v1/agent/runs \
  -H "Authorization: Bearer $RAPID_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "goal": "Read the release notes and summarize the breaking changes.",
    "tool_names": ["files__read_file"],
    "execution": "server"
  }'
```

The create call returns HTTP 202 and a run view. Poll the run or request only
events after the last sequence already consumed:

```bash
curl -sS \
  -H "Authorization: Bearer $RAPID_API_KEY" \
  http://127.0.0.1:8000/v1/agent/runs/RUN_ID

curl -sS \
  -H "Authorization: Bearer $RAPID_API_KEY" \
  'http://127.0.0.1:8000/v1/agent/runs/RUN_ID/events?after=0'
```

`output` appears on the authenticated run view after completion. It is never
copied into the durable event contract. Events likewise omit the goal, raw
argument values, tool-result content, credentials, and model reasoning.

## Approve or deny an action

When status is `awaiting_approval`, the run view contains one transient
`pending_action` with the call ID, tool name, risk, an empty `arguments` object,
and the bounded `approval_summary` described above. Approval is tied to its
exact call ID:

```bash
curl -sS http://127.0.0.1:8000/v1/agent/runs/RUN_ID/approval \
  -H "Authorization: Bearer $RAPID_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"call_id":"CALL_ID","approved":true}'
```

Use `approved:false` to deny it. Rapid returns the denial to the model as an
unexecuted tool result, then asks for a safe final response. A stale or wrong
call ID returns HTTP 409 and cannot release an action.

After `approved:true`, a client-executed run moves to
`awaiting_tool_result` and only then exposes the original arguments. This
prevents a client executor from receiving an actionable side-effect payload
before the approval boundary.

## Client execution for Desktop

Set `execution:"client"` when an authenticated client owns execution. Rapid
still chooses tools from the server-owned registry, validates the model call,
and applies the same approval policy. The client reads `pending_action`, runs
the matching local adapter, then returns exactly one result:

```bash
curl -sS http://127.0.0.1:8000/v1/agent/runs/RUN_ID/tool-result \
  -H "Authorization: Bearer $RAPID_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "call_id":"CALL_ID",
    "content":"tool output supplied to the model",
    "is_error":false,
    "executed":true
  }'
```

Server-executed runs reject client results. Client-authored result content is
transient; clients cannot author the event `safe_summary` field. This prevents
secrets or tool output from being copied into the event stream.
The `executed` boolean is required: Desktop must explicitly distinguish a
pre-dispatch failure from a call it actually attempted. When `executed` is
`false`, Rapid forces the result to an error and replaces client-authored
content with a stable "not executed" observation, so the model cannot mistake
an unexecuted action for success.

Tool completion events use `executed:true` when dispatch occurred,
`executed:false` for a known pre-dispatch rejection, and `executed:null` when a
third-party registry failure leaves the outcome uncertain. Treat `null` as
potentially executed and never retry it automatically.

## Cancel a run

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $RAPID_API_KEY" \
  http://127.0.0.1:8000/v1/agent/runs/RUN_ID/cancel
```

Cancellation aborts the current model request and normally makes the run
terminal within one second. A custom generation driver that ignores
cancellation receives an HTTP `409`; the run remains non-terminal and
non-evictable until that work stops, after which the caller can retry cancel.
If an MCP call has already been dispatched, Rapid does not pretend it was
cancelled: it waits for that call's existing timeout/result, records the
execution outcome, and only then marks the run cancelled. This makes completed
or failed external actions visible before an operator retries them. In `client`
execution mode, once arguments have been released to the caller, Rapid cannot
know whether that caller committed the action. Cancelling then records a
`tool.completed` event with `executed:null` before `run.cancelled`; do not retry
that action automatically.
Server shutdown cancels all active runs before MCP connections and the model
engine are torn down. Dispatched MCP calls retain their configured execution
timeout so Rapid can record their outcome. Cancelled model-generation tasks
are joined for up to 30 seconds; if a custom driver still refuses cancellation,
shutdown fails explicitly instead of tearing the engine down underneath live
work.

## MiniCPM5-2B defaults

The MiniCPM5-2B profile is the measured low-memory path:

- at most 6 visible tools;
- at most 8 tool rounds, followed by one tools-disabled synthesis turn;
- at most 2 identical calls before loop blocking;
- one tool call per model turn;
- a hard 900-token output ceiling (larger request values are clamped),
  temperature 0.7, top-p 0.95, thinking disabled;
- 300-second model-turn timeout.

The sampling and timeout fields can be overridden on run creation. Safety and
tool-count limits cannot be widened by a client. Other models use the same
runtime with their own profile limits.

Profile selection uses the loaded model identity, not the public served name.
For an arbitrarily named local MiniCPM directory, Rapid verifies the exact
released 2B architecture shape from its bounded local `config.json` together
with the configured `minicpm` tool parser. Renaming the directory or API model
therefore cannot widen its limits.

A run is also bound to the concrete model/engine generation selected when it
starts. If that model is unloaded or replaced while the run is paused for an
approval or client result, the run fails rather than silently continuing on a
new default model under stale profile limits.

## HTTP surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/v1/agent/runs` | Create and start a bounded run |
| `GET` | `/v1/agent/runs/{id}` | Read status, transient pending action, or final output |
| `GET` | `/v1/agent/runs/{id}/events?after=N` | Poll versioned events after a sequence |
| `POST` | `/v1/agent/runs/{id}/approval` | Approve or deny the exact pending action |
| `POST` | `/v1/agent/runs/{id}/tool-result` | Return one client-executed tool result |
| `POST` | `/v1/agent/runs/{id}/cancel` | Cancel an active run |

Every route uses the existing Bearer API-key and rate-limit dependencies.
Capacity is 32 process-local runs; a new run evicts the oldest terminal run but
never an active one. When all slots are active, creation returns HTTP 503.
