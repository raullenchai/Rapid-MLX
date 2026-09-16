# Rapid Agent Runtime

Rapid Agent Runtime is a bounded tool loop shared by headless Server and the
Desktop client. It reuses the model, tool-call parser, admission control, MCP
connections, sandbox, and authentication already owned by `rapid-mlx serve`.
It does not add a second model server, tool ecosystem, database, or daemon.

This first server surface is process-local. Restarting the server cancels all
in-flight runs, and completed runs expire after 15 minutes. Use ordinary
`/v1/chat/completions` or `/v1/responses` when a managed tool loop is not
needed; those endpoints are unchanged.

## Use Personal Intelligence in Desktop

Desktop presents the runtime as **Personal Intelligence**, not as a separate
Agent product. The single four-point icon beside the Chat attachment button is
the per-conversation control:

1. Hover the icon to see what the capability does.
2. On first use, choose **Turn on** after reviewing the local-data and approval
   contract, or **Not now** to keep ordinary Chat.
3. After that introduction, click the icon once to turn Personal Intelligence
   on for the current conversation and click it again to turn it off. A filled
   indigo icon means on; a neutral icon means off.
4. New conversations start on after the user has chosen **Turn on**. Any
   consequential action still receives its own approval prompt. Existing
   conversations are not opted in retroactively; their choice is stored per
   conversation.

Personal Intelligence is fail-closed and model-specific. Desktop enables it
only when the live `/v1/models/{id}` response binds the exact selected model to
a harness profile that has been tuned and dogfooded with that model. The server
is the single source of truth; Desktop does not maintain a parallel model
allowlist. Tool-call support by itself is not enough,
and selecting Personal Intelligence never changes or downloads a different
model. Every qualified pairing owns its tool visibility, loop budget, repeat
guard, output ceiling, parser, and prompt behavior. The popover names the
current model and explains when its profile is not ready. Switching back to a qualified model
restores that
conversation's prior on/off choice, unless attachments were staged while it
was on ordinary Chat; in that case Personal Intelligence stays off and explains
why instead of stranding those attachments.

The introduction copy is the user contract:

> Use your Mac’s tools and local context to get things done.
>
> Reads only what you choose. Asks before making changes. Runs locally by
> default.

Personal Intelligence currently accepts text tasks. Turn it off before adding
a normal Chat attachment; model-native image and document attachment behavior
is otherwise unchanged.

The qualification field is additive to the OpenAI model card:

```json
{
  "id": "minicpm5-2b-4bit",
  "personal_intelligence_profile": "minicpm5-2b",
  "personal_intelligence_qualification": "minicpm5-2b-q4-v1"
}
```

`null` means ordinary Chat, including for models that otherwise advertise tool
support. At run creation Desktop also checks that the returned model, runtime
profile, and versioned qualification ID match the model-card values; a mismatch
is cancelled rather than falling back to a generic harness. Qualification also
requires the tested canonical backing repository identity and live native
parser. A public alias is not accepted as proof of the backing artifact: an
alias reused for other weights, an incompatible parser override, or
`--no-tool-call-parser` returns `null`.

Each admitted pairing is a versioned
`PersonalIntelligenceQualification`: public model identities, backing artifact
identities, parser, harness profile, and a repository-relative evidence report.
Quantizations remain separate qualifications even when they share a parser and
harness. Adding a name to a broad family matcher is therefore insufficient to
enable the product. Current admitted builds are MiniCPM5-2B MLX Q4,
Qwen3.5-4B Q4, Qwen3.5-9B Q4, Qwen3.6-35B-A3B Q8, and LFM2.5-1.2B Q4. Other
quantizations, renamed local copies, and unlisted models remain ordinary Chat
until their exact build passes qualification. MiniCPM Q8 is a 16 GB candidate
and does not inherit Q4's strict qualification.

Maintainers can run the same live Agent API qualification used for those
receipts:

```bash
python scripts/qualify_personal_intelligence.py MODEL \
  --base-url http://127.0.0.1:8000 \
  --seeds 11,22,33 \
  --hardware 'Mac model, chip, memory' \
  --os 'macOS version and build' \
  --runtime 'rapid-mlx, MLX, and mlx-lm versions' \
  --source-revision 'exact Git commit' \
  --server-command 'complete launch command and flags' \
  --expected-profile 'model-specific harness profile' \
  --expected-parser 'live native tool parser' \
  --expected-qualification 'versioned exact-build qualification ID' \
  --output reports/benchmarks/personal-intelligence-MODEL.json
```

For an authenticated server, export `RAPID_MLX_API_KEY` before running the
suite. The script sends it as a bearer credential but never records it in argv,
the JSON receipt, or the reproduction command.

The exact-build target matrix and receipts live in
`docs/engineering/performance/2026-09-15-personal-intelligence-top-model-qualification.md`.
Only the complete canonical matrix can set `qualified:true`; subset `--tasks`
runs are diagnostics. Receipts must use a full source SHA and a reconstruction
command that checks out that revision before starting the server.
The suite first verifies that the live model card matches all four expected
identity fields, then verifies every created run returns the same profile and
qualification ID. The qualification ID represents the exact public alias,
backing artifact, parser, and harness admission record; behavior from a
different mounted model therefore cannot certify the requested build.

## Start the server

Agent runs use the MCP servers already configured for `serve`:

```bash
rapid-mlx serve minicpm5-2b-4bit \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$RAPID_API_KEY" \
  --mcp-config ./mcp.json
```

Rapid does not trust a tool name as a safety policy. It therefore requires
approval for every MCP tool unless the operator explicitly declares an exact
namespaced tool as read-only. An operator can separately mark a tool as a
local change so the approval dialog accurately says it changes this Mac rather
than implying it contacts another person or service:

```json
{
  "mcpServers": {
    "files": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/safe/root"],
      "agent_read_only_tools": [
        "files__read_file",
        "files__list_directory"
      ],
      "agent_local_change_tools": [
        "files__write_file",
        "files__create_directory"
      ]
    }
  }
}
```

Do not put a write, delete, shell, messaging, payment, or other consequential
tool in that server's `agent_read_only_tools`. Local changes still pause for
approval; `agent_local_change_tools` changes only the explanation shown to the
user. A tool cannot appear in both lists, and each declaration must use its
own server's namespace. Name patterns are intentionally not trusted: a
connector could call a mutating tool `get_and_delete`, so undeclared tools are
treated as external side effects and always pause for approval. The existing
MCP sandbox still runs immediately before execution and may reject an approved
call.

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

Rapid automatically exposes two local, read-only helpers when tools are
selected automatically:

- `rapid__calculate` evaluates up to 16 basic arithmetic expressions using a
  restricted parser. It cannot execute names, functions, or code.
- `rapid__batch_read_only` runs up to eight independent calls concurrently,
  but only when every nested tool was already classified read-only. A batch
  containing a write or undeclared tool is rejected before anything runs.

These helpers let a compact model read several files and calculate totals in
fewer model turns. They do not bypass MCP schemas, the sandbox, or approval
policy. If `tool_names` is supplied explicitly, include either helper by name
when the task needs it.

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

Use `approved:false` to deny it. Rapid records the action as unexecuted and
finishes with `That action wasn’t approved, so it wasn’t run.` This wording is
scoped to the pending action because earlier approved actions in the same run
may already have completed. Rapid does not ask the model to reinterpret the
refusal, so compact models cannot leak planning text or imply the pending
action happened. A stale or wrong call ID returns HTTP 409 and cannot release
an action.

After `approved:true`, a client-executed run moves to
`awaiting_tool_result` and only then exposes the original arguments. This
prevents a client executor from receiving an actionable side-effect payload
before the approval boundary.

## Client execution for Desktop

Set `execution:"client"` when an authenticated client owns execution. Rapid
still chooses tools from server-owned schemas, validates the model call, and
applies the same approval policy. Desktop selects only enabled names from the
official `web_search`, `browse`, and `weather` catalog; it cannot submit a new
schema or change a risk label. The client reads `pending_action`, validates the
arguments against its matching native schema, runs the existing local adapter,
then returns exactly one result:

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

Desktop sends global and conversation custom instructions through the dedicated
`trusted_instructions` create field; the runtime keeps them in the leading
system message, after its fixed safety/tool policy, with conversation
instructions taking precedence over conflicting global instructions. Memory
and at most the last eight completed user/assistant messages travel separately
through `local_context` as explicitly untrusted quoted data. The server caps
that field at 32,768 characters; Desktop caps the complete serialized context,
including wrappers and separators, at 24,000 Unicode scalars with newest
messages first. Neither transient field is copied into public Agent events.

The harness exposes only tools relevant to the current request. Recall,
writing, explicit no-network requests, and transformation tasks see no
live-data tool; weather tasks see only `weather`; web tasks deterministically
run `web_search`, extract ranked HTTP(S) result lines, then run `browse`. Long
pages follow the tool's `next_offset`, and explicit comparison tasks can read
up to three ranked pages. Explicit URLs remain browseable for transformation
requests such as “summarize this URL”; an explicit offline instruction still
wins. Referential follow-ups such as “open that link” use the newest HTTP(S)
URL in bounded recent context. Terse current-weather requests such as “Paris
weather?” use live weather, while dated forecasts continue through web
evidence. Explicitly time-sensitive questions such as yesterday's game result
also route through current web evidence. Search and browse still execute only
in Desktop, and `browse` retains
its existing cache, SSRF guard, and per-fetch approval. Tool output remains
untrusted data, and the final synthesis turn has no tools visible.

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

- at most 6 configured connector tools plus 2 bounded Rapid host helpers;
- at most 12 tool rounds, followed by one tools-disabled synthesis turn;
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
