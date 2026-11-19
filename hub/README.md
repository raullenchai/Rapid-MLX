# Rapid Hub (MVP)

Rapid Hub is a separate local process that supervises trusted apps and gives them
an authenticated model gateway, a private model-call journal, and a small jobs
registry. Engines remain independent OpenAI-compatible services. Hub never
imports the inference engine or MLX and never manages engine processes.

## Run

Requires Python 3.12+, FastAPI, uvicorn, and httpx. `hub/pyproject.toml` defines the
independent `rapid-hub` distribution and console script. With dependencies already
available, run from the repository root without installing anything:

```sh
PYTHONPATH=hub /private/tmp/rapid-mlx-grokdemo/env/bin/python -m rapid_hub up \
  --engine q35=http://127.0.0.1:8482/v1 \
  --engine embed=http://127.0.0.1:8483/v1 \
  --host 127.0.0.1 --port 4800 \
  --data-dir /private/tmp/rapid-mlx-hub/data --plugins-dir apps

PYTHONPATH=hub /private/tmp/rapid-mlx-grokdemo/env/bin/python -m rapid_hub status \
  --data-dir /private/tmp/rapid-mlx-hub/data
```

After installing the distribution, the equivalent command starts with
`rapid-hub up`. The default data directory is `~/.rapid-hub`. Engine names must be
unique; URLs end in `/v1`. The first advertised chat-capable model is the default.
No inference requests are made to detect capabilities: `text`/`chat` map to chat,
`embedding`/`embeddings` map to embeddings; missing capability information stays
unknown. `/health` supplies the engine's explicit `ready` field.

Use the printed single-use login link to open the admin home page. It exchanges
a nonce for an HttpOnly, SameSite=Strict session cookie and redirects to `/`.
The random admin bearer credential is stored in `admin-token` with mode 0600;
it is never included in the link. Access logging is disabled. Ctrl-C/SIGTERM stops
Hub and plugin groups. A separate plugin guardian handles abrupt Hub parent death.
Engine processes and direct `/v1` clients are unaffected.

## Plugin manifest

Discovery reads exactly `<plugins-dir>/*/rapid-plugin.toml`. All valid manifests
are enabled in this MVP. Invalid manifests are reported in status without taking
down Hub. A missing required capability blocks the plugin; an optional capability
ending in `?` does not block startup. Restart is explicit after a blocked startup.

```toml
id = "research-loop"                # lowercase slug; unique
name = "Research Loop"
version = "1.0.0"
description = "Project research workbench"
hub_api = "^1"                      # supported major: 1
requires = ["models.chat", "models.embeddings?"]

[run]
command = ["{python}", "{plugin_dir}/server.py", "--hub"]

[ui]
kind = "web"
health = "/healthz"                 # cheap GET returning 200

[permissions]
network = true
files = "plugin"
```

Commands are argv arrays, never shell commands. The only substitutions are
`{python}` (Hub's interpreter) and `{plugin_dir}` (absolute plugin directory).
Processes receive `RAPID_HUB_URL`, a scoped `RAPID_HUB_TOKEN`, `RAPID_PLUGIN_ID`,
`RAPID_PLUGIN_HOST`, `RAPID_PLUGIN_PORT`, and `RAPID_PLUGIN_DATA_DIR`. Each plugin
binds its own loopback port and serves its own UI origin. Hub serves no plugin JS.
The free-port reservation has a bind race; a collision enters the restart policy.

Readiness is polled for up to 20 seconds. Failed processes/readiness checks have
five restart attempts with 1, 2, 4, 8, 16 second delays, then become `failed`.
The restart button resets the attempt budget. Status includes the supervised
server PID, guardian PID, state, port, and last log lines. Logs live in
`plugins/<id>/logs/{stdout,stderr}.log`; app data lives in `plugins/<id>/data`.

## Shared services

All `/hub/v1` endpoints require a bearer credential (the home page also uses its
admin cookie). Plugin credentials can use models and manage their own jobs and
journal entries. Admin credentials can inspect all plugins, restart them, and
request job cancellation. Cross-origin browser requests and unexpected Host
headers are rejected; credentials are never forwarded to engines.

| Endpoint | Behavior |
| --- | --- |
| `GET /hub/v1/models` | Merged model catalog, engine readiness, default chat model |
| `GET /hub/v1/status` | Admin engine/plugin states and discovery errors |
| `GET,POST /hub/v1/engines/{engine}/v1/{path}` | OpenAI passthrough; preserves status, Retry-After and SSE delivery |
| `GET /hub/v1/journal?plugin=&project=&limit=50` | Own entries for plugin tokens; admin may filter all entries |
| `POST,PATCH /hub/v1/jobs` | Create/update own job by `id` |
| `GET,PATCH /hub/v1/jobs/{id}` | Read/update own job |
| `GET /hub/v1/jobs` | Own jobs, or all for admin |
| `POST /hub/v1/jobs/{id}/cancel` | Admin sets running job to `cancel_requested` |
| `POST /hub/v1/plugins/{id}/restart` | Admin restarts a plugin |

Model calls accept `X-Rapid-Project` and `X-Rapid-Priority` (`interactive`, the
default, or `background`). Hub forwards these and `X-Rapid-Plugin` to the engine.
Per-engine background concurrency defaults to 1 and can be set with
`--background-concurrency`. Excess background calls wait FIFO. Interactive calls
bypass this admission gate. This does not preempt work already executing inside
an engine or reserve engine GPU capacity.

Journal files are mode-0600 append-only JSONL in
`journal/<plugin>/<UTC-date>.jsonl`. Requests are recorded **before** forwarding,
then a second row with the same `id` records completion. The read API folds those
rows to the latest state. A hard crash leaves a `phase=started` entry. Rows include
the full request, identity/project/engine/path/model/priority, status, latency,
queue wait, usage when supplied, and up to 64 KiB of response bytes (raw SSE for
streams). Requests and journals contain model-visible project text. Jobs persist
by atomic replacement of a private `jobs.json` file. Plugins poll cancellation;
Hub does not directly execute plugin-specific cancellation code.

## Research Loop

The manifest starts `server.py --hub`. This mode discovers chat/embedding models,
routes discussion, PDF metadata, retrieval embeddings and experiment model calls
through the gateway, and tags each call with its project slug. Discussion is
interactive; experiment proposal/repair subprocesses inherit background priority
and the scoped credential through environment variables, never argv. Literature
indexing and retrieval propagate project context through their worker threads.
The existing pure Python local-search imports remain until a Hub retrieval
service replaces them.

Hub mode puts projects under `plugins/research-loop/data/projects`; each project
retains its existing `runs/` history layout. `--data-dir` selects another data
root; a top-level `runs/` directory is also created for future shared run storage.
Standalone mode retains its original flags, ports, paths and endpoint behavior.
Experiment jobs report lifecycle/progress and poll Hub cancellation every second;
cancel creates `STOP` and clears `PAUSE`, just like the existing stop button.
Cancellation remains cooperative and can wait for the current experiment step.

To reuse a project, copy it; never move a live standalone project:

```sh
mkdir -p /private/tmp/rapid-mlx-hub/data/plugins/research-loop/data/projects
cp -R /private/tmp/rapid-mlx-grokdemo/wt/apps/research-loop/projects/grokking \
  /private/tmp/rapid-mlx-hub/data/plugins/research-loop/data/projects/
```

## Security and MVP limits

**Trusted plugins only. No sandbox. No per-plugin environments yet.** Manifest
network/files permissions describe intent and are not OS-enforced. Plugins and
experiment scripts run as the current account, can read account files, and share
Hub's Python interpreter. Common credential environment variables are filtered
from plugin startup; Hub provides no engine credentials. Engine URLs containing
credentials are rejected. Engine credential configuration is not implemented.
Plugin HTTP UIs currently retain Research Loop's local-only access model.

Separate ports provide browser origin isolation, not account/process isolation.
Browser cookies are host-scoped rather than port-scoped: a same-host trusted
plugin can receive the admin session cookie when visited. Host/Origin checks do
not turn localhost into a boundary against malicious native software. Bind Hub
to loopback; there is no remote TLS/session-management product in this MVP.

The background queue and current supervision state are in memory. There are no
queue bounds, journal/log retention, filesystem quotas, per-plugin Python envs,
distributed scheduling, automatic adoption of pre-existing orphan runs, or
cross-process data-directory locking. Run one Hub per data directory. Manually
detached plugin descendants can escape process-group cleanup; Research Loop
explicitly cleans its experiment groups on Hub termination. Jobs survive restart,
but plugins must reconcile stale job states. Journal writes are synchronous;
there is no fsync durability guarantee against machine/power failure. SSE usage
is available only when the engine actually emits a usage chunk.

## Verification

```sh
/private/tmp/rapid-mlx-grokdemo/env/bin/python -m pytest -q hub/tests
cd apps/research-loop
/private/tmp/rapid-mlx-grokdemo/env/bin/python -m pytest -q tests
node tests/test_web.mjs
```

Tests cover manifest errors, import boundaries, JSON/SSE/status passthrough,
scoped auth/journals/jobs, FIFO background admission with interactive bypass,
incremental delivery and disconnect cleanup, readiness/restarts/process groups,
abrupt parent death, model discovery, subprocess headers and cooperative cancel.
They use fake upstreams and ephemeral local ports; no weights or packages needed.
