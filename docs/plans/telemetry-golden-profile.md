# Telemetry → Golden Profile design doc

Status: **draft** · Author: raullen + Claude · Date: 2026-06-06

This doc covers Phase 2 of the opt-in telemetry pipeline first sketched
in [Issue #236](https://github.com/raullenchai/Rapid-MLX/issues/236).
Phase 1 (consent + redaction + schema) already landed in
`vllm_mlx/telemetry/` and the Cloudflare Worker at
`~/work/Rapid-MLX-telemetry-worker/`. This doc is **only** about wiring
client transport, event call sites, server DNS, and the downstream
Golden Profile aggregation. Re-litigating the consent UX or schema
shape is out of scope.

## 1 · Goals (3 data streams, 1 pipeline)

The user-facing pitch is three categories — but they share the same
schema envelope, the same consent gate, the same Worker, and the same
R2 bucket. They differ only in the `event` discriminator and which
optional payload is populated.

| Stream | `event` value | Optional payload | Feeds |
|---|---|---|---|
| **Performance** — `(model, hardware, batch) → tok/s, ttft` | `request` | `RequestPayload` | Golden Profile speed table |
| **Usage habits** — which models/subcommands/flags are popular | `session_start` / `session_end` | `SessionPayload` | Roadmap prioritization (Top-N alias support, parser coverage) |
| **Errors / crashes** — broken parsers, OOM, tool failures, model-load failures | `error` | `ErrorPayload` | Bug triage, regression watchlist |

What we explicitly do **not** want:
- Prompt content. Ever.
- Generated content. Ever.
- API keys, webhook URLs, file paths.
- Per-event IPs or UAs (Worker invariants pin this).
- Anything that can re-identify a single user across sessions beyond
  the opaque `client_id` UUID the user can rotate at will.

## 2 · Current state (what's already done)

### 2.1 Client (`vllm_mlx/telemetry/`) — Phase 1 complete

| File | Owns |
|---|---|
| `state.py` | `~/.rapid-mlx/telemetry-client-id`, `~/.rapid-mlx/telemetry-consent.yaml`, kill switch precedence (`--no-telemetry` > `RAPID_MLX_TELEMETRY=0` > file > default OFF). No env-var force-on (CI would skew aggregates). |
| `consent.py` | First-run prompt, schema-version-aware re-prompt. |
| `schema.py` | `TelemetryPayload` envelope, `PlatformInfo` / `SessionPayload` / `RequestPayload` / `ErrorPayload` dataclasses, `sample_preview_payload()`. `SCHEMA_VERSION = 1`. |
| `redact.py` | Bucket primitives (`bucket_tokens`, `bucket_ttft_ms`, `bucket_tps`, `bucket_memory_gb`), `normalize_model_path` (passes `org/name`, redacts local paths), `hash_flag_names` (names only, never values), `fingerprint_traceback` (16-hex of `class_name + basename:func:lineno`, no message text, no module path), `platform_info` (chip + memory rounded to GB + OS major.minor + python major.minor). |
| `cli.py:3133` | `rapid-mlx telemetry {status,enable,disable,preview,reset}` subcommand. |

Phase 1 has **no event call sites** — `is_enabled()` exists, but
nothing calls it to actually emit. The package compiles, has tests,
and ships dark.

### 2.2 Server (`~/work/Rapid-MLX-telemetry-worker/`) — Phase 1 complete, deploy blocked

| Component | State |
|---|---|
| Worker handler `src/index.js` (165 LOC) | POST `/v1/events`, validates `schema_version == 1`, body cap 256 KB, batch cap 100 events, 50ms CPU cap, stamps `received_at`. Writes one NDJSON object per batch to R2 key `events/YYYY/MM/DD/HH/<rand12>.ndjson`. |
| Privacy invariants pinned by `test/worker.test.js` | (1) does not read `CF-Connecting-IP` / `X-Forwarded-For` / `X-Real-IP` / UA; (2) does not forward any request header to R2; (3) rejects `schema_version != 1`; (4) "do not log bodies" is code-review-only. |
| `wrangler.toml` | `r2_buckets.binding = "EVENTS"`, `bucket_name = "rapid-mlx-telemetry-events"`, `[observability] enabled = true`. |
| Deploy state | **Blocked**. Per memory `project_telemetry_worker_deploy_blocked.md`: existing OAuth token lacks R2 + Workers scopes. Either re-auth `wrangler login` interactively in user shell, or mint a scoped API token (Workers Scripts:Edit, Account R2:Edit). |

### 2.3 rapidmlx.com infrastructure — already serves *other* workloads

Three Cloudflare-attached surfaces in production today:

- **`rapid-pro.pages.dev`** — Big-AGI vendored static (Pages, no Worker).
- **`rapidserver.quicksilverpro.io`** — Worker `rapidserver` (712 LOC),
  hosts the share WebSocket relay (Durable Object `TunnelHub`) +
  `/tool/*` proxies (weather/wiki/currency/web_search) + `/r/<id>/*`
  reverse-proxy. Custom domain via the Workers Custom Domains API
  (OAuth grants workers-only, no zone:dns:edit needed).
- **`chat.rapidmlx.com`** — static splash + vendored BCG, served via
  cloudflared tunnel from M3 Ultra (NOT a Worker).

**Decision: do NOT fold telemetry into `rapidserver`.** Different
security postures (share-tunnel intentionally sees client IP to
fingerprint abuse; telemetry must never see IP). Different deploy
cadences. Different blast radius if a route is misconfigured. Keep
`rapid-mlx-telemetry` as its own Worker, attached to its own subdomain.

## 3 · Phase 2 client work

### 3.1 New module: `vllm_mlx/telemetry/transport.py`

Single responsibility: take a list of `TelemetryPayload` dicts, ship
them to the Worker, fail silently on error. Nothing about consent or
event construction lives here.

```python
# vllm_mlx/telemetry/transport.py — sketch

DEFAULT_ENDPOINT = "https://telemetry.rapidmlx.com/v1/events"
TIMEOUT_S = 3.0  # short — telemetry must NEVER slow a real request
RETRY_BACKOFFS_S = (0.5, 2.0)  # 3 attempts total, ~2.5s ceiling

def post_batch(events: list[dict], *, endpoint: str = DEFAULT_ENDPOINT) -> bool:
    """Best-effort. Returns True on 2xx. False on anything else.
    NEVER raises. NEVER blocks the foreground for more than TIMEOUT_S * 3."""
    body = json.dumps({"batch": events}).encode("utf-8")
    if len(body) > 200 * 1024:  # Worker caps at 256 KB; leave headroom
        return False
    for attempt, backoff in enumerate((*RETRY_BACKOFFS_S, None)):
        try:
            req = Request(endpoint, data=body, method="POST",
                          headers={"content-type": "application/json"})
            with urlopen(req, timeout=TIMEOUT_S) as resp:
                return 200 <= resp.status < 300
        except (URLError, HTTPError, TimeoutError):
            if backoff is None:
                return False
            time.sleep(backoff)
    return False
```

Notes:
- `urllib.request` (stdlib) over `httpx` / `requests` — no new
  dependency for a telemetry path. We already import stdlib `urllib`
  elsewhere.
- Endpoint overridable via `RAPID_MLX_TELEMETRY_ENDPOINT` env var
  (mirrors PRT debug practice) but **only** when telemetry is otherwise
  enabled. The env var alone never opts a user in.
- HTTPS-only check (mirror `share` PR #504 codex round 4 fix —
  `URLError` is **NOT** a `TimeoutError` subclass; this code already
  catches both explicitly).

### 3.2 New module: `vllm_mlx/telemetry/queue.py`

A bounded in-process queue + flush daemon. Events accumulate locally;
a background thread flushes either when the queue hits N events, when
T seconds elapse, or on graceful shutdown.

```python
# vllm_mlx/telemetry/queue.py — sketch

MAX_QUEUE_LEN = 100   # match Worker batch cap
FLUSH_INTERVAL_S = 60 # idle flush
FLUSH_THRESHOLD = 10  # eager flush when this many pending

class TelemetryQueue:
    """Thread-safe, lossy under back-pressure (drops oldest)."""
    def __init__(self): ...
    def enqueue(self, payload: dict) -> None: ...
    def start_flush_daemon(self) -> None: ...
    def shutdown(self, *, timeout_s: float = 2.0) -> None: ...
```

Invariants:
- **Lossy by design**: queue drops oldest event on overflow. Telemetry
  must never grow unbounded if the network is down.
- **Daemon thread** so SIGTERM does not hang on flush. Final flush is
  scheduled via `atexit` with a 2 s budget.
- **Hot path stays sync-free**: `enqueue()` is a single deque append
  inside one `Lock`. No HTTP, no JSON dump on the request path.
- **Coalesce session events**: emitting `session_start` + immediate
  `session_end` of the same session_id within the same flush window
  collapses to a single `session_summary` event (out of Phase 2 scope
  but worth noting for Phase 3).

### 3.3 Event call sites (the 3 streams wired)

Surgical instrumentation only. Every site touches:
1. `if not telemetry.is_enabled(): return` — no event constructor runs
   when disabled.
2. Wrap event construction in `try/except Exception: pass` — telemetry
   failures must never surface to the user.
3. Use redaction primitives. Do not construct payloads directly.

| Stream | File | Hook | Notes |
|---|---|---|---|
| **session_start** | `vllm_mlx/cli.py` `main()` | After argparse, before subcommand dispatch | Captures subcommand + redacted `flag_names` from `sys.argv`. |
| **session_end** | `vllm_mlx/cli.py` `main()` | `atexit` / `finally` | Carries `duration_seconds`. For `serve`, this fires at server shutdown so `models_loaded` is final. |
| **request** | `vllm_mlx/routes/chat.py` (and embeddings/audio mirrors) | After response sent, before context exit | Bucket ttft, tps, prompt/completion tokens. `tool_call_used` from response inspection. **Skip streaming partials** — emit once per completed request. |
| **error** (model load) | `vllm_mlx/engine/loader.py` exception handler | On any load exception | `category="model_load_failure"`, `phase="startup"`. |
| **error** (oom) | `vllm_mlx/scheduler.py` `MemoryError` handler | On OOM | `category="oom"`, `phase="request"`. |
| **error** (tool parse) | `vllm_mlx/parsers/*.py` failure paths | When a parser falls back to text | `category="tool_parse"`. Carries fingerprint of the parser path, not the offending input. |
| **error** (shutdown traceback) | `vllm_mlx/api/server.py` lifespan exit handler | On exception during shutdown | `category="shutdown_traceback"`, `phase="shutdown"`. |

### 3.4 Threading + lifecycle integration

- `vllm_mlx/cli.py` `main()` instantiates a process-singleton
  `TelemetryQueue` after consent check. The instance is also held by
  `vllm_mlx/api/server.py` `lifespan()` (FastAPI) and surfaced via
  request-state for the route layer.
- One flush daemon per CLI invocation. Daemon thread, `daemon=True`,
  joined with 2 s budget on `atexit`.
- `vllm_mlx/api/server.py` `lifespan()` enqueues `session_start` on
  startup, `session_end` on shutdown; route layer enqueues `request`.

## 4 · Server-side: DNS + Worker deploy

### 4.1 DNS plan

Target hostname: **`telemetry.rapidmlx.com`** (NOT `rapidserver.…` —
see §2.3 reasoning).

Two-zone reality:
- `rapidmlx.com` zone — owned by raullen, hosts Pages + `chat.…`.
- `quicksilverpro.io` zone — Iotex-owned, hosts `rapidserver.…`.

Because the telemetry Worker has zero coupling to the share tunnel,
it can attach to `rapidmlx.com` cleanly:

1. Move the telemetry Worker's `wrangler.toml` to add:
   ```toml
   routes = [
     { pattern = "telemetry.rapidmlx.com", custom_domain = true }
   ]
   ```
2. Once deployed, the Workers Custom Domains API mints the placeholder
   `100::` AAAA record automatically — no `zone:dns:edit` token needed
   (same trick as `rapidserver`).
3. CORS stays off (clients are CLIs, not browsers). If we ever expose
   `/v1/events` to a browser surface (e.g. a debug button in chat.…),
   add an explicit allowlist mirroring the `rapidserver` worker's
   atomic CORS-replace pattern (`gotcha_worker_cors_atomic_replace`).

### 4.2 Token / OAuth unblock

The original blocker is fully captured in memory
`project_telemetry_worker_deploy_blocked.md`. Two paths:

**Path A — interactive re-auth (recommended, low blast radius):**
```bash
# In a regular user terminal, NOT inside claude-code (see memory
# gotcha_wrangler_oauth_claude_code — claude-code's bash strips
# Authorization headers):
cd ~/work/Rapid-MLX-telemetry-worker
npx wrangler login       # browser opens, grants Workers + R2
npx wrangler r2 bucket create rapid-mlx-telemetry-events --location enam
npx wrangler deploy
```

**Path B — long-lived scoped API token:**
Mint at https://dash.cloudflare.com/profile/api-tokens with:
- Account · Workers Scripts · Edit
- Account · Workers R2 Storage · Edit
- Zone · `rapidmlx.com` · Workers Routes · Edit
- Zone · `rapidmlx.com` · DNS · Edit (only if Workers Custom Domains
  fails to mint the placeholder, which would surprise me)

Export `CLOUDFLARE_API_TOKEN=…` and `npx wrangler deploy`.

### 4.3 Operational dashboard

Cloudflare's `[observability] enabled = true` already captures
per-request URL + status + CPU time. No app-level metrics added.

For the data side, point DuckDB at the bucket from a local dev
machine:

```sql
INSTALL httpfs; LOAD httpfs;
SET s3_endpoint = '404b05ab....r2.cloudflarestorage.com';
SET s3_access_key_id = '<R2 key>';
SET s3_secret_access_key = '<R2 secret>';

SELECT event, count(*) AS n
FROM read_json_auto('s3://rapid-mlx-telemetry-events/events/2026/**/*.ndjson')
GROUP BY event;
```

## 5 · Golden Profile aggregation

Where the user's perf data actually goes after R2.

### 5.1 Schema mapping (R2 NDJSON → Golden Profile row)

The `RequestPayload` envelope already carries the right fields:

| `RequestPayload` field | Golden Profile column |
|---|---|
| `model_alias` | `alias` (PK part) |
| (top-level `platform.chip`) | `chip` (PK part) |
| (top-level `platform.memory_gb`) | `memory_gb` |
| `prompt_tokens_bucket` | `prompt_bucket` |
| `completion_tokens_bucket` | `completion_bucket` |
| `ttft_ms_bucket` | `ttft_bucket` |
| `tps_bucket` | `tps_bucket` |
| `tool_call_used` | `tool_call` (0/1) |
| `stream` | `stream` (0/1) |
| `status` | `status` (filter to 200 for the perf table) |
| (top-level `rapid_mlx_version`) | `rapid_mlx_version` (so a regression on a single release is visible) |

Per-row count is the aggregation primary key, not per-event. The whole
point of bucketing is that the (alias, chip, *_bucket, ...) tuple
collapses to a count + percentile target.

### 5.2 Aggregation jobs (where + when)

Three options, picking the cheapest:

| Option | How | Cadence | Cost |
|---|---|---|---|
| **A** — Nightly DuckDB on a dev box | cron + `read_json_auto('s3://…/2026/MM/DD/*.ndjson')` → write parquet to `golden_profile/YYYY-MM-DD.parquet` | nightly | Free (compute on dev box) |
| **B** — CF Workers Cron Triggers | Second Worker reads R2, writes aggregate parquet back to R2 | nightly | Free tier OK, but parquet roundtrip is awkward in JS |
| **C** — Cloudflare D1 | Worker INSERTs aggregated counts into D1 from each batch (no nightly job) | per-batch | D1 free tier limits at 5M reads/day — fine |

**Recommendation: A for MVP**, C for v2 once volume justifies it.
Option A is one bash script + one DuckDB call + one `aws s3 cp`
equivalent — debuggable, no new infra. Option C is the "real-time
dashboard" version once we have enough events to want one.

### 5.3 Publishing the Golden Profile

Once a parquet is built, ship it as a static artifact:

- **`https://golden.rapidmlx.com/profile-v1.parquet`** — static, signed
  by the nightly job, served by Pages or R2 public bucket.
- **`https://golden.rapidmlx.com/profile-v1.json`** — same data, JSON,
  for browser surfaces (chat.rapidmlx.com Big-AGI splash could read
  this to pre-populate "your hardware → recommended alias").
- **Versioning**: `profile-v1` is the schema. Bump to `v2` when the
  column set changes incompatibly.

`rapid-mlx suggest` (future CLI subcommand) reads the JSON, intersects
with `aliases.json`, and emits a ranked list. This is the
"rapid-mlx-flavored whichllm" the user described — but powered by
**real measurements** instead of spec-sheet algebra.

## 6 · Privacy + security posture (must stay true)

These invariants are the user-facing contract. Breaking any of them
breaks the deal we made when we asked them to opt in.

1. **No event before consent.** `is_enabled()` is the only gate. No
   call site constructs a payload without first calling it.
2. **No prompt or generated content, ever.** The schema dataclasses
   are the wire shape — they have no field for free-form text. Code
   review on any schema addition must justify the new field.
3. **No raw model paths.** `normalize_model_path` redacts anything
   that isn't `org/name`. Local checkouts surface as `<local>`.
4. **No flag values, ever.** Only flag names. The `redact.hash_flag_names`
   regex does not capture the value half of `--foo=bar`.
5. **No exception messages, no module paths.** `fingerprint_traceback`
   hashes class name + `basename:func:lineno` only.
6. **No IP, no UA.** Pinned at the Worker layer (vitest tests in
   `Rapid-MLX-telemetry-worker/test/worker.test.js`).
7. **No header forwarding.** Worker never copies a request header
   into the R2 object body.
8. **Opaque rotatable identity.** `client_id` is a local UUID4. User
   can `rm ~/.rapid-mlx/telemetry-client-id` to rotate or replace it
   with the all-zero UUID to anonymize while still contributing.
9. **No env-var force-on.** CI agents cannot silently opt in. This is
   asymmetric with the kill-switch on purpose: hostile defaults
   matter more than hostile sites.

Audit-friendly defaults: `rapid-mlx telemetry preview` already exists
and prints exactly what a future event would look like, so a security
reviewer can grep the binary's actual wire shape without strace.

## 7 · MVP scope (what to ship first)

**Phase 2.0 (1 week):**
- Land `transport.py` + `queue.py` (no event sites yet, no Worker
  deploy yet).
- Add unit tests pinning: silent failure on Worker 5xx, bounded queue,
  daemon thread joins on atexit, lossy under back-pressure.
- Behind `RAPID_MLX_TELEMETRY_DEBUG=1` env: log "would have sent N
  events to <endpoint>" without actually POSTing. Lets us validate
  the queue with zero server dependency.

**Phase 2.1 (1 week):**
- Deploy Worker to `telemetry.rapidmlx.com` (Path A from §4.2).
- Wire `session_start` + `session_end` only — no request events yet.
  Lowest-risk surface to validate the end-to-end pipeline.
- Land `rapid-mlx telemetry status` reporting last-flush success/fail.

**Phase 2.2 (2 weeks):**
- Wire `request` event in `routes/chat.py`.
- Wire `error` events at the 4 sites in §3.3.
- Nightly DuckDB → parquet job (Option A) on a dev box, cron'd.

**Phase 3 (deferred):**
- Publish `golden.rapidmlx.com/profile-v1.{parquet,json}`.
- Build `rapid-mlx suggest` CLI subcommand that reads it.
- Browser-side hook into chat.rapidmlx.com Big-AGI splash to
  recommend an alias based on `WebGPU.getAdapter()` chip hint.

## 8 · Open questions for review

1. **Endpoint hostname.** `telemetry.rapidmlx.com` proposed. Could
   also be `events.rapidmlx.com` or `api.rapidmlx.com/telemetry`.
   First wins on dedicated-purpose clarity.
2. **Flush cadence vs. ingestion cost.** R2 writes are $0.36 per
   million Class A operations. At 10 events/batch, 100K users, 5
   sessions/day = 50 K writes/day = ~$0.018/day. Negligible. Can
   eagerly flush.
3. **Sampling.** Phase 2 ships **no sampling** — every consenting
   user's every event lands. We're at 0 telemetry today; under-counting
   isn't a problem yet. Add sampling when volume justifies, and put
   the sample rate on the wire so aggregations can unbias.
4. **`models_loaded` cardinality bound.** A server with 50 aliases
   would emit a 50-element list per `session_end`. Cap at 32 for now
   (almost always 1–3 in practice; serve+pin is the only multi-load
   surface).
5. **What about `share`?** PR #504 share sessions are interesting
   (which model? which chat frontend? did the tunnel die?). Out of
   Phase 2 scope but `subcommand="share"` already covers the basic
   counter; per-share details land later.
6. **Where does the dev-box cron live?** M3 Ultra is the natural home
   (already running tunnels). One launchd plist + a 30-line bash
   script. Bus factor: doc the recipe in `docs/runbooks/`.

## 9 · Decision log

| Decision | Why |
|---|---|
| Keep telemetry Worker separate from `rapidserver` | Different security posture (IP/UA discipline) + different blast radius |
| `urllib` over `httpx` | No new dependency on the consent-gated path |
| Daemon thread, not asyncio | Works inside both `rapid-mlx serve` (async) and `rapid-mlx chat` (sync) without coupling |
| Lossy queue (drop oldest) | Telemetry must never grow unbounded; a stuck Worker should not crash the CLI |
| Three event types, one envelope | Schema simplicity; one Worker code path; one R2 directory; one aggregation query |
| Bucketed numerics | Soft-fingerprint resistance + cheap aggregation |
| No env-var force-on | CI synthetic skew is worse than slow opt-in |
| Per-request event after response sent | Don't measure ourselves measuring |
| MVP without sampling | Volume too low to need it; add later with sample rate on the wire |

## Appendix A · File layout after Phase 2

```
vllm_mlx/telemetry/
├── __init__.py          # Phase 1, exports
├── consent.py           # Phase 1, first-run prompt
├── schema.py            # Phase 1, wire shape
├── redact.py            # Phase 1, bucket + fingerprint
├── state.py             # Phase 1, consent + client_id
├── transport.py         # Phase 2.0, urllib POST
├── queue.py             # Phase 2.0, bounded queue + flush daemon
└── emit.py              # Phase 2.1+, event constructor helpers
```

`vllm_mlx/cli.py:main()` calls into `telemetry.emit.session_start()` /
`session_end()`. `vllm_mlx/routes/chat.py` calls `emit.request()`.
Parsers + scheduler + loader call `emit.error(...)`.

## Appendix B · Reusable patterns this lands

- `transport.py` — first opt-in HTTPS phone-home in rapid-mlx. The
  retry + silent-fail shape is reusable for any future low-stakes
  outbound (model-popularity beacon, "phone home from share session",
  etc).
- `queue.py` — bounded, lossy, daemon-flushed queue. Same shape would
  work for buffering Prometheus metric pushes if we ever want them.
- DuckDB-on-R2 pattern — same recipe would work for the eval scorecard
  history we already collect in `evals/`. Could merge the dev-box cron.

## Appendix C · How this relates to whichllm (project intel)

Whichllm uses **spec-sheet algebra** for its speed predictions
(bandwidth × bytes-per-weight × efficiency constant). No measured
per-(model, hardware) entries. See `data/gpu.py` for the bandwidth
table source — all theoretical peaks from vendor datasheets.

Golden Profile is the **measured** counterpart. Same use case as
whichllm's speed column ("what should I run?"), entirely different
underlying data. Two integration paths once Golden Profile exists:

1. **Internal**: `rapid-mlx suggest` reads
   `https://golden.rapidmlx.com/profile-v1.json` and ranks
   `aliases.json` entries. No new ranking logic — Golden Profile
   already carries TPS, the ranker is `argmax`.
2. **External**: ship a `--source rapid-mlx-golden` flag for whichllm
   that swaps its Apple-Silicon speed predictions for our real
   measurements. PR upstream once `golden.rapidmlx.com` is live + has
   a month of data.
