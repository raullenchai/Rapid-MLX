# OpenAI-Compatible Server

rapid-mlx provides a FastAPI server with full OpenAI API compatibility.
Continuous batching is always on.

## Starting the Server

### Default

```bash
rapid-mlx serve qwen3.5-4b-4bit --port 8000
```

Short aliases (see `rapid-mlx models`) work everywhere a model name is
accepted. Full HuggingFace repo IDs (`mlx-community/...`) work too.

### With Paged Cache

Memory-efficient caching for production / shared system prompts:

```bash
rapid-mlx serve qwen3.5-9b-4bit --port 8000 --use-paged-cache
```

### FLUX.2 Klein full-precision weights

On a Mac with at least 32 GB unified memory, explicitly select the bf16 image
checkpoint with either spelling below. The existing `flux2-klein-4b` alias
remains q4 and no hardware-based switch happens automatically.

```bash
rapid-mlx serve flux2-klein-4b --image-weight-precision bf16
# Equivalent:
rapid-mlx serve flux2-klein-4b-bf16
```

Use `--image-weight-precision q4` to force the compact checkpoint. The option
currently rejects Z-Image, DiffusionGemma, and other diffusion families because
their q4/bf16 end-to-end paths have not completed the same qualification.

## Server Options

The most consequential `rapid-mlx serve` flags. The exhaustive list — every
flag visible in `rapid-mlx serve --help`, grouped by category — lives in the
[CLI reference](../reference/cli.md#rapid-mlx-serve).

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Server port; when omitted, uses the first free port from 8000 through 8009; an explicit port never falls back | First free in 8000–8009 |
| `--host` | Server host (loopback-only by default; pass `0.0.0.0` to expose on LAN) | 127.0.0.1 |
| `--listen-fd` | Adopt a pre-bound listening socket (3-1023) from a supervisor instead of binding; `--host`/`--port` are then ignored. Native MTP, DSpark K4, DFlash, and DDTree reject this option with rc 2 (see the socket-activation section below) | None |
| `--log-level` | Log level for Python logging and uvicorn (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | INFO |
| `--served-model-name` | Model name reported by the API; when unset the `model` argument is used | None |
| `--api-key` | API key for authentication (falls back to `RAPID_MLX_API_KEY`) | None |
| `--cors-origins` | Allowed CORS origins (also via `RAPID_MLX_CORS_ALLOW_ORIGINS`) | `*` (all origins) |
| `--trusted-hosts` | Opt-in Host-header allowlist; non-matching requests get HTTP 400 | None (not enforced) |
| `--rate-limit` | Requests per minute per client (0 = disabled) | 0 |
| `--max-request-bytes` | Max HTTP request body size; oversized requests get HTTP 413 before parsing (0 disables) | 8 MiB (8388608) |
| `--timeout` | Default request timeout in seconds; per-request `timeout: null` or `timeout: 0` uses this value | 1800 |
| `--max-num-seqs` | Max concurrent sequences | 256 |
| `--max-concurrent-requests` | Admission cap on in-flight requests (queued + running); excess requests get HTTP 503 with `Retry-After` | 256 |
| `--prefill-batch-size` | Max prompts prefilled together in one cold wave; lower for better first-token latency under concurrent cold load | 8 |
| `--completion-batch-size` | Completion batch size | 32 |
| `--prefill-step-size` | Chunk size for prompt prefill processing | 2048 |
| `--gpu-memory-utilization` | Fraction of device memory for the Metal allocation limit (0.0-1.0); advanced override of the automatic per-model budget | auto |
| `--context-length` | Operator-selected per-request window in tokens (prompt plus output), up to the model's declared limit; requests still need to fit available memory | auto |

| `--image-weight-precision` | Explicit FLUX.2 Klein weight source (`q4` or `bf16`); no automatic hardware switch | alias default |
| `--kv-cache-dtype` | KV cache dtype (`bf16`, `int8`, `int4`); int8/int4 shrink the KV cache 2x/4x at a long-context decode cost. See the [CLI reference](../reference/cli.md#kv-cache-dtype-and-quantization) for the full quantization family (`--kv-cache-quantization*`, `--kv-cache-turboquant*`). | bf16 |
| `--enable-prefix-cache` / `--disable-prefix-cache` | Toggle prefix caching for repeated prompts | enabled |
| `--prefix-cache-index` | Prefix-cache lookup index: `radix` (token trie) or `hash` (legacy) | radix |
| `--use-paged-cache` | Enable paged KV cache | False |
| `--cache-memory-mb` | Cache memory limit in MB | Auto |
| `--cache-memory-percent` | Fraction of available RAM for cache. When the flag is not passed, the 0.20 default is raised to the agent-session floor (a third of the Metal headroom left after the weights, at most 4 GiB) when that is larger. An explicit value is always kept | 0.20 |
| `--idle-cache-clear-seconds` | Clear reusable KV cache after idle time; model weights remain loaded | Disabled |
| `--max-tokens` | Default max tokens | 32768 |
| `--default-temperature` | Default temperature when not specified (companions: `--default-top-k`, `--default-min-p`, `--default-repetition-penalty`, `--default-presence-penalty`, `--default-frequency-penalty`) | None |
| `--default-top-p` | Default top_p when not specified | None |
| `--stream-interval` | Tokens per stream chunk | 1 |
| `--mcp-config` | Path to MCP config file | None |
| `--reasoning-parser` | Reasoning parser (`qwen3`, `deepseek_r1`, `deepseek_r1_distill`, `deepseek_v4`, `gemma4`, `glm4`, `gpt_oss`, `harmony`, `hy3`/`hy_v3`, `minimax`, `muse`, `ui_tars`, `vibethinker`). Auto-detected from the alias profile; explicit flag overrides. There is no literal `auto` value — omit the flag for auto-detection. | None (auto-detected) |
| `--embedding-model` | Pre-load an embedding model at startup (requires `pip install 'rapid-mlx[embeddings]'`; companions: `--embedding-max-length`, `--embedding-overflow-policy`) | None |
| `--enable-auto-tool-choice` | Enable automatic tool calling | False |
| `--tool-call-parser` | Tool call parser (see [Tool Calling](tool-calling.md)) | None |
| `--mllm` / `--no-mllm` | Force multimodal (vision) loading / force text-only loading, overriding auto-detection | auto-detect |
| `--enable-audio` | Mount `/v1/audio/*` routes on a text-only server (audio-capable models auto-mount them) | False |
| `--disk-stream` | Stream MoE routed-expert weights from disk instead of holding them resident (opt-in; budget via `--disk-stream-cache-gb`) | False |
| `--resident-memory-limit-gb` | Process-wide resident model ceiling in GiB (multi-model serving); LRU idle unpinned models are evicted first; 0 disables (companion: `--resident-model-idle-ttl`) | 0 |
| `--lazy-load` | Keep the endpoint online in `standby` until the configured primary receives its first text-generation request | False |
| `--idle-unload-seconds` | Unload the configured primary after an idle interval, preserving its route and reloading on demand; 0 disables | 0 |
| `--pflash` | PFlash long-prompt prefill compression (`off`, `auto`, `always`); tuning knobs in the [CLI reference](../reference/cli.md#pflash-long-prompt-compression) | `always` for verified aliases, `off` otherwise |

For example, `rapid-mlx serve ling-3.0-tiny-4bit --context-length 65536`
selects a 64K-token request window for Ling. The model's declared maximum
remains unchanged. With no flag, `/v1/models.max_model_len` reports the
current memory-based estimate; with the flag, it reports the selected window.
An explicit window does not reserve RAM or bypass the Metal memory gate, so a
request that cannot fit at the time of admission can still return HTTP 503.

Primary standby applies to Chat Completions, legacy Completions, Responses,
and Anthropic Messages/counting. `/v1/embeddings` has an independent engine
selected by `--embedding-model`; it neither wakes nor uses the primary model.

## API Endpoints

### Endpoint Index

The complete route surface. Endpoints with a detailed section in this guide
are marked; multimodal and MCP surfaces link to their own guides.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion, streaming and non-streaming (detailed below) |
| `/v1/completions` | POST | Text completion (detailed below) |
| `/v1/responses` | POST | OpenAI Responses API (the surface Codex CLI uses) |
| `/v1/messages` | POST | Anthropic Messages API — Claude Code / OpenCode compatible (detailed below) |
| `/v1/messages/count_tokens` | POST | Count input tokens for an Anthropic-format request (detailed below) |
| `/v1/embeddings` | POST | Text embeddings — see the [Embeddings Guide](embeddings.md) |
| `/v1/models` | GET | List available models (detailed below) |
| `/v1/models/{id}` | GET | Metadata for a single model |
| `/v1/models/residency` | GET | Residency status of every model loaded in the process |
| `/v1/models/load` | POST | Load an additional model into the running server |
| `/v1/models/{id}/pin` | PUT | Pin a resident model so it is never auto-evicted |
| `/v1/models/{id}` | DELETE | Unload a resident (non-startup) model |
| `/v1/audio/speech` | POST | Text-to-speech — see the [Audio Guide](audio.md) |
| `/v1/audio/transcriptions` | POST | Speech-to-text — see the [Audio Guide](audio.md) |
| `/v1/audio/translations` | POST | Speech translation to English — see the [Audio Guide](audio.md) |
| `/v1/audio/music` | POST | Music generation — see the [Audio Guide](audio.md) |
| `/v1/audio/voices` | GET | List available TTS voices — see the [Audio Guide](audio.md) |
| `/v1/images/generations` | POST | Image generation |
| `/v1/images/edits` | POST | Image editing |
| `/v1/images/progress` | GET | Denoise progress of the in-flight image render (`step / total`) |
| `/v1/images/cancel` | POST | Stop the in-flight image render at the next denoise step |
| `/v1/videos` | POST / GET | Start a video-generation job / list jobs — see the [Video Generation Guide](video-generation.md) |
| `/v1/videos/capabilities` | GET | Video-lane capability report |
| `/v1/videos/{id}` | GET / DELETE | Video job status / delete a job and its artifact |
| `/v1/videos/{id}/content` | GET | Download the finished video |
| `/v1/mcp/tools` | GET | List tools discovered from the MCP config — see the [MCP Tools Guide](mcp-tools.md) |
| `/v1/mcp/servers` | GET | List configured MCP servers |
| `/v1/mcp/status` | GET | MCP subsystem status (including init errors) |
| `/v1/mcp/execute` | POST | Execute an MCP tool by name |
| `/v1/mcp/reload` | POST | Re-read the MCP config file without a server restart |
| `/v1/status` | GET | Real-time server statistics (detailed below) |
| `/v1/models/activate` | POST | Authenticated pre-warm/qualification of the configured primary model |
| `/v1/cache/stats` | GET | Cache statistics |
| `/v1/cache/clear` | POST | Clear reusable prompt KV state without unloading model weights |
| `/v1/cache/export` | POST | Export the prefix cache to a disk snapshot |
| `/v1/cache/import` | POST | Import a previously exported prefix-cache snapshot |
| `/v1/cache/info` | GET | Read the manifest of an exported cache snapshot |
| `/v1/requests/{id}/cancel` | POST | Cancel an active or queued request by its `chatcmpl-...` id (`DELETE /v1/requests/{id}` is an alias) |
| `/health` | GET | Full health view (queries engine stats on every hit) |
| `/health/ready` | GET | Endpoint readiness; lazy `standby` is ready, lifecycle `error` is 503 |
| `/healthz` | GET | Constant-cost liveness probe (k8s convention; 503 while draining) |
| `/readyz` | GET | Alias for `/health/ready` |
| `/livez` | GET | Process liveness only (does not check model readiness) |
| `/metrics` | GET | Prometheus metrics |
| `/v1/cua/capabilities` | GET | Authenticated computer-use protocol and host availability |
| `/v1/cua/permissions` | GET | Authenticated macOS Accessibility and Screen Recording readiness |
| `/v1/cua/permissions/request` | POST | Request one macOS CUA permission after an explicit user action |
| `/v1/cua/apps` | GET | Authenticated running-app discovery for custom CUA clients |
| `/v1/cua/apps/{app}/windows` | GET | Authenticated window discovery for an app |
| `/v1/cua/observations` | POST | Fresh, authenticated observation of an exact app process and window |
| `/v1/cua/runs` | GET/POST | List or create supervised high-level computer-use runs |
| `/v1/cua/runs/by-request/{id}` | GET | Recover a run created with a client request ID |
| `/v1/cua/runs/{id}` | GET | Poll typed events, terminal state, and any pending approval gate |
| `/v1/cua/runs/{id}/approval` | POST | Resolve the current gate with `{"gate_id": "...", "approved": true|false}` |
| `/v1/cua/runs/{id}/cancel` | POST | Cancel a run |

### Custom computer-use clients

To host only the authenticated Computer Use control plane, without resolving,
downloading, or loading a chat model, start the server in CUA-only mode:

```bash
RAPID_MLX_API_KEY=replace-me rapid-mlx serve --cua-only \
  --host 127.0.0.1 --port 8000 \
  --cors-origins http://127.0.0.1 http://localhost
```

This mode mounts health and `/v1/cua/*` routes only. It does not expose chat,
model, image, audio, or video inference routes, and it rejects model and
residency flags. `GET /health/ready` reports `ready: true`, `model: null`, and
`model_loaded: false` once the listener is ready. Clients should then verify an
authenticated `GET /v1/cua/capabilities` before enabling Computer Use. An API
key is mandatory, including for loopback listeners.

`GET /v1/cua/permissions` is always read-only. A native client may request one
grant in direct response to a permission button by posting
`{"permission":"accessibility"}` or
`{"permission":"screen_recording"}` to
`/v1/cua/permissions/request`. The endpoint requires a loopback connection plus
the same bearer and rate limit as every CUA route, accepts no prompt-control
options, and serializes requests. Its response contains `permission`, `granted`, and a fresh full
`permissions` snapshot. A user denial is a successful response with
`granted:false`; clients must continue to block observation or input until the
read-only status reports the required grant. Discovery and polling never open a
system prompt. The server process must be the signed native Computer Use helper
that owns the macOS privacy grants; a standalone child process does not inherit
another application's grants.

The Desktop sidecar includes the native macOS framework bindings. Standalone
Python installs that use local macOS Computer Use should install the matching
extra:

```bash
pip install 'rapid-mlx[computer-use]'
```

The extra is Darwin-only. Remote or Linux servers remain import-safe and report
the local desktop capability as unavailable.

A client can target one discovered window by taking its opaque `window_id` from
`GET /v1/cua/apps/{app}/windows` and including it in the run request:

```bash
curl -X POST http://127.0.0.1:8000/v1/cua/runs \
  -H "Authorization: Bearer $RAPID_MLX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "app": "Safari",
    "window_id": "cg:12345",
    "client_request_id": "desktop-018f5d2a",
    "goal": "Open the account settings",
    "allowed_domain": "example.com"
  }'
```

The server validates that the window belongs to the resolved app process before
accepting the run, freezes the canonical ID, and returns `window_id` in create,
list, run-view, and event-poll responses. Keep the ID opaque and rediscover
windows before retrying a stopped run.

A run may instead freeze two or three exact PID and window anchors. Each target
has a client-chosen opaque `target_id`; browser targets require a reviewed
`allowed_domain`. The top-level `app` must match the initial PID selector:

```json
{
  "app": "pid:1234",
  "goal": "Read the source, then add a note",
  "client_request_id": "desktop-018f5d2b",
  "initial_target_id": "source",
  "targets": [
    {
      "target_id": "source",
      "app": "pid:1234",
      "pid": 1234,
      "window_id": "cg:12345",
      "allowed_domain": "example.com"
    },
    {
      "target_id": "notes",
      "app": "pid:5678",
      "pid": 5678,
      "window_id": "cg:67890",
      "allowed_domain": ""
    }
  ]
}
```

Multi-target requests cannot include top-level `window_id`, `allowed_domain`,
or `open_url`. The create response and request-ID lookup echo the complete
canonical target list and `active_target_id`; clients should compare both
before accepting control authority. During the run, the planner may request a
no-input `switch_target` to one frozen ID. The server observes only the active
target, emits `target_switched` with the old and new IDs, and includes
`target_id` on subsequent events and approval gates. A process restart, window
replacement, unknown target, domain mismatch, or approval for an obsolete gate
stops or rejects the operation without dispatching input. Target order and
per-target domains are part of the idempotent request identity.

Clients that must recover from a lost create response should send a unique,
opaque `client_request_id` of at most 128 characters. It must be one URL path
segment and cannot contain `/`. The `202` response echoes that ID. Repeating the
same normalized request with the same ID returns the
original `run_id` and does not start another task. Reusing the ID with a
different request returns `409` with code `request_identity_conflict`.

After a timeout, disconnect, or undecodable response, recover the accepted run
before allowing another Start action:

```bash
curl http://127.0.0.1:8000/v1/cua/runs/by-request/desktop-018f5d2a \
  -H "Authorization: Bearer $RAPID_MLX_API_KEY"
```

The lookup returns the create-response shape (`run_id`, current `status`,
`window_id`, `client_request_id`, `targets`, and `active_target_id`). An unknown or expired ID returns typed
`404` code `request_identity_not_found`. Request IDs and runs are held only in
the server process, expire together under the 100-run retention limit, and do
not survive a server restart. An ID no longer present in that registry has no
continuing idempotency guarantee; generate a new ID only when starting a new
task.

Selected-window runs fail closed if the app identity changes or the window is
closed, replaced, moved, or resized. They also stop if the planned control
changes before input is dispatched. Domain-restricted browser runs stop when a
trusted URL cannot be tied unambiguously to the selected browser process,
including when multiple browser processes expose the same application bundle.
`open_url` cannot be combined with `window_id`, because opening a URL can change
which window is targeted.

To render a selected window without starting a run, send the exact app identity,
PID, and opaque window ID returned by discovery. Observations always bypass the
snapshot cache and do not activate the app:

```bash
curl -X POST http://127.0.0.1:8000/v1/cua/observations \
  -H "Authorization: Bearer $RAPID_MLX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "app": "com.apple.Safari",
    "pid": 1234,
    "window_id": "cg:12345",
    "screenshot": false
  }'
```

The response contains `snapshot_id`, `observed_at`, canonical app identity,
window identity and geometry, coordinate space, typed accessibility elements,
element count, and truncation status. It deliberately omits the backend's raw
tree text and redacts secure-text-field labels. `screenshot` defaults to
`false` and the response image is `null`.

Accessibility permission is required for every observation. PNG output also
requires Screen Recording permission, request field `"screenshot": true`, and
the server opt-in `RAPID_MLX_CUA_EXPOSE_SCREENSHOTS=1`. PNGs larger than 4 MiB
are rejected before base64 encoding. Success and typed error responses use
`Cache-Control: no-store` and `Pragma: no-cache`; clients should not persist
observations that may contain private UI labels or pixels.

For lazy or idle-unload deployments, `/metrics` always exposes primary-model
residency and lifecycle series even while the engine is in standby:
`rapid_mlx_model_loaded`, the one-hot `rapid_mlx_model_lifecycle_state`, load
attempt/failure counters, the most recent load duration, and successful unloads
by reason. These process-local series make `ready but cold` distinguishable from
`loaded and ready` without probing an inference route. See the
[headless macOS service guide](headless-macos-service.md) for the complete metric
names and status output contract. If the lifecycle snapshot itself is
temporarily unavailable, the state is `unknown` and its numeric samples are
`NaN` rather than false zeroes.

The `/v1/audio/*` routes are mounted when the loaded model is audio-capable
or `--enable-audio` is passed; on a plain text-only server they return 404.
Image and video routes are always mounted and answer with a structured 409
when no image/video model is loaded.

### Chat Completions

```bash
POST /v1/chat/completions
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Non-streaming
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

# Streaming
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Completions

```bash
POST /v1/completions
```

```python
response = client.completions.create(
    model="default",
    prompt="The capital of France is",
    max_tokens=50
)
```

### Models

```bash
GET /v1/models
```

Returns available models.

### Embeddings

```bash
POST /v1/embeddings
```

```python
response = client.embeddings.create(
    model="mlx-community/multilingual-e5-small-mlx",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions
```

See [Embeddings Guide](embeddings.md) for details.

### Health Check

```bash
GET /health
```

Returns server status.

### Anthropic Messages API

```bash
POST /v1/messages
```

Anthropic-compatible endpoint that allows tools like Claude Code and OpenCode to connect directly to rapid-mlx. Internally it translates Anthropic requests to OpenAI format, runs inference through the engine, and converts the response back to Anthropic format.

Capabilities:
- Non-streaming and streaming responses (SSE)
- System messages (plain string or list of content blocks)
- Multi-turn conversations with user and assistant messages
- Tool calling with `tool_use` / `tool_result` content blocks
- Token counting for budget tracking
- Multimodal content (images via `source` blocks)
- Client disconnect detection (returns HTTP 499)
- Automatic special token filtering in streamed output

#### Non-streaming

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key="not-needed")

response = client.messages.create(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
# Response includes: response.id, response.model, response.stop_reason,
# response.usage.input_tokens, response.usage.output_tokens
```

#### Streaming

Streaming follows the Anthropic SSE event protocol. Events are emitted in this order:
`message_start` -> `content_block_start` -> `content_block_delta` (repeated) -> `content_block_stop` -> `message_delta` -> `message_stop`

```python
with client.messages.stream(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

#### System messages

System messages can be a plain string or a list of content blocks:

```python
# Plain string
response = client.messages.create(
    model="default",
    max_tokens=256,
    system="You are a helpful coding assistant.",
    messages=[{"role": "user", "content": "Write a hello world in Python"}]
)

# List of content blocks
response = client.messages.create(
    model="default",
    max_tokens=256,
    system=[
        {"type": "text", "text": "You are a helpful assistant."},
        {"type": "text", "text": "Be concise in your answers."},
    ],
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
```

#### Tool calling

Define tools with `name`, `description`, and `input_schema`. The model returns `tool_use` content blocks when it wants to call a tool. Send results back as `tool_result` blocks.

```python
# Step 1: Send request with tools
response = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }]
)

# Step 2: Check if model wants to use tools
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}, Input: {block.input}, ID: {block.id}")
        # response.stop_reason will be "tool_use"

# Step 3: Send tool result back
response = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "Sunny, 22C"
            }
        ]}
    ],
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }]
)
print(response.content[0].text)  # "The weather in Paris is sunny, 22C."
```

Tool choice modes:

| `tool_choice` | Behavior |
|---------------|----------|
| `{"type": "auto"}` | Model decides whether to call tools (default) |
| `{"type": "any"}` | Model must call at least one tool |
| `{"type": "tool", "name": "get_weather"}` | Model must call the specified tool |
| `{"type": "none"}` | Model will not call any tools |

#### Multi-turn conversations

```python
messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What's my name?"},
]

response = client.messages.create(
    model="default",
    max_tokens=100,
    messages=messages
)
```

#### Token counting

```bash
POST /v1/messages/count_tokens
```

Counts input tokens for an Anthropic request using the model's tokenizer. Useful for budget tracking before sending a request. Counts tokens from system messages, conversation messages, tool_use inputs, tool_result content, and tool definitions (name, description, input_schema).

```python
import requests

resp = requests.post("http://localhost:8000/v1/messages/count_tokens", json={
    "model": "default",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "system": "You are helpful.",
    "tools": [{
        "name": "search",
        "description": "Search the web",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}
    }]
})
print(resp.json())  # {"input_tokens": 42}
```

#### curl examples

Non-streaming:

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Streaming:

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 256,
    "stream": true,
    "messages": [{"role": "user", "content": "Tell me a joke"}]
  }'
```

Token counting:

```bash
curl http://localhost:8000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
# {"input_tokens": 12}
```

#### Request fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | - | Model name (use `"default"` for the loaded model) |
| `messages` | list | yes | - | Conversation messages with `role` and `content` |
| `max_tokens` | int | yes | - | Maximum number of tokens to generate |
| `system` | string or list | no | null | System prompt (string or list of `{"type": "text", "text": "..."}` blocks) |
| `stream` | bool | no | false | Enable SSE streaming |
| `temperature` | float | no | *(resolved — see below)* | Sampling temperature (0.0 = deterministic, 1.0 = creative). Must be in `[0, 1]` — out-of-range values are rejected with HTTP 422 |
| `top_p` | float | no | *(resolved — see below)* | Nucleus sampling threshold. Must be in `(0, 1]` — out-of-range values are rejected with HTTP 422 |
| `top_k` | int | no | null | Top-k sampling |
| `stop_sequences` | list | no | null | Sequences that stop generation |
| `tools` | list | no | null | Tool definitions with `name`, `description`, `input_schema` |
| `tool_choice` | dict | no | null | Tool selection mode (`auto`, `any`, `tool`, `none`) |
| `metadata` | dict | no | null | Arbitrary metadata (passed through, not used by server) |

When `temperature` / `top_p` are omitted, the server resolves them through a
cascade — first value set wins:

1. the request field,
2. the CLI overrides (`--default-temperature` / `--default-top-p`),
3. the alias profile's `recommended_sampling`,
4. the model's `generation_config.json`,
5. last-resort fallbacks `0.7` (temperature) / `0.9` (top_p).

Independently of the cascade, this surface enforces the Anthropic spec ranges:
`temperature` must be in `[0, 1]` and `top_p` in `(0, 1]`; violations return
HTTP 422 before any inference runs.

#### Response format

Non-streaming response:

```json
{
  "id": "msg_abc123...",
  "type": "message",
  "role": "assistant",
  "model": "default",
  "content": [
    {"type": "text", "text": "Hello! How can I help?"}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 8
  }
}
```

When tools are called, `content` includes `tool_use` blocks and `stop_reason` is `"tool_use"`:

```json
{
  "content": [
    {"type": "text", "text": "Let me check the weather."},
    {
      "type": "tool_use",
      "id": "call_abc123",
      "name": "get_weather",
      "input": {"city": "Paris"}
    }
  ],
  "stop_reason": "tool_use"
}
```

Stop reasons:

| `stop_reason` | Meaning |
|---------------|---------|
| `end_turn` | Model finished naturally |
| `tool_use` | Model wants to call a tool |
| `max_tokens` | Hit the `max_tokens` limit |
| `stop_sequence` | A user-supplied `stop_sequences` entry matched; the matched string is returned in the response's `stop_sequence` field (which is `null` for every other stop reason) |

#### Using with Claude Code

Point Claude Code directly at your rapid-mlx server:

```bash
# Start the server
rapid-mlx serve mlx-community/Qwen3-Coder-Next-235B-A22B-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# In another terminal, configure Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

### Server Status

```bash
GET /v1/status
```

Real-time monitoring endpoint that returns server-wide statistics and per-request details. Useful for debugging performance, tracking cache efficiency, and monitoring Metal GPU memory.

```bash
curl -s http://localhost:8000/v1/status | python -m json.tool
```

Example response:

```json
{
  "status": "generating",
  "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
  "uptime_s": 342.5,
  "steps_executed": 1247,
  "num_running": 1,
  "num_waiting": 0,
  "total_requests_processed": 15,
  "total_prompt_tokens": 28450,
  "total_completion_tokens": 3200,
  "generation_tps": 45.2,
  "prompt_tps": 812.0,
  "adaptive_prefill": {
    "chunk_size": 2048,
    "protected_chunks": 0,
    "reduced_chunks": 0
  },
  "idle_cache_clear": {
    "enabled": false,
    "seconds": 0,
    "clear_count": 0,
    "last_clear_at": null
  },
  "metal": {
    "active_memory_gb": 5.2,
    "peak_memory_gb": 8.1,
    "cache_memory_gb": 2.3
  },
  "cache": {
    "entries": 5,
    "hit_rate": 0.87,
    "memory_mb": 2350
  },
  "requests": [
    {
      "request_id": "req_abc123",
      "status": "running",
      "phase": "generation",
      "elapsed_s": 3.42,
      "prompt_tokens": 1850,
      "completion_tokens": 85,
      "max_tokens": 256,
      "progress": 0.332,
      "tokens_per_second": 45.2,
      "ttft_s": 0.8,
      "cache_hit_type": "prefix",
      "cached_tokens": 1200
    }
  ]
}
```

Response fields:

| Field | Description |
|-------|-------------|
| `status` | Server state: `generating` (at least one request in flight), `idle` (model loaded, nothing running), or `not_loaded` (no engine yet) |
| `model` | Name of the loaded model |
| `uptime_s` | Seconds since the server started |
| `steps_executed` | Total inference steps executed |
| `num_running` | Number of requests currently generating tokens |
| `num_waiting` | Number of requests queued for prefill |
| `total_requests_processed` | Total requests completed since startup |
| `total_prompt_tokens` | Total prompt tokens processed since startup |
| `total_completion_tokens` | Total completion tokens generated since startup |
| `generation_tps` | Current aggregate decode throughput (tokens/s; `0.0` when idle) |
| `prompt_tps` | Current aggregate prefill throughput (tokens/s; `0.0` when idle) |
| `adaptive_prefill` | Adaptive prefill state: `chunk_size`, `protected_chunks`, `reduced_chunks` |
| `idle_cache_clear` | Idle cache-clear supervisor state: `enabled`, `seconds`, `clear_count`, `last_clear_at` |
| `metal.active_memory_gb` | Current Metal GPU memory in use (GB) |
| `metal.peak_memory_gb` | Peak Metal GPU memory usage (GB) |
| `metal.cache_memory_gb` | Metal cache memory usage (GB) |
| `cache` | Cache statistics; the exact keys vary by cache backend, and it is `{"enabled": false}` when the prefix cache is disabled |
| `requests` | List of active requests with per-request details |

Per-request fields in `requests`:

| Field | Description |
|-------|-------------|
| `request_id` | Unique request identifier |
| `status` | `waiting` (queued) or `running` |
| `phase` | Current phase: `queued`, `prefill`, or `generation` |
| `elapsed_s` | Seconds since the request arrived |
| `prompt_tokens` | Prompt tokens for this request |
| `completion_tokens` | Tokens generated so far |
| `max_tokens` | Maximum tokens requested |
| `progress` | `completion_tokens / max_tokens` (0.0 to 1.0) |
| `tokens_per_second` | Generation throughput for this request (`null` until the first token) |
| `ttft_s` | Time to first token in seconds (`null` until the first token) |
| `cache_hit_type` | Cache match type: `exact`, `prefix`, `supersequence`, `lcp`, or `miss` |
| `cached_tokens` | Number of tokens served from cache |

## Tool Calling

Enable OpenAI-compatible tool calling with `--enable-auto-tool-choice`:

```bash
rapid-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

Use the `--tool-call-parser` option to select the parser for your model:

| Parser | Models |
|--------|--------|
| `auto` | Auto-detect (tries all parsers) |
| `mistral` | Mistral, Devstral |
| `qwen` | Qwen, Qwen3 |
| `llama` | Llama 3.x, 4.x |
| `hermes` | Hermes, NousResearch |
| `deepseek` | DeepSeek V3, R1 |
| `kimi` | Kimi K2, Moonshot |
| `granite` | IBM Granite 3.x, 4.x |
| `nemotron` | NVIDIA Nemotron |
| `xlam` | Salesforce xLAM |
| `functionary` | MeetKai Functionary |
| `glm47` | GLM-4.7, GLM-4.7-Flash |

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
)

if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"{tc.function.name}: {tc.function.arguments}")
```

See [Tool Calling Guide](tool-calling.md) for full documentation.

## Reasoning Models

For models that show their thinking process (Qwen3, DeepSeek-R1), use `--reasoning-parser` to separate reasoning from the final answer:

```bash
# Qwen3 models
rapid-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# DeepSeek-R1 models
rapid-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

The API response includes a `reasoning` field with the model's thought process:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)

print(response.choices[0].message.reasoning)  # Step-by-step thinking
print(response.choices[0].message.content)    # Final answer
```

For streaming, reasoning chunks arrive first, followed by content chunks:

```python
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.reasoning:
        print(f"[Thinking] {delta.reasoning}")
    if delta.content:
        print(delta.content, end="")
```

See [Reasoning Models Guide](reasoning.md) for full details.

## Structured Output (JSON Mode)

Force the model to return valid JSON using `response_format`:

### JSON Object Mode

Returns any valid JSON:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={"type": "json_object"}
)
# Output: {"colors": ["red", "blue", "green"]}
```

### JSON Schema Mode

Returns JSON matching a specific schema:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "colors",
            "schema": {
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["colors"]
            }
        }
    }
)
# Output validated against schema
data = json.loads(response.choices[0].message.content)
assert "colors" in data
```

### Curl Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "List 3 colors"}],
    "response_format": {"type": "json_object"}
  }'
```

## Curl Examples

### Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Streaming Configuration

Control streaming behavior with `--stream-interval`:

| Value | Behavior |
|-------|----------|
| `1` (default) | Send every token immediately |
| `2-5` | Batch tokens before sending |
| `10+` | Maximum throughput, chunkier output |

```bash
# Smooth streaming
rapid-mlx serve model --stream-interval 1

# Batched streaming (better for high-latency networks)
rapid-mlx serve model --stream-interval 5
```

## Open WebUI Integration

```bash
# 1. Start rapid-mlx server
rapid-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# 2. Start Open WebUI
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main

# 3. Open http://localhost:3000
```

## Production Deployment

### With systemd

Create `/etc/systemd/system/rapid-mlx.service`:

```ini
[Unit]
Description=Rapid-MLX Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/rapid-mlx serve qwen3.5-27b-4bit \
  --use-paged-cache --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable rapid-mlx
sudo systemctl start rapid-mlx
```

### Authentication and bind→auth ordering

When `--api-key` (or the `RAPID_MLX_API_KEY` env var) is set, every
request to the OpenAI-style routes (`/v1/chat/completions`,
`/v1/embeddings`, `/v1/audio/*`, `/v1/models`, ...) must carry a valid
`Authorization: Bearer <key>` header — anonymous requests get `401`.

The auth check is wired via FastAPI route dependencies at app
construction time, **before** uvicorn binds the listening socket.
There is no window where the port is accepting connections but the
auth dependency has not yet been registered. A regression test
(`tests/test_server_auth_ordering.py`) pins this invariant so a
future refactor can't silently reopen it.

### Socket activation (`--listen-fd`) for strongest guarantee

On a multi-tenant box, the strongest closure of the bind→auth race is
to let an external supervisor (launchd, systemd, or a parent process)
bind the listening socket and validate the auth secret **before**
`execve`-ing into `rapid-mlx`. That way the only process holding the
fd at any point is one with auth in place.

`rapid-mlx serve <alias> --listen-fd N` adopts the inherited fd
instead of binding fresh. `--host` and `--port` are ignored when
`--listen-fd` is set. Native MTP, DSpark K4, DFlash, and DDTree do not
support inherited listeners and reject `--listen-fd` with rc 2 before
model loading.

Example (parent-process style, mirroring `LISTEN_FDS=1` conventions):

```python
import os
import socket

# Supervisor binds 127.0.0.1:8000 and validates the auth secret.
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 8000))
s.listen(128)

# Move the listening socket to fd 3 (the systemd / launchd convention
# for the first inherited socket). Order matters:
#   1. ``dup2(src, 3)`` clones src onto fd 3 (and clears CLOEXEC on 3).
#      When ``s.fileno() == 3`` already, ``dup2`` is a no-op.
#   2. Mark ONLY fd 3 inheritable — the child should see the listener
#      via fd 3 and nothing else.
#   3. Close the original fd ONLY when it isn't already 3, otherwise
#      we'd close the inherited fd out from under ``execvpe``.
src_fd = s.fileno()
os.dup2(src_fd, 3)
os.set_inheritable(3, True)
if src_fd != 3:
    s.close()
os.execvpe(
    "rapid-mlx",
    [
        "rapid-mlx", "serve", "qwen3.5-4b-4bit",
        "--api-key", os.environ["RAPID_MLX_API_KEY"],
        "--listen-fd", "3",
    ],
    {**os.environ, "LISTEN_FDS": "1"},
)
```

Validation: `--listen-fd` accepts integers in `[3, 1023]`. Stdio fds
(0/1/2), negatives, and out-of-range values are rejected with `rc=2`
at the argparse layer.

### Recommended Settings

For production with 50+ concurrent users:

```bash
rapid-mlx serve qwen3.5-27b-4bit \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --port 8000
```
