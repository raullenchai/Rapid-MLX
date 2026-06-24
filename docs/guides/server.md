# OpenAI-Compatible Server

rapid-mlx provides a FastAPI server with full OpenAI API compatibility.
Continuous batching is on by default; `--continuous-batching` is accepted as
a no-op for back-compat.

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

## Server Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Server port | 8000 |
| `--host` | Server host (loopback-only by default; pass `0.0.0.0` to expose on LAN) | 127.0.0.1 |
| `--api-key` | API key for authentication | None |
| `--rate-limit` | Requests per minute per client (0 = disabled) | 0 |
| `--timeout` | Request timeout in seconds | 300 |
| `--continuous-batching` | Accepted for back-compat (continuous batching is always on) | on |
| `--use-paged-cache` | Enable paged KV cache | False |
| `--cache-memory-mb` | Cache memory limit in MB | Auto |
| `--cache-memory-percent` | Fraction of RAM for cache | 0.20 |
| `--max-tokens` | Default max tokens | 32768 |
| `--default-temperature` | Default temperature when not specified | None |
| `--default-top-p` | Default top_p when not specified | None |
| `--stream-interval` | Tokens per stream chunk | 1 |
| `--mcp-config` | Path to MCP config file | None |
| `--reasoning-parser` | Reasoning parser (`gemma4`, `qwen3`, `deepseek_r1`, `glm4`, `gpt_oss`, `harmony`, `minimax`). Auto-detected; explicit flag overrides. | auto |
| `--embedding-model` | Pre-load an embedding model at startup (requires `pip install 'rapid-mlx[embeddings]'`) | None |
| `--enable-auto-tool-choice` | Enable automatic tool calling | False |
| `--tool-call-parser` | Tool call parser (see [Tool Calling](tool-calling.md)) | None |

## API Endpoints

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
| `temperature` | float | no | 0.7 | Sampling temperature (0.0 = deterministic, 1.0 = creative) |
| `top_p` | float | no | 0.9 | Nucleus sampling threshold |
| `top_k` | int | no | null | Top-k sampling |
| `stop_sequences` | list | no | null | Sequences that stop generation |
| `tools` | list | no | null | Tool definitions with `name`, `description`, `input_schema` |
| `tool_choice` | dict | no | null | Tool selection mode (`auto`, `any`, `tool`, `none`) |
| `metadata` | dict | no | null | Arbitrary metadata (passed through, not used by server) |

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

#### Using with Claude Code

Point Claude Code directly at your rapid-mlx server:

```bash
# Start the server
rapid-mlx serve mlx-community/Qwen3-Coder-Next-235B-A22B-4bit \
  --continuous-batching \
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
  "status": "running",
  "model": "mlx-community/Qwen3-8B-4bit",
  "uptime_s": 342.5,
  "steps_executed": 1247,
  "num_running": 1,
  "num_waiting": 0,
  "total_requests_processed": 15,
  "total_prompt_tokens": 28450,
  "total_completion_tokens": 3200,
  "metal": {
    "active_memory_gb": 5.2,
    "peak_memory_gb": 8.1,
    "cache_memory_gb": 2.3
  },
  "cache": {
    "type": "memory_aware_cache",
    "entries": 5,
    "hit_rate": 0.87,
    "memory_mb": 2350
  },
  "requests": [
    {
      "request_id": "req_abc123",
      "phase": "generation",
      "tokens_per_second": 45.2,
      "ttft_s": 0.8,
      "progress": 0.35,
      "cache_hit_type": "prefix",
      "cached_tokens": 1200,
      "generated_tokens": 85,
      "max_tokens": 256
    }
  ]
}
```

Response fields:

| Field | Description |
|-------|-------------|
| `status` | Server state: `running`, `stopped`, or `not_loaded` |
| `model` | Name of the loaded model |
| `uptime_s` | Seconds since the server started |
| `steps_executed` | Total inference steps executed |
| `num_running` | Number of requests currently generating tokens |
| `num_waiting` | Number of requests queued for prefill |
| `total_requests_processed` | Total requests completed since startup |
| `total_prompt_tokens` | Total prompt tokens processed since startup |
| `total_completion_tokens` | Total completion tokens generated since startup |
| `metal.active_memory_gb` | Current Metal GPU memory in use (GB) |
| `metal.peak_memory_gb` | Peak Metal GPU memory usage (GB) |
| `metal.cache_memory_gb` | Metal cache memory usage (GB) |
| `cache` | Cache statistics (type, entries, hit rate, memory usage) |
| `requests` | List of active requests with per-request details |

Per-request fields in `requests`:

| Field | Description |
|-------|-------------|
| `request_id` | Unique request identifier |
| `phase` | Current phase: `queued`, `prefill`, or `generation` |
| `tokens_per_second` | Generation throughput for this request |
| `ttft_s` | Time to first token (seconds) |
| `progress` | Completion percentage (0.0 to 1.0) |
| `cache_hit_type` | Cache match type: `exact`, `prefix`, `supersequence`, `lcp`, or `miss` |
| `cached_tokens` | Number of tokens served from cache |
| `generated_tokens` | Tokens generated so far |
| `max_tokens` | Maximum tokens requested |

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
rapid-mlx serve model --continuous-batching --stream-interval 1

# Batched streaming (better for high-latency networks)
rapid-mlx serve model --continuous-batching --stream-interval 5
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
Description=vLLM-MLX Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/rapid-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching --use-paged-cache --port 8000
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
`--listen-fd` is set.

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
rapid-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --port 8000
```
