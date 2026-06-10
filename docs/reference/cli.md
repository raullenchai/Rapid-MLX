# CLI Reference

## Commands Overview

| Command | Description |
|---------|-------------|
| `rapid-mlx serve` | Start OpenAI-compatible server |
| `rapid-mlx chat` | Interactive chat REPL with a model |
| `rapid-mlx bench` | Run performance benchmarks |
| `rapid-mlx models` | List available model aliases |
| `rapid-mlx info` | Show the per-model profile for an alias or repo |
| `rapid-mlx pull` | Download a model into the HuggingFace cache |
| `rapid-mlx rm` | Remove a cached model |
| `rapid-mlx ps` | List running rapid-mlx servers |
| `rapid-mlx agents` | List, configure, and test agent integrations |
| `rapid-mlx doctor` | Run self-diagnostic / regression harness |
| `rapid-mlx telemetry` | Manage anonymous usage telemetry (opt-in) |
| `rapid-mlx upgrade` | Upgrade rapid-mlx (brew / pip / install.sh) |
| `rapid-mlx version` | Show version number |
| `rapid-mlx help <cmd>` | Show help for a subcommand |

Run `rapid-mlx <cmd> --help` for the full flag list of any subcommand.

## `rapid-mlx serve`

Start the OpenAI-compatible API server.

### Usage

```bash
rapid-mlx serve <model> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Server port | 8000 |
| `--host` | Server host | 0.0.0.0 |
| `--api-key` | API key for authentication | None |
| `--rate-limit` | Requests per minute per client (0 = disabled) | 0 |
| `--timeout` | Request timeout in seconds | 300 |
| `--continuous-batching` | Enable batching for multi-user | False |
| `--cache-memory-mb` | Cache memory limit in MB | Auto |
| `--cache-memory-percent` | Fraction of RAM for cache | 0.20 |
| `--no-memory-aware-cache` | Use legacy entry-count cache | False |
| `--use-paged-cache` | Enable paged KV cache | False |
| `--max-tokens` | Default max tokens | 32768 |
| `--stream-interval` | Tokens per stream chunk | 1 |
| `--mcp-config` | Path to MCP config file | None |
| `--paged-cache-block-size` | Tokens per cache block | 64 |
| `--max-cache-blocks` | Maximum cache blocks | 1000 |
| `--max-num-seqs` | Max concurrent sequences | 256 |
| `--gpu-memory-utilization` | Fraction of device memory for Metal allocation limit (0.0-1.0) | 0.90 |
| `--default-temperature` | Default temperature when not specified in request | None |
| `--default-top-p` | Default top_p when not specified in request | None |
| `--reasoning-parser` | Reasoning parser (`gemma4`, `qwen3`, `deepseek_r1`, `glm4`, `gpt_oss`, `harmony`, `minimax`). Auto-detected; explicit flag overrides. | auto |
| `--embedding-model` | Pre-load an embedding model at startup | None |
| `--enable-auto-tool-choice` | Enable automatic tool calling | False |
| `--tool-call-parser` | Tool call parser (e.g. `hermes`, `llama`, `deepseek`, `deepseek_v31`, `glm47`, `gemma4`, `minimax`, `kimi`, `harmony`, `qwen3_coder_xml`). Auto-detected from the model name; explicit flag overrides. | auto |

### Examples

```bash
# Default — continuous batching is on by default; short aliases work
rapid-mlx serve qwen3.5-4b-4bit

# A larger general-purpose model (5 GB)
rapid-mlx serve qwen3.5-9b-4bit --port 8000

# Paged KV cache (memory-efficient prefix sharing)
rapid-mlx serve qwen3.5-9b-4bit --use-paged-cache --port 8000

# With MCP tools
rapid-mlx serve qwen3.5-9b-4bit --mcp-config mcp.json

# Multimodal (vision) model — requires the [vision] extra
rapid-mlx serve gemma-4-26b-4bit --mllm

# Reasoning model — parser is auto-detected, but you can pin it
rapid-mlx serve qwen3.5-9b-4bit --reasoning-parser qwen3

# DeepSeek reasoning model
rapid-mlx serve deepseek-r1-8b-4bit --reasoning-parser deepseek_r1

# Tool calling with Mistral/Devstral
rapid-mlx serve devstral-24b-4bit --enable-auto-tool-choice --tool-call-parser hermes

# DFlash speculative decoding (single-user, single supported alias)
rapid-mlx serve qwen3.5-27b-8bit --enable-dflash --port 8000

# API key authentication
rapid-mlx serve qwen3.5-9b-4bit --api-key your-secret-key

# Production setup with security options
rapid-mlx serve qwen3.5-9b-4bit \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120
```

### Security

When `--api-key` is set, protected API routes require the
`Authorization: Bearer <api-key>` header. Anthropic-compatible routes
(`/v1/messages` and `/v1/messages/count_tokens`) also accept
`x-api-key: <api-key>` for SDK compatibility; if both headers are sent, both
must match.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key"  # Must match --api-key
)
```

Or with curl:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-secret-key"
```

## `rapid-mlx bench`

Run a built-in performance benchmark against a model.

### Usage

```bash
rapid-mlx bench <model> [options]
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `<model>` | Model alias (e.g. `qwen3.5-4b-4bit`) or HF repo (positional) | *(required)* |
| `--num-prompts` | Number of prompts | 5 |
| `--max-tokens` | Max tokens per prompt | 256 |
| `--enable-prefix-cache` / `--disable-prefix-cache` | Toggle prefix caching | enabled |
| `--use-paged-cache` | Use paged KV cache layout | off |
| `--kv-cache-quantization` | Quantize prefix cache entries | off |

Run `rapid-mlx bench --help` for the full list (memory limits, batch sizes, etc.).

### Examples

```bash
# Quick LLM benchmark using a short alias
rapid-mlx bench qwen3.5-4b-4bit

# Bench a vision-language model by full HF repo
rapid-mlx bench mlx-community/Qwen3-VL-8B-Instruct-4bit
```

## `rapid-mlx chat`

Spawn (or attach to) a server and start an interactive REPL with a model. This
is a terminal chat — not a web UI. (For the Gradio web UI, install the optional
`[chat]` extra: `pip install 'rapid-mlx[chat]'`.)

### Usage

```bash
rapid-mlx chat [model] [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `model` | Model alias or HF repo (positional, optional) | `qwen3.5-4b-4bit` |
| `--system` | System prompt prepended to the conversation | *(none)* |
| `--think` / `--no-think` | Enable / disable reasoning output in the REPL | off |
| `--max-tokens` | Max tokens per assistant response | 2048 |
| `--temperature` | Sampling temperature | 0.7 |
| `--port` | Connect to an existing server on `127.0.0.1:<port>` instead of spawning | *(spawn)* |
| `--base-url` | Connect to an existing server URL (overrides `--port`) | *(spawn)* |
| `--ready-timeout` | Seconds to wait for the spawned server to become ready | 600 |
| `--response-timeout` | Seconds to wait for a single response | 600 |

> The REPL defaults to `--no-think` because reasoning models (Qwen3.5, etc.)
> otherwise leak raw chain-of-thought and can loop until `max-tokens`. Pass
> `--think` to surface reasoning.

### Examples

```bash
# Fastest path — defaults to qwen3.5-4b-4bit, spawns its own server
rapid-mlx chat

# A reasoning model with thinking surfaced
rapid-mlx chat qwen3.5-9b-4bit --think

# Attach to a server you're already running on :8000
rapid-mlx serve qwen3.5-27b-4bit --port 8000 &
rapid-mlx chat --port 8000

# Pin a system prompt
rapid-mlx chat qwen3.5-4b-4bit --system "You are a terse, friendly Mac shell tutor."
```

In-REPL slash commands: `/help`, `/reset` (alias `/clear`), `/model <alias>`,
`/save <path>` (write conversation to markdown), `/exit` (alias `/quit`).
Type `"""` on its own line to start/end a multi-line block (pasting code).

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | Model for tests |
| `HF_TOKEN` | HuggingFace token |
