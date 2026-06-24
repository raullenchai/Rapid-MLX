# Configuration Reference

## Server Configuration

### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Server host address (loopback-only by default; pass `0.0.0.0` to expose on LAN) | `127.0.0.1` |
| `--port` | Server port | `8000` |
| `--max-tokens` | Default max tokens | `32768` |
| `--default-temperature` | Default temperature when not specified in request | None |
| `--default-top-p` | Default top_p when not specified in request | None |

### Security Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-key` | API key for authentication | None |
| `--rate-limit` | Requests per minute per client (0 = disabled) | `0` |
| `--timeout` | Request timeout in seconds | `300` |

### Batching Options

| Option | Description | Default |
|--------|-------------|---------|
| `--continuous-batching` | Enable batching | `false` |
| `--stream-interval` | Tokens per stream chunk | `1` |
| `--max-num-seqs` | Max concurrent sequences | `256` |

### Cache Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-memory-mb` | Cache memory limit in MB | Auto |
| `--cache-memory-percent` | Fraction of RAM for cache | `0.20` |
| `--no-memory-aware-cache` | Use legacy entry-count cache | `false` |
| `--use-paged-cache` | Enable paged KV cache | `false` |
| `--paged-cache-block-size` | Tokens per block | `64` |
| `--max-cache-blocks` | Maximum blocks | `1000` |

### Tool Calling Options

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-auto-tool-choice` | Enable automatic tool calling | `false` |
| `--tool-call-parser` | Tool call parser (see [Tool Calling](../guides/tool-calling.md)) | None |

### Reasoning Options

| Option | Description | Default |
|--------|-------------|---------|
| `--reasoning-parser` | Parser for reasoning models (`qwen3`, `deepseek_r1`) | None |

### Embedding Options

| Option | Description | Default |
|--------|-------------|---------|
| `--embedding-model` | Pre-load an embedding model at startup (requires `pip install 'rapid-mlx[embeddings]'`) | None |

### MCP Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mcp-config` | Path to MCP config file | None |

## MCP Configuration

Create `mcp.json`:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-name", "arg1"],
      "env": {
        "ENV_VAR": "value"
      }
    }
  }
}
```

### MCP Server Options

| Field | Description | Required |
|-------|-------------|----------|
| `command` | Executable command | Yes |
| `args` | Command arguments | Yes |
| `env` | Environment variables | No |

## API Request Options

### Chat Completions

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Model name | Required |
| `messages` | Chat messages | Required |
| `max_tokens` | Max tokens to generate | 256 |
| `temperature` | Sampling temperature | Model default |
| `top_p` | Nucleus sampling | Model default |
| `stream` | Enable streaming | `true` |
| `stop` | Stop sequences | None |
| `tools` | Tool definitions | None |
| `response_format` | Output format (`json_object`, `json_schema`) | None |

### Multimodal Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_fps` | Frames per second | 2.0 |
| `video_max_frames` | Max frames | 32 |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | Default model for tests |
| `HF_TOKEN` | HuggingFace authentication token |
| `OPENAI_API_KEY` | Set to any value for SDK compatibility |

## Example Configurations

### Development (Single User)

```bash
rapid-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Production (Multiple Users)

```bash
rapid-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --port 8000
```

### With Tool Calling

```bash
rapid-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral \
  --continuous-batching
```

### With MCP Tools

```bash
rapid-mlx serve mlx-community/Qwen3-4B-4bit \
  --mcp-config mcp.json \
  --enable-auto-tool-choice \
  --tool-call-parser qwen \
  --continuous-batching
```

### Reasoning Model

```bash
rapid-mlx serve mlx-community/Qwen3-8B-4bit \
  --reasoning-parser qwen3 \
  --continuous-batching
```

### With Embeddings

```bash
rapid-mlx serve mlx-community/Qwen3-4B-4bit \
  --embedding-model mlx-community/multilingual-e5-small-mlx \
  --continuous-batching
```

### High Throughput

```bash
rapid-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --stream-interval 5 \
  --max-num-seqs 256
```
