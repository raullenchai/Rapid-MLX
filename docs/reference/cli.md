# CLI Reference

## Commands Overview

| Command | Description |
|---------|-------------|
| `vllm-mlx serve` | Start OpenAI-compatible server |
| `vllm-mlx-bench` | Run performance benchmarks |
| `vllm-mlx-chat` | Start Gradio chat interface |

## `vllm-mlx serve`

Start the OpenAI-compatible API server.

### Usage

```bash
vllm-mlx serve <model> [options]
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
| `--reasoning-parser` | Parser for reasoning models (`qwen3`, `deepseek_r1`) | None |
| `--embedding-model` | Pre-load an embedding model at startup | None |
| `--enable-auto-tool-choice` | Enable automatic tool calling | False |
| `--tool-call-parser` | Tool call parser (`auto`, `mistral`, `qwen`, `llama`, `hermes`, `deepseek`, `kimi`, `granite`, `nemotron`, `xlam`, `functionary`, `glm47`) | None |

### Examples

```bash
# Simple mode (single user, max throughput)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit

# Continuous batching (multiple users)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

# With memory limit for large models
vllm-mlx serve mlx-community/GLM-4.7-Flash-4bit \
  --continuous-batching \
  --cache-memory-mb 2048

# Production with paged cache
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --port 8000

# With MCP tools
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

# Multimodal model
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit

# Reasoning model (separates thinking from answer)
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# DeepSeek reasoning model
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1

# Tool calling with Mistral/Devstral
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral

# Tool calling with Granite
vllm-mlx serve mlx-community/granite-4.0-tiny-preview-4bit \
  --enable-auto-tool-choice --tool-call-parser granite

# With API key authentication
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --api-key your-secret-key

# Large models (200GB+) — raise memory limit to avoid cache thrashing
vllm-mlx serve mlx-community/Qwen3.5-397B-A17B-nvfp4 \
  --continuous-batching \
  --gpu-memory-utilization 0.95

# Production setup with security options
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --continuous-batching
```

### Security

When `--api-key` is set, all API requests require the `Authorization: Bearer <api-key>` header:

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

## `vllm-mlx-bench`

Run performance benchmarks.

### Usage

```bash
vllm-mlx-bench --model <model> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name | Required |
| `--prompts` | Number of prompts | 5 |
| `--max-tokens` | Max tokens per prompt | 256 |
| `--quick` | Quick benchmark mode | False |
| `--video` | Run video benchmark | False |
| `--video-url` | Custom video URL | None |
| `--video-path` | Custom video path | None |

### Examples

```bash
# LLM benchmark
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit

# Quick benchmark
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --quick

# Image benchmark (auto-detected for VLM models)
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Video benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video

# Custom video
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit \
  --video --video-url https://example.com/video.mp4
```

## `vllm-mlx-chat`

Start Gradio chat interface.

### Usage

```bash
vllm-mlx-chat --model <model> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name | Required |
| `--port` | Gradio port | 7860 |
| `--text-only` | Disable multimodal | False |

### Examples

```bash
# Multimodal chat (text + images + video)
vllm-mlx-chat --model mlx-community/Qwen3-VL-4B-Instruct-3bit

# Text-only chat
vllm-mlx-chat --model mlx-community/Llama-3.2-3B-Instruct-4bit --text-only
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | Model for tests |
| `HF_TOKEN` | HuggingFace token |
