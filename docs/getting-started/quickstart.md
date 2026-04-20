# Quick Start

## Option 1: OpenAI-Compatible Server

Start the server:

```bash
# Simple mode - maximum throughput for single user
rapid-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# Continuous batching - for multiple concurrent users
rapid-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

Use with OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Or with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Option 2: Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

## Option 3: Gradio Chat UI

```bash
rapid-mlx-chat --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

Opens a web interface at http://localhost:7860

## Multimodal Models

For image/video understanding, use a VLM model:

```bash
rapid-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }],
    max_tokens=256
)
```

## Reasoning Models

Separate the model's thinking process from the final answer:

```bash
rapid-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)
print(response.choices[0].message.content)  # Final answer
```

## Embeddings

Generate text embeddings for semantic search and RAG:

```bash
rapid-mlx serve mlx-community/Qwen3-4B-4bit --embedding-model mlx-community/multilingual-e5-small-mlx
```

```python
response = client.embeddings.create(
    model="mlx-community/multilingual-e5-small-mlx",
    input="Hello world"
)
```

## Tool Calling

Enable function calling with any supported model:

```bash
rapid-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral
```

## Next Steps

- [Server Guide](../guides/server.md) - Full server configuration
- [Python API](../guides/python-api.md) - Direct API usage
- [Multimodal Guide](../guides/multimodal.md) - Images and video
- [Audio Guide](../guides/audio.md) - Speech-to-Text and Text-to-Speech
- [Embeddings Guide](../guides/embeddings.md) - Text embeddings
- [Reasoning Models](../guides/reasoning.md) - Thinking models
- [Tool Calling](../guides/tool-calling.md) - Function calling
- [Supported Models](../reference/models.md) - Available models
