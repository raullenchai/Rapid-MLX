# Quick Start

## Option 1: Interactive Chat (fastest first taste)

The shortest path to talking to a model — `chat` spawns its own server,
downloads the model on first run (~2.5 GB for the default `qwen3.5-4b`), and
drops you into a REPL.

```bash
rapid-mlx chat                  # defaults to qwen3.5-4b
rapid-mlx chat qwen3.5-9b       # a larger model (5 GB)
rapid-mlx chat --think          # surface chain-of-thought reasoning
```

In-REPL: `/help`, `/reset`, `/save <path>`, `/model <alias>`, `/exit`. Type
`"""` on its own line to start/end a multi-line block. See the
[CLI reference](../reference/cli.md#rapid-mlx-chat) for all flags.

## Option 2: OpenAI-Compatible Server

Start the server:

```bash
rapid-mlx serve qwen3.5-4b --port 8000
```

Use with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
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

## Option 3: Gradio Web UI

A browser-based chat UI ships in the optional `[chat]` extra:

```bash
pip install 'rapid-mlx[chat]'
# Then launch — see `rapid-mlx help` for the UI entry point in your install.
```

## Multimodal Models

For image / video understanding, use a VLM (requires the `[vision]` extra —
`pip install 'rapid-mlx[vision]'`):

```bash
rapid-mlx serve gemma-4-26b --mllm --port 8000
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

Reasoning parsers are auto-detected from the model name. The server splits
chain-of-thought into a separate `reasoning_content` field, leaving `content`
clean.

```bash
rapid-mlx serve qwen3.5-9b --port 8000   # qwen3 reasoning parser auto-detected
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)
print(response.choices[0].message.content)            # final answer
print(response.choices[0].message.reasoning_content)  # thinking trace
```

## Embeddings

Generate text embeddings for semantic search and RAG (install the
`[embeddings]` extra first):

```bash
rapid-mlx serve qwen3.5-4b --embedding-model mlx-community/multilingual-e5-small-mlx
```

```python
response = client.embeddings.create(
    model="mlx-community/multilingual-e5-small-mlx",
    input="Hello world"
)
```

## Tool Calling

Tool/function calling is on by default for supported model families (Qwen3.x,
GLM-4.7, GPT-OSS, Llama, Mistral, etc.) — the right parser is auto-detected:

```bash
rapid-mlx serve qwen3.5-9b --port 8000
```

If you need to pin the parser manually:

```bash
rapid-mlx serve devstral-24b \
  --enable-auto-tool-choice --tool-call-parser hermes
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
