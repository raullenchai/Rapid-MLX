# Python API

Use the OpenAI-compatible server via any OpenAI client library.

## Start Server

```bash
rapid-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

## Basic Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

## Streaming

```python
stream = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Tell me a story about a robot"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Tool Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)
print(response.choices[0].message.tool_calls)
```

## Engine API (Advanced)

For embedding rapid-mlx in your own application:

```python
from vllm_mlx.engine import BatchedEngine

engine = BatchedEngine("mlx-community/Llama-3.2-3B-Instruct-4bit")
await engine.start()

# Generate via the engine directly
async for output in engine.stream_chat(
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
):
    print(output.text, end="")

await engine.stop()
```

## Error Handling

```python
try:
    response = client.chat.completions.create(
        model="invalid-model",
        messages=[{"role": "user", "content": "test"}],
    )
except Exception as e:
    print(f"Error: {e}")
```
