# Python API

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start a local API:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3.5-4b-4bit --port 8000
```

The server prints its address. In another Terminal window, install the client and run the examples with the installer Python:

```bash
~/.rapid-mlx/bin/python -m pip install openai
```

## Basic Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen3.5-4b-4bit",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

## Streaming

```python
stream = client.chat.completions.create(
    model="qwen3.5-4b-4bit",
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
    model="qwen3.5-4b-4bit",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)
print(response.choices[0].message.tool_calls)
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
