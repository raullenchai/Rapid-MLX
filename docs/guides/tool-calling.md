# Tool Calling

rapid-mlx supports OpenAI-compatible tool calling (function calling) with automatic parsing for many popular model families.

## Quick Start

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start a tool-capable model:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3.5-9b-4bit
```

The alias profile selects its tool-call parser. Use `rapid-mlx info qwen3.5-9b-4bit` to inspect its capabilities.

Install the client with `~/.rapid-mlx/bin/python -m pip install openai`, then use tools with the standard OpenAI API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

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
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }]
)

# Check for tool calls
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Function: {tc.function.name}")
        print(f"Arguments: {tc.function.arguments}")
```

## Parser selection

A catalog alias can supply its parser. For another checkpoint, inspect its model card and select a parser with `--tool-call-parser` if needed. Run `rapid-mlx serve --help` for the current flags. The [model catalog](https://rapidmlx.com/docs/aliases.html) lists supported aliases.

## Streaming Tool Calls

Tool calls work with streaming. Parsers emit tool-call deltas
incrementally via each parser's `extract_tool_calls_streaming` method --
the function name arrives first, then argument fragments follow as the
model generates them:

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's 25 * 17?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        for tc in chunk.choices[0].delta.tool_calls:
            print(f"Tool call: {tc.function.name}({tc.function.arguments})")
```

## Handling Tool Results

After receiving a tool call, execute the function and send the result back:

```python
import json

# First request - model decides to call a tool
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[weather_tool]
)

# Get the tool call
tool_call = response.choices[0].message.tool_calls[0]
tool_call_id = tool_call.id
function_name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

# Execute the function (your implementation)
result = get_weather(**arguments)  # {"temperature": 22, "condition": "sunny"}

# Send result back to model
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)}
    ],
    tools=[weather_tool]
)

print(response.choices[0].message.content)
# "The weather in Tokyo is sunny with a temperature of 22C."
```

## Think Tag Handling

Models that produce `<think>...</think>` reasoning tags (like DeepSeek-R1, Qwen3, GLM-4.7) are handled automatically. The parser strips thinking content before extracting tool calls, so reasoning tags never interfere with tool call parsing.

This works even when `<think>` was injected in the prompt (implicit think tags with only a closing `</think>`).

## Calls to Tools the Request Did Not Declare

rapid-mlx only returns `tool_calls` for tools the request declared (and, with a
named `tool_choice`, only that tool). A model can still write a call to a tool
that does not exist, for example `read` when the client offered `shell` and
`write`. Such a call is never executed.

With `qwen3_coder_xml`, a complete call block to an undeclared tool
(`<tool_call>`, one `<function=NAME>…</function>`, `</tool_call>`) is removed
from the response text, and the server logs a warning naming the tool. The
block is removed only when the request declared tools, it is not inside
Markdown code (a ```` ``` ```` or `~~~` fence, or inline code on its line), and
no other tool-call markup came earlier in the same response. The response then has `finish_reason: "stop"`
and whatever text the model wrote around the block. Other text that only looks
like tool-call markup, such as a bare `<function=…>` example, a block inside a
code example, or an unfinished block, is returned unchanged, so prose about the
format keeps working.

## CLI reference

Run `rapid-mlx serve --help` for tool-call flags and [CLI reference](../reference/cli.md) for related server options.
