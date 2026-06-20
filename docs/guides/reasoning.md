# Reasoning Models

rapid-mlx supports reasoning models that show their thinking process before giving an answer. Models like Qwen3 and DeepSeek-R1 wrap their reasoning in `<think>...</think>` tags, and rapid-mlx can parse these tags to separate the reasoning from the final response.

## Why Use Reasoning Parsing?

When a reasoning model generates output, it typically looks like this:

```
<think>
Let me analyze this step by step.
First, I need to consider the constraints.
The answer should be a prime number less than 10.
Checking: 2, 3, 5, 7 are all prime and less than 10.
</think>
The prime numbers less than 10 are: 2, 3, 5, 7.
```

Without reasoning parsing, you get the raw output with the tags included. With reasoning parsing enabled, the thinking process and final answer are separated into distinct fields in the API response.

## Getting Started

### Start the Server with Reasoning Parser

```bash
# For Qwen3 models
rapid-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# For DeepSeek-R1 models
rapid-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

### API Response Format

When reasoning parsing is enabled, the API response includes a `reasoning` field:

**Non-streaming response:**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The prime numbers less than 10 are: 2, 3, 5, 7.",
      "reasoning": "Let me analyze this step by step.\nFirst, I need to consider the constraints.\nThe answer should be a prime number less than 10.\nChecking: 2, 3, 5, 7 are all prime and less than 10."
    }
  }]
}
```

**Streaming response:**

Chunks are sent separately for reasoning and content. During the reasoning
phase, chunks have `reasoning_content` (with a `reasoning` alias for
backward compatibility) populated. When the model transitions to the final
answer, chunks have `content` populated:

```json
{"delta": {"reasoning_content": "Let me analyze"}}
{"delta": {"reasoning_content": " this step by step."}}
{"delta": {"reasoning_content": "\nFirst, I need to"}}
...
{"delta": {"content": "The prime"}}
{"delta": {"content": " numbers less than 10"}}
{"delta": {"content": " are: 2, 3, 5, 7."}}
```

> **Note (L-03):** `delta.reasoning_content` is not part of the official
> OpenAI Chat Completions streaming schema. The official `openai-python`
> SDK tolerates the extra key (`chunk.choices[0].delta.model_extra` or
> attribute access), but **strict / hand-rolled parsers may reject it**.
> The naming intentionally mirrors the non-stream
> `message.reasoning_content` field. See
> [SDK Compatibility Notes — L-03](sdk-compat.md#l-03--streaming-deltareasoning_content-is-a-non-standard-openai-key).
>
> **Note (L-06):** Do not assume `reasoning_content` deltas fully
> precede `content` deltas — they can interleave for many bytes after
> the first `content` delta arrives. Buffer both streams separately and
> join at the end. See
> [SDK Compatibility Notes — L-06](sdk-compat.md#l-06--streaming-reasoning_content-and-content-deltas-interleave).

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Non-streaming
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What are the prime numbers less than 10?"}]
)

message = response.choices[0].message
print("Reasoning:", message.reasoning)  # The thinking process
print("Answer:", message.content)        # The final answer
```

### Streaming with Reasoning

```python
reasoning_text = ""
content_text = ""

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Solve: 2 + 2 = ?"}],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'reasoning') and delta.reasoning:
        reasoning_text += delta.reasoning
        print(f"[Thinking] {delta.reasoning}", end="")
    if delta.content:
        content_text += delta.content
        print(delta.content, end="")

print(f"\n\nFinal reasoning: {reasoning_text}")
print(f"Final answer: {content_text}")
```

## Supported Parsers

### Qwen3 Parser (`qwen3`)

For Qwen3 models that use explicit `<think>` and `</think>` tags.

- Requires **both** opening and closing tags
- If tags are missing, output is treated as regular content
- Best for: Qwen3-0.6B, Qwen3-4B, Qwen3-8B and similar models

```bash
rapid-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

### DeepSeek-R1 Parser (`deepseek_r1`)

For DeepSeek-R1 models that may omit the opening `<think>` tag.

- More lenient than Qwen3 parser
- Handles cases where `<think>` is implicit
- Content before `</think>` is treated as reasoning even without `<think>`

```bash
rapid-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

## How It Works

The reasoning parser uses text-based detection to identify thinking tags in the model output. During streaming, it tracks the current position in the output to correctly route each token to either `reasoning` or `content`.

```
Model Output:        <think>Step 1: analyze...</think>The answer is 42.
                     ├─────────────────────┤├─────────────────────┤
Parsed:              │     reasoning       ││       content       │
                     └─────────────────────┘└─────────────────────┘
```

The parsing is stateless and uses the accumulated text to determine context, making it robust for streaming scenarios where tokens may arrive in arbitrary chunks.

## Tips for Best Results

### Prompting

Reasoning models work best when you encourage step-by-step thinking:

```python
messages = [
    {"role": "system", "content": "Think through problems step by step before answering."},
    {"role": "user", "content": "What is 17 × 23?"}
]
```

### Handling Missing Reasoning

Some prompts may not trigger reasoning. In these cases, `reasoning` will be `None` and all output goes to `content`:

```python
message = response.choices[0].message
if message.reasoning:
    print(f"Model's thought process: {message.reasoning}")
print(f"Answer: {message.content}")
```

### Temperature and Reasoning

Lower temperatures tend to produce more consistent reasoning patterns:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    temperature=0.3  # More focused reasoning
)
```

## Backward Compatibility

When `--reasoning-parser` is not specified, the server behaves as before:
- Thinking tags are included in the `content` field
- No `reasoning` field is added to responses

This ensures existing applications continue to work without changes.

## Example: Math Problem Solver

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def solve_math(problem: str) -> dict:
    """Solve a math problem and return reasoning + answer."""
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a math tutor. Show your work."},
            {"role": "user", "content": problem}
        ],
        temperature=0.2
    )

    message = response.choices[0].message
    return {
        "problem": problem,
        "work": message.reasoning,
        "answer": message.content
    }

result = solve_math("If a train travels 120 km in 2 hours, what is its average speed?")
print(f"Problem: {result['problem']}")
print(f"\nWork shown:\n{result['work']}")
print(f"\nFinal answer: {result['answer']}")
```

## Curl Examples

### Non-streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}]
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}],
    "stream": true
  }'
```

## Troubleshooting

### No reasoning field in response

- Make sure you started the server with `--reasoning-parser`
- Check that the model actually uses thinking tags (not all prompts trigger reasoning)

### Reasoning appears in content

- The model may not be using the expected tag format
- Try a different parser (`qwen3` vs `deepseek_r1`)

### Truncated reasoning

- Increase `--max-tokens` if the model is hitting the token limit mid-thought

## Related

- [Supported Models](../reference/models.md) - Models that support reasoning
- [Server Configuration](server.md) - All server options
- [CLI Reference](../reference/cli.md) - Command line options
- [SDK Compatibility Notes](sdk-compat.md) - Non-standard streaming keys, `usage` chunk behavior, and other SDK gotchas
