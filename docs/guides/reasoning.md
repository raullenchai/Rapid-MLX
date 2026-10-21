# Reasoning Models

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start a reasoning-capable model:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3.5-9b-4bit
```

The alias profile selects its reasoning parser. Use `rapid-mlx chat qwen3.5-9b-4bit --think` for an interactive session that shows thinking, or send an API request to the printed endpoint.

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

### Start the server

The recommended alias carries its reasoning parser, so the opening `serve` command needs no parser flag. Run `rapid-mlx info qwen3.5-9b-4bit` to inspect its profile. For a custom checkpoint without a catalog profile, run `rapid-mlx serve --help` and select a matching `--reasoning-parser` explicitly.

### API Response Format

When reasoning parsing is enabled, the API response includes a `reasoning_content` field:

**Non-streaming response:**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The prime numbers less than 10 are: 2, 3, 5, 7.",
      "reasoning_content": "Let me analyze this step by step.\nFirst, I need to consider the constraints.\nThe answer should be a prime number less than 10.\nChecking: 2, 3, 5, 7 are all prime and less than 10."
    }
  }]
}
```

The field is named `reasoning_content` in both streaming and non-streaming
responses — there is no `reasoning` key on chat-completion messages or
deltas.

**Streaming response:**

Chunks are sent separately for reasoning and content. During the reasoning
phase, chunks have `reasoning_content` populated. When the model transitions
to the final answer, chunks have `content` populated:

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
# reasoning_content is a non-standard extra field: openai-python exposes it
# via attribute access (or message.model_extra) when present, but omits the
# attribute entirely when the model produced no reasoning — use getattr.
print("Reasoning:", getattr(message, "reasoning_content", None))
print("Answer:", message.content)
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
    reasoning_delta = getattr(delta, "reasoning_content", None)
    if reasoning_delta:
        reasoning_text += reasoning_delta
        print(f"[Thinking] {reasoning_delta}", end="")
    if delta.content:
        content_text += delta.content
        print(delta.content, end="")

print(f"\n\nFinal reasoning: {reasoning_text}")
print(f"Final answer: {content_text}")
```

## Parser selection

Catalog aliases carry their parser in the model profile. Use `rapid-mlx info <alias>` to inspect it. For other checkpoints, choose a parser from `rapid-mlx serve --help` that matches the checkpoint's output format.

## How It Works

The reasoning parser uses text-based detection to identify thinking tags in the model output. During streaming, it tracks the current position in the output to correctly route each token to either `reasoning_content` or `content`.

```
Model Output:        <think>Step 1: analyze...</think>The answer is 42.
                     ├─────────────────────┤├─────────────────────┤
Parsed:              │  reasoning_content  ││       content       │
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

Some prompts may not trigger reasoning. In these cases the `reasoning_content`
key is omitted from the response entirely (the server prunes `null` fields), so
plain attribute access would raise `AttributeError` — always use `getattr`:

```python
message = response.choices[0].message
reasoning = getattr(message, "reasoning_content", None)
if reasoning:
    print(f"Model's thought process: {reasoning}")
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

## Controlling Reasoning Length

Reasoning models can think for a while before answering. Two request parameters bound this:

- **`reasoning_max_tokens`** — caps the *thinking* portion. Once the model has produced this many tokens inside `<think>...</think>`, thinking is closed and the model moves on to its answer. It does **not** limit the answer.
- **`max_tokens`** — caps the *total* response (thinking + answer).

Use `reasoning_max_tokens` to stop a model from over-thinking, and pair it with `max_tokens` to bound the whole reply:

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
  "reasoning_max_tokens": 256,
  "max_tokens": 1024
}
```

If `reasoning_max_tokens` is unset, the model decides how long to think.

### `reasoning_effort` (OpenAI-style)

The OpenAI `reasoning_effort` knob (`none` / `minimal` / `low` / `medium` / `high` / `xhigh`, also `reasoning.effort` on `/v1/responses`) is translated per model:

- `none` turns thinking off (`chat_template_kwargs.enable_thinking=false`).
- On a model whose chat template has its own effort vocabulary (Qwen3.8 accepts `low` / `medium` / `xhigh`), the nearest native level is written into the prompt and **no** token cap is added: `low` → `low`, `medium` → `medium`, `high` → `xhigh`, `minimal` → `low`.
- On every other thinking model the value becomes a `reasoning_max_tokens` cap: `minimal` 256, `low` 512, `medium` 2048, `high` 8192, `xhigh` 24000.

Explicit knobs on the same dimension always win: `chat_template_kwargs.reasoning_effort` over the mapped level, `reasoning_max_tokens` over the tier cap, `enable_thinking` over `none`.

A chat template that never reads `enable_thinking` but branches on its own boolean `reasoning` variable in an eagerly evaluated template scope (Cohere's convention; North Mini Code defaults it to on) still turns off: a resolved `enable_thinking=false` (Desktop's default, `rapid-mlx chat` without `--think`, `reasoning_effort: none`) is passed as `reasoning=false`. `reasoning` is the only such switch recognised today; it is detected from the template's own reads and reachable `if` tests, not from the model name. Deferred macro/call-block bodies and loops that may not execute do not establish a switch. A request that already sets `reasoning` or `reasoning_effort` in `chat_template_kwargs` keeps control.

## Chat-template controls

Catalog aliases can select their reasoning parser automatically. Run `rapid-mlx info <alias>` to inspect model capabilities. To explicitly disable reasoning parsing, use `--no-reasoning-parser` with `serve`.

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
        "work": getattr(message, "reasoning_content", None),
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

### No `reasoning_content` field in response

- Check the alias profile with `rapid-mlx info <alias>`; custom checkpoints may need `--reasoning-parser`.
- Check that the model actually uses thinking tags (not all prompts trigger reasoning)

### Reasoning appears in content

- The model may not be using the expected tag format
- Try a different parser (`qwen3` vs `deepseek_r1`)

### Truncated reasoning

- Increase `--max-tokens` if the model is hitting the token limit mid-thought
- If you set `reasoning_max_tokens`, thinking is capped there — see [Controlling Reasoning Length](#controlling-reasoning-length)

## Related

- [Supported Models](../reference/models.md) - Models that support reasoning
- [Server Configuration](server.md) - All server options
- [CLI Reference](../reference/cli.md) - Command line options
- [SDK Compatibility Notes](sdk-compat.md) - Non-standard streaming keys, `usage` chunk behavior, and other SDK gotchas
