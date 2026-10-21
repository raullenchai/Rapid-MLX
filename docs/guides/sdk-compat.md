# SDK Compatibility Notes

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start the local API:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3.5-4b-4bit
```

Use the printed host and port. OpenAI-compatible clients use the `/v1` base URL; Anthropic-compatible clients use the server root. The sections below cover response fields and streaming behavior relevant to strict SDK parsers.

## L-01 — Anthropic SDK `base_url` must NOT include `/v1`

When configuring the official [Anthropic Python SDK][anthropic-sdk]
(or any port that follows the same URL conventions) against rapid-mlx,
point it at the server root, **not** the `/v1` prefix.

The SDK appends `/v1/messages` (or `/v1/messages/count_tokens`) to
whatever `base_url` you give it. If you also supply `/v1`, the request
goes to `/v1/v1/messages` and the server returns `404`.

**Correct:**

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",   # no trailing /v1
    api_key="not-needed",
)

message = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Say hello"}],
)
print(message.content[0].text)
```

**Wrong (returns 404):**

```python
client = Anthropic(
    base_url="http://localhost:8000/v1",   # SDK will append /v1/messages
    api_key="not-needed",
)
```

This is a property of the Anthropic SDK itself, not rapid-mlx — the
same rule applies when pointing the SDK at any custom Messages-API
backend. The OpenAI SDK has the opposite convention and **does**
require the `/v1` suffix (it does not append a versioned segment).

See also:

- [Anthropic Python SDK source][anthropic-sdk] (`client.py` builds the URL by joining `base_url` with the route).
- [Quick Start — OpenAI Server](../getting-started/quickstart.md#option-2-openai-compatible-server)
- [AI Client Compatibility — Quick Configuration Pattern](ai-clients.md#quick-configuration-pattern)

[anthropic-sdk]: https://github.com/anthropics/anthropic-sdk-python

---

## L-03 — Streaming `delta.reasoning_content` is a non-standard OpenAI key

When you run a reasoning model (Qwen3, DeepSeek-R1, phi-4-mini-reasoning,
VibeThinker, etc.) with `stream=true`, rapid-mlx emits an extra key in
each SSE chunk's delta:

```json
{
  "choices": [{
    "index": 0,
    "delta": {
      "role": "assistant",
      "reasoning_content": " step by step"
    }
  }]
}
```

The official OpenAI Chat Completions streaming schema only defines
`delta.role`, `delta.content`, `delta.tool_calls`, `delta.refusal`, and
the legacy `delta.function_call` (the pre-1.0 function-calling field,
deprecated in favor of `delta.tool_calls`). It does **not** define
`delta.reasoning_content`.

This mirrors the non-stream
`message.reasoning_content` field that Rapid-MLX adds. Keeping the field name identical
across streaming and non-streaming makes client code symmetric.

**What this means for SDK consumers:**

- The official `openai-python` SDK tolerates unknown delta keys — it
  exposes them via `chunk.choices[0].delta.model_extra` (or as
  attribute access on lenient builds). No 4xx, no client crash.
- **Strict parsers** — hand-rolled validators, code generators, or
  SDKs that pin to a sealed schema (`additionalProperties: false`) —
  may raise on this key. You will need to either pre-filter the chunks
  or loosen the parser.
- Treat any future "strict OpenAI delta" toggle as opt-in; we will
  not change the default emission.

For chunk-by-chunk consumption patterns, see
[Streaming with Reasoning](reasoning.md#streaming-with-reasoning) — and
note the interleave caveat below in [L-06](#l-06--streaming-reasoning_content-and-content-deltas-interleave).

OpenAI reference for what the canonical schema actually contains:

- [OpenAI Chat Completions — Streaming Reference][openai-stream-ref]

[openai-stream-ref]: https://platform.openai.com/docs/api-reference/chat-streaming

---

## L-04 — Streaming `usage` requires `stream_options.include_usage`

The OpenAI spec says token-usage data is only delivered on the wire
when the client opts in with:

```json
{
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

When `include_usage` is `true`, OpenAI sends a **dedicated trailer
chunk** after `finish_reason` is set, with an empty `choices` array
and a populated `usage` object — followed by `data: [DONE]`.

Rapid-MLX sends usage as follows:

1. When `include_usage` is unset or `false`, `usage` is omitted from
   **every** chunk.
2. When `include_usage: true` is set, `usage` appears **only** in the
   dedicated trailer chunk (empty `choices`) after the finish chunk;
   terminal content / tool-call chunks carry `"usage": null`.

**What this means for clients:** Set `stream_options.include_usage=True` and read usage from the trailer chunk with an empty `choices` array.

Example of spec-aligned consumption:

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hi"}],
    stream=True,
    stream_options={"include_usage": True},
)

usage = None
for chunk in stream:
    if chunk.usage is not None:
        usage = chunk.usage  # set only on the dedicated trailer chunk
    if chunk.choices:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)

print(f"\nprompt_tokens={usage.prompt_tokens} completion_tokens={usage.completion_tokens}")
```

OpenAI reference:

- [OpenAI Chat Completions — `stream_options.include_usage`][openai-stream-options]

[openai-stream-options]: https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options

---

## L-06 — Streaming `reasoning_content` and `content` deltas interleave

It is tempting to assume the stream is structured in two contiguous
phases:

> phase A — all `reasoning_content` deltas, then
> phase B — all `content` deltas

This holds for some prompts and some models, but it is **not
guaranteed**, and you should not write code that depends on it. In
production we have seen reasoning-model streams where, after the
first `content` delta has already arrived, another ~7 KB of
`reasoning_content` deltas continued to flow before the model went
back to emitting `content`.

This interleave mirrors the underlying token stream from the model
(the `<think>` / `</think>` boundary in the raw output is what the
parser uses to label each delta — and models can re-open thought
blocks). It is intentional, and we will not flatten it on the server
side.

**Don't do this:**

```python
# WRONG — assumes reasoning_content fully precedes content
seen_content = False
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        seen_content = True
    if getattr(delta, "reasoning_content", None):
        if seen_content:
            raise RuntimeError("reasoning after content?!")   # will fire on some models
```

**Do this instead — buffer both streams separately, join at
`message_stop` / `[DONE]`:**

```python
reasoning_buf = []
content_buf = []

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}],
    stream=True,
)

for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    rc = getattr(delta, "reasoning_content", None)
    if rc:
        reasoning_buf.append(rc)
    if delta.content:
        content_buf.append(delta.content)

reasoning = "".join(reasoning_buf)
answer = "".join(content_buf)
```

For Anthropic SDK consumers on `/v1/messages`, the equivalent is to
buffer `thinking` content blocks separately from `text` content blocks
and join them at the `message_stop` event — the same interleave
caveat applies.

See also:

- [Reasoning Models — Streaming with Reasoning](reasoning.md#streaming-with-reasoning)

---

## Quick reference

- For Anthropic-compatible SDKs, use the server root as `base_url`.
- For OpenAI-compatible SDKs, include `/v1` in `base_url`.
- Request streaming usage explicitly with `stream_options.include_usage=True`.
- Buffer reasoning and answer deltas independently.
