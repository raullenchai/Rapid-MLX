# SDK Compatibility Notes

This page collects small but important details about how rapid-mlx's
OpenAI-compatible and Anthropic-compatible surfaces deviate from — or
sit slightly outside of — the upstream specifications. None of these
break common SDK usage in practice, but each has tripped at least one
user during 0.7.x / 0.8.0 dogfooding. If you are debugging a strict-mode
parser, building your own SDK wrapper, or pinning a vendored client to
a specific schema, read through these before assuming you have hit a
bug.

Each entry is tagged with its tracking id (e.g. `L-01`) so you can grep
back to the upstream issue list.

---

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
(very recently) `delta.function_call`. It does **not** define
`delta.reasoning_content`.

For us this is intentional: it mirrors the non-stream
`message.reasoning_content` field that rapid-mlx already adds (also
non-standard, but widely adopted by other reasoning-model providers
like DeepSeek and Together AI). Keeping the field name identical
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
note the interleave caveat below in [L-06](#l-06--streaming-reasoning_content--content-deltas-interleave).

OpenAI reference for what the canonical schema actually contains:

- [OpenAI Chat Completions — Streaming Reference][openai-stream-ref]

[openai-stream-ref]: https://platform.openai.com/docs/api-reference/chat-streaming

---

## L-04 — Streaming emits `usage` in the last content chunk without `stream_options.include_usage`

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

rapid-mlx currently does two things at once:

1. **Always** embeds `usage` directly into the last content chunk
   (the one that carries `finish_reason`), **regardless** of whether
   `stream_options.include_usage` was set. This is non-spec.
2. When `include_usage: true` **is** set, it additionally emits the
   dedicated trailer chunk after the content chunk. This part **is**
   spec-aligned.

**What this means for clients:**

- If you are using the official `openai-python` SDK to read
  `chunk.usage`, you'll see usage data with or without
  `include_usage`. Reading it works today.
- For forward compatibility, **explicitly set
  `stream_options.include_usage=True`** and rely on the dedicated
  trailer chunk. We may gate the inline-on-last-content emission
  behind the same opt-in flag in a future minor release to align with
  the spec; clients that already opt in will be unaffected.
- Strict parsers that disallow `usage` on a chunk with a non-empty
  `choices` array will reject our current emission. The same parsers
  will accept the trailer chunk fine, so opting in is the
  forward-compatible workaround.

Example of safe consumption (works on rapid-mlx today *and* on a
spec-strict future):

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
        usage = chunk.usage  # may be set on last content chunk OR trailer
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

| Tag | Surface | Symptom | Mitigation |
|-----|---------|---------|------------|
| **L-01** | Anthropic SDK | `404` on every request | `base_url="http://host:port"` — no `/v1` suffix |
| **L-03** | OpenAI SDK / streaming | Strict parser rejects `delta.reasoning_content` | Loosen parser, or pre-filter chunks |
| **L-04** | OpenAI SDK / streaming | `usage` arrives on last content chunk without opt-in | Opt in via `stream_options.include_usage=True` and read from trailer |
| **L-06** | OpenAI / Anthropic streaming | `reasoning_content` deltas after first `content` delta | Buffer separately, join at end of stream |
