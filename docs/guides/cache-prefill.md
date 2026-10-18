# Cache prefill API

`POST /v1/cache/prefill` warms a loaded model's KV cache before a normal chat
request. It is a Rapid-MLX endpoint, separate from the OpenAI and Anthropic
APIs. It consumes a prompt through the usual chat scheduler, discards the
generated token, and returns counts only. It does not return assistant text,
tool calls, or a streaming response.

Send the same system message, tools, and thinking setting in the later chat
request. Cache reuse depends on an exact token prefix and available cache
space; a successful warm response is not a promise that a later request will
hit the cache. Normal batching and MTP settings remain in effect. The endpoint
does not reserve a private scheduler lane or change the later response shape.

The examples below set `allow_disk_cache` and therefore require the user's
consent to possible KV persistence. To warm without that consent, start the
server with `--disable-disk-caches` and omit the field.

## Warm a fixed system and tools prefix

```sh
curl -X POST http://127.0.0.1:8000/v1/cache/prefill \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "prefix",
    "messages": [{"role": "system", "content": "You are a helpful assistant."}],
    "tools": [{"type": "function", "function": {
      "name": "lookup", "description": "Look up a value",
      "parameters": {"type": "object", "properties": {"key": {"type": "string"}}}
    }}],
    "enable_thinking": false,
    "allow_disk_cache": true
  }'
```

`prefix` accepts exactly one system message. `tools` is optional. If the
server's disk cache is enabled, add `"allow_disk_cache": true` only after the
user has consented to storing this prompt on disk: a normal shutdown may save
the shared cache even when `persist` is false. `"persist": true` requests an
immediate checkpoint and also requires `allow_disk_cache` when disk caching is
enabled. `persist` defaults to false.

## Warm a replaceable draft

```sh
curl -X POST http://127.0.0.1:8000/v1/cache/prefill \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "draft",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize my draft"}
    ],
    "draft_key": "conversation-42",
    "revision": 7,
    "enable_thinking": false,
    "allow_disk_cache": true
  }'
```

A draft must end in a user message. Use a stable `draft_key` for one draft and
increase its nonnegative `revision` after each edit. A newer revision cancels
an active older revision of the same key. Stale or repeated revisions get
`409`. Revision history expires after five minutes and retains at most 128
keys; clients should use fresh keys after that interval. A draft cannot request
an immediate disk checkpoint. If disk caching is enabled, `allow_disk_cache`
must be true after user consent because a later shutdown may still save it.

## Response and limits

```json
{
  "status": "prefilled",
  "prompt_tokens": 14000,
  "cached_tokens": 13900,
  "cache_saved": false
}
```

`cache_saved` means an explicit checkpoint saved at least one cache entry; it
does not prove this particular prefix survived a partial save. The endpoint
accepts text and tool-call messages only, up to 64 messages and 64 tools, and
at most 500,000 characters across their serialized representations. The
server's request body and JSON depth caps also apply, as does the model's
context limit. Only one prefill runs per server process. Another key or prefix
request during an active prefill gets `429`; a disconnected client cancels its
warm operation. Normal chat requests can continue while a prefill runs.

The usual API key check and configured `--rate-limit` apply. Rate limiting is
disabled by default; set it for exposed servers. A request gets `503` while
the model is unavailable, `409` for missing disk consent or a stale revision,
and `400` when the prompt cannot be rendered or counted. Request schema and
body-size failures use the server's normal validation errors.
