# Claude Code

Point [Anthropic's Claude Code](https://docs.anthropic.com/en/docs/claude-code)
at a local rapid-mlx server. Claude Code speaks the **Anthropic Messages
API** (`POST /v1/messages`); rapid-mlx implements that route natively, so
you can drive Claude Code with any local model.

Verified end-to-end against Qwen 3.6, Gemma 4, and gpt-oss in the Tier-1
agent matrix (`tests/integrations/test_agents_matrix.py::TestClaudeCode`,
with the deep Anthropic-SDK flow in `test_anthropic_sdk.py`). See the
[support matrix](matrix.md) for the current per-family status.

## How Claude Code reaches rapid-mlx

| Item | Value |
|---|---|
| Wire | `POST /v1/messages` (Anthropic Messages API) |
| Base URL | `http://localhost:8000` — **bare host, no `/v1`** |
| Config | `ANTHROPIC_BASE_URL` env var |

> **Critical (L-01):** the Anthropic SDK appends `/v1/messages` to whatever
> base URL you give it. Pass the **bare host** (`http://localhost:8000`). If
> you include `/v1`, requests go to `/v1/v1/messages` and the server 404s.
> Full write-up in
> [SDK Compatibility Notes — L-01](../guides/sdk-compat.md#l-01--anthropic-sdk-base_url-must-not-include-v1).

## TL;DR

```bash
# 1. Install Claude Code
npm install -g @anthropic-ai/claude-code

# 2. Start rapid-mlx
rapid-mlx serve qwen3.6-35b-4bit --port 8000

# 3. Point Claude Code at the local server (note: bare host, no /v1)
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=not-needed \
  claude
```

To make it permanent, export the two env vars in your shell profile:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed   # any non-empty string
```

If you started the server with `--api-key`, set `ANTHROPIC_API_KEY` to that
key instead of `not-needed`.

## Per-family setup

The parser is auto-detected from the alias — the `serve` line is all you
need. The parser column documents what the matrix test exercises.

### Qwen 3.6

```bash
rapid-mlx serve qwen3.6-35b-4bit --port 8000   # ~20 GB
# smaller: qwen3.6-27b-4bit
```

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=not-needed claude
```

- tool-call parser: `qwen3_coder_xml` · reasoning parser: `qwen3`
- matrix cell: **PASS** ✅

### Gemma 4

```bash
rapid-mlx serve gemma-4-12b-4bit --port 8000   # ~7 GB at 4-bit
# larger: gemma-4-26b-4bit
```

- tool-call parser: `gemma4` · reasoning parser: `gemma4`
- matrix cell: **PASS** ✅

### gpt-oss

```bash
rapid-mlx serve gpt-oss-20b --port 8000        # ~11 GB (MXFP4-Q8)
# larger: gpt-oss-120b
```

- tool-call parser: `harmony` · reasoning parser: `harmony`
- matrix cell: **PASS** ✅ — the harmony `analysis` channel is routed to
  the reasoning block, not into visible content.

## Verifying the route

Confirm the Anthropic route is reachable without booting Claude Code:

```bash
curl -sS http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-3-5-sonnet","max_tokens":64,
       "messages":[{"role":"user","content":"Say hello in one word."}]}' | jq .
```

Like Codex, any `claude-*` model name resolves to the loaded engine, so the
request reaches whatever alias you started `rapid-mlx serve` with.

## Troubleshooting

- **404 on every request** — you included `/v1` in `ANTHROPIC_BASE_URL`.
  Use the bare host `http://localhost:8000` (see L-01 above).
- **Reasoning models emit a thinking block first** — that's expected;
  rapid-mlx returns a `thinking` content block before the `text` block. The
  matrix cell walks to the first `text` block.
- **Client rejects `not-needed` as an API key** — use any non-empty string
  (`sk-local`, `rapid-mlx`).

## See also

- [Agent support matrix](matrix.md)
- [AI client compatibility](../guides/ai-clients.md)
- [SDK compatibility notes](../guides/sdk-compat.md) · [Server setup](../guides/server.md)
