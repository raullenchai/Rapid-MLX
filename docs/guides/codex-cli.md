# Codex CLI

Use [OpenAI's Codex CLI](https://github.com/openai/codex) with rapid-mlx
as the local backend. Codex is a Rust-based coding agent that talks to
the OpenAI Responses API (`POST /v1/responses`); rapid-mlx implements
that endpoint as a stateless shim — every Codex turn re-sends the full
conversation history, so no response-store layer is needed on the
server side.

Requires **rapid-mlx >= 0.7.10**.

## TL;DR

```bash
# 1. Install Codex CLI
brew install codex   # or: npm install -g @openai/codex

# 2. Start rapid-mlx with a strong-enough model
rapid-mlx serve qwen3.6-35b-4bit --port 8000

# 3. Point Codex at the local server
rapid-mlx agents codex --setup     # writes ~/.codex/config.toml for you

# 4. Run Codex
codex                              # interactive
codex exec "explain this repo"     # one-shot
```

## Model recommendations

Codex's workflow leans on multi-tool calls + `apply_patch` for file
edits. Small models underperform. On Apple Silicon, in rough order:

| Model | Size | Notes |
|---|---|---|
| `qwen3.6-35b-4bit` | ~20 GB | Recommended workhorse for M3 Max / M4 Pro 24 GB+ |
| `qwen3-coder-30b-4bit` | ~17 GB | Code-specialized; great for narrower coding tasks |
| `qwen3.5-9b-4bit` | ~5 GB | Practical floor — works on 16 GB Macs but expect more retries |

Smaller models (≤8B) tend to hallucinate `apply_patch` shapes; not
recommended.

## Manual config

If `rapid-mlx agents codex --setup` didn't fit your layout (e.g. you
already have a `~/.codex/config.toml`), the relevant block is:

```toml
model = "default"            # or any rapid-mlx alias
model_provider = "rapid-mlx"

[model_providers.rapid-mlx]
name = "Rapid-MLX (local)"
base_url = "http://localhost:8000/v1"
api_key = "not-needed"
```

Codex picks the provider from `model_provider` and resolves its
`base_url` from the matching `[model_providers.NAME]` block. The
`api_key` value is ignored by rapid-mlx unless you started the server
with `--api-key`.

## Model name passthrough

Codex sends model names like `gpt-5` or `gpt-5-codex` in the request
body even when you've configured a different one. Rapid-mlx's route
recognises any `gpt-*` / `claude-*` model name as "the loaded engine,
not a strict alias lookup" — so the request reaches the model you
actually started the server with, instead of 404'ing on the name
mismatch. The response's `model` field carries the loaded model's name
(consistent with the Anthropic-compat route).

This means: **whatever model you start `rapid-mlx serve` with is what
Codex will talk to**, regardless of what Codex thinks it's talking to.

## What's mapped, what's not

The shim is intentionally minimal — it covers Codex's hot path and
nothing more.

**Translated:**

- `instructions` → system message
- `input[]` polymorphic items: `message` / `function_call` /
  `function_call_output` → assistant / tool messages
- `tools` (Responses-flat shape) → Chat-nested tools
- `text.format` JSON-schema → `response_format`
- `max_output_tokens` → `max_tokens`
- SSE: 7 events Codex actually parses (`response.created`,
  `response.output_item.added`, `response.output_text.delta`,
  `response.function_call_arguments.delta`,
  `response.output_item.done`, `response.completed`,
  `response.failed`)

**Not translated (v1):**

- `previous_response_id` → returns 400. Codex doesn't use this field
  (openai/codex#3841 confirms it's not implemented client-side), so the
  400 is a safety net for any other client that tries.
- `reasoning.effort` → ignored. Set thinking on the server with
  `rapid-mlx serve --enable-thinking` instead.
- `input_image` → dropped. Codex doesn't send images.
- Non-function tool types (`web_search`, `code_interpreter`,
  `image_generation`, `file_search`, `computer_use`) → dropped. Codex
  doesn't send these to third-party backends.

## Probing the endpoint

If you want to verify the shim is reachable without booting Codex:

```bash
curl -sS http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "input": "Say hello in one word.",
    "stream": false
  }' | jq .
```

You should see a `response` object with an `output` array containing a
`message` item. With `--api-key`, add `-H "Authorization: Bearer <key>"`.

## Troubleshooting

**Codex says "stream closed before response.completed"** — this should
not happen on rapid-mlx >= 0.7.10. If it does, the engine likely
crashed mid-generation; check the server logs. Re-running the query
usually works.

**Codex 404s on `/v1/responses`** — you're on rapid-mlx < 0.7.10.
Upgrade with `rapid-mlx upgrade` (or `pip install -U rapid-mlx`).

**Tool calls don't apply** — make sure `--enable-auto-tool-choice` is
passed when starting the server, and that the model's tool parser is
auto-detected correctly (`rapid-mlx serve ... --log-level DEBUG` shows
it during boot).

**Codex hangs** — first run prompts for sandbox permissions
(Landlock on Linux, Seatbelt on macOS). Accept them in the Codex
prompt; the second run is non-interactive.

## See also

- [Server setup](server.md)
- [Tool calling](tool-calling.md)
- [Reasoning models](reasoning.md)
- Issue [#549](https://github.com/raullenchai/Rapid-MLX/issues/549) — the request that drove this integration
