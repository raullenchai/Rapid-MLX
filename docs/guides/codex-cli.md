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
| `qwen3.6-35b-4bit` | ~20 GB | Recommended workhorse for 48 GB+ Macs (swaps and aborts on 32 GB) |
| `qwen3-coder-30b-4bit` | ~17 GB | Code-specialized; great for narrower coding tasks |
| `qwen3.5-9b-4bit` | ~5 GB | Practical floor — works on 16 GB Macs but expect more retries |

Smaller models (≤8B) tend to hallucinate `apply_patch` shapes; not
recommended.

## Manual config

If `rapid-mlx agents codex --setup` didn't fit your layout (e.g. you
already have a `~/.codex/config.toml`), the relevant block is:

```toml
model = "qwen3.6-35b-4bit"   # or any rapid-mlx alias
model_provider = "rapid-mlx"

[model_providers.rapid-mlx]
name = "Rapid-MLX (local)"
base_url = "http://localhost:8000/v1"
```

Codex picks the provider from `model_provider` and resolves its
`base_url` from the matching `[model_providers.NAME]` block.

### With `--api-key` enabled on the server

Current Codex CLI (>= 0.135) reads the credential via **env-var
indirection**, not as an inline literal — Codex's `--strict-config`
rejects `api_key = "..."` as an unknown field. Use `env_key` instead:

```toml
[model_providers.rapid-mlx]
name = "Rapid-MLX (local)"
base_url = "http://localhost:8000/v1"
env_key = "RAPID_MLX_API_KEY"
```

And in your shell:

```bash
export RAPID_MLX_API_KEY=your-secret
```

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
- `reasoning.effort` (or the top-level `reasoning_effort` shorthand) →
  rapid-mlx's thinking controls: `"none"` disables thinking
  (`chat_template_kwargs.enable_thinking=false`); `minimal` / `low` /
  `medium` / `high` set the matching `reasoning_max_tokens` tier cap.
  Explicit client knobs on the same dimension always win. Values are
  validated against the OpenAI closed set (garbage → 400).
- `input_image` content blocks → Chat `image_url` parts (needs a
  multimodal model to do anything useful)
- SSE: the streaming lifecycle events Codex parses
  (`response.created`, `response.in_progress`,
  `response.output_item.added` / `.done`,
  `response.content_part.added` / `.done`,
  `response.output_text.delta` / `.done`,
  `response.function_call_arguments.delta`, the
  `response.reasoning_summary_*` family, `response.completed`,
  `response.failed`)

**Not translated (v1):**

- `previous_response_id` → returns 400. Codex doesn't use this field
  (openai/codex#3841 confirms it's not implemented client-side), so the
  400 is a safety net for any other client that tries.
- Hosted tool types (`web_search`, `file_search`, `code_interpreter`,
  `image_generation`) → rejected with 400 listing the supported types
  (`function`, `computer_20251022`). Exception: Codex-shaped requests
  (fingerprinted by their `namespace` tool groups) get the ambient
  hosted-tool noise silently dropped so the rest of the request still
  runs.

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

## Codex CLI versions

Verified against **Codex CLI 0.136.0** on **rapid-mlx 0.7.12+** with
Qwen3.5-9B and Qwen3.6-27B. Codex 0.135 → 0.136 reshaped the request
in three ways that earlier rapid-mlx releases mishandled:

- the per-turn instruction channel switched from `system` to the new
  Responses-API `developer` role,
- multiple system-equivalent messages are now interleaved with user
  turns, and
- the agent loop terminates silently if the `function_call` item is
  missing — no error, no partial output, just a closed stream.

If you're on rapid-mlx **< 0.7.12** with a recent Codex, you may see
"stream disconnected before completion" or a turn that ends with no
visible output. Upgrade with `rapid-mlx upgrade`.

## Troubleshooting

**Codex says "stream closed before response.completed"** — this should
not happen on rapid-mlx >= 0.7.12 with Codex 0.136. If it does, the
engine likely crashed mid-generation; check the server logs.
Re-running the query usually works.

**Codex 404s on `/v1/responses`** — you're on rapid-mlx < 0.7.10.
Upgrade with `rapid-mlx upgrade` (or `pip install -U rapid-mlx`).

**Codex turn ends with no output** — on rapid-mlx 0.7.10–0.7.11 with
Codex 0.136, tool-call XML was filtered before the parser ran and the
agent loop saw zero items. Fixed in 0.7.12.

**Tool calls don't apply** — tool calling is auto-enabled when the
model's tool parser is auto-detected (boot logs show
`Auto-configured --tool-call-parser ...`). For a model the
auto-detector doesn't recognise, pass
`--tool-call-parser <name> --enable-auto-tool-choice` explicitly
(`rapid-mlx serve ... --log-level DEBUG` shows the parser during boot).

**Codex hangs** — first run prompts for sandbox permissions
(Landlock on Linux, Seatbelt on macOS). Accept them in the Codex
prompt; the second run is non-interactive.

## See also

- [Server setup](server.md)
- [Tool calling](tool-calling.md)
- [Reasoning models](reasoning.md)
- Issue [#549](https://github.com/raullenchai/Rapid-MLX/issues/549) — the request that drove this integration
