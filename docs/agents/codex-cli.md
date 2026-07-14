# Codex CLI

Point [OpenAI's Codex CLI](https://github.com/openai/codex) at a local
rapid-mlx server. Codex is a Rust-based coding agent that talks to the
OpenAI **Responses API** (`POST /v1/responses`); rapid-mlx implements that
endpoint as a stateless shim, so any local model can drive Codex.

Requires **rapid-mlx >= 0.7.10**. Verified end-to-end against Qwen 3.6,
Gemma 4, and gpt-oss in the Tier-1 agent matrix
(`tests/integrations/test_agents_matrix.py::TestCodexCLI`). See the
[support matrix](matrix.md) for the current per-family status.

## How Codex reaches rapid-mlx

| Item | Value |
|---|---|
| Wire | `POST /v1/responses` (OpenAI Responses API shim) |
| Base URL | `http://localhost:8000/v1` |
| Config file | `~/.codex/config.toml` |
| Auto-setup | `rapid-mlx agents codex --setup` |

## TL;DR

```bash
# 1. Install Codex CLI
brew install codex   # or: npm install -g @openai/codex

# 2. Start rapid-mlx with a strong-enough model (see per-family below)
rapid-mlx serve qwen3.6-35b-4bit --port 8000

# 3. Point Codex at the local server (writes ~/.codex/config.toml)
rapid-mlx agents codex --setup

# 4. Run Codex
codex                              # interactive
codex exec "explain this repo"     # one-shot
```

## Manual config

`rapid-mlx agents codex --setup` writes this block; write it by hand if you
already have a `~/.codex/config.toml` you want to keep:

```toml
model = "qwen3.6-35b-4bit"   # or any rapid-mlx alias
model_provider = "rapid-mlx"

[model_providers.rapid-mlx]
name = "Rapid-MLX (local)"
base_url = "http://localhost:8000/v1"
```

Codex picks the provider from `model_provider` and resolves its `base_url`
from the matching `[model_providers.NAME]` block. Do **not** add an inline
`api_key = "..."` — Codex's `--strict-config` rejects it. If you started the
server with `--api-key`, use env-var indirection instead:

```toml
[model_providers.rapid-mlx]
name = "Rapid-MLX (local)"
base_url = "http://localhost:8000/v1"
env_key = "RAPID_MLX_API_KEY"
```

```bash
export RAPID_MLX_API_KEY=your-secret
```

## Per-family setup

Each family needs its matching tool-call parser and reasoning parser. These
are auto-detected from the alias, so the `serve` line below is all you need;
the parser column documents what rapid-mlx wires (and what the matrix test
exercises).

### Qwen 3.6 — recommended for Codex

Codex leans hard on multi-tool calls + `apply_patch`; Qwen 3.6 is the
recommended workhorse.

```bash
rapid-mlx serve qwen3.6-35b-4bit --port 8000   # ~20 GB, M3 Max / M4 Pro 24 GB+
# smaller: qwen3.6-27b-4bit
```

```toml
model = "qwen3.6-35b-4bit"
model_provider = "rapid-mlx"

[model_providers.rapid-mlx]
name = "Rapid-MLX (local)"
base_url = "http://localhost:8000/v1"
```

- tool-call parser: `qwen3_coder_xml` · reasoning parser: `qwen3`
- matrix cell: **PASS** ✅

### Gemma 4

```bash
rapid-mlx serve gemma-4-12b-4bit --port 8000   # ~7 GB at 4-bit
# larger: gemma-4-26b-4bit
```

- tool-call parser: `gemma4` · reasoning parser: `gemma4`
- matrix cell: **PASS** ✅ (Codex uses the text-only `/v1/responses` route)

> **Heads-up:** `gemma-4-12b-it` (any quant) can hang Codex on ~60% of
> tool-use prompts due to a model-side degenerate `thought\n…` loop
> (huggingface.co/google/gemma-4-12B-it/discussions/41). For sustained
> Codex agent workflows prefer the Qwen 3.5 / 3.6 models.

### gpt-oss

```bash
rapid-mlx serve gpt-oss-20b --port 8000        # ~11 GB (MXFP4-Q8)
# larger: gpt-oss-120b
```

- tool-call parser: `harmony` · reasoning parser: `harmony`
- matrix cell: **PASS** ✅ — rapid-mlx strips the harmony `analysis`
  channel out of visible content (no `<|channel|>analysis` leak).

## Model name passthrough

Codex sends model names like `gpt-5` or `gpt-5-codex` in the request body.
Rapid-mlx treats any `gpt-*` / `claude-*` model name as "the loaded engine,
not a strict alias lookup", so the request reaches whatever model you
started `rapid-mlx serve` with, regardless of what Codex thinks it's
talking to.

## Probing the endpoint

Verify the shim is reachable without booting Codex:

```bash
curl -sS http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5","input":"Say hello in one word.","stream":false}' | jq .
```

You should see a `response` object with an `output` array containing a
`message` item. With `--api-key`, add `-H "Authorization: Bearer <key>"`.

## Troubleshooting

- **Codex 404s on `/v1/responses`** — you're on rapid-mlx < 0.7.10. Upgrade
  with `rapid-mlx upgrade` (or `pip install -U rapid-mlx`).
- **Turn ends with no output / stream closes early** — upgrade to rapid-mlx
  >= 0.7.12; the 0.7.10–0.7.11 shim mishandled Codex 0.136's request shape.
- **Codex hangs on first run** — it's prompting for sandbox permissions
  (Landlock on Linux, Seatbelt on macOS). Accept them; the second run is
  non-interactive.

## See also

- [Agent support matrix](matrix.md)
- [Codex CLI guide (full shim internals)](../guides/codex-cli.md)
- [Server setup](../guides/server.md) · [Tool calling](../guides/tool-calling.md)
