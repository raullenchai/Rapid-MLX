# Qwen Code

Point [Qwen Code](https://github.com/QwenLM/qwen-code) at a local rapid-mlx
server. Qwen Code is Alibaba's `gemini-cli` fork tuned for Qwen tool-calling;
it speaks the **OpenAI-compatible chat completions API**
(`POST /v1/chat/completions`) via an `openaiCompatible.baseUrl` config that
maps 1:1 onto rapid-mlx's default endpoint.

Verified via wire smoke against Qwen 3.6, Gemma 4, and gpt-oss in the Tier-1
agent matrix (`tests/integrations/test_agents_matrix.py::TestQwenCode`). See
the [support matrix](matrix.md) for the current per-family status.

## How Qwen Code reaches rapid-mlx

| Item | Value |
|---|---|
| Wire | `POST /v1/chat/completions` (OpenAI-compatible) |
| Base URL | `http://localhost:8000/v1` — **must include `/v1`** |
| Config file | `~/.qwen/settings.json` (or `./qwen.json`) |
| Auto-setup | `rapid-mlx agents qwen-code --setup` |

## TL;DR

```bash
# 1. Install Qwen Code
npm install -g @qwenlm/qwen-code

# 2. Start rapid-mlx (Qwen 3.6 is the sweet spot)
rapid-mlx serve qwen3.6-35b-4bit --port 8000

# 3. Point Qwen Code at the local server
rapid-mlx agents qwen-code --setup    # writes ~/.qwen/settings.json

# 4. Run it
qwen                          # interactive
qwen -p "explain this repo"   # one-shot
```

## Manual config

`~/.qwen/settings.json` — substitute your alias for `{model_id}`:

```json
{
  "openaiCompatible": {
    "baseUrl": "http://localhost:8000/v1",
    "apiKey": "not-needed",
    "model": "qwen3.6-35b-4bit"
  }
}
```

> **Note:** Qwen Code hardcodes the OpenAI-compat `/v1` suffix expectation —
> set `baseUrl` to `http://localhost:8000/v1`, **not** the root URL.

## Per-family setup

Swap the `serve` alias and the `model` field. The parser is auto-detected
from the alias; the parser column documents what the matrix test exercises.

### Qwen 3.6 — the native fit

Qwen Code was trained on Qwen3-Coder / Qwen3.x tool-calling, so the Qwen 3.6
family is the sweet spot.

```bash
rapid-mlx serve qwen3.6-35b-4bit --port 8000   # ~20 GB
# smaller: qwen3.6-27b-4bit
```

```json
{ "openaiCompatible": { "baseUrl": "http://localhost:8000/v1",
  "apiKey": "not-needed", "model": "qwen3.6-35b-4bit" } }
```

- tool-call parser: `qwen3_coder_xml` · reasoning parser: `qwen3`
- matrix cell: **PASS** ✅

### Gemma 4

```bash
rapid-mlx serve gemma-4-12b-4bit --port 8000   # ~7 GB at 4-bit
# larger: gemma-4-26b-4bit
```

```json
{ "openaiCompatible": { "baseUrl": "http://localhost:8000/v1",
  "apiKey": "not-needed", "model": "gemma-4-12b-4bit" } }
```

- tool-call parser: `gemma4` · reasoning parser: `gemma4`
- matrix cell: **PASS** ✅ (non-Qwen models run at reduced quality)

### gpt-oss

```bash
rapid-mlx serve gpt-oss-20b --port 8000        # ~11 GB (MXFP4-Q8)
# larger: gpt-oss-120b
```

```json
{ "openaiCompatible": { "baseUrl": "http://localhost:8000/v1",
  "apiKey": "not-needed", "model": "gpt-oss-20b" } }
```

- tool-call parser: `harmony` · reasoning parser: `harmony`
- matrix cell: **PASS** ✅

## Troubleshooting

- **`<think>` traces leaking into content on Qwen 3.6** — upgrade Qwen Code
  to >= 1.2; older releases stripped `chat_template_kwargs` before sending.
- **404 / connection issues** — confirm `baseUrl` ends in `/v1`, not the
  root host.

## See also

- [Agent support matrix](matrix.md)
- [AI client compatibility](../guides/ai-clients.md) · [Tool calling](../guides/tool-calling.md)
- [Server setup](../guides/server.md)
