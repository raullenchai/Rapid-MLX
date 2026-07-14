# OpenCode

Point [OpenCode](https://github.com/sst/opencode) at a local rapid-mlx
server. OpenCode is a Claude-Code-like terminal coding agent that speaks the
**OpenAI-compatible chat completions API** (`POST /v1/chat/completions`) via
the `@ai-sdk/openai-compatible` provider.

Verified via wire smoke against Qwen 3.6, Gemma 4, and gpt-oss in the Tier-1
agent matrix (`tests/integrations/test_agents_matrix.py::TestOpenCode`). See
the [support matrix](matrix.md) for the current per-family status.

## How OpenCode reaches rapid-mlx

| Item | Value |
|---|---|
| Wire | `POST /v1/chat/completions` (OpenAI-compatible) |
| Base URL | `http://localhost:8000/v1` |
| Config file | `~/.config/opencode/opencode.json` (or `./opencode.json`) |
| Auto-setup | `rapid-mlx agents opencode --setup` |

## TL;DR

```bash
# 1. Install OpenCode
brew install sst/tap/opencode   # or: npm install -g opencode-ai

# 2. Start rapid-mlx
rapid-mlx serve qwen3.6-35b-4bit --port 8000

# 3. Point OpenCode at the local server
rapid-mlx agents opencode --setup    # writes ~/.config/opencode/opencode.json

# 4. Run OpenCode
opencode
```

## Manual config

`rapid-mlx agents opencode --setup` writes this `~/.config/opencode/opencode.json`.
Substitute your alias for `{model_id}`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "rapid-mlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Rapid-MLX",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "qwen3.6-35b-4bit": {}
      }
    }
  },
  "model": "rapid-mlx/qwen3.6-35b-4bit"
}
```

## Per-family setup

Swap the `serve` alias and the `models` / `model` entries. The parser is
auto-detected from the alias; the parser column documents what the matrix
test exercises.

### Qwen 3.6

```bash
rapid-mlx serve qwen3.6-35b-4bit --port 8000   # ~20 GB
# smaller: qwen3.6-27b-4bit
```

```json
{ "models": { "qwen3.6-35b-4bit": {} }, "model": "rapid-mlx/qwen3.6-35b-4bit" }
```

- tool-call parser: `qwen3_coder_xml` · reasoning parser: `qwen3`
- matrix cell: **PASS** ✅

### Gemma 4

```bash
rapid-mlx serve gemma-4-12b-4bit --port 8000   # ~7 GB at 4-bit
# larger: gemma-4-26b-4bit
```

```json
{ "models": { "gemma-4-12b-4bit": {} }, "model": "rapid-mlx/gemma-4-12b-4bit" }
```

- tool-call parser: `gemma4` · reasoning parser: `gemma4`
- matrix cell: **PASS** ✅

### gpt-oss

```bash
rapid-mlx serve gpt-oss-20b --port 8000        # ~11 GB (MXFP4-Q8)
# larger: gpt-oss-120b
```

```json
{ "models": { "gpt-oss-20b": {} }, "model": "rapid-mlx/gpt-oss-20b" }
```

- tool-call parser: `harmony` · reasoning parser: `harmony`
- matrix cell: **PASS** ✅

## Troubleshooting

- **First run prompts for an Anthropic API key** — choose "skip". Rapid-mlx
  supplies the model through the OpenAI-compatible provider above.
- **Config template doesn't load** — the OpenCode config schema shifts
  across versions. Run `opencode --help` and check
  `~/.config/opencode/opencode.json` against the docs at opencode.ai.
- **OpenCode is interactive-only** — there's no headless one-shot query
  mode, so `rapid-mlx agents opencode --test` skips the query check.

## See also

- [Agent support matrix](matrix.md)
- [AI client compatibility](../guides/ai-clients.md) · [Tool calling](../guides/tool-calling.md)
- [Server setup](../guides/server.md)
