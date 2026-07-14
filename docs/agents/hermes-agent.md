# Hermes Agent

Point [Nous Research's Hermes Agent](https://github.com/NousResearch/hermes-agent)
at a local rapid-mlx server. Hermes is a tool-heavy CLI agent (it injects up
to 62 tools per request) that speaks the **OpenAI-compatible chat completions
API** (`POST /v1/chat/completions`).

Verified via wire smoke in the Tier-1 agent matrix
(`tests/integrations/test_agents_matrix.py::TestHermesAgent`), with the deep
62-tool E2E flow in `test_hermes.py`. See the [support matrix](matrix.md) for
the current per-family status.

## How Hermes reaches rapid-mlx

| Item | Value |
|---|---|
| Wire | `POST /v1/chat/completions` (OpenAI-compatible) |
| Base URL | `http://localhost:8000/v1` |
| Config file | `~/.hermes/config.yaml` |
| Auto-setup | `rapid-mlx agents hermes --setup` |
| Test suite | `rapid-mlx agents hermes --test` (runs `test_hermes.py`) |

## TL;DR

```bash
# 1. Install Hermes Agent (a venv is recommended — see troubleshooting)
pip install hermes-agent

# 2. Start rapid-mlx
rapid-mlx serve qwen3.6-35b-4bit --port 8000

# 3. Point Hermes at the local server
rapid-mlx agents hermes --setup    # writes ~/.hermes/config.yaml

# 4. Run it
hermes chat -q "explain this repo" -Q
```

## Manual config

`~/.hermes/config.yaml` — substitute your alias for `{model_id}`:

```yaml
model:
  provider: "custom"
  default: "qwen3.6-35b-4bit"
  base_url: "http://localhost:8000/v1"
  context_length: 32768
  max_tokens: 4096
platform_toolsets:
  cli: [terminal, file, code_execution, web, browser, skills]
```

## Per-family setup

Swap the `serve` alias and the `default` model field. The parser is
auto-detected from the alias; the parser column documents what the matrix
test exercises.

### Qwen 3.6

```bash
rapid-mlx serve qwen3.6-35b-4bit --port 8000   # ~20 GB
```

```yaml
model:
  provider: "custom"
  default: "qwen3.6-35b-4bit"
  base_url: "http://localhost:8000/v1"
```

- tool-call parser: `qwen3_coder_xml` · reasoning parser: `qwen3`
- matrix cell: **PASS** ✅

### Gemma 4

```bash
rapid-mlx serve gemma-4-12b-4bit --port 8000   # ~7 GB at 4-bit
# larger: gemma-4-26b-4bit
```

```yaml
model:
  provider: "custom"
  default: "gemma-4-12b-4bit"
  base_url: "http://localhost:8000/v1"
```

- tool-call parser: `gemma4` · reasoning parser: `gemma4`
- matrix cell: **PASS** ✅ (see the `todo`-tool note below)

### gpt-oss

```bash
rapid-mlx serve gpt-oss-20b --port 8000        # ~11 GB (MXFP4-Q8)
# larger: gpt-oss-120b
```

```yaml
model:
  provider: "custom"
  default: "gpt-oss-20b"
  base_url: "http://localhost:8000/v1"
```

- tool-call parser: `harmony` · reasoning parser: `harmony`
- matrix cell: **PASS** ✅

## Troubleshooting

- **`todo` tool conflicts with Gemma 4** — Hermes injects a `todo` tool that
  errors on Gemma 4. Exclude it from `platform_toolsets` or pass the `-t`
  flag.
- **Gemma 4 `[Calling tool` mimicry** — after many tool turns Gemma 4 may
  parrot Hermes' UI text. Fixed in rapid-mlx v0.4.3+.
- **`pip install hermes-agent` fails** — clone and install from source into
  a venv; some systems can't resolve the PyPI wheel.

## See also

- [Agent support matrix](matrix.md)
- [AI client compatibility](../guides/ai-clients.md) · [Tool calling](../guides/tool-calling.md)
- [Server setup](../guides/server.md)
