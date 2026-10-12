# OpenHands

Point [OpenHands CLI](https://github.com/OpenHands/OpenHands-CLI) at a local
rapid-mlx server. OpenHands 1.x reaches the model over the **OpenAI-compatible
chat completions API** (`POST /v1/chat/completions`) via LiteLLM. The CLI can
run local tools without Docker; its full GUI server uses Docker. The current
CLI sends OpenAI-shaped tool calls for file and shell actions.

The existing [support matrix](matrix.md) exercises a pinned OpenHands 0.9.0
Docker image with its older CodeAct text-action format. Those historical
results do not verify the current 1.x CLI. Use `rapid-mlx agents openhands
--test` to exercise the installed CLI against your running model.

## How OpenHands reaches rapid-mlx

| Item | Value |
|---|---|
| Wire | `POST /v1/chat/completions` (via LiteLLM) |
| Base URL | `http://localhost:8000/v1` for the host CLI |
| Config | `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` env vars, applied with `--override-with-envs`; saved CLI settings live at `~/.openhands/agent_settings.json` |
| Auto-setup | `rapid-mlx agents openhands --setup` |
| Prereq | Python 3.12+ and `uv` for the CLI; Docker for `openhands serve` |

## TL;DR

```bash
# 1. Install the current OpenHands CLI
uv tool install openhands --python 3.12

# 2. Start rapid-mlx
rapid-mlx serve qwen3.6-35b-4bit --port 8000

# 3. Run setup and paste its three export lines into this shell
rapid-mlx agents openhands --setup
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_API_KEY=not-needed
export LLM_MODEL=openai/qwen3.6-35b-4bit

# 4. Apply those env vars explicitly (required by OpenHands 1.x)
openhands --override-with-envs --headless --always-approve -t "Fix the bug in add.py"
```

If you started the server with `--api-key`, set `LLM_API_KEY` to that key.
Without `--override-with-envs`, OpenHands ignores these exports and uses its
saved settings instead. `rapid-mlx agents openhands --test` now drives the
current CLI in a temporary HOME.

## Per-family setup

Swap the `serve` alias and `LLM_MODEL`. The parser is auto-detected from the
alias; the parser column documents what the matrix test exercises.

### Qwen 3.6

```bash
rapid-mlx serve qwen3.6-35b-4bit --port 8000   # ~20 GB
export LLM_BASE_URL=http://localhost:8000/v1 LLM_API_KEY=not-needed LLM_MODEL=openai/qwen3.6-35b-4bit
```

- tool-call parser: `qwen3_coder_xml` · reasoning parser: `qwen3`
- matrix cell: **PASS** ✅ (real Docker E2E: read → edit → finish)

### Gemma 4

```bash
rapid-mlx serve gemma-4-12b-4bit --port 8000   # ~7 GB at 4-bit
export LLM_BASE_URL=http://localhost:8000/v1 LLM_API_KEY=not-needed LLM_MODEL=openai/gemma-4-12b-4bit
```

- tool-call parser: `gemma4` · reasoning parser: `gemma4`
- matrix cell: **PASS** ✅

### gpt-oss

```bash
rapid-mlx serve gpt-oss-20b --port 8000        # ~11 GB (MXFP4-Q8)
export LLM_BASE_URL=http://localhost:8000/v1 LLM_API_KEY=not-needed LLM_MODEL=openai/gpt-oss-20b
```

- tool-call parser: `harmony` · reasoning parser: `harmony`
- matrix cell: **XFAIL (format)** — see note below.

> **gpt-oss + OpenHands is a known XFAIL.** gpt-oss's native harmony output
> (analysis + final channels, plain-markdown code in the final channel) does
> **not** emit the `<execute_bash>` / `<execute_ipython>` text-action XML
> tags that OpenHands' CodeActAgent parses. CodeActAgent treats the reply as
> an empty action, prompts for user input, and the non-interactive harness
> times out. This is an **upstream OpenHands parser gap**, not a rapid-mlx
> bug — the rapid-mlx wire-level harmony-stop fix landed in PR #1051. Tracked
> at
> [All-Hands-AI/OpenHands#15167](https://github.com/All-Hands-AI/OpenHands/issues/15167).
> Use Qwen 3.6 or Gemma 4 with OpenHands.

## Troubleshooting

- **Small local models struggle** — OpenHands sends a large system prompt and
  multiple tool definitions; prefer a larger model for file edits.
- **Full GUI server needs Docker** — start Docker Desktop / `dockerd` before
  running `openhands serve`. The host CLI quick start above does not need it.

## See also

- [Agent support matrix](matrix.md)
- [AI client compatibility](../guides/ai-clients.md)
- [Server setup](../guides/server.md) · [Tool calling](../guides/tool-calling.md)
