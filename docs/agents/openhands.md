# OpenHands

Point [OpenHands](https://github.com/All-Hands-AI/OpenHands) (formerly
OpenDevin) at a local rapid-mlx server. OpenHands drives its `CodeActAgent`
inside a Docker sandbox and reaches the model over the **OpenAI-compatible
chat completions API** (`POST /v1/chat/completions`) via LiteLLM.

Unlike the OpenAI-tool-calling agents, OpenHands uses a **text-action
format** — the CodeActAgent parses the model's plaintext reply for
`<execute_bash>` / `<execute_ipython>` action tags and file edits, so the
correctness signal is "did OpenHands rewrite the file", not "did tool_calls
fire". Verified via the real Docker E2E harness
(`tests/integrations/test_agents_matrix.py::TestOpenHands`, which shells out
to `test_openhands.sh`). See the [support matrix](matrix.md) for status.

## How OpenHands reaches rapid-mlx

| Item | Value |
|---|---|
| Wire | `POST /v1/chat/completions` (via LiteLLM, inside Docker) |
| Base URL | `http://localhost:8000/v1` (host) — the sandbox rewrites the host to `host.docker.internal` internally |
| Config | `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` env vars |
| Auto-setup | `rapid-mlx agents openhands --setup` |
| Prereq | Docker daemon running (sandbox runtime) |

## TL;DR

```bash
# 1. Install OpenHands (needs Docker for the sandbox)
pip install openhands

# 2. Start rapid-mlx
rapid-mlx serve qwen3.6-35b-4bit --port 8000

# 3. Export the LiteLLM env vars OpenHands reads
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_API_KEY=not-needed
export LLM_MODEL=qwen3.6-35b-4bit

# 4. Run OpenHands (or `rapid-mlx agents openhands --setup`)
python -m openhands.core.main -t "Fix the bug in add.py"
```

If you started the server with `--api-key`, set `LLM_API_KEY` to that key.

## Per-family setup

Swap the `serve` alias and `LLM_MODEL`. The parser is auto-detected from the
alias; the parser column documents what the matrix test exercises.

### Qwen 3.6

```bash
rapid-mlx serve qwen3.6-35b-4bit --port 8000   # ~20 GB
export LLM_BASE_URL=http://localhost:8000/v1 LLM_API_KEY=not-needed LLM_MODEL=qwen3.6-35b-4bit
```

- tool-call parser: `qwen3_coder_xml` · reasoning parser: `qwen3`
- matrix cell: **PASS** ✅ (real Docker E2E: read → edit → finish)

### Gemma 4

```bash
rapid-mlx serve gemma-4-12b-4bit --port 8000   # ~7 GB at 4-bit
export LLM_BASE_URL=http://localhost:8000/v1 LLM_API_KEY=not-needed LLM_MODEL=gemma-4-12b-4bit
```

- tool-call parser: `gemma4` · reasoning parser: `gemma4`
- matrix cell: **PASS** ✅

### gpt-oss

```bash
rapid-mlx serve gpt-oss-20b --port 8000        # ~11 GB (MXFP4-Q8)
export LLM_BASE_URL=http://localhost:8000/v1 LLM_API_KEY=not-needed LLM_MODEL=gpt-oss-20b
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

- **"Cannot connect to Docker daemon"** — OpenHands needs Docker for its
  sandbox runtime. Start Docker Desktop / `dockerd` first.
- **Small local models struggle** — the CodeActAgent text-action format is
  demanding; prefer the 27–35B tier (`qwen3.6-35b-4bit`).
- **First cold-cache run is slow** — OpenHands pulls two sandbox images
  (~3.4 GB + ~9 GB uncompressed) and builds a runtime layer; subsequent runs
  reuse the hash-tagged image (30–75 s per task).

## See also

- [Agent support matrix](matrix.md)
- [AI client compatibility](../guides/ai-clients.md)
- [Server setup](../guides/server.md) · [Tool calling](../guides/tool-calling.md)
