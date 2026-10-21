# Hermes Agent

## One-command local launch

Install Rapid-MLX on an Apple Silicon Mac with `curl -fsSL https://rapidmlx.com/install.sh | bash` and restart the terminal (or run `export PATH="$HOME/.local/bin:$PATH"`), then install Hermes Agent. Run `rapid-mlx launch hermes`. The same command works after a restart; `rapid-mlx launch hermes --remove` deletes only Rapid's private profile. Existing Hermes Agent settings are left untouched. Use `--model qwen3.5-9b-4bit` to choose that model on a Mac with enough RAM; the default recommendation follows the Mac's memory tier. The first reply may wait for a download, model load, or a long agent prompt. The manual setup steps below are for users who want to edit their regular agent configuration.


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

## Current release (v0.21.6)

The [v0.21.6 source release](https://github.com/NousResearch/hermes-agent/releases/tag/v0.21.6)
requires Python 3.14 for its runtime dependencies. The
[PyPI package](https://pypi.org/project/hermes-agent/) is currently older than
v0.21.6, so `pip install hermes-agent` does not install this release. In a
fresh directory, install the source release with `uv`:

```bash
# Start the server in one terminal.
rapid-mlx serve qwen3.5-4b-4bit --port 8000

# In another terminal, install Hermes and keep this terminal for setup and chat.
git clone --depth 1 --branch v0.21.6 https://github.com/NousResearch/hermes-agent.git "$HOME/hermes-agent"
uv venv --python 3.14 "$HOME/hermes-agent/.venv"
uv pip install --python "$HOME/hermes-agent/.venv/bin/python" -e "$HOME/hermes-agent"
export PATH="$HOME/hermes-agent/.venv/bin:$PATH"
rapid-mlx agents hermes --setup --yes
hermes chat -q "Run cat README.md using the terminal tool and report its title" -Q -t terminal
```

The `--setup` command reads the model ID and context window from the running
server, then writes `~/.hermes/config.yaml` (or `$HERMES_HOME/config.yaml`).
Hermes 0.21.6 reads `model.default`, `model.provider`, `model.base_url`, and
`model.context_length` for this local endpoint. It ignores the former
`model.max_tokens` output setting, so a fresh setup omits it. Setup preserves
that field in an existing config; remove it there if you want a clean file.

## Manual config

`~/.hermes/config.yaml` — substitute your alias for `{model_id}`:

```yaml
model:
  provider: "custom"
  default: "qwen3.5-4b-4bit"
  base_url: "http://localhost:8000/v1"
  context_length: 262144
platform_toolsets:
  cli: [terminal, file, code_execution, web, browser, skills, image_gen]
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
- **A Python 3.11 or 3.12 source install starts without its runtime
  dependencies** — use Python 3.14 for the current source release.

## See also

- [Agent support matrix](matrix.md)
- [AI client compatibility](../guides/ai-clients.md) · [Tool calling](../guides/tool-calling.md)
- [Server setup](../guides/server.md)
