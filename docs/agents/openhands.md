# OpenHands

Point the current [OpenHands agent-canvas](https://github.com/All-Hands-AI/OpenHands)
at a Rapid-MLX OpenAI-compatible endpoint. Current OpenHands routes completions
through LiteLLM and requires the model name to carry the `openai/` prefix.
`rapid-mlx launch openhands` adds that prefix automatically.

## How configuration works

| Item | Value |
|---|---|
| Wire | `POST /v1/chat/completions` via LiteLLM |
| Rapid base URL | `http://127.0.0.1:<rapid-port>/v1` |
| Stored config | `~/.openhands/settings.json` |
| Setup command | `rapid-mlx launch openhands` |
| Prerequisite | OpenHands agent-canvas must be running |

OpenHands no longer reads `LLM_BASE_URL`, `LLM_API_KEY`, or `LLM_MODEL`.
Its stored API key is encrypted with an app-owned key, so writing
`settings.json` directly is not valid. The launch adapter authenticates to the
running app's `/api/settings` endpoint and lets OpenHands encrypt and persist
the credential itself.

## Setup

OpenHands and Rapid cannot both use port 8000. Keep Rapid on 8000 and start the
agent-canvas ingress on another port such as 8010:

```bash
# Terminal 1: Rapid-MLX
RAPID_MLX_API_KEY=your-secret rapid-mlx serve qwen3.6-35b-4bit --port 8000

# Terminal 2: OpenHands agent-canvas
npx @openhands/agent-canvas --port 8010

# Terminal 3: write the provider settings through the running app
RAPID_MLX_API_KEY=your-secret rapid-mlx launch openhands \
  --server-url http://127.0.0.1:8000 \
  --model qwen3.6-35b-4bit
```

The adapter probes common agent-canvas ports. For a custom address, set
`OPENHANDS_URL`, for example:

```bash
OPENHANDS_URL=http://127.0.0.1:9123 \
RAPID_MLX_API_KEY=your-secret \
rapid-mlx launch openhands --server-url http://127.0.0.1:8000 \
  --model qwen3.6-35b-4bit
```

To have the CLI start Rapid as well, omit both port flags. The command reserves
8001 for Rapid because OpenHands normally owns 8000:

```bash
RAPID_MLX_API_KEY=your-secret \
rapid-mlx launch openhands --start-server --model qwen3.6-35b-4bit
```

## Manual setup

In OpenHands, open **Settings → LLM** and enter:

| Field | Value |
|---|---|
| Model | `openai/qwen3.6-35b-4bit` |
| Base URL | `http://127.0.0.1:8000/v1` |
| API key | The value used in `RAPID_MLX_API_KEY` |

The `openai/` model prefix is required for LiteLLM routing. A bare Rapid model
alias fails before a request reaches Rapid-MLX.

## Troubleshooting

- **OpenHands session key not found** — start agent-canvas once so it creates
  `~/.openhands/agent-canvas/api-key.txt`.
- **Something else is on port 8000** — keep Rapid on 8000 and move the
  agent-canvas ingress, or set `OPENHANDS_URL` to its actual port.
- **The saved key stopped working after Rapid restarted** — the desktop app
  rotates its bearer each launch; copy and run the Launch-page command again.
- **Model/provider not found** — confirm the model is stored as
  `openai/<rapid-alias>`, not a bare alias.
- **The agent loop stalls** — use a model that reliably performs native tool
  calls; small chat-oriented models often answer without invoking a tool.

## See also

- [Agent support matrix](matrix.md)
- [AI client compatibility](../guides/ai-clients.md)
- [Server setup](../guides/server.md)
