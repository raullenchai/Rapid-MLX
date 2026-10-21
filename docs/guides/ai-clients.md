# AI clients

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, then run an installed coding agent with a local model:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
rapid-mlx launch codex
```

Use `claude-code`, `opencode`, `pi`, `hermes`, `crush`, or `goose` in place of `codex`. The command creates a private Rapid profile, starts the server if needed, and opens the agent. Use the same command on later days. `rapid-mlx launch codex --remove` removes Rapid's Codex profile. Run `rapid-mlx launch list` for additional integrations.

The [agent guides](https://rapidmlx.com/docs/agents/) cover each path. Claude Desktop uses its [official in-app Gateway settings](../integrations/claude-desktop.md).

## Connect an app or SDK

Start a local server:

```bash
rapid-mlx serve qwen3.5-4b-4bit --port 8000
```

Wait for the ready address. An OpenAI-compatible client uses `http://localhost:8000/v1`; an Anthropic-compatible client uses `http://localhost:8000`. Set the model to `default` when the client accepts a model ID. A client that requires an API key can use a nonempty placeholder unless server authentication is enabled.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

## Other clients

Apps and frameworks that let you set a local compatible endpoint can use the same server address. [Aider](../agents/aider.md) and [Qwen Code](../agents/qwen-code.md) have short setup guides. Consult the [server guide](server.md) for authentication and network options and the [CLI reference](../reference/cli.md) for flags.

## Troubleshooting

- Run `rapid-mlx doctor` to check the installation.
- If a connection fails, check that the server printed a ready address and use its actual port.
- For an Anthropic SDK client, use the server root. The SDK adds `/v1/messages` itself.
- If a model cannot call tools, choose a tool-capable model with `rapid-mlx recipe` and consult the [tool-calling guide](tool-calling.md).
