# Codex CLI

Install Rapid-MLX on an Apple Silicon Mac and install Codex CLI. Then run:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx launch codex
```

Rapid-MLX selects a model for this Mac, starts a local server, and opens Codex with a private profile. Run the same `rapid-mlx launch codex` command next time. The first reply may take longer while the model downloads and loads. If the command is not found after installation, run `export PATH="$HOME/.local/bin:$PATH"` in the current terminal.

## Choose a model

Run `rapid-mlx recipe` to see models that fit this Mac. To use a particular model, run `rapid-mlx launch codex --model <alias>`. The alias must be supported by the current [model catalog](https://rapidmlx.com/docs/aliases.html).

## Manage the connection

`rapid-mlx launch codex --remove` removes Rapid-MLX's private profile. Your existing Codex settings stay in place. Run `rapid-mlx doctor` if the local service does not start. See the [Codex agent guide](../agents/codex-cli.md) for launch options and the [server guide](server.md) for API clients.
