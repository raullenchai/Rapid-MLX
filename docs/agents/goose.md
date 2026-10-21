# Goose with Rapid-MLX

## One-command local launch

Install Rapid-MLX on an Apple Silicon Mac with `curl -fsSL https://rapidmlx.com/install.sh | bash`, restart Terminal, and install Goose. Then run:

```bash
rapid-mlx launch goose
```

The command saves a private Rapid profile, starts a local server if needed, and launches Goose. Run it again next time. Use `rapid-mlx launch goose --remove` to remove the private profile.

Run `rapid-mlx recipe` for model recommendations or `rapid-mlx launch goose --model <alias>` to select a model. The first reply may wait for a download or model load. See the [current agent guide](https://rapidmlx.com/docs/agents/goose) for more detail.
