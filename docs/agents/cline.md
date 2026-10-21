# Cline

Install Rapid-MLX on an Apple Silicon Mac and restart Terminal:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3.5-4b-4bit
```

Keep the server running. In another terminal, run `rapid-mlx launch cline` to configure the detected Cline client. Run `rapid-mlx launch cline --dry-run` to preview changes. See the [current integration guide](https://rapidmlx.com/docs/agents/) for editor steps.
