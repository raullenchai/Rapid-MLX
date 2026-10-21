# Continue

Install Rapid-MLX on an Apple Silicon Mac and restart Terminal:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3.5-4b-4bit
```

Keep the server running. In another terminal, run `rapid-mlx launch continue-dev` to configure the detected Continue client. Run `rapid-mlx launch continue-dev --dry-run` to preview changes. See the [current integration guide](https://rapidmlx.com/docs/agents/) for editor steps.
