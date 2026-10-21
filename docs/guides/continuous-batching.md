# Continuous batching

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start a local server:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3.5-4b-4bit
```

Continuous batching is enabled while the server handles concurrent requests. No extra flag is needed. The server prints the selected port; send compatible clients to its `/v1` endpoint.

## Memory controls

Run `rapid-mlx serve --help` for the cache and admission flags in your installed version. Start with defaults, then measure your own workload before changing cache capacity or batching limits. The [performance flags guide](https://rapidmlx.com/docs/perf-flags.html) covers tuning choices.
