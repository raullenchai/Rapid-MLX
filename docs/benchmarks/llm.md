# Text generation benchmark

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and benchmark a catalog alias that fits your Mac:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx bench qwen3.5-4b-4bit --num-prompts 5 --max-tokens 256
```

The output measures the model on the Mac that ran it. Run `rapid-mlx recipe` to select another alias, then repeat the same command with that alias. Run `rapid-mlx bench --help` for current prompt count, token, cache, and concurrency options.

For a useful comparison, keep the model, prompt count, token limit, and cache settings the same across runs. Record chip, unified memory, model alias, and Rapid-MLX version with your result. The [model recommendation methodology](model-recommendations.md) explains how catalog picks are evaluated.
