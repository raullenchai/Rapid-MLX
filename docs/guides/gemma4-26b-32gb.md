# Gemma 4 26B

Install Rapid-MLX on an Apple Silicon Mac. Run `rapid-mlx recipe` to choose a model for your Mac, then start this Gemma alias:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve gemma-4-26b-4bit
```

The server prints its local API address. Run `rapid-mlx info gemma-4-26b-4bit` to inspect its capabilities and download size. If this model does not fit your workload, run `rapid-mlx recipe` for another pick. The [Gemma model guide](https://rapidmlx.com/docs/models/families/gemma.html) covers the family.
