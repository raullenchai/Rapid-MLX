# Gemma 4 12B

Install Rapid-MLX on an Apple Silicon Mac. Run `rapid-mlx recipe` to choose a model for your Mac, then start this Gemma alias:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve gemma-4-12b-4bit
```

The server prints its local API address. Run `rapid-mlx info gemma-4-12b-4bit` to inspect its capabilities and download size. Keep other memory-heavy apps closed if the model approaches your Mac's limit. For other Gemma sizes, see the [Gemma model guide](https://rapidmlx.com/docs/models/families/gemma.html) and `rapid-mlx recipe`.
