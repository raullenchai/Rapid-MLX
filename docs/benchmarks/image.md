# Image measurement

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start a catalog vision alias that fits your Mac:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3-vl-2b-4bit --port 8000
```

If port 8000 is busy, choose another `--port` and use that address in the requests. Follow the [multimodal guide](../guides/multimodal.md) to send an image request. Measure elapsed time with the same image, prompt, and model when comparing runs. Record image dimensions, chip, unified memory, model alias, and Rapid-MLX version. The shipped `rapid-mlx bench` command measures text generation.
