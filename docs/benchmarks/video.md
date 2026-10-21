# Video measurement

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start a catalog vision alias that fits your Mac:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen3-vl-2b-4bit --port 8000
```

Follow the [video guide](../guides/video-generation.md) for video generation or the [multimodal guide](../guides/multimodal.md) for video understanding. Measure elapsed time with the same input, prompt, model, and options when comparing runs. Record media duration or output frames, chip, unified memory, model alias, and Rapid-MLX version. The shipped `rapid-mlx bench` command measures text generation.
