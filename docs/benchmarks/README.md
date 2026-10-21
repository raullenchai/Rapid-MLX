# Benchmarks

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and run a short text-generation benchmark with a catalog alias that fits your Mac:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx bench qwen3.5-4b-4bit --num-prompts 5 --max-tokens 256
```

The command prints measurements for this Mac and this model. Use `rapid-mlx recipe` to choose another RAM-fit alias, and `rapid-mlx bench --help` for current benchmark options. Model loading happens on the machine running the command.

## What to measure

- [Text generation](llm.md): use the shipped `rapid-mlx bench` command.
- [Audio](audio.md), [images](image.md), and [video](video.md): use representative requests and record latency on your own Mac. The text benchmark command does not measure these modalities.
- [Model recommendation methodology](model-recommendations.md): read the measurement gates behind the catalog recommendations.

Record the model alias, chip, unified memory, prompt or media input, and run settings with any result. Those details matter more than a speed number copied from another machine.
