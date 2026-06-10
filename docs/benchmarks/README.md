# Benchmarks

Performance benchmarks for rapid-mlx on Apple Silicon.

## Benchmark Types

- [LLM Benchmarks](llm.md) - Text generation performance
- [Image Benchmarks](image.md) - Image understanding performance
- [Video Benchmarks](video.md) - Video understanding performance

## Quick Commands

```bash
# LLM benchmark — short aliases work
rapid-mlx bench qwen3.5-4b-4bit

# Or by full HF repo (vision/multimodal benches live in scripts/ — they are
# dev-only and not shipped with `pip install rapid-mlx`)
rapid-mlx bench mlx-community/Qwen3.5-9B-4bit
```

## Standalone Test Defaults

Standalone benchmark test scripts have built-in default models, so you can run:

```bash
python tests/test_continuous_batching.py
python tests/test_prefix_cache.py
```

Defaults:
- `tests/test_continuous_batching.py` → `mlx-community/Qwen3-8B-6bit`
- `tests/test_prefix_cache.py` → `mlx-community/Qwen3-0.6B-8bit`

To test different models, use the optional `--model` flag:

```bash
python tests/test_continuous_batching.py --model mlx-community/Qwen3-0.6B-8bit
python tests/test_prefix_cache.py --model mlx-community/Qwen3-8B-6bit
```

## Hardware

Benchmarks have been collected on the following Apple Silicon configurations:

| Chip | Memory | Python |
|------|--------|--------|
| Apple M4 Max | 128 GB unified | 3.13 |
| Apple M1 Max | 64 GB unified | 3.12 |

Results will vary on different Apple Silicon chips.

## Contributing Benchmarks

If you have a different Apple Silicon chip, please share your results:

```bash
rapid-mlx bench qwen3.5-4b-4bit | tee results.txt
```

Open an issue with your results at [GitHub Issues](https://github.com/raullenchai/Rapid-MLX/issues).
