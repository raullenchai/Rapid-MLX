# Model recommendations

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and see the model
choices for that Mac:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx recipe
```

Choose a model that fits your memory, then measure it with the
[text benchmark](llm.md). Results depend on the chip, available memory, prompt,
and model revision. Treat measurements from another machine as context, not as
an expected speed for yours.

## How recommendations are maintained

The CLI and Desktop read the same checked-in
[recommendation policy](../../rapid_mlx/model_recommendations.json). It defines
memory tiers, model aliases, and the two roles shown for each tier. The model
catalog defines the aliases and their requirements.

Maintainers review capability, interactive speed, and memory use together.
Record the model revision, machine, operating system, run settings, and raw
results when proposing a change. Keep measured results separate from the
current recommendation policy so a past run is not presented as today's
performance promise.

The repository's [measurement script](../../scripts/benchmark_model_recommendations.py)
runs one cached model at a time through `rapid-mlx serve` on macOS. It needs a
[development environment](../../CONTRIBUTING.md) and model weights in the local
cache. Use `--allow-download` if the weights are not already cached:

```bash
python3 scripts/benchmark_model_recommendations.py qwen3.5-4b-4bit \
  --allow-download --output /tmp/model-recommendations.json
```

Run `python3 scripts/benchmark_model_recommendations.py --help` for current
options. The script writes raw JSON for review; it does not update the product
recommendations by itself.
