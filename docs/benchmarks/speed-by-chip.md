# How fast will it be on my Mac?

Text generation speed depends mostly on memory bandwidth: how quickly the chip reads model weights. The same model can run at very different speeds on base, Pro, Max, and Ultra chips from the same generation. A result from a Pro or Max Mac is not a speed guarantee for a base chip. Every speed in this table names the chip it was measured on.

## Community measurements

These are median decode speeds in tokens per second, with unified memory in parentheses. Decode is the generation of output after the prompt is processed. The community submitted these results to the [leaderboard](https://rapidmlx.com/leaderboard); the data was fetched on October 28, 2026, using the `rapid-community-speed v2` protocol. Each median combines the available runs for that chip, memory size, and exact model, including short and long prompt cases. Your speed can differ with the prompt, settings, and other work on your Mac. A dash means no measurement for that exact model, not that it cannot run.

| Chip | Qwen3.5 4B 4-bit | Qwen3.5 9B 4-bit | Gemma 4 E4B 4-bit | Qwen3.6 35B-A3B 4-bit | Qwen3.8 27B 4-bit MTP |
|---|---:|---:|---:|---:|---:|
| M4 | 38.4 (16 GB) | 21.9 (16 GB) | 35.1 (16 GB) | — | — |
| M5 | 46.5 (32 GB) | 25.9 (32 GB) | 42.2 (32 GB) | — | — |
| M3 Pro | 48.3 (36 GB) | 28.0 (36 GB) | 42.5 (36 GB) | 58.9 (36 GB) | 8.8 (36 GB) |
| M4 Pro | 82.7 (48 GB)† | 49.2 (48 GB)† | — | — | — |
| M5 Pro | — | — | — | — | 17.6 (64 GB) |
| M1 Max | 83.3 (32 GB) | — | — | — | 18.1 (32 GB)* |
| M3 Max | 90.7 (36 GB) | 53.8 (36 GB) | 73.5 (36 GB) | 101.1 (36 GB) | 17.6 (36 GB) |
| M4 Max | — | — | — | 129.2 (128 GB) | 24.8 (128 GB) |
| M5 Max | — | — | — | — | 32.5 (128 GB) |
| M5 Ultra | — | — | — | — | 53.6 (96 GB); 54.2 (256 GB) |

The exact model IDs are `mlx-community/Qwen3.5-4B-MLX-4bit`, `mlx-community/Qwen3.5-9B-4bit`, `mlx-community/gemma-4-e4b-it-4bit`, `mlx-community/Qwen3.6-35B-A3B-4bit`, and `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX`. *The M1 Max 27B result uses the distinct `rapid-mlx/Qwen3.8-27B-4bit-MTP-fp16-MLX` checkpoint. M4 Pro submissions used different 35B checkpoints and are not combined with this table's model. †M4 Pro 4B and 9B values are the median of the short and long cases in the [M4 Pro 48 GB benchmark record](m4-pro-48gb-community.md), measured with Rapid-MLX 0.14.1 using the same protocol (the `qwen3.5-4b-4bit` and `qwen3.5-9b-4bit` aliases). That record also measured the dense Qwen3.8 27B 4-bit without MTP at about 15.4 tok/s.

The M4 4B result includes Rapid-MLX versions 0.16.8 and 0.16.10. Other M4, M1 Max, M3 Pro, M3 Max, and M4 Max results use 0.16.8. M5 and M5 Pro results use 0.16.5; M5 Max uses 0.16.9. M5 Ultra uses 0.16.8 for 96 GB and 0.15.7 for 256 GB.

## Memory bandwidth

Apple publishes these figures for [M4, M4 Pro, and M4 Max configurations](https://support.apple.com/en-us/121553). Max bandwidth depends on the chip configuration.

| Chip | Memory bandwidth (GB/s) |
|---|---:|
| M4 | 120 |
| M4 Pro | 273 |
| M4 Max | 410 or 546 |

## Rules of thumb

- For a dense model, decode tokens per second are roughly memory bandwidth divided by the model's size in memory. This is a rough upper bound, not a promised speed.
- Mixture-of-experts (MoE) models such as 35B-A3B can be much faster than their total size suggests because only the active parameters are read for each token.
- Prompt processing also depends on GPU compute and core count. Long prompts can take longer to process on base chips before the first token appears.

## Memory bandwidth vs internet bandwidth

Once a model is downloaded, running it locally does not require the internet. Internet bandwidth controls how quickly data travels over your network, including the initial download. Memory bandwidth controls how quickly the chip can read model weights from unified memory as it generates each token. More RAM lets a larger model fit. More memory bandwidth can make generation faster. That is one reason an M4 Max can generate faster than an M4 Pro even when both have the same amount of RAM.

## Measure your Mac

List the available benchmark models, then run one on your Mac:

```bash
rapid-mlx benchmark catalog
rapid-mlx benchmark run <model>
```

To submit a completed run to the community leaderboard, use its run ID:

```bash
rapid-mlx benchmark share <run-id>
```

See the [community benchmark guide](m4-pro-48gb-community.md) for more details.

`benchmark run` uses the same acceleration profile that `serve <alias>` selects
by default. Use `--baseline` to measure without accelerations. The acceleration
flags have the same names and meanings as for `serve`: `--speculative-config`,
`--no-spec-decode`, `--mtp-max-k`, `--kv-cache-quantization*`,
`--kv-cache-dtype`, `--prefill-step-size`, `--pflash`,
`--pflash-threshold`, and `--pflash-keep-ratio`. Quantized KV is recorded only
when the engine actually ran with it. Dedicated DFlash, native MTP and companion DSpark runtimes are measured too;
TensorFold profiles use the same serial runtime as serve. Select a DFlash drafter with
`--speculative-config '{"method":"dflash","model":"<drafter>"}'`, or disable
speculation with `--no-spec-decode` or all accelerations with `--baseline`.
DDTree is refused because its non-streaming runtime cannot measure time to first
token. Suffix decoding is refused because the workload repeats one prompt.
Prefix cache remains off. The result records the effective configuration. Engine
logs are hidden by default so the output shows only progress, warnings and the
summary; add `--log-level INFO` (or set `RAPID_MLX_LOG_LEVEL`) to see them.

With `--json`, stdout contains exactly one JSON document; progress goes to stderr.
Results are stored under `$RAPID_MLX_HOME/benchmarks` when `RAPID_MLX_HOME` is
set. Older results under `~/.rapid-mlx/benchmarks` remain readable.
