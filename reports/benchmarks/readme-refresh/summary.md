# README benchmark refresh — B=4 concurrent throughput

Generated: 2026-06-06
M3 Ultra 256 GB · macOS 25.3.0
Engines: rapid-mlx v0.6.80 · mlx-lm 0.31.3 · Ollama 0.24.0

Workload: 4 concurrent streaming requests, ~32 input tokens, 256 max output
tokens each, temperature 0.7, top_p 0.95.
Thinking-off requested via `chat_template_kwargs.enable_thinking=False`,
which rapid-mlx / mlx-lm / mlx-vlm honour; Ollama 0.24 ignores it for
Qwen3 and keeps streaming `delta.reasoning` chunks — those chunks decode
at the same model rate as content tokens so we count them, which means
the Qwen3 Ollama numbers reflect CoT-on throughput in practice.
Metric: aggregate tok/s = sum(output_tokens across 4 streams) / wall_clock
(authoritative token count comes from the streaming `usage` chunk, not
from counting SSE frames).
Each engine reported as the median of 3 measured rounds after 1 discarded
warmup. Engines were swapped sequentially (8 s cooldown between) so Metal
contention never crossed engine boundaries.

## Results

| Model (MLX alias)                  | rapid-mlx | mlx-lm    | Ollama tag                        | Ollama | vs mlx-lm | vs Ollama |
|------------------------------------|----------:|----------:|-----------------------------------|-------:|----------:|----------:|
| qwen3.5-4b                         |     261.1 |     173.2 | qwen3:4b                          |  119.5 |     1.51x |     2.18x |
| qwen3.5-9b                         |     180.0 |     136.3 | qwen3:8b                          |   84.1 |     1.32x |     2.14x |
| qwen3.5-27b                        |      65.9 |      54.9 | qwen3:32b¹                        |   27.1 |     1.20x |     2.43x |
| gemma-4-12b                        |      55.4 |     crash²| gemma3:12b                        |   56.1 |       —   |     0.99x |
| gpt-oss-20b                        |     220.5 |     162.0 | gpt-oss:20b                       |   96.5 |     1.36x |     2.29x |
| qwen3.6-35b (A3B 4-bit)            |     176.4 |     128.6 | qwen3:30b-a3b                     |   87.1 |     1.37x |     2.02x |
| qwen3.5-35b (A3B 8-bit)            |     151.4 |     112.0 | qwen3:30b-a3b                     |   87.1 |     1.35x |     1.74x |

Aggregate tok/s = sum across 4 concurrent streams ÷ wall-clock seconds
(includes first-token latency). The JSON artifacts also carry
`median_per_stream_tps`, which is each request's **decode-only** rate
(`output_tokens / (e2e − ttft)`) — that is typically higher than
`aggregate / concurrency` because it excludes the prefill phase.

### Notes

1. The qwen3.5-27b Ollama row is benched against `qwen3:32b`
   (dense Qwen3 32B, the closest tag Ollama 0.24 can actually load).
   The first attempt mapped to `hf.co/unsloth/Qwen3.6-27B-GGUF:Q4_K_M`
   but every load raised HTTP 500 "unable to load model" — Qwen3.6
   dense hasn't landed in llama.cpp yet — so we switched and recorded
   the working comparator. The row above and the README cite
   `qwen3:32b`.
2. mlx-lm 0.31.3 can't run Gemma 4 (its Gemma 4 loader lives in mlx-vlm).
3. Architecture caveats — Ollama can't load Qwen3.5/3.6 DeltaNet or
   Gemma 4 natively; the comparison tag is the closest available arch:
   - qwen3.5-4b/9b → qwen3:4b/8b (Qwen3 base, not 3.5; same model family)
   - qwen3.5-27b → qwen3:32b (closest dense Qwen3 that Ollama 0.24 will load; see note 1)
   - qwen3.6-35b / qwen3.5-35b → qwen3:30b-a3b (closest MoE A3B)
   - gemma-4-12b → gemma3:12b (Gemma 3, prior generation)
4. gpt-oss-20b is the only direct apples-to-apples row: same model
   weights both sides. The 2.29x is unmodified by arch gap.
