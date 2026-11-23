# Inkling 2 Mini on M5 Max

These are Rapid-MLX serving and in-process performance measurements for two
Inkling 2 Mini artifacts on a 36 GB Mac Studio. The results cover text
generation, context scaling, and two experiments that were not shipped.

## Environment

| Component | Value |
| --- | --- |
| Machine | Mac Studio (`Mac17,14`) |
| Chip | Apple M5 Max |
| Unified memory | 36 GB |
| GPU architecture | `applegpu_g17s` |
| macOS | 27.0.1 |
| Python | 3.12.13 |
| MLX | 0.32.4 |
| MLX-LM | 0.31.3 |
| Rapid-MLX, 4-bit run | `main` at `40dddb7d6` |
| Rapid-MLX, official checkpoint | Branch that became [#5924](https://github.com/raullenchai/Rapid-MLX/pull/5924) |
| 4-bit artifact | `mlx-community/Inkling-2-Mini-4bit` at `9856dc53edead71405f8348930937c3638ff21a2` |
| 4-bit alias / quantization | `inkling-2-mini-4bit` / affine 4-bit, group size 64, 16 GB on disk |
| Official checkpoint | `thinkingmachines/Inkling-2-Mini` at `b85c3cfa615fc413cc9af3379f491b963f3b586a` (NVFP4, 19 GB on disk including the DSpark drafter) |

The official checkpoint needs #5924 to load on the text lane. Its results
therefore use that branch rather than the `main` revision of the 4-bit run.

## Methodology

Each artifact ran in a fresh server process with `HF_HUB_OFFLINE=1` and no
other model server resident:

```bash
HF_HUB_OFFLINE=1 rapid-mlx serve <model> --port <p>
```

- Requests used OpenAI-compatible streaming chat completions, 256 output
  tokens, and temperature 0.
- Each request had a unique random-word prompt, so there were no prefix-cache
  hits. Each prompt length had three repetitions; tables report medians.
- TTFT is time to the first streamed token. Prefill tok/s is prompt tokens
  divided by TTFT, including request and first-token overhead.
- Decode tok/s excludes TTFT.
- The concurrency measurement used four simultaneous requests and reports
  aggregate output tok/s, with per-request throughput shown separately.

## Results

| Prompt tokens | 4-bit TTFT | Official TTFT | 4-bit prefill tok/s | Official prefill tok/s | 4-bit decode tok/s | Official decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 659 | 0.54 s | 0.56 s | 1,216 | 1,182 | 55.6 | 45.6 |
| 4,996 | 3.19 s | 3.19 s | 1,568 | 1,569 | 52.9 | 44.0 |
| 19,845 | 13.57 s | 13.47 s | 1,463 | 1,472 | 49.3 | 41.1 |

| Serving measurement | 4-bit | Official |
| --- | ---: | ---: |
| Four concurrent requests, aggregate output | 72.5 tok/s | 67.7 tok/s |
| Four concurrent requests, per request | 24.8 tok/s | 22.9 tok/s |
| Metal peak memory | 22.8 GB | 24.0 GB |

### In-process decode and prefill

Without the server, decode measured prefill followed by 256 pipelined greedy
steps. Each value is the median of three runs.

| Context | 4-bit decode tok/s | Official decode tok/s |
| --- | ---: | ---: |
| 512 tokens | 56.9 | 46.7 |
| 4K tokens | 54.2 | 44.9 |

One 2,048-token prefill chunk of the 4-bit artifact, measured in process:

| Starting context | Chunk time | Prefill tok/s |
| --- | ---: | ---: |
| 0 | 1,151 ms | 1,779 |
| 16K | 1,526 ms | 1,342 |

The component profile at context 0 attributes 427 ms to routed experts,
316 ms to sliding-window attention, 265 ms to other work, 144 ms to loop
message gates, 88 ms to full attention, and 44 ms to depth residual.
Full attention rises to 442 ms at 16K context. Components were profiled
eagerly, so their times sum above the fused chunk total.

## Notes

- Decode with the official NVFP4 checkpoint is about 18% slower than with
  the affine 4-bit artifact; prefill is the same.
- **Thinking:** the chat template defaults to a high reasoning effort (0.9).
  On hard multiple-choice science questions, the model often reasons for
  80K–110K characters before answering, coherently and without repetition.
  This matches the model card's warning about long reasoning. Set
  `max_tokens`, or lower the reasoning effort, for evaluations.
- On M5-class GPUs, MLX fp32 matmul with more than one row runs at about
  1e-3 relative error; single-row GEMV is exact. The model's parity tests
  detect this at runtime ([#5925](https://github.com/raullenchai/Rapid-MLX/pull/5925)).

## Closed experiments

### DSpark block drafter

The checkpoint's DSpark block drafter used block size 8. At 512-token context,
a verify forward of 2/4/8 tokens cost 32.5/42.5/57.8 ms, against 24.7 ms for
one token. Accepted tokens per round were 1.0–1.3 on code and prose and about
3 on JSON.

Across four realistic prompts, median decode was 43.9 tok/s against
45.6 tok/s with plain decoding. JSON alone improved by 39%. The experiment
was not shipped.

**Reopen if** verify costs fall to about 30 ms for two tokens and 37 ms for
four tokens, or for workloads dominated by structured output.

### Full-attention prefill with additive bias

Routing full-attention prefill through the platform SDPA with an additive
bias mask on accelerator GPUs reduced attention-kernel time from 47.1 to
23.6 ms at 18K keys. A 2,048-token chunk became 13.7% faster at 16K context.
Served TTFT improved only 3.3% at 11.5K and 6.9% at 23K prompt tokens,
because early chunks of a long prompt sit at short context. The experiment
was not shipped.

**Reopen for** prompts of 64K+ tokens, or with an accelerator-native attention
kernel that carries the relative bias.
