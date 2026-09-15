# K2 Horizon 7B performance qualification

Date: 2026-09-14

## Decision

The Rapid-owned K2 Horizon adapter is not a decode bottleneck. On a 48 GB M4
Pro Mac, its fixed-length decode throughput matched the checkpoint-bundled
implementation within measurement noise and was about 2% faster than Qwen3.5
9B at a similar memory footprint. No model-specific performance patch is
justified by this evidence.

The earlier 7.06 tokens/second observation came from a very short arithmetic
response whose elapsed time was dominated by fixed prompt-prefill and request
startup work. It was an end-to-end output rate, not steady-state decode
throughput, and must not be used to characterize the model's generation speed.

## Environment

- Mac mini with Apple M4 Pro and 48 GB unified memory
- macOS 26.5.1 (25F80)
- Rapid-MLX `132e4fee62bdfbb568250635461e99eafd897629`
- MLX 0.32.2, MLX-LM 0.31.3, Transformers 5.15.1
- offline Hugging Face snapshots; no model downloads during measurements
- greedy decoding; machine otherwise idle; one model resident at a time

Snapshots:

- K2 Horizon 7B 4-bit: `4cf7a13154070004f56e3530246e94df2832ef8a`
- Qwen3.5 4B 4-bit: `32f3e8ecf65426fc3306969496342d504bfa13f3`
- Qwen3.5 9B 4-bit: `8b2b98c00a6b4d291155e4890773ca8f769aee53`

## Fixed-length decode isolation

Each model received exactly 256 input tokens. The input was the first 256 tokens
obtained by repeating this text with one trailing space:

```text
You are testing deterministic local inference on Apple silicon. Explain how prefix caching and continuous batching reduce latency, then give a short numbered implementation checklist.
```

The probe used the installed language runtime's `generate_step`, greedy
sampling, a fresh prompt cache for each run, and `time.perf_counter` around
synchronized token iteration. After one 16-token warm-up, three measured runs
generated exactly 128 tokens each. The first-token interval was reported
separately as TTFT; decode throughput used the remaining 127 token intervals.
Within each model/path, all three measured runs produced the same token hash.

For each run, the probe called `gc.collect`, `mx.clear_cache`, and
`mx.reset_peak_memory`; recorded the start immediately before constructing the
`generate_step` iterator; recorded TTFT when its first token was yielded; called
`mx.synchronize` after its 128th token; and calculated decode as
`127 / (synchronized_end - first_token_time)`. Token hashes are SHA-256 over
UTF-8 `json.dumps(output_token_ids)`.

| Model/path | Load | Median TTFT | Median decode | Peak MLX memory |
| --- | ---: | ---: | ---: | ---: |
| K2 Horizon 7B, Rapid-owned adapter | 1.43 s | 0.693 s | 51.12 tok/s | 5.52 GB |
| K2 Horizon 7B, checkpoint-bundled adapter | 1.53 s | 0.693 s | 51.12 tok/s | 5.52 GB |
| Qwen3.5 4B | 1.35 s | 0.389 s | 83.51 tok/s | 3.08 GB |
| Qwen3.5 9B | 1.98 s | 0.708 s | 50.18 tok/s | 5.66 GB |

The three decode measurements in tokens/second were:

| Model/path | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| K2 Horizon 7B, Rapid-owned adapter | 51.2152 | 51.1196 | 51.1098 |
| K2 Horizon 7B, checkpoint-bundled adapter | 51.1120 | 51.1242 | 51.1808 |
| Qwen3.5 4B | 83.4932 | 83.5079 | 83.5464 |
| Qwen3.5 9B | 50.1826 | 50.1734 | 50.1765 |

The Rapid and checkpoint-bundled K2 paths also emitted the same token hash:
`cf50ce0dd5d1675738f4947ea425a124936d503c9acf7f0c8b813a6231d0b1e5`.
Their measured medians differ by 0.0089%, which rules out a material adapter
regression under this workload.

## Shipped server path

The standard product benchmark was run with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  rapid-mlx bench k2-horizon-7b-4bit --tier speed
```

It passed five prompts, generated 640 tokens, and reported 49.1 tokens/second
over 13.0 seconds. Attaching the same tier to an already-warm server produced
the same range. Individual 128-token OpenAI-compatible completion requests ran
at 48.4-50.0 tokens/second, showing that routing, reasoning sanitization, and
response serialization do not introduce a material throughput loss.

For context, the same standard tier reported 65.5 tokens/second for Qwen3.5 4B
and 47.9 tokens/second for Qwen3.5 9B. Those two models naturally stopped after
166 and 169 total generated tokens, so the fixed-length table above is the
primary cross-model comparison.

## Product interpretation

K2 Horizon 7B is not a 7-token/second model on this hardware. Its practical
decode speed is approximately the same as the existing 9B option, while the 4B
option remains substantially faster and lighter. K2 remains experimental
because its broader quality, compatibility, and hardware coverage are still
limited, not because the Rapid runtime is slow.
