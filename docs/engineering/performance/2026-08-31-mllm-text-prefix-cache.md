# MLLM text-prefix cache validation (2026-08-31)

This note records the user-side validation for text-only multi-turn requests
served through the multimodal scheduler. The optimization retains exact hybrid
prompt-cache state at the stable conversation boundary; requests that contain
images bypass it.

## Environment

- Apple M3 Ultra, 256 GB unified memory
- macOS 26.0, Python 3.12
- Offline model resolution (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)
- Baseline runtime: Rapid-MLX 0.13.1, MLX 0.32.2, MLX-LM 0.31.3,
  MLX-VLM 0.6.16, transformers 5.12.1
- Candidate runtime: the #2524 worktree, MLX 0.32.2, MLX-LM 0.31.3,
  MLX-VLM 0.6.16, transformers 5.15.1
- One model resident at a time

The 9B prompt was generated deterministically from 220 lines of the form
`File module_N.py defines function transform_N(value) returning value + N.`
The 27B prompt used the first 120 lines. Each turn streamed 16 tokens at
temperature 0; the actual assistant content was appended before the next user
message. TTFT is request start to the first non-empty content or reasoning
delta. All results are one quiet-window run intended to prove the request path,
not a general model-performance claim.

Servers:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 rapid-mlx serve \
  qwen3.5-9b-4bit --host 127.0.0.1 --port 18462 --mllm --no-thinking

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 rapid-mlx serve \
  qwen3.8-27b-4bit --host 127.0.0.1 --port 18462 --mllm --no-thinking
```

Each request used this OpenAI-compatible shape, with `messages` extended after
every response:

```json
{
  "model": "<alias>",
  "messages": "<growing conversation described above>",
  "max_tokens": 16,
  "temperature": 0,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

## Results

### Qwen3.5 9B, 5.2K-token conversation

| Runtime | Turn | Prompt tokens | Cached tokens | TTFT |
| --- | ---: | ---: | ---: | ---: |
| 0.13.1 baseline | 1 | 5,210 | 0 | 4.948 s |
| 0.13.1 baseline | 2 | 5,250 | 0 | 4.874 s |
| 0.13.1 baseline | 3 | 5,290 | 0 | 4.941 s |
| #2524 candidate | 1 | 5,210 | 0 | 4.963 s |
| #2524 candidate | 2 | 5,250 | 5,195 | 0.260 s |
| #2524 candidate | 3 | 5,290 | 5,235 | 0.256 s |

Warm-turn TTFT improved by 18.7–19.3x. The three deterministic response strings
matched the 0.13.1 baseline byte for byte.

### Qwen3.8 27B, 2.7K-token conversation

| Turn | Prompt tokens | Cached tokens | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 2,706 | 0 | 8.785 s |
| 2 | 2,746 | 2,691 | 0.563 s |
| 3 | 2,786 | 2,731 | 0.549 s |

Warm turns were 15.6–16.0x faster than the cold first turn. This row validates
the same request path on the second issue acceptance model; it is not a
cross-version comparison.

## Correctness and observability

- A real 16x16 red PNG sent through the same 9B server returned `red`.
- The image request did not change prefix-cache hit, miss, or saved-token
  counters; media-bearing requests remain on the cold vision path.
- After the three 9B text turns, `/metrics` reported 2 hits, 1 miss, and 10,430
  tokens saved. OpenAI usage reported 5,195 and 5,235 cached tokens on turns 2
  and 3.
- After the three 27B turns, `/metrics` reported 2 hits, 1 miss, and 5,422
  tokens saved.

