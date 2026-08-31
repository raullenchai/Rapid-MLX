# Recent large models on M3 Ultra

This note is the evidence behind the concise large-model table in the project
README. It brings the latest measured Qwen3.8-27B, Qwen3.8-Flash-Next, and
GLM-5.3-Flash serving results together without pretending that unlike
workloads are directly comparable.

## Environment

| Component | Value |
| --- | --- |
| Machine | Mac Studio (`Mac15,14`) |
| Chip | Apple M3 Ultra, 28 CPU cores |
| Unified memory | 256 GB |
| macOS | 26.5.2 (25F84) |
| Architecture | arm64 |
| Serving shape | Batch size 1; one model resident |
| Quantization | The Rapid-MLX 4-bit alias named in each row |

All rates are medians of three measured requests after model warmup. TTFT is
measured to the first visible streamed token. Prefill is server-reported
prompt tokens divided by TTFT. Decode excludes TTFT. MLX active memory is
allocator-active unified memory, not process RSS; RSS materially undercounts
Metal allocations.

The Qwen context curves use temperature zero, thinking disabled, a cold prefix
cache for every request, and 256 requested decode tokens. The harness targets
128, 2,048, 8,192, and 32,768 prompt tokens; after applying the chat template,
the server reports 92, 2,012, 8,156, and 32,732 tokens.

## At a glance

| Model | Model shape | Measured workload | Median TTFT | Median prefill | Median decode | Maximum MLX active memory for row |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3.8-27b-4bit` | 27B dense | 8,156 → 256 | 24.246s | 336.4 tok/s | 37.38 tok/s | 17.6 GB |
| `qwen3.8-flash-next-4bit` | 180B total / 6B active | 8,156 → 256 | 9.236s | 883.1 tok/s | 23.40 tok/s | about 103.4 GB |
| `qwen3.8-flash-next-4bit`, fixed MTP K=1 | Same checkpoint and target model | 8,156 → 256 | 14.581s | 559.4 tok/s | 32.20 tok/s | 107.2–107.5 GB |
| `glm5.3-flash-4bit` | 320B total / 18B active | 47 → 512 | Not captured | Not captured | 29.20 tok/s | 165.4 GB |

The Flash-Next parameter total is 125B language-model parameters plus a 51B
n-gram embedding and 4B MTP head; 6B language-model parameters are active per
token. The GLM shape is 320B total and 18B active. Those figures describe the
upstream architectures; throughput and memory in this document are Rapid-MLX
measurements. See the official model cards for the
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), and
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) architecture
descriptions.

The Flash-Next MTP row is a separate same-machine, same-checkpoint experiment
on the optimized sparse-mask engine. It is not paired with the later batched
compressed-key prefill result. MTP is an explicit opt-in and ordinary
autoregressive decode remains the default.

## Qwen3.8-27B context curve

Artifact: `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` at
`aa985c29ff5b334cbfdcbbc787d47e66e9d9e456`. The reference server used the
Rapid-MLX 0.13.1 release commit `819db66767ac7e16722315122600d0855a6981c8`
with speculative decoding off.

| Target (reported) prompt tokens | Median TTFT | Median prefill | Median decode | MLX active memory |
| ---: | ---: | ---: | ---: | ---: |
| 128 (92) | 0.429s | 214.6 tok/s | 40.29 tok/s | 15.6 GB |
| 2,048 (2,012) | 5.904s | 340.8 tok/s | 39.50 tok/s | 16.0 GB |
| 8,192 (8,156) | 24.246s | 336.4 tok/s | 37.38 tok/s | 17.6 GB |
| 32,768 (32,732) | 107.244s | 305.2 tok/s | 32.58 tok/s | 24.1 GB |

The model-recommendation qualification used the same M3 Ultra and measured a
20.0 GB peak for the complete server process tree at roughly 8K, with zero new
swap. That is intentionally a different memory boundary from MLX allocator
active memory.

## Qwen3.8-Flash-Next context curve

Artifact: `rapid-mlx/Qwen3.8-Flash-Next-4bit` at
`dcf657e4acda2aae72da99cde65b6c491cd96998`. The rows below are the final
batched compressed-key result delivered in Rapid-MLX 0.13.2. They retain the
same checkpoint, prompts, cache-clear procedure, and three-run methodology as
the release baseline.

| Target (reported) prompt tokens | Median TTFT | Median prefill | Median decode |
| ---: | ---: | ---: | ---: |
| 128 (92) | 0.346s | 266.3 tok/s | 25.67 tok/s |
| 2,048 (2,012) | 2.262s | 889.4 tok/s | 24.27 tok/s |
| 8,192 (8,156) | 9.236s | 883.1 tok/s | 23.40 tok/s |
| 32,768 (32,732) | 44.659s | 732.9 tok/s | 21.72 tok/s |

MLX active memory remained approximately 102.8–103.8 GB. The process also
reported a 148.1 GB allocator peak inherited from model loading; it is not the
steady active footprint.

The separate fixed-K=1 MTP qualification measured decode at 34.85, 33.53,
32.20, and 28.82 tok/s at the same four prompt lengths: a 36–42% improvement
over its exact-run ordinary-decode baseline. All 45 functional outcomes
matched ordinary decode, with a 76.41% aggregate proposal acceptance ratio.
MTP added as much as 6.6 GB of active memory and increased 2K–32K TTFT by
5–8%, so it remains an explicit workload-dependent choice.

The model's quantized weights occupy about 99 GB before cache and allocator
headroom. A 192 GB Mac is the practical recommended tier. A 128 GB Mac is
tight and was not physically tested.

Full Flash-Next methodology and correctness evidence:

- [ordinary decode and QSA prefill](qwen38-flash-next-m3-ultra.md)
- [native MTP qualification](qwen38-flash-next-mtp-m3-ultra.md)

## GLM-5.3-Flash qualification row

Artifact: `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` at
`06d6c7530e8290e20fabdc37a825ce07bdfc490c`. Rapid-MLX implementation:
`4acffa71df832eb7865e5c76e1ce8295bd6f074b`.

The server ran with thinking and speculative decoding disabled and temperature
zero. After warmup, three 512-token requests decoded at 26.11, 29.20, and
30.70 tok/s; the median is 29.20 tok/s. The prompt was 47 server-reported
tokens:

> You are a senior software engineer writing a blog post. Explain the
> difference between threads and processes in operating systems, covering
> address space, scheduling, context switch cost, IPC, and a worked example of
> when each is appropriate. Be concrete and specific.

The target process reported 165.4 GB MLX active memory and approximately
167.9 GB peak during the run. The alias therefore has a 192 GB memory floor.
This short-prompt qualification did not capture TTFT or prefill, so the README
leaves those cells blank instead of deriving them from wall time.

Resolve the exact measured checkpoint revision first, then serve that immutable
local snapshot. This avoids accidentally benchmarking a newer cached revision:

```bash
MODEL_DIR="$(
  python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="Vontra/GLM-5.3-Flash-MLX-4bit-MTP",
    revision="06d6c7530e8290e20fabdc37a825ce07bdfc490c",
))
PY
)"

HF_HUB_OFFLINE=1 rapid-mlx serve \
  "$MODEL_DIR" --served-model-name glm5.3-flash-4bit \
  --host 127.0.0.1 --port 8465 --no-thinking
```

In another shell, send one discarded warmup followed by three identical
requests. This is the exact payload shape; set `run=warmup` for the discarded
request and then repeat it with `run=1`, `run=2`, and `run=3` so the output can
be archived separately:

```bash
PROMPT='You are a senior software engineer writing a blog post. Explain the difference between threads and processes in operating systems, covering address space, scheduling, context switch cost, IPC, and a worked example of when each is appropriate. Be concrete and specific.'
run=warmup

jq -nc --arg prompt "$PROMPT" '{
  model: "glm5.3-flash-4bit",
  messages: [{role: "user", content: $prompt}],
  temperature: 0,
  max_tokens: 512,
  enable_thinking: false,
  stream: true,
  stream_options: {include_usage: true}
}' | curl --no-buffer --silent --show-error \
  http://127.0.0.1:8465/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @- \
  | tee "glm53-${run}.sse"

curl --silent http://127.0.0.1:8465/v1/status | jq '{
  generation_tps,
  prompt_tps,
  active_memory_gb: .metal.active_memory_gb,
  peak_memory_gb: .metal.peak_memory_gb
}'
```

The streamed final usage event supplies the prompt and completion token
counts. `/v1/status` supplies `generation_tps`, `prompt_tps`, and the
`metal.active_memory_gb` / `metal.peak_memory_gb` allocator readings. Do not
run another model server concurrently.

The checkpoint contains a native MTP head, but the qualification experiment
did not produce a speedup: 32.00 tok/s ordinary decode versus 31.65 tok/s with
MTP for the sustained 512-token comparison, despite 72.97% acceptance and
5.71 GB additional active memory. GLM MTP is therefore disabled for this
alias; neither that no-go result nor an unqualified acceleration mode is used
in the README headline.
