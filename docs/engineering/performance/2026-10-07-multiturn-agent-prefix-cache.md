# Multi-turn agent prefix-cache check

Date: 2026-10-07. Host: M2 Pro Mac mini, 32 GB unified memory. Base commit:
`e77bbeb03`. Model: `mlx-community/Qwen3.5-4B-MLX-4bit` at
`32f3e8ecf65426fc3306969496342d504bfa13f3`. The MTP runs attached
`mlx-community/Qwen3.5-4B-MTP-4bit` at
`ab6f59bc6627196c611ab8851638651078170485` with a two-token draft.

## Workload

`scripts/bench_multiturn_prefix.py` sends ten streaming agent turns to each
route. The system prompt uses this repository's `AGENTS.md` and the first
25,000 characters of `README.md` as varied coding guidance and project
context, followed by 30 OpenAI or Anthropic tool schemas. Each response's
structured tool call is appended to the next request with a tool result and
another user instruction. There are no timestamps or changing tool
definitions. Time to first token is measured at the first content or
tool-call delta, excluding the opening role or message event. The harness
rejects a stream without its terminal event and usage fields.

Each route/mode ran on its own server with a fresh temporary `HOME`, the
shared read-only Hugging Face model cache, and default prefix-cache memory
settings. Servers bound only to `127.0.0.1`, on ports 18123–18126. The base
model used `--no-mllm --enable-prefix-cache --enable-auto-tool-choice
--tool-call-parser qwen3`. MTP runs added `--force-spec-decode` and
`--speculative-config` with `method=mtp`, the local sidecar path,
`num_speculative_tokens=2`, and `disable_auto_k=true`.

## Current-main result

The before and after runs use the same serving code, each with a fresh server
and `HOME`. The branch adds the benchmark and regression tests, but makes no
serving change. These paired runs show repeatability, not a performance gain.
Each cold prompt has 15,704 tokens on chat completions or 15,560 on messages.
All responses contained one structured tool call.

| Mode | Route | Warm hits before/after | Cold TTFT before/after | Warm median TTFT before/after | Mean cached share before/after |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | Chat completions | 9/9 · 9/9 | 63.470 · 56.751 s | 2.197 · 2.368 s | 99.46 · 99.46% |
| Base | Messages | 9/9 · 9/9 | 53.757 · 57.622 s | 2.329 · 2.056 s | 99.68 · 99.68% |
| MTP | Chat completions | 9/9 · 9/9 | 57.843 · 62.064 s | 2.477 · 2.360 s | 99.46 · 99.46% |
| MTP | Messages | 9/9 · 9/9 | 58.298 · 62.354 s | 2.402 · 2.427 s | 99.51 · 99.51% |

TTFT in seconds for every turn; each cell is before / after:

| Turn | Base chat | Base messages | MTP chat | MTP messages |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 63.470 / 56.751 | 53.757 / 57.622 | 57.843 / 62.064 | 58.298 / 62.354 |
| 2 | 1.855 / 2.368 | 2.953 / 2.203 | 2.567 / 2.188 | 2.383 / 2.239 |
| 3 | 3.288 / 2.395 | 2.252 / 3.437 | 2.553 / 2.839 | 1.972 / 3.021 |
| 4 | 2.145 / 1.702 | 2.329 / 1.722 | 2.011 / 3.132 | 1.906 / 2.331 |
| 5 | 1.988 / 2.468 | 1.630 / 1.698 | 3.805 / 2.025 | 4.028 / 2.590 |
| 6 | 3.313 / 3.862 | 3.495 / 1.889 | 1.780 / 2.266 | 2.639 / 2.562 |
| 7 | 1.767 / 1.557 | 2.959 / 2.198 | 2.477 / 2.360 | 2.709 / 2.384 |
| 8 | 2.281 / 3.575 | 1.456 / 2.471 | 2.430 / 2.503 | 2.854 / 2.427 |
| 9 | 2.197 / 1.916 | 2.627 / 2.056 | 3.494 / 2.400 | 2.402 / 2.924 |
| 10 | 2.326 / 1.958 | 1.749 / 1.697 | 1.826 / 2.300 | 1.995 / 2.107 |

Cached-prefix tokens reported by the API on each turn were identical in the
before and after trials:

| Turn | Base chat | Base messages | MTP chat | MTP messages |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 15,680 | 15,586 | 15,680 | 15,545 |
| 3 | 15,744 | 15,647 | 15,744 | 15,606 |
| 4 | 15,808 | 15,672 | 15,808 | 15,670 |
| 5 | 15,872 | 15,736 | 15,872 | 15,734 |
| 6 | 15,954 | 15,800 | 15,954 | 15,798 |
| 7 | 16,018 | 15,904 | 16,018 | 15,862 |
| 8 | 16,082 | 15,967 | 16,082 | 15,926 |
| 9 | 16,146 | 16,027 | 16,146 | 15,986 |
| 10 | 16,210 | 16,091 | 16,210 | 16,050 |

The older failures in [#2310](https://github.com/raullenchai/Rapid-MLX/issues/2310)
and [#2061](https://github.com/raullenchai/Rapid-MLX/issues/2061) did not
reproduce on current main. Both were addressed by earlier changes. Older
entries were sometimes evicted under cache pressure, but the latest message
boundary remained available and all nine growing turns hit it. TTFT varied
between runs while other workloads used the host.

## Exact completion reuse limit

The cache also stores prompt-plus-completion entries when supported. For the
first OpenAI chat turn, the model generated a JSON tool envelope inside
`<tool_call>`, while the checkpoint's chat template renders a replayed
structured tool call in XML `<function=...><parameter=...>` form. The two
token streams differ at the assistant tool call. Reusing that completion's KV
would give the model state for different tokens. The message-boundary snapshot
correctly supplies a hit instead: the chat run cached 15,680 of 15,766
prompt tokens on turn two. It left 86 tokens to prefill.

The Anthropic base run sometimes extended the completion entry (turn two
cached 15,586 tokens, equal to turn one's prompt plus its 26 output tokens),
and later sometimes used the boundary. The MTP path stored boundary snapshots
and hit them on both routes. The synthetic template test verifies that an XML
tool completion is an exact prefix of the next turn and that a JSON completion
keeps the safe boundary fallback. A separate check using the checkpoint's
actual tokenizer confirmed the XML prefix for all ten constructed turns on
both route translations and fetched the saved boundary after divergent JSON
output.

No cache memory default was changed. The measured workload already has a
100% warm-turn hit rate. Changing tool-call replay to prefer one syntax would
make other valid model outputs diverge, so the server keeps the conservative
boundary snapshot when the generated completion cannot be re-rendered exactly.

## Reproduction

After serving the model as above, run each command against a fresh server:

```bash
python scripts/bench_multiturn_prefix.py \
  --base-url http://127.0.0.1:18123 \
  --model /path/to/Qwen3.5-4B-MLX-4bit/snapshot --api chat
python scripts/bench_multiturn_prefix.py \
  --base-url http://127.0.0.1:18124 \
  --model /path/to/Qwen3.5-4B-MLX-4bit/snapshot --api messages
```

Each JSON line reports the turn, prompt tokens, cached-prefix tokens, TTFT,
tool-call count, and output length. The server's `[cache_fetch]` and
`[boundary_snapshot]` logs give an independent view of each match.
