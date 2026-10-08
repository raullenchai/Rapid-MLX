# Multi-turn agent prefix-cache check

Date: 2026-10-07. Host: M2 Pro Mac mini, 32 GB unified memory. Base commit:
`e77bbeb03`. Model: `mlx-community/Qwen3.5-4B-MLX-4bit` at
`32f3e8ecf65426fc3306969496342d504bfa13f3`. The MTP runs attached
`mlx-community/Qwen3.5-4B-MTP-4bit` at
`ab6f59bc6627196c611ab8851638651078170485` with a two-token draft.

## Workload

`scripts/bench_multiturn_prefix.py` sends ten streaming agent turns to each
route. The prompt starts with a repeated repository instruction and 30 OpenAI
or Anthropic tool schemas. Each response's structured tool call is appended
to the next request with a tool result and another user instruction. There
are no timestamps or changing tool definitions. The first prompt is 14,616
tokens on `/v1/chat/completions` and 14,472 on `/v1/messages`. Every response
in this run made one tool call. Time to first token is measured at the first
content or tool-call delta, excluding the opening role or message event.

Each route/mode ran on its own server with a fresh temporary `HOME`, the
shared read-only Hugging Face model cache, and default prefix-cache memory
settings. Servers bound only to `127.0.0.1`, on ports 18123–18126. The base
model used `--no-mllm --enable-prefix-cache --enable-auto-tool-choice
--tool-call-parser qwen3`. MTP runs added `--force-spec-decode` and
`--speculative-config` with `method=mtp`, the local sidecar path,
`num_speculative_tokens=2`, and `disable_auto_k=true`.

## Current-main result

| Mode | Route | Cold turn TTFT | Warm hit rate | Warm median TTFT | Mean cached share of warm prompts |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | Chat completions | 58.137 s | 9/9 | 1.945 s | 99.46% |
| Base | Messages | 58.966 s | 9/9 | 2.107 s | 99.55% |
| MTP | Chat completions | 59.135 s | 9/9 | 1.812 s | 99.46% |
| MTP | Messages | 56.866 s | 9/9 | 2.259 s | 99.49% |

The older failures in [#2310](https://github.com/raullenchai/Rapid-MLX/issues/2310)
and [#2061](https://github.com/raullenchai/Rapid-MLX/issues/2061) did not
reproduce on current main. Both were addressed by earlier changes. An LRU
eviction count can rise during this run as older session entries leave the
eight-entry hybrid bound; the latest message boundary remains available, and
all nine growing turns hit it.

## Exact completion reuse limit

The cache also stores prompt-plus-completion entries when supported. For the
first OpenAI chat turn, the model generated a JSON tool envelope inside
`<tool_call>`, while the checkpoint's chat template renders a replayed
structured tool call in XML `<function=...><parameter=...>` form. The two
token streams differ at the assistant tool call. Reusing that completion's KV
would give the model state for different tokens. The message-boundary snapshot
correctly supplies a hit instead: the large chat run cached 14,592 of 14,676
prompt tokens on turn two. It left 84 tokens to prefill.

The Anthropic base run sometimes extended the completion entry (turn two
cached 14,496 tokens, equal to turn one's prompt plus its 24 output tokens),
and later sometimes used the boundary. The MTP path stored boundary snapshots
and hit them on both routes. The synthetic template test verifies that an XML
tool completion is an exact prefix of the next turn and that a JSON completion
keeps the safe boundary fallback. A separate check using the checkpoint's
actual tokenizer confirmed the XML prefix for all ten constructed turns on
both route translations.

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
