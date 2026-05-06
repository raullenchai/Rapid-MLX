# lightning-mlx

**lightning-mlx** is a high-speed Apple Silicon local LLM server.

This project is a fork of [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) with additional work inspired by and integrated from [MTPLX](https://github.com/youssofal/MTPLX/). The goal is simple: run local LLMs as fast as possible on Apple Silicon while keeping an OpenAI-compatible API for agentic tools, editors, CLIs, and local automation.

## Why lightning-mlx

Local coding agents are bottlenecked by repeated long-context tool turns. Raw decode speed matters, but short agentic turns are often where latency collapses: system prompt, tool schemas, previous tool output, and small next actions all get paid repeatedly.

lightning-mlx focuses on that workload:

- OpenAI-compatible chat completions API.
- MLX-native execution for Apple Silicon.
- MTPLX-style speculative decoding support.
- Tool calling and reasoning parser support.
- Fast streaming Server-Sent Events paths for agentic tool calls.
- Lower Python/Pydantic overhead on the hot streaming path.

## Benchmark Summary

The benchmark below uses the same agentic fixture across both runs:

```text
Create the snake game using react, vite and typescript
```

The agent was run from an empty directory through `pi`, with one local server running at a time, `max_num_seqs=1`, MTP enabled, and prefix cache disabled for a conservative apples-to-apples agentic comparison.

| Metric | MLX baseline | lightning-mlx | Gain |
| --- | ---: | ---: | ---: |
| Completion result | Timed out after 10 min | Finished successfully | Completed |
| Generated app build | Not completed | Passed | Valid artifact |
| All-turn average | 13.49 tok/s | 26.47 tok/s | +96.2% / 1.96x |
| Long-turn average | 28.02 tok/s | 38.60 tok/s | +37.8% / 1.38x |
| Short-turn average | 7.42 tok/s | 20.40 tok/s | +174.9% / 2.75x |
| MTP acceptance average | 92.02% | 94.30% | +2.28 pp |

The largest practical win is in short agentic turns: `7.42 -> 20.40 tok/s`, a `2.75x` improvement. That is the path that matters most for local agents repeatedly deciding, calling tools, reading outputs, and continuing.

## What Was Added

lightning-mlx keeps the Rapid-MLX foundation and adds performance work around MTPLX-style local speculative decoding and agentic streaming.

Current kept improvements include:

- Direct JSON serialization for tool-call SSE chunks.
- Reduced per-token streaming overhead by avoiding Pydantic serialization on the tool-call hot path.
- Debug-level SSE trace logging instead of info-level per-chunk logging.
- Benchmark-driven validation against real local agentic artifact generation.

Experimental changes that did not improve the benchmark were removed. The project intentionally keeps only changes with measured performance or reliability value.

## Install

Install from this repository:

```bash
python3 -m pip install git+https://github.com/samuelfaj/lightning-mlx.git
```

Or install from a local checkout:

```bash
git clone https://github.com/samuelfaj/lightning-mlx.git
cd lightning-mlx
python3 -m pip install -e .
```

After installation, the command is available directly in your shell:

```bash
lightning-mlx --help
```

For a self-contained install under `~/.lightning-mlx` with a symlink in `~/.local/bin`:

```bash
curl -fsSL https://raw.githubusercontent.com/samuelfaj/lightning-mlx/main/install.sh | bash
```

If `~/.local/bin` is not already in your `PATH`, add it:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Quick Start

Start a local OpenAI-compatible server:

```bash
lightning-mlx serve /path/to/model \
  --enable-mtp \
  --served-model-name local \
  --port 8010 \
  --default-temperature 0.6 \
  --default-top-p 0.95 \
  --disable-prefix-cache \
  --max-num-seqs 1 \
  --prefill-batch-size 1 \
  --completion-batch-size 1 \
  --stream-interval 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder_xml \
  --reasoning-parser qwen3 \
  --no-thinking \
  --enable-tool-logits-bias
```

Use it as an OpenAI-compatible endpoint:

```bash
curl http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "Write a tiny Python HTTP server."}
    ],
    "stream": true
  }'
```

## Recommended Agentic Settings

For single-agent local coding workloads on Apple Silicon:

```bash
--enable-mtp
--max-num-seqs 1
--prefill-batch-size 1
--completion-batch-size 1
--stream-interval 1
--default-temperature 0.6
--default-top-p 0.95
--disable-prefix-cache
--enable-auto-tool-choice
--enable-tool-logits-bias
```

For Qwen3-style tool-calling models:

```bash
--tool-call-parser qwen3_coder_xml
--reasoning-parser qwen3
--no-thinking
```

## Benchmark Protocol

The comparison table is based on a local agentic benchmark:

1. Start exactly one server.
2. Run `pi` from an empty directory.
3. Use the prompt: `Create the snake game using react, vite and typescript`.
4. Build the generated Vite app.
5. Compare throughput from server-side streaming metrics.
6. Keep only runtime changes that improve measured performance and preserve valid artifact generation.

Full benchmark notes for the current optimization branch are in [`REPORT.md`](REPORT.md).

## Upstream Credits

lightning-mlx builds on:

- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX), the upstream Apple Silicon local inference server.
- [MTPLX](https://github.com/youssofal/MTPLX/), which informed the MTP/speculative decoding direction.
- [MLX](https://github.com/ml-explore/mlx), Apple's array framework for Apple Silicon machine learning.

## Status

This repository is optimized for local experimentation and high-speed Apple Silicon agentic inference. APIs and performance work may move quickly. The north star remains stable: make local LLMs feel fast enough for real coding-agent loops on Mac hardware.
