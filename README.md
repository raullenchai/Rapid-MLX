# lightning-mlx

**Run local LLM agents faster on Apple Silicon.**

**lightning-mlx** is an OpenAI-compatible local LLM server optimized for **coding agents**, **tool calling**, and **fast short-turn loops** on Mac. It is a fork of [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) with MTPLX-style work inspired by [MTPLX](https://github.com/youssofal/MTPLX/).

## What You Get

- **2.75x faster short agentic turns** in the benchmark fixture.
- **1.96x higher all-turn throughput** versus the MLX baseline.
- **Successful artifact generation** where baseline timed out.
- **OpenAI-compatible API** for local tools, agents, editors, and CLIs.
- **Apple Silicon first**: built around MLX and local Mac inference.
- **MTPLX optimized preset** behind one simple command.

## Agentic Benchmarks

Same prompt, same agentic workflow, one server at a time:

```text
Create the snake game using react, vite and typescript
```

| Model | Metric | oMLX | Rapid MLX | **Lightning MLX (MTPLX)** |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6-27B | All Turns | 13.94 tok/s | 13.49 tok/s | **26.47 tok/s** |
| Qwen3.6-27B | Long | 20.35 tok/s | 28.02 tok/s | **38.60 tok/s** |
| Qwen3.6-27B | Short | 9.67 tok/s | 7.42 tok/s | **20.40 tok/s** |
| Qwen3.6-35B | All Turns | 49.60 tok/s | 27.73 tok/s | **64.85 tok/s** |
| Qwen3.6-35B | Long | 76.77 tok/s | 26.52 tok/s | **75.13 tok/s** |
| Qwen3.6-35B | Short | 35.11 tok/s | 29.18 tok/s | **52.50 tok/s** |

| Model / workflow | Baseline acceptance | **MTPLX acceptance** | Delta |
| --- | ---: | ---: | ---: |
| Qwen3.6 27B snake | 92.02% | **94.30%** | **+2.28 pp** |
| Qwen3.6 35B-A3B snake | N/A | **96.62%** | **MTP enabled** |

Artifact result:

- Qwen3.6-27B Lightning MLX generated app build: **passed**.
- Qwen3.6-35B base and MTPLX generated app build: failed in both runs.

oMLX agentic numbers were collected through its OpenAI-compatible server with Pi using the same snake-game prompt. Agentic numbers measure the developer loop: tool calls, growing context, file writes, retries, and build validation. They are not directly comparable with raw decode throughput.

Full benchmark notes are in [`REPORT.md`](REPORT.md).

## Raw Decode Benchmarks

Same machine, same local model paths, same microbenchmark shape:

```bash
bench <model> --num-prompts 3 --max-tokens 512 --disable-prefix-cache \
  --max-num-seqs 1 --prefill-batch-size 1 --completion-batch-size 1
```

For MTPLX Qwen3.6 benchmark runs, the preset uses MTP depth 3 with optimistic
acceptance and `--prefill-step-size 8192`.

| Model | mlx-lm | oMLX | Rapid MLX | **Lightning MLX (MTPLX)** |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | 29.80 tok/s | 31.80 tok/s | 32.37 tok/s | **40.67 tok/s** |
| Qwen3.6-35B | 110.37 tok/s | 114.59 tok/s | 106.00 tok/s | **119.68 tok/s** |

Raw decode numbers measure generation throughput only. oMLX was measured with `omlx serve --no-cache` and three OpenAI-compatible chat completion runs after warmup, using `completion_tokens / wall time`. They do not include tool calls, file writes, growing context, retries, or build validation.

The Qwen3.6-27B MTPLX row was rechecked locally after fixing `bench` to use the same MTPLX loader/preset path as `serve`. The README command produced 268 completion tokens at 40.67 tok/s, above the previous 32.37 tok/s MTPLX result.

## More Model Benchmarks

The models below use the same benchmark positioning as upstream [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX): Mac Studio M3 Ultra, Apple MLX backend, and tok/s as decode throughput. **lightning-mlx keeps the same model coverage**, and adds optimized **Qwen3.6 MTPLX** paths for local coding agents.

| Model | lightning-mlx | Best Alternative | Speedup |
| --- | ---: | ---: | ---: |
| **Qwen3.6 27B MTPLX** | **40.67 tok/s raw decode** / **26.47 tok/s agentic all-turn** | 31.80 tok/s oMLX / 32.37 tok/s Rapid MLX raw | **1.26x raw** / **1.96x agentic all-turn** |
| **Phi-4 Mini 14B** | **180 tok/s** | 77 tok/s mlx-lm / 56 tok/s Ollama | **2.3x** / **3.2x** |
| **Qwen3.5 4B** | **160 tok/s** | 155 tok/s mlx-lm serve | **1.0x** |
| **Nemotron-Nano 30B** | **141 tok/s** · 100% tools | - | - |
| **DeepSeek V4 Flash 158B-A13B** (2-bit DQ) | **56 tok/s** | Only MLX engine, day-0 | - |
| **DeepSeek V4 Flash 158B-A13B** (8-bit) | **31 tok/s** | Only MLX engine, day-0 | - |
| **GPT-OSS 20B** | **127 tok/s** · 100% tools | 79 tok/s mlx-lm serve | **1.6x** |
| **Qwen3.5 9B** | **108 tok/s** | 41 tok/s Ollama | **2.6x** |
| **Qwen3.6 35B-A3B MTPLX** | **119.68 tok/s raw decode** / **64.85 tok/s agentic all-turn** | 114.59 tok/s oMLX / 106.00 tok/s Rapid MLX raw | **2.34x agentic all-turn** |
| **Qwen3.6 35B-A3B** | **109.89 tok/s raw decode** · 100% tools | 114.59 tok/s oMLX / 106.00 tok/s Rapid MLX raw | ~**1.0x** |
| **Kimi-Linear 48B** | **94 tok/s** · 100% tools | Only engine | - |
| **Gemma 4 26B-A4B** | **85 tok/s** | 68 tok/s Ollama | **1.3x** |
| **Gemma 4 E4B** | **83 tok/s** | - | - |
| **Qwen3.5 35B-A3B** | **83 tok/s** · 100% tools | 75 tok/s oMLX | **1.1x** |
| **Qwen3-Coder 80B** | **74 tok/s** · 100% tools | 69 tok/s mlx-lm serve | **1.1x** |
| **Qwen3.5 122B** | **44 tok/s** · 100% tools | 43 tok/s mlx-lm serve | ~**1.0x** |
| **Gemma 4 31B** | **31 tok/s** | - | - |

**Note:** raw decode and agentic throughput are different measurements. Raw decode is a microbenchmark. Agentic throughput includes tool calls, growing context, artifact generation, and build validation.

## Install

```bash
python3 -m pip install git+https://github.com/samuelfaj/lightning-mlx.git
```

Local checkout:

```bash
git clone https://github.com/samuelfaj/lightning-mlx.git
cd lightning-mlx
python3 -m pip install -e .
```

Self-contained install:

```bash
curl -fsSL https://raw.githubusercontent.com/samuelfaj/lightning-mlx/main/install.sh | bash
```

Then verify:

```bash
lightning-mlx --help
```

## Start Fast

```bash
lightning-mlx serve qwen3.6-27b
```

That one command expands to the optimized MTPLX model:

```text
Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed
```

and applies the agentic performance preset automatically: **MTP on**, **tool parser on**, **reasoning parser on**, **tool logits bias on**, **single-sequence agent mode**, and **OpenAI model name `local`**.

Local model path works too:

```bash
lightning-mlx serve /path/to/Qwen3.6-27B-MTPLX-Optimized-Speed
```

Run it without keeping a terminal open:

```bash
lightning-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit --daemon
```

Daemon mode starts a detached supervisor, writes logs under `~/.lightning-mlx/logs/`, and restarts the server if the model process exits unexpectedly.

```bash
lightning-mlx status
lightning-mlx tui <PID-or-model-name>
lightning-mlx kill <PID-or-model-name>
```

Use `status` to see running daemons, `tui` to attach the live monitor, and `kill` to stop by supervisor PID, server PID, alias, or model name.

## Convert Local MTPLX Models

Use `convert-mtplx` to package a local model with an MTPLX MTP sidecar. This is useful when the serving model is quantized, but the MTP tensors come from the original full model:

```bash
lightning-mlx convert-mtplx \
  /path/to/Qwen3.6-35B-A3B-4bit \
  --mtp-source /path/to/Qwen3.6-35B-A3B
```

By default, the output is written next to the source model as:

```text
/path/to/Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed
```

Then serve it normally:

```bash
lightning-mlx serve /path/to/Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed
```

For `Qwen3.6-35B-A3B`, the default temperature is `0.6`. Thinking stays enabled by default for agentic tool use. Pass `--no-thinking` only when you explicitly want to disable it.

## Use It Like OpenAI

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

## Why Developers Use It

**Local agents need different optimization than chat demos.** The hard path is not one long completion; it is dozens of short, tool-heavy turns with growing context. lightning-mlx focuses on that path:

- **Fast tool-call streaming**
- **Lower SSE/Pydantic overhead**
- **MTPLX-style speculative decoding**
- **Qwen3.6 MTPLX preset**
- **Benchmark-driven changes only**

## Built On

- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX)
- [MTPLX](https://github.com/youssofal/MTPLX/)
- [MLX](https://github.com/ml-explore/mlx)
