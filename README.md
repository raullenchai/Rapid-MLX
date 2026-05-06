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

## Benchmark Highlight

Same prompt, same agentic workflow, one server at a time:

```text
Create the snake game using react, vite and typescript
```

| Metric | MLX baseline | **lightning-mlx** | **Gain** |
| --- | ---: | ---: | ---: |
| Completion | Timeout after 10 min | **Finished** | **Completed** |
| Generated app build | Not completed | **Passed** | **Valid artifact** |
| All-turn avg | 13.49 tok/s | **26.47 tok/s** | **+96.2% / 1.96x** |
| Long-turn avg | 28.02 tok/s | **38.60 tok/s** | **+37.8% / 1.38x** |
| Short-turn avg | 7.42 tok/s | **20.40 tok/s** | **+174.9% / 2.75x** |
| MTP acceptance avg | 92.02% | **94.30%** | **+2.28 pp** |

**Biggest win:** short tool-heavy turns went from **7.42 tok/s** to **20.40 tok/s**. That is the loop developers feel most when a local coding agent reads files, calls tools, edits code, and continues.

Full benchmark notes are in [`REPORT.md`](REPORT.md).

## More Model Benchmarks

The models below use the same benchmark positioning as upstream [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX): Mac Studio M3 Ultra, Apple MLX backend, and tok/s as decode throughput. **lightning-mlx keeps the same model coverage**, and adds the optimized **Qwen3.6 27B MTPLX** path for local coding agents.

| Model | lightning-mlx | Best Alternative | Speedup |
| --- | ---: | ---: | ---: |
| **Qwen3.6 27B MTPLX** | **26.47 tok/s agentic all-turn** / **38.60 tok/s long-turn** / **20.40 tok/s short-turn** | 13.49 / 28.02 / 7.42 tok/s MLX baseline | **1.96x all-turn** / **2.75x short-turn** |
| **Phi-4 Mini 14B** | **180 tok/s** | 77 tok/s mlx-lm / 56 tok/s Ollama | **2.3x** / **3.2x** |
| **Qwen3.5 4B** | **160 tok/s** | 155 tok/s mlx-lm serve | **1.0x** |
| **Nemotron-Nano 30B** | **141 tok/s** · 100% tools | - | - |
| **DeepSeek V4 Flash 158B-A13B** (2-bit DQ) | **56 tok/s** | Only MLX engine, day-0 | - |
| **DeepSeek V4 Flash 158B-A13B** (8-bit) | **31 tok/s** | Only MLX engine, day-0 | - |
| **GPT-OSS 20B** | **127 tok/s** · 100% tools | 79 tok/s mlx-lm serve | **1.6x** |
| **Qwen3.5 9B** | **108 tok/s** | 41 tok/s Ollama | **2.6x** |
| **Qwen3.6 35B-A3B** | **95 tok/s** · 100% tools | - | - |
| **Kimi-Linear 48B** | **94 tok/s** · 100% tools | Only engine | - |
| **Gemma 4 26B-A4B** | **85 tok/s** | 68 tok/s Ollama | **1.3x** |
| **Gemma 4 E4B** | **83 tok/s** | - | - |
| **Qwen3.5 35B-A3B** | **83 tok/s** · 100% tools | 75 tok/s oMLX | **1.1x** |
| **Qwen3-Coder 80B** | **74 tok/s** · 100% tools | 69 tok/s mlx-lm serve | **1.1x** |
| **Qwen3.5 122B** | **44 tok/s** · 100% tools | 43 tok/s mlx-lm serve | ~**1.0x** |
| **Gemma 4 31B** | **31 tok/s** | - | - |

**Note:** the Qwen3.6 27B MTPLX row is an agentic benchmark, not a single raw decode pass. It measures the developer loop: tool calls, growing context, artifact generation, and build validation.

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
