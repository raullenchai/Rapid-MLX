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
