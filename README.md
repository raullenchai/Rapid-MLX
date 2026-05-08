# lightning-mlx

**🔥 The fastest local AI engine for Apple Silicon. Optimized for agentic use.**

It's [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) with [MTPLX](https://github.com/youssofal/MTPLX/) inspired work.

## Raw Decode Benchmarks

The Lightning MLX MTPLX raw-decode rows were run with explicit max-performance
benchmark settings:

| Model | mlx-lm | oMLX | Rapid MLX | **Lightning MLX (MTPLX)** |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | 29.80 tok/s | 31.80 tok/s | 32.37 tok/s | **40.67 tok/s** |
| Qwen3.6-35B | 110.37 tok/s | 114.59 tok/s | 106.00 tok/s | **220.86 tok/s** |

Used:

```bash
lightning-mlx bench qwen3.6-27b \
--num-prompts 3 --max-tokens 512 --disable-prefix-cache \
--max-num-seqs 1 --prefill-batch-size 1 --completion-batch-size 1 \
--prefill-step-size 8192 --mtp-num-draft-tokens 3 --mtp-optimistic

lightning-mlx bench qwen3.6-35b \
--num-prompts 3 --max-tokens 512 --disable-prefix-cache \
--max-num-seqs 1 --prefill-batch-size 1 --completion-batch-size 1 \
--prefill-step-size 8192 --mtp-num-draft-tokens 3 --mtp-optimistic
```

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


## More Model Benchmarks

The models below use the same benchmark positioning as upstream [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX): Mac Studio M3 Ultra, Apple MLX backend, and tok/s as decode throughput.

| Model | lightning-mlx | Best Alternative | Speedup |
| --- | ---: | ---: | ---: |
| **Phi-4 Mini 14B** | **180 tok/s** | 77 tok/s mlx-lm / 56 tok/s Ollama | **2.3x** / **3.2x** |
| **Qwen3.5 4B** | **160 tok/s** | 155 tok/s mlx-lm serve | **1.0x** |
| **Nemotron-Nano 30B** | **141 tok/s** · 100% tools | - | - |
| **DeepSeek V4 Flash 158B-A13B** (2-bit DQ) | **56 tok/s** | Only MLX engine, day-0 | - |
| **DeepSeek V4 Flash 158B-A13B** (8-bit) | **31 tok/s** | Only MLX engine, day-0 | - |
| **GPT-OSS 20B** | **127 tok/s** · 100% tools | 79 tok/s mlx-lm serve | **1.6x** |
| **Qwen3.5 9B** | **108 tok/s** | 41 tok/s Ollama | **2.6x** |
| **Kimi-Linear 48B** | **94 tok/s** · 100% tools | Only engine | - |
| **Gemma 4 26B-A4B** | **85 tok/s** | 68 tok/s Ollama | **1.3x** |
| **Gemma 4 E4B** | **83 tok/s** | - | - |
| **Qwen3.5 35B-A3B** | **83 tok/s** · 100% tools | 75 tok/s oMLX | **1.1x** |
| **Qwen3-Coder 80B** | **74 tok/s** · 100% tools | 69 tok/s mlx-lm serve | **1.1x** |
| **Qwen3.5 122B** | **44 tok/s** · 100% tools | 43 tok/s mlx-lm serve | ~**1.0x** |
| **Gemma 4 31B** | **31 tok/s** | - | - |

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

Best optimized models:

```bash
lightning-mlx serve qwen3.6-27b
lightning-mlx serve qwen3.6-35b
```

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

## What You Get

- **2.75x faster short agentic turns** in the benchmark fixture.
- **1.96x higher all-turn throughput** versus the MLX baseline.
- **Successful artifact generation** where baseline timed out.
- **OpenAI-compatible API** for local tools, agents, editors, and CLIs.
- **Apple Silicon first**: built around MLX and local Mac inference.
- **MTPLX optimized preset** behind one simple command.

## Built On

- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX)
- [MTPLX](https://github.com/youssofal/MTPLX/)
- [MLX](https://github.com/ml-explore/mlx)
