# Supported Models

Most quantized models from [mlx-community on HuggingFace](https://huggingface.co/mlx-community/models) work out of the box — any architecture mlx-lm/mlx-vlm knows can be served by HF path (unknown `model_type`s fail at load). The curated alias registry (`rapid-mlx models`) is the supported surface: every alias there is profiled (parser, hybrid flags, KV codec, spec-decode gates — `rapid-mlx info <alias>`).

## Language Models (via mlx-lm)

Families with registered aliases (run `rapid-mlx models` for the full, current list):

| Model Family | Registered sizes | Quantization |
|--------------|-------|--------------|
| Qwen 3.5 / 3.6 / 3.8 / Coder | 4B to 122B | 4/6/8-bit, mxfp4, mixed |
| Gemma 4 | E2B, E4B, 12B, 26B, 31B (+ QAT builds) | 4/6/8-bit |
| DeepSeek V4 Flash, R1, Coder V2 Lite | 16B, 8B/32B (R1), V4 Flash MoE | 2/4/8-bit, mxfp4 |
| Llama 3.x | 1B, 3B, 8B | 4/8-bit |
| Mistral / Devstral | 24B, 119B (Small 4) | 4/8-bit |
| GLM | 4.5 Air, 4.7 9B, 5.2 REAP-50 | 4-bit |
| Kimi | K2.6 | 4-bit |
| Phi 3.5 / 4 | mini, 14B | 4-bit |
| Granite 4 / 4.2 | tiny, h-micro (4.0-H); 3B, 8B, 30B dense (4.2) | 4/8-bit |
| Nemotron 3 / 3.5 | Nano 30B, Lightning 30B | 4-bit |
| LFM 2 / 2.5 | 1B, 2.6B, 8B-A1B, 24B-A2B | 4-bit |
| MiniCPM 5 | 1B, 2B | 4-bit, OptiQ 4-bit |
| GPT-OSS | 20B, 120B | 4/8-bit, mxfp4 |
| Ternary Bonsai | 1.7B, 27B | 2-bit (ternary) |
| Hunyuan 3 (Hy3) | 295B MoE (21B active) — **Ultra-only** | 4-bit |
| NeoHorse 1 | 9B (experimental Chat candidate) | 4-bit |
| G9v3 (AI9Stars) | 39B MoE (5B active) | 4-bit |

### Experimental 256 GB lane: DeepSeek V4.1 Flash

`deepseek-v41-flash-reap-2bit` serves the pinned Rapid-MLX REAP 2-bit
checkpoint with its pinned DSpark K4 sidecar. The target contains about 199 GiB
of tensors; the sidecar download is narrowed to the three required MTP shards
plus its data index and config (4.47 GB), rather than fetching the full source
repository.

```bash
rapid-mlx pull deepseek-v41-flash-reap-2bit
rapid-mlx serve deepseek-v41-flash-reap-2bit
```

This is a deliberately narrow product lane:

- 256 GB Apple Silicon is the supported hardware; startup fails closed below
  the measured 224 GiB unified-memory floor.
- Requests serialize behind one model worker. Continuous batching and prefix
  caching are not claimed for this architecture-specific cache.
- Greedy generation is supported, with at most 8,192 input tokens and 4,096
  output tokens. Sampling, images, tools, MCP, and structured output are not
  yet qualified.
- The four-workload 128-token suite was deterministic across two repeats. It
  measured 10.72 tok/s overall; the isolated stable K4 run measured 11.76
  tok/s. The product wrapper loaded in 300.81 seconds and peaked at 217.97 GB
  on the qualification Studio.

The runtime verifies immutable revisions, expected file sizes, SHA-256 values
for every sidecar data file, and the complete tensor/config contract before it
reports healthy.

### MiniCPM5 2B

`minicpm5-2b-4bit` is the compact, tool-capable option for latency-sensitive
local assistants. It uses the official Apple Silicon 4-bit checkpoint (about
1.3 GiB), the native MiniCPM XML tool-call parser, and the Qwen-style reasoning
parser. It is available in both the CLI and Desktop model picker, but it does
not replace the RAM-tier Smart defaults.

```bash
rapid-mlx serve minicpm5-2b-4bit
```

The exact checkpoint passed 24 of 31 tool-calling scenarios in product-path
qualification. It completed the same suite in 7.17 seconds of summed request
time versus 32.54 seconds for the current 4B comparison model, while the 4B
model passed 26 of 31. Treat MiniCPM5 2B as the faster, smaller trade-off rather
than a blanket quality replacement. Speculative decoding remains disabled
until its separately published draft architecture is supported and qualified.

### Recommended Models

Recommendations live in one catalog (`vllm_mlx/model_recommendations.json`) shared by the installer, the desktop app, and `rapid-mlx recipe` — run `rapid-mlx recipe` to see the Smart and Fast picks for *this* Mac. The RAM-tier smart picks:

| RAM | Alias | ~8K-prompt peak |
|-----|-------|-----------------|
| 8–15 GB | `lfm2.5-2.6b-4bit` | 3.0 GB |
| 16–17 GB | `qwen3.5-4b-4bit` | 6.0 GB |
| 18–23 GB | `qwen3.5-9b-4bit` | 8.7 GB |
| 24–31 GB | `bonsai-27b-2bit` | 13.0 GB |
| 32 GB+ | `qwen3.8-27b-4bit` | 20.0 GB |

### Experimental Chat candidate: NeoHorse 1 9B

`neohorse-9b-4bit` is an opt-in, text-only Chat model for Macs with at
least 18 GB of unified memory. It is not a Smart/Fast recommendation and does
not change any default. Product-path qualification found unresolved gaps in
multi-step tool use, executable coding, reasoning, and instruction following,
so broader comparative dogfood is required before any default change.

```bash
rapid-mlx serve neohorse-9b-4bit
```

The alias deliberately enables none of the Qwen-specific speculative or
cache optimizations until that exact checkpoint has separate compatibility
evidence. See the
[reproducible qualification note](../engineering/performance/2026-09-08-neohorse-9b-chat-qualification.md)
for the current evidence and limitations.

### Experimental research model: Qwen3.8 27B Abliterated

`qwen3.8-27b-abliterated-4bit` serves the `oQ4e/` Apple-Silicon build of
[`windowsxp811203/Qwen3.8-27B-Abliterated-MLX-MTP`](https://huggingface.co/windowsxp811203/Qwen3.8-27B-Abliterated-MLX-MTP).
The alias downloads only that 16.99 GB checkpoint rather than every build in
the multi-quant repository. It supports text and image input and is deliberately
not a Smart/Fast default.

```bash
# Text-only (does not require the optional vision runtime).
rapid-mlx serve qwen3.8-27b-abliterated-4bit --no-mllm

# Image input requires the exact optional vision stack shipped with Rapid-MLX.
rapid-mlx serve qwen3.8-27b-abliterated-4bit --mllm
```

The publisher reports 81.75% on its seeded 400-question MMLU sample for this
quantization, versus 82.50% for its bf16 conversion, and 0/80 AdvBench plus
0/119 HarmBench-safety refusals. Those are checkpoint-author measurements, not
Rapid-MLX qualification results. Abliteration removes refusal behavior; it does
not make answers more accurate or safe.

The repository also carries an MTP head, but Rapid-MLX keeps speculative
decoding off for this alias until the exact abliterated target/drafter pair is
qualified on our runtime. Do not pair it with the curated base-Qwen DFlash2
drafter: the publisher modified both the trunk and MTP head, so a draft trained
for the original target is not an equivalent artifact.

See the
[reproducible qualification note](../engineering/performance/2026-09-09-qwen38-27b-abliterated-qualification.md)
for the pinned artifact, product-path smoke results, measured throughput, and
the remaining speculative-decoding gate.

### Ultra-only: Hunyuan 3 (Hy3)

> ⚠️ **Validated only on an M3 Ultra with 256 GB unified memory.** The
> runtime enforces a **192 GB** unified-memory floor (`min_memory_gb`) and
> prints a loud warning below it — it does *not* check the chip
> generation, so a 192 GB non-Ultra Mac is not blocked but is untested.
> Do not attempt on a smaller Mac — it will OOM the Metal allocator (or,
> on macOS < 15.2, kernel-panic) before the first token generates.

Tencent's **Hunyuan 3** is a 295B-parameter Mixture-of-Experts model
(21B active per token). Only a 4-bit quant is shipped:

| Alias | HF path | Weights | Peak RAM | Hardware |
|-------|---------|---------|----------|----------|
| `hy3-preview-4bit` | `mlx-community/Hy3-preview-4bit` | ~166 GB | ~156 GB | M3 Ultra 256 GB |

```bash
rapid-mlx serve hy3-preview-4bit
```

The alias carries a `min_memory_gb: 192` floor. Before the 166 GB
download begins, rapid-mlx checks your machine's total unified memory and
prints a loud warning if it is below the floor:

```
⚠  Ultra-only alias 'hy3-preview-4bit' declares a 192 GB unified-memory
   floor, but this Mac reports 128.0 GB.
   The model weights are large enough to OOM the Metal allocator (or
   kernel-panic on macOS < 15.2, issue #324) before the first token
   generates.
   Recommended: pick a Tier-1 alias sized for this machine
   (`rapid-mlx models` for the full list). Proceeding anyway…
```

The warning never aborts (an operator with an unusual allocator setup can
still opt in), but on any non-Ultra Mac you should pick a smaller alias
instead — `rapid-mlx models` lists every alias with its size. Hy3's tool
calling and reasoning are exercised in CI without booting the model via
an offline parser-level integration test; real-inference coverage runs in
the weekly Golden Path job on M3 Ultra hardware.

### Granite 4.2 (IBM) 3B / 8B / 30B dense

Granite 4.2 is IBM's dense line (`model_type: granite`, native in mlx-lm; the
4.0-H `tiny` / `h-micro` aliases are the hybrid Mamba line). The aliases point
at IBM's own MLX conversions:

| Alias | HF path | Weights |
|-------|---------|---------|
| `granite-4.2-30b-4bit` | `ibm-granite/granite-4.2-30b-q4-mlx` | ~15 GB |
| `granite-4.2-30b-8bit` | `ibm-granite/granite-4.2-30b-q8-mlx` | ~29 GB |
| `granite-4.2-8b-4bit` | `ibm-granite/granite-4.2-8b-q4-mlx` | ~4.6 GB |
| `granite-4.2-8b-8bit` | `ibm-granite/granite-4.2-8b-q8-mlx` | ~8.7 GB |
| `granite-4.2-3b-4bit` | `ibm-granite/granite-4.2-3b-q4-mlx` | ~1.9 GB |
| `granite-4.2-3b-8bit` | `ibm-granite/granite-4.2-3b-q8-mlx` | ~3.6 GB |

```bash
rapid-mlx serve granite-4.2-8b-4bit
```

The Granite 4.2 chat template is ChatML with a Qwen3-style `<think>` block
and Qwen3-Coder-style `<function=…><parameter=…>` tool calls (the tokenizer
config declares `tool_parser_type: qwen3_coder`), so the aliases pin the
`qwen3` reasoning parser and the `qwen3_coder_xml` tool-call parser.
**Thinking is on by default** in this template; `enable_thinking: false`
(what Desktop sends with the thinking toggle off, and what
`reasoning_effort: "none"` maps to) turns it off. `reasoning_effort: "low"`
uses the reasoning-token budget; to send the template's own
`{reasoning effort: low}` marker instead, pass
`chat_template_kwargs: {"reasoning_effort": "low"}`.

### G9v3 (AI9Stars) 39B-A5B MoE

`ai9stars/G9v3-39A5B` is a 39B-parameter Mixture-of-Experts model with 5B
parameters active per token (32 of 320 routed experts plus one shared
expert), gated GQA attention and a 128K context. The architecture ships
only as `trust_remote_code` transformers code, so rapid-mlx vendors the MLX
backbone (`vllm_mlx/models/g9v3.py`) and publishes its own conversion.
The 4-bit export keeps the routed experts at 4-bit and everything else
(attention, dense/shared MLP, embeddings) at 8-bit: uniform 4-bit was
measured to hurt this architecture's attention projections badly.

| Alias | HF path | Weights | Hardware |
|-------|---------|---------|----------|
| `g9v3-39a5b-4bit` | `rapid-mlx/G9v3-39A5B-MLX-4bit` | ~21 GB | 32 GB+ unified memory |

```bash
rapid-mlx serve g9v3-39a5b-4bit
```

The chat template is ChatML with Qwen3-style `<think>` blocks and
MiniCPM-style XML tool calls, so the alias pins the `qwen3` reasoning parser
and the `minicpm` tool-call parser; `enable_thinking` toggles reasoning the
same way it does for the Qwen family. Speculative decoding is not enabled
for this alias (the checkpoint has no draft model or MTP head).

## Multimodal Models (via mlx-vlm)

| Model Family | Example Models |
|--------------|----------------|
| **Qwen-VL** | `Qwen3-VL-4B-Instruct-3bit`, `Qwen3-VL-8B-Instruct-4bit`, `Qwen2-VL-2B/7B-Instruct-4bit` |
| **LLaVA** | `llava-1.5-7b-4bit`, `llava-v1.6-mistral-7b-4bit`, `llava-llama-3-8b-v1_1-4bit` |
| **Idefics** | `Idefics3-8B-Llama3-4bit`, `idefics2-8b-4bit` |
| **PaliGemma** | `paligemma2-3b-mix-224-4bit`, `paligemma-3b-mix-224-8bit` |
| **Pixtral** | `pixtral-12b-4bit`, `pixtral-12b-8bit` |
| **Molmo** | `Molmo-7B-D-0924-4bit`, `Molmo-7B-D-0924-8bit` |
| **Phi-3 Vision** | `Phi-3-vision-128k-instruct-4bit` |
| **DeepSeek-VL** | `deepseek-vl-7b-chat-4bit`, `deepseek-vl2-small-4bit` |

### Recommended VLM Models

| Use Case | Model | Memory |
|----------|-------|--------|
| Fast/Light | `mlx-community/Qwen3-VL-4B-Instruct-3bit` | ~3 GB |
| Balanced | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | ~6 GB |
| Quality | `mlx-community/Qwen3-VL-30B-A3B-Instruct-6bit` | ~20 GB |

## Embedding Models (via mlx-embeddings)

| Model Family | Example Models |
|--------------|----------------|
| **BERT** | `mlx-community/bert-base-uncased-mlx` |
| **XLM-RoBERTa** | `mlx-community/multilingual-e5-small-mlx`, `multilingual-e5-large-mlx` |
| **ModernBERT** | `mlx-community/ModernBERT-base-mlx` |

## Audio Models (via mlx-audio)

| Type | Model Family | Example Models |
|------|--------------|----------------|
| **STT** | Whisper | `mlx-community/whisper-large-v3-turbo` |
| **STT** | Parakeet | `mlx-community/parakeet-tdt-0.6b-v2` |
| **TTS** | Kokoro | `mlx-community/Kokoro-82M-bf16` (alias `kokoro`) |
| **TTS** | Chatterbox | `mlx-community/chatterbox-turbo-fp16` (alias `chatterbox`) |

## Model Detection

rapid-mlx auto-detects multimodal models by name patterns:
- Contains "VL", "Vision", "vision"
- Contains "llava", "idefics", "paligemma"
- Contains "pixtral", "molmo", "deepseek-vl"
- Contains "MedGemma", "Gemma-3" (vision variants)

## Using Models

### From HuggingFace

```bash
rapid-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Local Path

```bash
rapid-mlx serve /path/to/local/model
```

## Finding Models

Filter mlx-community models by:
- **LLM**: `Llama`, `Qwen`, `Mistral`, `Phi`, `Gemma`, `DeepSeek`, `GLM`, `Kimi`, `Granite`, `Nemotron`
- **VLM**: `-VL-`, `llava`, `paligemma`, `pixtral`, `molmo`, `idefics`, `deepseek-vl`, `MedGemma`
- **Embedding**: `e5`, `bert`, `ModernBERT`
- **Size**: `1B`, `3B`, `7B`, `8B`, `70B`
- **Quantization**: `4bit`, `8bit`, `bf16`
