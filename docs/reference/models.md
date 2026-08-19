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
| Granite 4 | tiny, h-micro | 4-bit |
| Nemotron 3 / 3.5 | Nano 30B, Lightning 30B | 4-bit |
| LFM 2 / 2.5 | 1B, 2.6B, 8B-A1B, 24B-A2B | 4-bit |
| GPT-OSS | 20B, 120B | 4/8-bit, mxfp4 |
| Ternary Bonsai | 1.7B, 27B | 2-bit (ternary) |
| Hunyuan 3 (Hy3) | 295B MoE (21B active) — **Ultra-only** | 4-bit |

### Recommended Models

Recommendations live in one catalog (`vllm_mlx/model_recommendations.json`) shared by the installer, the desktop app, and `rapid-mlx recipe` — run `rapid-mlx recipe` to see the Smart and Fast picks for *this* Mac. The RAM-tier smart picks:

| RAM | Alias | ~8K-prompt peak |
|-----|-------|-----------------|
| 8–15 GB | `lfm2.5-2.6b-4bit` | 3.0 GB |
| 16–17 GB | `qwen3.5-4b-4bit` | 6.0 GB |
| 18–23 GB | `qwen3.5-9b-4bit` | 8.7 GB |
| 24–31 GB | `bonsai-27b-2bit` | 13.0 GB |
| 32 GB+ | `qwen3.8-27b-4bit` | 20.0 GB |

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
