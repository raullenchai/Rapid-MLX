# Supported Models

All quantized models from [mlx-community on HuggingFace](https://huggingface.co/mlx-community/models) are compatible.

Browse thousands of pre-optimized models at: **https://huggingface.co/mlx-community/models**

## Language Models (via mlx-lm)

| Model Family | Sizes | Quantization |
|--------------|-------|--------------|
| Llama 3.x, 4.x | 1B, 3B, 8B, 70B | 4-bit |
| Mistral / Devstral | 7B, Mixtral 8x7B | 4-bit, 8-bit |
| Qwen2/Qwen3 | 0.5B to 72B | Various |
| DeepSeek V3, R1 | 7B, 33B, 67B | 4-bit |
| Gemma 2, 3 | 2B, 9B, 27B | 4-bit |
| GLM-4.7 | Flash, Base | 4-bit, 8-bit |
| Kimi K2 | Various | 4-bit |
| Phi-3 | 3.8B, 14B | 4-bit |
| Granite 3.x, 4.x | Various | 4-bit |
| Nemotron | 3 Nano 30B | 6-bit |
| Hunyuan 3 (Hy3) | 295B MoE (21B active) — **Ultra-only** | 4-bit |

### Recommended Models

| Use Case | Model | Memory |
|----------|-------|--------|
| Fast/Light | `mlx-community/Qwen3-0.6B-8bit` | ~0.7 GB |
| Balanced | `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~1.8 GB |
| Quality | `mlx-community/Llama-3.1-8B-Instruct-4bit` | ~4.5 GB |
| Large | `mlx-community/Qwen3-30B-A3B-4bit` | ~16 GB |

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
| **TTS** | Kokoro | `prince-canuma/Kokoro-82M` |
| **TTS** | Chatterbox | `chatterbox/chatterbox-tts-0.1` |

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
