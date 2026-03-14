# Rapid-MLX Optimization Roadmap

> Goal: For every popular model on Apple Silicon, Rapid-MLX should be the fastest engine — **zero configuration required**. Users pick a model, we auto-apply the best optimizations.

## Strategy

1. **Profile** each popular model — measure decode speed, TTFT, memory usage
2. **Apply** the best optimization techniques per model (MTP, prompt cache, KV quant, speculative decode, EAGLE)
3. **Benchmark** against Ollama, llama.cpp, mlx-lm, LM Studio
4. **Publish** comparison table in README — users see the speed advantage and switch

---

## Optimization Techniques

### Already Implemented

| Technique | Speedup | Status | Notes |
|-----------|---------|--------|-------|
| **Prompt Cache** | 5-30x TTFT | Shipped (SimpleEngine) | Core advantage. Always on. |
| **KV Cache Quantization** | 1.0-1.3x + 4x memory | Shipped | `--kv-bits 4/8`. KV4 can be faster than unquantized on Apple Silicon. |
| **MTP (Qwen3-Next)** | 1.2-2.1x decode | Shipped (BatchedEngine only) | Need to port to SimpleEngine. |
| **Tool Call Recovery** | N/A (reliability) | Shipped | 17 parsers, auto-recovery. |

### To Implement

| Priority | Technique | Expected Speedup | Effort | Applicable Models |
|----------|-----------|-----------------|--------|-------------------|
| **P0** | MTP in SimpleEngine | 1.4x decode | Low | Qwen3-Next, Qwen3.5, DeepSeek-V3, Nemotron |
| **P1** | Standard Speculative Decode | 1.5-2.3x decode | Medium | Any model with small draft variant |
| **P1** | Auto-Optimization per model | N/A | Medium | All models — auto-detect and apply best technique |
| **P2** | EAGLE-3 on Metal | 3-6.5x decode | High | Qwen3-32B, Qwen3-8B, GPT-OSS, Llama-3 |
| **P3** | ReDrafter | 1.4-1.5x | Medium | Needs training heads per model (Apple has MLX code) |

### Not Pursuing

| Technique | Reason |
|-----------|--------|
| Medusa | Superseded by EAGLE-3, only old Vicuna heads available |
| Prompt Lookup Decoding | Benchmarked — minimal benefit, Qwen ArraysCache not trimmable |

---

## Model Optimization Matrix

For each model: which techniques apply, expected speedup, and benchmark status.

| # | Model | Active Params | RAM (4bit) | Optimization Plan | vs Ollama Target | Status |
|---|-------|--------------|-----------|-------------------|-----------------|--------|
| 1 | Qwen3.5-9B | 9B | 5.1 GB | Prompt cache + KV quant + auto-config | **2.7x** | **DONE** ✅ 109 tok/s, 100% tool, 0% leak |
| 2 | Llama 3.2 3B | 3B | ~2 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 3 | Phi-4 Mini 14B | 14B | 2.4 GB | Prompt cache + KV quant | TBD | **Benchmarked** ✅ 174 tok/s, 0% tool (no template support), 100% leak (needs reasoning parser) |
| 4 | Mistral 7B | 7B | ~4.4 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 5 | Mistral Small 24B | 24B | 12.7 GB | Prompt cache + KV quant + mistral parser | TBD | **Benchmarked** ⚠️ 48 tok/s, 0% tool (chat template strips tools), 0% leak |
| 6 | Gemma 3 12B | 12B | 8.5 GB | Prompt cache + KV quant | TBD | **Benchmarked** ⚠️ 49 tok/s, 0% tool (chat template strips tools), no prompt cache benefit |
| 7 | DeepSeek-R1-Distill 14B | 14B | ~9 GB | Prompt cache + KV quant + reasoning parser | 1.5-2x | Not started |
| 8 | Qwen 2.5 Coder 14B | 14B | ~9 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 9 | GPT-OSS 20B | 20B | 11.8 GB | Prompt cache + KV quant + seed_oss parser | TBD | **Benchmarked** ⚠️ 123 tok/s, 0% tool (model doesn't produce tool calls), 0% leak |
| 10 | GLM-4.7-Flash 9B | 9B | 30.1 GB (8bit) | Prompt cache + KV quant + glm47 parser | TBD | **Benchmarked** ✅ 60 tok/s, 100% tool, 0% leak |
| 11 | Llama 3.3 70B | 70B | ~40 GB | Prompt cache + KV quant + spec decode (Llama-8B draft) | 1.5-2x | Not started |
| 12 | Gemma 3 27B | 27B | ~16 GB | Prompt cache + KV quant | 1.5-2x | Not started |
| 13 | Qwen3.5-35B-A3B | 3B active | 34.8 GB (8bit) | Prompt cache + KV quant + MTP | TBD | **Benchmarked** ✅ 82 tok/s, 100% tool, 0% leak |
| 14 | Qwen3.5-122B-A10B | 10B active | 121.3 GB | Prompt cache + KV quant + MTP | TBD | **Benchmarked** ✅ 44 tok/s, 100% tool, 0% leak, TTFT slow (2.4s cached) |
| 15 | Qwen3-Coder-Next 80B | 3B active | 42.2 GB (4bit) | Prompt cache + KV quant + MTP | TBD | **Benchmarked** ✅ 74 tok/s, 100% tool, 0% leak |
| 16 | Qwen3.5-4B | 4B | 2.7 GB | Prompt cache + KV quant | TBD | **Benchmarked** ✅ ~158 tok/s, 100% tool, 0% leak |
| 17 | Qwen3.5-27B | 27B | 14.5 GB | Prompt cache + KV quant | TBD | **Benchmarked** ✅ 39 tok/s, 100% tool, 0% leak |
| 18 | Hermes-3-Llama-8B | 8B | 4.7 GB | Prompt cache + KV quant | TBD | **Benchmarked** ⚠️ 123 tok/s, 0% tool |
| 19 | Devstral-Small-2 24B | 24B | 12.7 GB | Prompt cache + KV quant + mistral parser | TBD | **Benchmarked** ⚠️ 48 tok/s, partial tool ([ARGS] bug) |
| 20 | GLM-4.5-Air | MoE? | 56.4 GB | Prompt cache + KV quant + glm47 parser | TBD | **Benchmarked** ⚠️ ~46 tok/s, 100% tool, long gen broken |
| 21 | Llama 4 Scout 109B | 17B active | ~55 GB | Prompt cache + KV quant | 1.5-2x | ❌ FAILED (dimension mismatch) |
| 22 | DeepSeek R1 671B | 37B active | ~404 GB | Prompt cache + KV quant + MTP | 1.5-2x | Not started |
| 23 | Mixtral 8x7B | 13B active | ~26 GB | Prompt cache + KV quant | 1.5-2x | Not started |

### Auto-Optimization Vision

When a user runs:
```bash
rapid-mlx serve mlx-community/Qwen3.5-9B-4bit --port 8000
```

The engine should automatically:
1. Detect model family → Qwen3.5
2. Apply best tool parser → `hermes`
3. Apply best reasoning parser → `qwen3`
4. Enable prompt cache → always on
5. Set optimal `--prefill-step-size` → based on model size
6. Apply KV cache quantization if beneficial → auto KV4/8
7. Enable MTP if model has MTP head → auto-detect
8. Set optimal temperature/sampling defaults

**Zero flags needed. Just `serve <model>` and get the best performance.**

---

## Benchmark Table (README Target)

The goal is to publish this table in README. Each cell = tok/s decode speed on the same hardware.

### Decode Speed (tok/s) — Apple M3 Ultra 256GB

| Model | Quant | Rapid-MLX | Upstream | Ollama | llama.cpp | mlx-lm | Best Speedup |
|-------|-------|----------|----------|--------|-----------|--------|-------------|
| Phi-4 Mini 14B | 4bit | **174** | 170 | 51 | 55 | 77 | **3.4x** vs Ollama |
| Qwen3.5-4B | 4bit | **~158** | 155 | - | - | 168 | ~1.0x |
| GPT-OSS 20B | MXFP4 | **123** | 79 | - | - | 79 | **1.56x** vs upstream |
| Hermes-3-Llama 8B | 4bit | **123** | 122 | - | - | 127 | ~1.0x |
| Qwen3.5-9B | 4bit | **109** | 104 | 26 | N/A | 61 | **4.2x** vs Ollama |
| Qwen3.5-35B-A3B | 8bit | **82** | 80 | - | - | 85 | ~1.0x |
| Qwen3-Coder-Next 80B | 4bit | **74** | 69 | - | - | 76 | 1.07x vs upstream |
| Qwen3-Coder-Next 80B | 6bit | **68** | - | - | - | 69 | ~1.0x |
| GLM-4.7-Flash 9B | 8bit | **60** | 56 | - | - | - | 1.07x vs upstream |
| Gemma 3 12B | qat-4bit | 49 | N/A* | 54 | N/A | 73 | 0.7x |
| Devstral-Small-2 24B | 4bit | **48** | 48 | - | - | 49 | ~1.0x |
| Mistral Small 24B | 4bit | **48** | 47 | - | - | 41 | **1.2x** vs mlx-lm |
| GLM-4.5-Air | 4bit | ~46 | 54 | - | - | 56 | 0.8x |
| Qwen3.5-122B-A10B | 8bit | 44 | 43 | - | - | 45 | ~1.0x |
| Qwen3.5-27B | 4bit | 39 | 38 | - | - | 39 | ~1.0x |

*N/A = upstream generated only 1 token (artifact)

### TTFT (cached) — Prompt Cache Advantage

| Model | Rapid-MLX | Upstream | Ollama | llama.cpp | Δ vs Upstream |
|-------|----------|----------|--------|-----------|--------------|
| Hermes-3-Llama 8B | **0.080s** | 0.106s | - | - | **1.3x** |
| Qwen3-Coder-Next 80B (4bit) | **0.099s** | 0.141s | - | - | **1.4x** |
| Phi-4 Mini 14B | **0.101s** | 0.119s | 0.058s | 0.026s | **1.2x** |
| Devstral-Small-2 24B | **0.103s** | 0.205s | - | - | **2.0x** |
| Mistral Small 24B | **0.107s** | 1.120s | - | - | **10.5x** |
| GLM-4.5-Air | **0.108s** | 0.201s | - | - | **1.9x** |
| GLM-4.7-Flash 9B | **0.110s** | 0.146s | - | - | **1.3x** |
| GPT-OSS 20B | **0.112s** | 0.234s | - | - | **2.1x** |
| Qwen3.5-9B | **0.145s** | 0.174s | 0.243s | N/A | **1.2x** |
| Gemma 3 12B | **0.147s** | 2.922s | 0.152s | N/A | **19.9x** |

### Tool Calling

| Model | Rapid-MLX | Ollama | llama.cpp |
|-------|----------|--------|-----------|
| Qwen3.5-9B | **100%** | 100% | N/A |
| Qwen3.5-4B | **100%** | - | - |
| Qwen3.5-27B | **100%** | - | - |
| Qwen3.5-35B-A3B | **100%** | - | - |
| Qwen3.5-122B-A10B | **100%** | - | - |
| Qwen3-Coder-Next 80B | **100%** | - | - |
| GLM-4.7-Flash 9B | **100%** | - | - |
| GLM-4.5-Air | **100%** | - | - |
| Devstral-Small-2 24B | partial | - | - |
| Phi-4 Mini 14B | 0% | 0% | 80% |
| Hermes-3-Llama 8B | 0% | - | - |
| Mistral Small 24B | 0% | - | - |
| Gemma 3 12B | 0% | - | - |
| GPT-OSS 20B | 0% | - | - |

---

## Progress Log

### 2026-03-13: Roadmap created
- Identified 18 most popular models on Mac (from Reddit r/LocalLLaMA, r/ollama, benchmarks)
- Mapped optimization techniques to each model
- Starting with Qwen3.5-9B as first model to profile + benchmark

### 2026-03-13: Qwen3.5-9B benchmark complete
- **Rapid-MLX: 108 tok/s decode, 0.14s cached TTFT**
- **Ollama: 41 tok/s decode, 0.28s cached TTFT**
- **Result: 2.7x faster decode, 2.0x faster multi-turn TTFT**
- Hardware: Mac Studio M3 Ultra 256GB
- Quantization: both 4-bit

### 2026-03-14: Multi-model benchmark sweep
- Benchmarked 8 models across 4 engines (Rapid-MLX, Ollama, llama.cpp, mlx-lm)
- **Key wins**: Phi-4 174 tok/s (3.4x vs Ollama), GPT-OSS 123 tok/s (1.6x vs mlx-lm), Qwen3.5-9B 109 tok/s (4.2x vs Ollama)
- **Tool calling**: Qwen family 100% perfect; Phi-4/Mistral/Gemma/GPT-OSS all 0%
- **Root cause for 0% tool calls**: Mistral/Gemma chat templates don't accept `tools` param (silently stripped); Phi-4/GPT-OSS templates support tools but models don't produce them
- **Gemma 3**: Rapid-MLX slower than mlx-lm (49 vs 73) — VLM pipeline overhead
- **llama.cpp**: Can't load Qwen3.5 (rope.dimension_sections mismatch); Phi-4 works at 55 tok/s with 80% tool calls
- **Llama-4-Scout**: Failed with dimension mismatch (mlx-lm compatibility)
- **GLM-4.7-Flash**: Loaded from external SSD, 60 tok/s, 100% tool calling
- Created model profiles: `memory/knowledge/model_profiles.md`
- Updated README with multi-model comparison table

### 2026-03-14: Extended benchmark sweep (6 more models)
- Benchmarked 6 additional models: Qwen3.5-4B (~158 tok/s), Qwen3.5-35B-A3B (82 tok/s), Qwen3.5-27B (39 tok/s), Qwen3-Coder-Next-4bit (74 tok/s), Hermes-3-Llama (123 tok/s), Devstral-Small-2 (48 tok/s), GLM-4.5-Air (~46 tok/s)
- **Qwen3.5 family**: All sizes (4B, 9B, 27B, 35B-A3B, 122B-A10B) achieve 100% tool calling
- **Qwen3-Coder-Next 4bit > 6bit**: 74 vs 68 tok/s, 42 vs 61GB RAM — 4bit is strictly better
- **Hermes-3**: Fast (123 tok/s) but 0% tool calling despite Hermes fine-tune
- **Devstral**: Tool calls produced with mistral parser but `[ARGS]` suffix bug — needs parser fix
- **GLM-4.5-Air**: 100% tool calling but long decode broken (0.1 tok/s — infinite thinking loop)
- Total: 15 models benchmarked on Rapid-MLX, 13 with mlx-lm comparison
