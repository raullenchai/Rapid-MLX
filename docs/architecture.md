# Rapid-MLX System Architecture

## Overview

Inference requests flow through tokenize → prefix-cache lookup → prefill → decode → detokenize, all driven by the scheduler over the mlx-lm public API (`insert`/`next`/`remove`/`close`). The engine layer (`engine/`) wraps mlx-lm with continuous batching; speculative drafters live in `speculative/` (DFlash, SuffixDecoding) and `spec_decode/` (MTP, DSpark); reasoning and tool-call parsing live in `reasoning/` and `tool_parsers/` and feed the streaming `PostProcessor`.

Design principles:

1. **No monkey-patching** — use mlx-lm's public API (`insert`/`next`/`remove`/`close`).
2. **mlx-lm version agnostic** — the public API is stable across versions.
3. **Per-request parsers** — reasoning + tool-call parsers are instantiated per request, never shared.

## System Overview

```
┌──────────────────────────────────────────────────────────┐
│                  Rapid-MLX API Layer                     │
│  (OpenAI-compatible: chat, completions, embeddings,      │
│   audio, tools, MCP, reasoning)                          │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│              BatchedEngine + Scheduler                   │
│     (native Apple Silicon inference and caching)         │
└──────────────────────────────────────────────────────────┘
                           │
       ┌──────────┬────────┴────────┬──────────┐
       ▼          ▼                 ▼          ▼
┌───────────┐┌───────────┐┌─────────────┐┌──────────────┐
│  mlx-lm   ││  mlx-vlm  ││  mlx-audio  ││mlx-embeddings│
│  (LLM)    ││  (Vision) ││  (STT/TTS)  ││ (Embeddings) │
└───────────┘└───────────┘└─────────────┘└──────────────┘
       │          │                 │          │
       └──────────┴────────┬────────┴──────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│                         MLX                              │
│          (Apple ML Framework - Metal kernels)            │
└──────────────────────────────────────────────────────────┘
```

`BatchedEngine` is the sole engine — the older `SimpleEngine` was deleted
(single-request workloads pay zero batching overhead, so the split was no
longer earning its keep). One engine instance handles both single-request
and multi-tenant workloads.

## Module Map

```
vllm_mlx/
├── server.py                  # App factory + model loading + CLI
│
├── config/                    # ServerConfig singleton
│   └── server_config.py
│
├── service/                   # Request processing
│   ├── helpers.py             # Shared request helpers (_resolve_*, get_engine, etc.)
│   └── postprocessor.py       # Streaming pipeline
│
├── routes/                    # HTTP endpoints
│   ├── chat.py                # /v1/chat/completions
│   ├── completions.py         # /v1/completions
│   ├── anthropic.py           # /v1/messages (Anthropic API)
│   ├── health.py              # /health, /v1/cache/*, /v1/status
│   ├── models.py, embeddings.py, audio.py, mcp_routes.py
│
├── engine/                    # Engine abstraction
│   ├── base.py                # BaseEngine ABC, GenerationOutput
│   ├── batched.py             # BatchedEngine (sole engine, continuous batching)
│
├── engine_core.py             # AsyncEngineCore (event loop + thread executor)
├── scheduler.py               # Scheduler (request queue + batch management)
├── mllm_scheduler.py          # MLLM (vision) request scheduler
├── mllm_batch_generator.py    # MLLM batch generation
├── paged_cache.py             # Paged KV cache (blocks, prefix hashing, COW)
├── prefix_cache.py            # Prefix cache manager
├── output_collector.py        # Request output collector
├── model_registry.py          # Model detection & registry
├── cli.py                     # CLI commands
│
├── api/                       # Pydantic request/response models, tool-call plumbing
├── audio/                     # Audio pipeline (STT, TTS, processing)
├── models/                    # Model implementations (MLLM, vendored architectures)
│
├── reasoning/                 # Reasoning parsers (Qwen3, DeepSeek, MiniMax, etc.)
├── tool_parsers/              # Tool call parsers (one module per model family)
├── spec_decode/               # Spec-decode drafters (MTP, DSpark) + registry
├── speculative/               # Speculative decoding (DFlash, SuffixDecoding)
├── agents/                    # Agent profiles (YAML, under profiles/)
├── runtime/                   # Model registry, cache persistence
├── middleware/                # Auth, rate limiting
├── telemetry/                 # Opt-in telemetry (consent, redaction, queue)
├── doctor/                    # User self-diagnostic
│
├── domain/                    # Domain types
│   └── events.py              # StreamEvent (seam between PostProcessor and SSE)
│
└── mcp/                       # MCP tool integration

scripts/                       # Dev-only (NOT shipped with pip)
├── dev_test.py                # Unified test entry point
├── stress_test.py             # 8-scenario stress test
└── agent_soak_test.py         # 10-min agent soak test

tests/                         # pytest unit tests
harness/                       # Regression baselines + thresholds
```

## Request Flow

### Streaming Chat Completion

```
Client POST /v1/chat/completions (stream=true)
    ↓
routes/chat.py: create_chat_completion()
    ├── Validate request
    ├── Apply chat template
    ├── Inject tool/reasoning system prompts
    ↓
routes/chat.py: stream_chat_completion()
    ├── Create StreamingPostProcessor (per-request parser instances)
    ├── engine.stream_chat() → engine.stream_generate()
    │       ↓
    │   engine_core.py: add_request() → scheduler
    │       ↓
    │   scheduler.py: _schedule_waiting() → decode.insert()
    │   scheduler.py: step() → decode.step() → TokenResult
    │       ↓
    │   engine_core.py: stream_outputs() → RequestOutput
    │       ↓
    │   batched.py: yield GenerationOutput
    ↓
    PostProcessor.process_chunk() → StreamEvent
    ↓
    SSE formatting → yield "data: {...}\n\n"
```

## Paged KV Cache Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PagedCacheManager                          │
├─────────────────────────────────────────────────────────────────┤
│  FreeKVCacheBlockQueue     │  BlockHashToBlockMap               │
│  (O(1) doubly linked list) │  (hash → block for prefix caching) │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐  │  {hash_0: block_5}                 │
│  │ 3 │↔│ 7 │↔│ 2 │↔│ 9 │  │  {hash_1: block_12}                │
│  └───┘ └───┘ └───┘ └───┘  │  {hash_2: block_5}  (shared!)      │
│   LRU ───────────▶ MRU    │                                     │
├─────────────────────────────────────────────────────────────────┤
│  CacheBlock[0..N]:                                              │
│  - block_id, ref_count, block_hash                              │
│  - prev_free_block, next_free_block (doubly linked)             │
│  - cache_data: List[(keys, values)] per layer                   │
└─────────────────────────────────────────────────────────────────┘
```

### Cache Flow

```
Request Completion                    Cache Storage
       │                                    │
       ▼                                    ▼
┌──────────────────┐              ┌─────────────────────┐
│ store_cache(     │ ───────────▶ │ Extract KVCache     │
│ tokens, cache)   │              │ .state (keys, vals) │
└──────────────────┘              └─────────────────────┘
                                            │
                                            ▼
                                  ┌─────────────────────┐
                                  │ Slice into fixed-   │
                                  │ size token blocks   │
                                  │ + chain hash        │
                                  └─────────────────────┘
                                            │
       New Request                          ▼
       │                          ┌─────────────────────┐
       ▼                          │ BlockHashToBlockMap │
┌──────────────────┐              │ deduplicate & share │
│ compute_block_   │ ◀─────────── └─────────────────────┘
│ hash(parent,     │
│ tokens)          │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Reconstruct via  │
│ mx.concatenate() │
│ + KVCache.from_  │
│ state()          │
└──────────────────┘
```

Key design points (following vLLM's paged-attention design):

| Feature | Benefit |
|---------|---------|
| **vLLM-style structures** | FreeKVCacheBlockQueue, BlockHashToBlockMap, chain hashing |
| **Real tensor storage** | Extracts actual KV data via mlx-lm's `KVCache.state` |
| **Block deduplication** | Hash-based detection prevents duplicate storage |
| **Copy-on-Write (COW)** | Shared blocks only copied when modified |
| **O(1) LRU eviction** | Doubly linked free-block list for efficient cleanup |

## Hardware Detection

Rapid-MLX auto-detects Apple Silicon:

- Chip name (M1–M4 series, including Pro/Max/Ultra variants)
- Total unified memory
- Memory bandwidth and GPU cores (from known chip profiles)

```python
from vllm_mlx.optimizations import detect_hardware
from vllm_mlx.chip_tier import detect_chip_tier

hw = detect_hardware()
print(f"{hw.chip_name} ({hw.total_memory_gb:.0f} GB, {hw.gpu_cores} GPU cores)")

tier = detect_chip_tier()
print(f"Apple Silicon: {tier.is_apple_silicon}, generation: M{tier.generation}")
```

## Performance Architecture

```
                    ┌─────────────────────────────┐
                    │     Metal GPU (Apple Silicon) │
                    │                               │
                    │  Model Forward   ← bottleneck │
                    │  (~10-50ms/step)              │
                    │                               │
                    └──────────┬────────────────────┘
                               │
                    ┌──────────▼────────────────────┐
                    │     Python Scheduler           │
                    │     (~0.5-1ms/step)            │
                    │                               │
                    │  Request queue                 │
                    │  Batch management              │
                    │  Cache lookup                  │
                    │  Token emission                │
                    └──────────┬────────────────────┘
                               │
                    ┌──────────▼────────────────────┐
                    │     API Layer (FastAPI)         │
                    │     (~0.1ms/request)           │
                    │                               │
                    │  SSE formatting                │
                    │  PostProcessor                 │
                    │  Response serialization        │
                    └───────────────────────────────┘

Bottleneck is always Metal GPU compute, not Python scheduling.
C/C++ scheduler rewrite would save <3% throughput.
```
