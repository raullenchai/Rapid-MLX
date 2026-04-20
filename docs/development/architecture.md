# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────┐
│                     vLLM API Layer                        │
│  (OpenAI-compatible: chat, completions, embeddings,      │
│   audio, tools, MCP, reasoning)                          │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                      MLXPlatform                          │
│       (vLLM platform plugin for Apple Silicon)           │
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
│                         MLX                               │
│          (Apple ML Framework - Metal kernels)            │
└──────────────────────────────────────────────────────────┘
```

## Engine Architecture

### Simple Engine
- Direct mlx-lm/mlx-vlm wrapper
- Maximum throughput for single user
- Zero batching overhead

### Batched Engine
- AsyncEngineCore with continuous batching
- Multiple concurrent requests
- Scheduler with priority queue

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
│ response.cache() │ ───────────▶ │ Extract .state      │
│ (KVCache objects)│              │ (keys, values)      │
└──────────────────┘              └─────────────────────┘
                                            │
                                            ▼
                                  ┌─────────────────────┐
                                  │ Slice into 64-token │
                                  │ blocks + chain hash │
                                  └─────────────────────┘
                                            │
       New Request                          ▼
       │                          ┌─────────────────────┐
       ▼                          │ BlockHashToBlockMap │
┌──────────────────┐              │ deduplicate & share │
│ compute_block_   │ ◀─────────── └─────────────────────┘
│ hash(parent, tok)│
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

## Key Features

| Feature | Benefit |
|---------|---------|
| **1.14x Speedup** | Faster inference by reusing cached KV computations |
| **80% Memory Savings** | Share system prompt blocks across concurrent users |
| **vLLM Architecture** | FreeKVCacheBlockQueue, BlockHashToBlockMap, chain hashing |
| **Real Tensor Storage** | Extracts actual KV data using `.state` |
| **Block Deduplication** | Hash-based detection prevents duplicate storage |
| **Copy-on-Write (COW)** | Shared blocks only copied when modified |
| **O(1) LRU Eviction** | Doubly linked list for efficient cleanup |

## Module Structure

```
vllm_mlx/
├── api/
│   ├── models.py         # Pydantic models
│   ├── utils.py          # Shared utilities
│   ├── streaming.py      # Streaming JSON encoder
│   └── tool_calling.py   # Tool call parsing
├── audio/
│   ├── processor.py      # Audio preprocessing
│   ├── stt.py            # Speech-to-Text
│   └── tts.py            # Text-to-Speech
├── engine/
│   ├── base.py           # BaseEngine ABC
│   └── batched.py        # BatchedEngine (continuous batching)
├── mcp/
│   ├── client.py         # MCP client
│   ├── config.py         # Config loading
│   ├── executor.py       # Tool execution
│   ├── security.py       # Command validation
│   ├── tools.py          # Tool sandbox
│   └── manager.py        # Server management
├── models/
│   └── mllm.py           # MLXMultimodalLM
├── tool_parsers/         # Tool call parsers (12 formats)
├── reasoning_parsers/    # Reasoning parsers (qwen3, deepseek_r1)
├── server.py             # FastAPI server
├── engine_core.py        # AsyncEngineCore
├── scheduler.py          # LLM request scheduler
├── mllm_scheduler.py     # MLLM request scheduler
├── mllm_batch_generator.py # MLLM batch generation
├── paged_cache.py        # Paged KV cache
├── prefix_cache.py       # Prefix cache manager
├── output_collector.py   # Request output collector
├── model_registry.py     # Model detection & registry
└── cli.py                # CLI commands
```

## Request Flow

1. **API Request** → FastAPI endpoint (auth, rate limit)
2. **Engine Selection** → Simple or Batched based on config
3. **Template Application** → Chat template formatting (with tool definitions if enabled)
4. **Generation** → mlx-lm, mlx-vlm, mlx-audio, or mlx-embeddings
5. **Post-processing** → Tool call parsing, reasoning extraction
6. **Streaming** → SSE response chunks
7. **Caching** → KV cache storage for reuse

## Hardware Detection

rapid-mlx auto-detects Apple Silicon:
- Chip name (M1, M2, M3, M4)
- Total memory
- Neural engine cores
- GPU cores

```python
from vllm_mlx.hardware import get_hardware_info

hw = get_hardware_info()
print(f"{hw.chip_name} ({hw.total_memory_gb:.0f} GB)")
```
