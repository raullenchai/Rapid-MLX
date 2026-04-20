# Rapid-MLX System Architecture

## Pipeline Architecture

Inference requests flow through a composable pipeline of pluggable stages:

```
Request → [Tokenize] → [PrefixCache] → [Prefill] → [Decode] → [Detokenize] → Response
                              ↑              ↑           ↑
                         Pluggable:      Pluggable:   Pluggable:
                         - LRU cache    - Chunked    - Standard (bg.next())
                         - TurboQuant   - Paged      - MTP draft
                         - RadixTree    - Offload    - Speculative
                                                     - Medusa heads
```

Each stage implements a Python ABC (`vllm_mlx/pipeline/interfaces.py`). Adding a new optimization = implementing the interface in a new file. No scheduler changes needed.

### Stage Interfaces

| Stage | Interface | Default Implementation | What it does |
|-------|-----------|----------------------|--------------|
| **Cache** | `CacheStrategy` | `LRUPrefixCache` | Look up/store KV cache by token prefix |
| **Decode** | `DecodeStrategy` | `StandardDecode` | Generate tokens via mlx-lm BatchGenerator |
| **Plugin** | `DecodePlugin` | *(none)* | Wrap decode step with MTP/speculative/Medusa |

### Key Design Principles

1. **No monkey-patching** — use mlx-lm's public API (`insert/next/remove/close`)
2. **One model = zero scheduler changes** — all model architectures work through the same interface
3. **Composable plugins** — `MTPPlugin(StandardDecode(...))` wraps without replacing
4. **mlx-lm version agnostic** — public API is stable across versions

## Module Map

```
vllm_mlx/
├── server.py                  # App factory + model loading + CLI (1047 lines)
│
├── pipeline/                  # Composable inference pipeline
│   ├── interfaces.py          # Stage ABCs: CacheStrategy, DecodeStrategy, DecodePlugin
│   └── decode.py              # StandardDecode (mlx-lm public API wrapper)
│
├── config/                    # ServerConfig singleton
│   └── server_config.py
│
├── service/                   # Request processing
│   ├── helpers.py             # Shared request helpers (_resolve_*, get_engine, etc.)
│   └── postprocessor.py       # Streaming pipeline (100% test coverage)
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
│   ├── batched.py             # BatchedEngine (default, continuous batching)
│
├── engine_core.py             # AsyncEngineCore (event loop + thread executor)
├── scheduler.py               # Scheduler (request queue + batch management)
│
├── reasoning/                 # 7 reasoning parsers (Qwen3, DeepSeek, MiniMax, etc.)
├── tool_parsers/              # 20+ tool call parsers
├── agents/                    # 11 agent profiles (YAML)
├── runtime/                   # Model registry, cache persistence
├── middleware/                # Auth, rate limiting
├── doctor/                    # User self-diagnostic
│
├── domain/                    # Domain types
│   └── events.py              # StreamEvent (seam between PostProcessor and SSE)
│
└── mcp/                       # MCP tool integration

scripts/                       # Dev-only (NOT shipped with pip)
├── dev_test.py                # Unified test entry point
├── stress_test.py             # 8-scenario stress test
├── agent_soak_test.py         # 10-min agent soak test
└── cross_model_stress.py      # Multi-model validation

tests/                         # pytest unit tests (2100+)
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

## Adding New Optimizations

### Example: TurboQuant KV Cache Compression

```python
# vllm_mlx/pipeline/turbo_quant.py

class TurboQuantCache(CacheStrategy):
    """TurboQuant KV cache compression (Google, 2025).
    
    Compresses KV cache entries to reduce memory footprint,
    enabling longer context windows on the same hardware.
    """
    
    def __init__(self, base_cache: CacheStrategy, compression_ratio: float = 4.0):
        self._base = base_cache
        self._ratio = compression_ratio
    
    def lookup(self, token_ids: list[int]) -> CacheResult:
        result = self._base.lookup(token_ids)
        if result.hit:
            result.cache = self._decompress(result.cache)
        return result
    
    def store(self, token_ids: list[int], cache: Any) -> None:
        compressed = self._compress(cache)
        self._base.store(token_ids, compressed)
```

### Example: Speculative Decode Plugin

```python
# vllm_mlx/pipeline/speculative.py

class SpeculativePlugin(DecodePlugin):
    """Draft model speculative decoding.
    
    Uses a small draft model to predict N tokens, then verifies
    with the full model in a single forward pass.
    """
    
    def __init__(self, draft_model, num_draft_tokens: int = 4):
        self._draft = draft_model
        self._n = num_draft_tokens
    
    def wrap_step(self, base_step):
        # 1. Generate N draft tokens with small model
        drafts = self._draft_tokens()
        # 2. Verify all N+1 tokens in one forward pass
        verified = self._verify(drafts)
        # 3. Return accepted tokens
        return verified
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
