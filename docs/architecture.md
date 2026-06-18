# Rapid-MLX System Architecture

## Overview

Inference requests flow through tokenize → prefix-cache lookup → prefill → decode → detokenize, all driven by the scheduler over the mlx-lm public API (`insert`/`next`/`remove`/`close`). The engine layer (`engine/`) wraps mlx-lm with continuous batching; speculative drafters (DFlash, SuffixDecoding, MTP) live in `speculative/`; reasoning and tool-call parsing live in `reasoning/` and `tool_parsers/` and feed the streaming `PostProcessor`.

Design principles:

1. **No monkey-patching** — use mlx-lm's public API (`insert`/`next`/`remove`/`close`).
2. **mlx-lm version agnostic** — the public API is stable across versions.
3. **Per-request parsers** — reasoning + tool-call parsers are instantiated per request, never shared.

## Module Map

```
vllm_mlx/
├── server.py                  # App factory + model loading + CLI (1047 lines)
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
├── agents/                    # 12 agent profiles (YAML)
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
