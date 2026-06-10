# Continuous Batching

Continuous batching enables higher throughput when serving multiple concurrent
users. It is **on by default** — the `--continuous-batching` flag is accepted
for back-compat but is a no-op.

## Default Behaviour

```bash
rapid-mlx serve qwen3.5-4b-4bit
```

## With Paged Cache

For memory-efficient prefix sharing:

```bash
rapid-mlx serve qwen3.5-4b-4bit --use-paged-cache
```

## How It Works

### Continuous Batching (always on)
- Multiple requests processed together when concurrency > 1
- Single-request workloads pay zero overhead
- Implemented in `BatchedEngine` (the sole engine; the old `SimpleEngine` was
  removed)

### Paged Cache
- KV cache stored in fixed-size blocks
- Shared system prompts use same blocks
- Memory savings: 80%+ for 10+ concurrent users

## Performance Results

**Continuous Batching Results (M4 Max, 128GB):**

| Model | Single Request | Batch (5 req) | Speedup |
|-------|----------------|---------------|---------|
| Llama-3.2-1B-Instruct-4bit | 299.1 tok/s | 613.0 tok/s | **2.05x** |
| Llama-3.2-3B-Instruct-4bit | 137.6 tok/s | 208.1 tok/s | **1.51x** |
| Qwen3-0.6B-8bit | 328.1 tok/s | 1111.8 tok/s | **3.39x** |
| Qwen3-30B-A3B-4bit | 98.1 tok/s | 233.3 tok/s | **2.38x** |
| Qwen2.5-1.5B-Instruct-4bit | 196.9 tok/s | 322.2 tok/s | **1.64x** |

*Batching 5 concurrent requests shows 1.5-3x throughput improvement.*

## Streaming Performance

**Streaming Performance (M4 Max, 128GB):**

| Model | TTFT | Generation Speed |
|-------|------|------------------|
| Llama-3.2-1B-Instruct-4bit | ~4.6ms | 218.9 tok/s |
| Llama-3.2-3B-Instruct-4bit | ~10.7ms | 93.6 tok/s |
| Qwen3-0.6B-8bit | ~3.0ms | 328.5 tok/s |
| Qwen3-30B-A3B-4bit | ~10.2ms | 98.4 tok/s |
| Qwen2.5-1.5B-Instruct-4bit | ~7.1ms | 140.3 tok/s |

*TTFT = Time to First Token*

## Streaming Configuration

Control token delivery with `--stream-interval`:

```bash
# Every token (smoothest)
rapid-mlx serve model --stream-interval 1

# Batch tokens (better for high-latency)
rapid-mlx serve model --stream-interval 5
```

| Value | Behavior |
|-------|----------|
| `1` | Send every token immediately |
| `2-5` | Batch tokens before sending |
| `10+` | Maximum throughput, chunkier output |

## Memory Management

For large models, the prefix cache can consume significant memory. The memory-aware cache automatically manages this:

```bash
# Auto-detect (uses 20% of available RAM)
rapid-mlx serve model

# Explicit limit
rapid-mlx serve model --cache-memory-mb 2048

# Custom percentage
rapid-mlx serve model --cache-memory-percent 0.10
```

| Option | Description |
|--------|-------------|
| `--cache-memory-mb` | Set explicit limit in MB |
| `--cache-memory-percent` | Fraction of available RAM (default: 0.20) |
| `--no-memory-aware-cache` | Use legacy entry-count based cache |

## Prefix Cache

Prefix caching reuses KV cache for repeated prompts.

### How It Works

```
User 1: System prompt (500 tokens) → Creates 8 blocks
User 2: Same system prompt → Shares 8 blocks (ref_count++)
User N: Same system prompt → Shares 8 blocks (ref_count++)

Memory savings: 80%+ for 10+ concurrent users
```

### Cache Key Strategy

- **LLM**: `hash(prompt)`
- **Images**: `hash(image_content) + hash(prompt)`
- **Videos**: `hash(video_path) + hash(fps) + hash(max_frames) + hash(prompt)`

### Testing Prefix Cache

```bash
python tests/test_prefix_cache.py
```

```
======================================================================
  LLM PREFIX CACHE TEST
======================================================================
  Model: mlx-community/Qwen3-0.6B-8bit
  Expected behavior:
    - Same prompt → cache HIT
    - Different prompt → cache MISS or PREFIX_HIT (shared template tokens)
----------------------------------------------------------------------
  Results:
  Step   | Description         | Expected | Actual | Status
  -------+---------------------+----------+--------+-------
  1a     | First request       | MISS     | MISS   | PASS
  1b     | Same prompt         | HIT      | HIT    | PASS
  1c     | Different prompt    | MISS     | MISS   | PASS
  1d     | Return to prompt 1  | HIT      | HIT    | PASS
======================================================================
```

## Running Benchmarks

```bash
# Continuous batching benchmark
python tests/test_continuous_batching.py

# Prefix cache test
python tests/test_prefix_cache.py
```

## When to Use

| Scenario | Flags |
|----------|-------|
| Default — any concurrency, any model | *(none — batching is on)* |
| Large models on tight RAM | `--cache-memory-mb 2048` |
| Production with shared prompts | `--use-paged-cache` |

## Production Setup

```bash
rapid-mlx serve qwen3.5-9b-4bit \
  --use-paged-cache \
  --port 8000
```
