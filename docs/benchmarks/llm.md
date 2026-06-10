# LLM Benchmarks

## Running LLM Benchmarks

```bash
rapid-mlx bench qwen3.5-4b-4bit --num-prompts 5 --max-tokens 256
```

## Results (M4 Max, 128GB)

| Model | Gen Speed | TTFT* | Memory |
|-------|-----------|-------|--------|
| Qwen3-0.6B-8bit | 402.3 tok/s | 58.6 ms | 0.68 GB |
| Llama-3.2-1B-Instruct-4bit | 463.6 tok/s | 49.2 ms | 0.69 GB |
| Qwen2.5-1.5B-Instruct-4bit | 308.5 tok/s | 86.2 ms | 0.84 GB |
| Llama-3.2-3B-Instruct-4bit | 200.1 tok/s | 81.4 ms | 1.79 GB |
| Qwen3-30B-A3B-4bit | 123.9 tok/s | 126.9 ms | 16.05 GB |
| NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-6Bit | 122.9 tok/s | 72.3 ms | 23.98 GB |

*TTFT = Time to First Token (latency until the model starts generating)

## Results (M1 Max, 64GB)

| Model | Runs | Prompt Tok | Gen Tok | Total Time (s) | TTFT Mean (ms) | TPOT Mean (ms) | Gen Speed (tok/s) | Total Throughput (tok/s) |
|-------|------|------------|---------|-----------------|-----------------|-----------------|-------------------|--------------------------|
| Qwen3-0.6B-8bit | 5 | 56 | 1280 | 5.66 | 119.0 | 3.97 | 251.9 | 236.1 |

## Continuous Batching Results

| Model | Single Request | Batch (5 req) | Speedup |
|-------|----------------|---------------|---------|
| Llama-3.2-1B-Instruct-4bit | 299.1 tok/s | 613.0 tok/s | **2.05x** |
| Llama-3.2-3B-Instruct-4bit | 137.6 tok/s | 208.1 tok/s | **1.51x** |
| Qwen3-0.6B-8bit | 328.1 tok/s | 1111.8 tok/s | **3.39x** |
| Qwen3-30B-A3B-4bit | 98.1 tok/s | 233.3 tok/s | **2.38x** |
| Qwen2.5-1.5B-Instruct-4bit | 196.9 tok/s | 322.2 tok/s | **1.64x** |

*Batching 5 concurrent requests shows 1.5-3x throughput improvement.*

### Continuous Batching (M1 Max, 64GB)

| Requests | Total Tokens | Total Time (s) | Throughput (tok/s) | Requests/sec |
|----------|--------------|-----------------|--------------------|--------------|
| 5 | 315 | 0.64 | 492.5 | 7.82 |

## Streaming Performance

| Model | TTFT | Generation Speed |
|-------|------|------------------|
| Llama-3.2-1B-Instruct-4bit | ~4.6ms | 218.9 tok/s |
| Llama-3.2-3B-Instruct-4bit | ~10.7ms | 93.6 tok/s |
| Qwen3-0.6B-8bit | ~3.0ms | 328.5 tok/s |
| Qwen3-30B-A3B-4bit | ~10.2ms | 98.4 tok/s |
| Qwen2.5-1.5B-Instruct-4bit | ~7.1ms | 140.3 tok/s |

### Streaming Detokenizer (M1 Max, 64GB)

`python3 scripts/bench_detok.py`:

| Tokens | Iterations | Naive Time | Streaming Time | Speedup |
|--------|------------|------------|----------------|---------|
| 742 | 5 | 1.69ms | 0.71ms | 2.39x |

`examples/benchmark_detokenizer.py`:

| Sequence | Tokens | decode() | Streaming | Speedup |
|----------|--------|----------|-----------|---------|
| Short | 8 | 0.029ms | 0.028ms | 1.04x |
| Medium | 103 | 0.206ms | 0.129ms | 1.59x |
| Long | 511 | 1.040ms | 0.502ms | 2.07x |
| 1K | 1191 | 2.446ms | 1.178ms | 2.08x |
| 2K | 2381 | 4.949ms | 2.356ms | 2.10x |
| 4K | 4761 | 9.887ms | 5.398ms | 1.83x |

Average speedup: 1.79x

## DFlash Speculative Decoding Results

### DFlash (M3 Ultra, 512GB) — Qwen3.6-35B-A3B-8bit

Community benchmark contributed by [@moc375](https://github.com/moc375) in [#472](https://github.com/raullenchai/Rapid-MLX/issues/472).
Model: `Qwen3.6-35B-A3B-8bit` (8-bit, group_size=64). Comparison vs the
no-spec-decode baseline on the same hardware.

| Output Tokens | Baseline (tok/s) | DFlash (tok/s) | Speedup | Accept Rate |
|---------------|-----------------:|---------------:|--------:|------------:|
| 256           | 74.72            | 194.80         | **2.61×** | 85.16%   |
| 512           | 73.76            | 214.78         | **2.91×** | 86.91%   |
| 1024          | 72.94            | 235.75         | **3.24×** | 88.87%   |
| 2048          | 72.66            | 234.52         | **3.23×** | 88.77%   |
| 4096          | 71.84            | 189.89         | **2.64×** | 84.69%   |

Notes:
- Accept rate stays in the **84–89%** band across all output lengths, peaking at 1024 tokens.
- Throughput peaks at **1024–2048** tokens, then dips at 4096 — consistent with KV-cache pressure as the rolling window fills.
- Memory delta (baseline → DFlash) tracks the draft model + verification overhead; see the source benchmark for the exact figure.

`Qwen3.6-35B-A3B-8bit` ships with DFlash enabled by default (`✨` in the alias table); these numbers reflect the out-of-the-box experience.

## Prefix Cache Results

### Prefix Cache (M4 Max, 128GB)

```
======================================================================
  LLM PREFIX CACHE TEST
======================================================================
  Model: mlx-community/Qwen3-0.6B-8bit
  Expected behavior:
    - Same prompt → cache HIT
    - Different prompt → cache MISS
----------------------------------------------------------------------
  Results:
  Step   | Description         | Expected | Actual | Status
  -------+---------------------+----------+--------+-------
  1a     | First request       | MISS     | MISS   | ✓
  1b     | Same prompt         | HIT      | HIT    | ✓
  1c     | Different prompt    | MISS     | MISS   | ✓
  1d     | Return to prompt 1  | HIT      | HIT    | ✓
======================================================================
```

### Prefix Cache (M1 Max, 64GB)

| Test | Expected | Actual | Time | Status |
|------|----------|--------|------|--------|
| First request | MISS | MISS | 203.5ms | PASS |
| Same prompt | HIT | HIT | 131.6ms | PASS |
| Different prompt | MISS or PREFIX_HIT | PREFIX_HIT (5 tok) | 135.3ms | PASS |

Final cache stats:

| Cache Hits | Cache Misses | Hit Rate | Tokens Saved | Cached Speedup |
|------------|--------------|----------|--------------|----------------|
| 2 | 1 | 66.7% | 20 | 1.55x |

## Paged Cache Results

*Test: 20 real inference requests in 2 rounds with ~286 token shared system prompt*

```
======================================================================
  PAGED KV CACHE - REAL INFERENCE TEST
======================================================================

--------------------------------------------------
Test 1: WITHOUT Paged Cache (2 rounds of 10)
--------------------------------------------------
  Time: 1.47s
  Throughput: 681.2 tok/s
  Cache hits: 0
  Tokens saved: 0

--------------------------------------------------
Test 2: WITH Paged Cache (2 rounds of 10)
--------------------------------------------------
  Time: 1.31s
  Throughput: 765.8 tok/s

  Paged Cache Stats:
    Blocks allocated: 25
    Shared blocks: 4
    Cache hits: 10
    Tokens saved: 2560

==================================================
SUMMARY
==================================================
  Without paged cache: 681.2 tok/s
  With paged cache:    765.8 tok/s

  Speedup: 1.12x
  Cache hits: 10 (all Round 2 requests)
  Tokens saved: 2,560 (~256 tokens × 10 requests)
==================================================
```

### Paged KV Cache (M1 Max, 64GB)

Inference benchmark (20 requests):

| Mode | Time (s) | Throughput (tok/s) |
|------|----------|--------------------|
| Without paged cache | 3.43 | 291.8 |
| With paged cache | 3.42 | 292.2 |

| Speedup | Blocks Allocated | Shared Blocks | Cache Hits | Tokens Saved |
|---------|------------------|---------------|------------|--------------|
| 1.00x | 45 | 4 | 10 | 2560 |

Real concurrent inference (20 requests):

| Mode | Time (s) | Throughput (tok/s) |
|------|----------|--------------------|
| Without paged cache | 4.32 | 231.7 |
| With paged cache | 4.35 | 229.7 |

| Speedup | Blocks Allocated | Shared Blocks | Cache Hits | Tokens Saved |
|---------|------------------|---------------|------------|--------------|
| 0.99x | 49 | 8 | 10 | 5120 |

Memory savings demo:

| Scenario | Memory Savings |
|----------|----------------|
| Shared system prompts | 70.8% |
| Concurrent memory efficiency | 83.5% |
| Prefix sharing branches | 38.5% |

## Streaming Detokenizer Analysis

*Phase 9.1 Investigation: mlx-lm's `BPEStreamingDetokenizer` vs naive `tokenizer.decode()`*

### Background

The naive approach calls `decode([token])` for each token. In theory, streaming detokenizers provide O(T) complexity vs O(T²) for naive decode.

### Isolated Benchmark Results

```bash
python3 scripts/bench_detok.py
```

When reusing the same detokenizer instance (with `reset()` between uses):

| Sequence | Tokens | Naive decode() | Streaming | Speedup |
|----------|--------|----------------|-----------|---------|
| Short | 8 | 0.020ms | 0.019ms | 1.05x |
| Medium | 103 | 0.155ms | 0.097ms | 1.59x |
| Long | 511 | 0.752ms | 0.371ms | **2.03x** |
| 1K tokens | 1191 | 1.743ms | 0.833ms | **2.09x** |
| 2K tokens | 2381 | 3.493ms | 1.737ms | **2.01x** |

### Critical Finding: Instance Creation Overhead

Creating a new `BPEStreamingDetokenizer` instance is **extremely expensive**:

```
100 tokenizer.detokenizer calls: 5.266s (52.7ms each!)
```

This means creating a new detokenizer per request adds **~52ms overhead**, negating any benefits.

### Real-World Impact

When integrated into the scheduler (one detokenizer per request):

| Metric | Naive decode() | Streaming (new instance) |
|--------|----------------|--------------------------|
| Throughput (20 req) | 681 tok/s | 275 tok/s |
| Impact | - | **-60% slower** |

### Conclusion

The streaming detokenizer is **not currently viable** for per-request usage due to instance creation cost. The naive `decode([token])` approach remains faster in practice.

**Future optimization**: Pre-create a pool of detokenizer instances at startup and reuse them across requests.

## Metrics Reference

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to First Token - latency until model starts responding (ms) |
| **TPOT** | Time Per Output Token - time between each generated token (ms/token) |
| **Generation TPS** | Output tokens per second (tok/s) |
| **Processing TPS** | Input/prompt tokens processed per second (tok/s) |
| **End-to-End Latency** | Total time from request to complete response |
| **Total Throughput** | Overall tokens (input + output) per second |

## Running Benchmarks

```bash
# Basic benchmark — short alias works
rapid-mlx bench qwen3.5-4b-4bit

# With more prompts
rapid-mlx bench qwen3.5-4b-4bit --num-prompts 10

# Save results
rapid-mlx bench qwen3.5-4b-4bit | tee results.txt

# Continuous batching test
python tests/test_continuous_batching.py

# Prefix cache test
python tests/test_prefix_cache.py

# Paged cache test
python tests/test_paged_cache_real_inference.py

# Streaming detokenizer benchmark (dev micro-bench)
python3 scripts/bench_detok.py
python3 scripts/bench_detok.py mlx-community/Llama-3.2-1B-Instruct-4bit --iterations 5
```
