# Image Benchmarks

## Running Image Benchmarks

The shipped `rapid-mlx bench` subcommand benches text generation only. The
multi-resolution image benchmarks below were collected with the dev-only
scripts under `scripts/` (not packaged with `pip install rapid-mlx`) — clone
the repo if you want to reproduce them.

```bash
rapid-mlx serve gemma-4-26b --mllm --port 8000   # then exercise the VLM via /v1/chat/completions
```

## Results - Qwen3-VL-8B-Instruct-4bit (M4 Max, 128GB)

| Resolution | Pixels | Time | Tokens | Speed |
|------------|--------|------|--------|-------|
| 224x224 | 50K | 1.04s | 78 | 74.8 tok/s |
| 336x336 | 113K | 0.94s | 64 | 68.3 tok/s |
| 448x448 | 201K | 1.45s | 70 | 48.1 tok/s |
| 512x512 | 262K | 1.58s | 99 | 62.8 tok/s |
| 672x672 | 452K | 1.83s | 83 | 45.3 tok/s |
| 768x768 | 590K | 2.05s | 91 | 44.3 tok/s |
| 896x896 | 803K | 2.61s | 90 | 34.5 tok/s |
| 1024x1024 | 1.0M | 2.79s | 76 | 27.2 tok/s |
| 1280x720 | 922K | 2.97s | 96 | 32.4 tok/s |
| 1920x1080 | 2.1M | 6.30s | 89 | 14.1 tok/s |

**Summary:** Average 45.2 tok/s across all resolutions. Fastest at 224x224 (74.8 tok/s), slowest at 1920x1080 (14.1 tok/s)

## Results - Qwen3-VL-8B-Instruct-4bit (M1 Max, 64GB)

Local MLLM benchmark:

| Resolution | Pixels | Time | Tokens | Speed |
|------------|--------|------|--------|-------|
| 224x224 | 50K | 1.84s | 78 | 42.5 tok/s |
| 448x448 | 201K | 2.28s | 70 | 30.7 tok/s |
| 768x768 | 590K | 4.39s | 91 | 20.7 tok/s |
| 1024x1024 | 1.0M | 6.41s | 76 | 11.9 tok/s |

| Configs | Total Time (s) | Total Tokens | Aggregate Tok/s |
|---------|-----------------|--------------|-----------------|
| 4 | 14.92 | 315 | 21.1 |

## Results - Qwen3-VL-4B-Instruct-3bit Server (M1 Max, 64GB)

| Resolution | Pixels | Time | Tokens | Speed |
|------------|--------|------|--------|-------|
| 224x224 | 50K | 1.65s | 113 | 68.4 tok/s |
| 448x448 | 201K | 2.09s | 120 | 57.5 tok/s |
| 768x768 | 590K | 2.93s | 106 | 36.2 tok/s |
| 1024x1024 | 1.0M | 4.12s | 100 | 24.3 tok/s |

| Configs | Total Time (s) | Total Tokens | Aggregate Tok/s |
|---------|-----------------|--------------|-----------------|
| 4 | 10.79 | 439 | 40.7 |

## MLLM Prefix Cache Results

```
======================================================================
  MLLM PREFIX CACHE TEST
======================================================================
  Model: mlx-community/Qwen3-VL-4B-Instruct-3bit
  Test: Verify KV cache reuse for repeated image/video + prompt combinations
  Expected behavior:
    - Same image + same prompt → cache HIT
    - Same image + different prompt → cache MISS
    - Different image + same prompt → cache MISS
----------------------------------------------------------------------
  SETUP: Loading Model
----------------------------------------------------------------------
    Model loaded in 0.11s

----------------------------------------------------------------------
  SETUP: Creating Test Images
----------------------------------------------------------------------
    Resized: 224x224, 336x336, 512x512, 768x768

----------------------------------------------------------------------
  TEST 1: Image Cache - Basic Hit/Miss
----------------------------------------------------------------------
    Results:
    Step   | Description               | Expected | Actual | Time   | Status
    -------+---------------------------+----------+--------+--------+-------
    1a     | First image+prompt        | MISS     | MISS   | 0.10ms | ✓
    1b     | Same image+prompt         | HIT      | HIT    | 0.18ms | ✓
    1c     | Different prompt          | MISS     | MISS   | 0.01ms | ✓
    1d     | Return to original        | HIT      | HIT    | 0.18ms | ✓

----------------------------------------------------------------------
  TEST 2: Different Images
----------------------------------------------------------------------
    Results:
    Step   | Description               | Expected | Actual | Time   | Status
    -------+---------------------------+----------+--------+--------+-------
    2a     | Image A first request     | MISS     | MISS   | 0.01ms | ✓
    2b     | Image B first request     | MISS     | MISS   | 0.01ms | ✓
    2c     | Image A cached            | HIT      | HIT    | 0.13ms | ✓

----------------------------------------------------------------------
  TEST 3: Image Resolutions
----------------------------------------------------------------------
    Results:
    Step   | Description           | Expected | Actual | Time   | Status
    -------+-----------------------+----------+--------+--------+-------
    3.1a | 224x224 first         | MISS     | MISS   | 0.01ms | ✓
    3.1b | 224x224 cached        | HIT      | HIT    | 0.20ms | ✓
    3.2a | 336x336 first         | MISS     | MISS   | 0.01ms | ✓
    3.2b | 336x336 cached        | HIT      | HIT    | 0.21ms | ✓
    3.3a | 512x512 first         | MISS     | MISS   | 0.12ms | ✓
    3.3b | 512x512 cached        | HIT      | HIT    | 0.20ms | ✓
    3.4a | 768x768 first         | MISS     | MISS   | 0.12ms | ✓
    3.4b | 768x768 cached        | HIT      | HIT    | 0.24ms | ✓
======================================================================
```

## Cache Key Strategy

- **Images**: `hash(image_content) + hash(prompt)`

Same image with same prompt will always hit cache. Different image or different prompt will miss.

## Performance Tips

- Smaller resolutions process faster (224x224 vs 1920x1080)
- Use appropriate resolution for your task
- Batch similar-sized images for consistent throughput

## Metrics Reference

| Metric | Description |
|--------|-------------|
| Resolution | Image dimensions (width x height) |
| Pixels | Total pixel count |
| Time | Generation time |
| Tokens | Output tokens generated |
| Speed | Tokens per second (tok/s) |
