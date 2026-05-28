# Video Benchmarks

## Running Video Benchmarks

The shipped `rapid-mlx bench` subcommand benches text generation only. The
video benchmarks below were collected with the dev-only scripts under
`scripts/` (not packaged with `pip install rapid-mlx`) — clone the repo to
reproduce them. The numbers shown here are kept as a reference data point.

## Results - Qwen3-VL-8B-Instruct-4bit (M4 Max, 128GB)

| Configuration | Frames | Time | Tokens | Speed | Memory |
|---------------|--------|------|--------|-------|--------|
| 2 frames @ 0.5fps | 2 | 4.48s | 256 | 57.1 tok/s | 6.4 GB |
| 4 frames @ 1fps | 4 | 4.65s | 256 | 55.0 tok/s | 6.4 GB |
| 6 frames @ 1fps | 6 | 5.15s | 197 | 38.2 tok/s | 6.6 GB |
| 8 frames @ 2fps | 8 | 6.45s | 240 | 37.2 tok/s | 6.8 GB |
| 12 frames @ 2fps | 12 | 8.73s | 256 | 29.3 tok/s | 7.1 GB |
| 16 frames @ 2fps | 16 | 10.96s | 256 | 23.4 tok/s | 7.6 GB |
| 24 frames @ 4fps | 24 | 14.95s | 226 | 15.1 tok/s | 8.4 GB |
| 32 frames @ 4fps | 32 | 20.00s | 256 | 12.8 tok/s | 9.2 GB |
| 48 frames @ 8fps | 48 | 31.11s | 246 | 7.9 tok/s | 11.1 GB |
| 64 frames @ 8fps | 64 | 59.81s | 256 | 4.3 tok/s | 12.9 GB |

**Summary:** Fastest at 2 frames (57.1 tok/s), slowest at 64 frames (4.3 tok/s). Memory scales from 6.4 GB to 12.9 GB.

> **Note:** 96+ frames causes GPU timeout on most hardware due to memory/compute limits

## Results - Qwen3-VL-8B-Instruct-4bit (M1 Max, 64GB)

| Configuration | Frames | FPS | Time | Tokens | Speed |
|---------------|--------|-----|------|--------|-------|
| 4 frames @ 1fps | 4 | 1.0 | 8.84s | 256 | 29.0 tok/s |
| 8 frames @ 2fps | 8 | 2.0 | 13.05s | 256 | 19.6 tok/s |
| 16 frames @ 2fps | 16 | 2.0 | 21.60s | 256 | 11.9 tok/s |

| Configs | Total Time (s) | Total Tokens | Aggregate Tok/s |
|---------|-----------------|--------------|-----------------|
| 3 | 43.48 | 768 | 17.7 |

## Results - Qwen3-VL-4B-Instruct-3bit (M1 Max, 64GB)

| Configuration | Frames | FPS | Time | Tokens | Speed |
|---------------|--------|-----|------|--------|-------|
| 4 frames @ 1fps | 4 | 1.0 | 5.09s | 150 | 29.5 tok/s |
| 8 frames @ 2fps | 8 | 2.0 | 8.36s | 150 | 17.9 tok/s |
| 16 frames @ 2fps | 16 | 2.0 | 15.21s | 150 | 9.9 tok/s |

| Configs | Total Time (s) | Total Tokens | Aggregate Tok/s |
|---------|-----------------|--------------|-----------------|
| 3 | 28.66 | 450 | 15.7 |

## Video Cache Results

```
----------------------------------------------------------------------
  TEST 4: Video Cache - fps/max_frames in Cache Key
----------------------------------------------------------------------
    Config: fps=2.0, max_frames=16

    Results:
    Step   | Description               | Expected | Actual | Time   | Status
    -------+---------------------------+----------+--------+--------+-------
    4a     | Video first request       | MISS     | MISS   | 0.03ms | ✓
    4b     | Same video+params         | HIT      | HIT    | 0.14ms | ✓
    4c     | Different fps (4.0)       | MISS     | MISS   | 0.01ms | ✓
    4d     | Different max_frames (32) | MISS     | MISS   | 0.01ms | ✓
    4.0.5a | fps=0.5 first             | MISS     | MISS   | 0.01ms | ✓
    4.0.5b | fps=0.5 cached            | HIT      | HIT    | 0.14ms | ✓
    4.1.0a | fps=1.0 first             | MISS     | MISS   | 0.01ms | ✓
    4.1.0b | fps=1.0 cached            | HIT      | HIT    | 0.14ms | ✓
    4.2.0a | fps=2.0 first             | MISS     | MISS   | 0.01ms | ✓
    4.2.0b | fps=2.0 cached            | HIT      | HIT    | 0.14ms | ✓
    4.4.0a | fps=4.0 first             | MISS     | MISS   | 0.01ms | ✓
    4.4.0b | fps=4.0 cached            | HIT      | HIT    | 0.14ms | ✓

----------------------------------------------------------------------
  TEST 5: Additional Videos
----------------------------------------------------------------------
    Results:
    Step   | Description               | Expected | Actual | Time   | Status
    -------+---------------------------+----------+--------+--------+-------
    5a     | Video 1 first             | MISS     | MISS   | 0.01ms | ✓
    5b     | Video 2 first             | MISS     | MISS   | 0.01ms | ✓
    5c     | Video 1 cached            | HIT      | HIT    | 0.13ms | ✓
    5d     | Video 2 cached            | HIT      | HIT    | 0.13ms | ✓
```

## Cache Key Strategy

- **Videos**: `hash(video_path) + hash(fps) + hash(max_frames) + hash(prompt)`

Same video with same fps, max_frames, and prompt will hit cache. Changing any parameter causes a miss.

## Performance Tips

- Lower FPS = faster processing
- Fewer frames = less memory usage
- 64 frames is practical maximum
- 96+ frames causes GPU timeout

## Frame Extraction

| FPS | 10s Video | 30s Video | 60s Video |
|-----|-----------|-----------|-----------|
| 0.5 | 5 frames | 15 frames | 30 frames |
| 1.0 | 10 frames | 30 frames | 60 frames |
| 2.0 | 20 frames | 60 frames | 120 frames* |
| 4.0 | 40 frames | 120 frames* | 240 frames* |

*May hit `max_frames` limit

## Metrics Reference

| Metric | Description |
|--------|-------------|
| Configuration | FPS and max frames settings |
| Frames | Actual frames extracted |
| Time | Total generation time |
| Tokens | Output tokens generated |
| Speed | Tokens per second (tok/s) |
| Memory | GPU memory usage |
