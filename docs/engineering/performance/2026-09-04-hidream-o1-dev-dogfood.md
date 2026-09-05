# HiDream-O1 Dev server dogfood

## Result

The `hidream-o1-dev` alias completed a real OpenAI-compatible image-generation
request at 1024×1024. The API returned HTTP 200 in 38.73 seconds and produced a
valid 1,015,295-byte PNG. Visual inspection found a coherent subject, lighting,
glass material, and composition with no regular grid artifact.

The complete server process peaked at 18,716,459,008 bytes (17.43 GiB) RSS.
Rapid therefore budgets 18.0 GiB resident memory and keeps the Desktop launch
warning at 32 GiB total system memory.

A final warm-cache regression run after the pinned-download and prompt-bound
hardening returned HTTP 200 in 29.46 seconds. It produced a valid 1,063,526-byte
1024×1024 RGB PNG; visual inspection again found a coherent composition with no
grid artifact.

## Environment

- Mac Studio (Mac15,14), Apple M3 Ultra, 28 CPU cores, 256 GB unified memory
- macOS 26.5.2 (25F84)
- Rapid-MLX source branch `raullenchai/vector-hidream-o1-dev`
- Model `mlx-community/HiDream-O1-Image-Dev-mlx-bf16`
- Pinned revision `33c7a00bce8e3410304f83ec408a15a1eb6782df`

## Reproduction

Start the server and capture whole-process resource usage:

```bash
RAPID_MLX_AUTO_PULL=1 /usr/bin/time -l \
  uv run rapid-mlx serve hidream-o1-dev --host 127.0.0.1 --port 8019
```

In another terminal, submit the measured request:

```bash
curl --fail-with-body --max-time 900 -sS \
  -o /tmp/hidream-response.json \
  -w 'http=%{http_code} total=%{time_total}\n' \
  -H 'Content-Type: application/json' \
  -d '{"model":"hidream-o1-dev","prompt":"A cinematic product photograph of a translucent glass cheetah sculpture on a black pedestal, warm rim light, crisp reflections, dark studio background","size":"1024x1024","n":1,"steps":28,"seed":42}' \
  http://127.0.0.1:8019/v1/images/generations
```

The request timing includes the first lazy model load. The server-process wall
time (253.07 seconds) additionally includes the initial 17.6 GB snapshot pull
and time spent waiting before shutdown, so it is not an inference latency.
