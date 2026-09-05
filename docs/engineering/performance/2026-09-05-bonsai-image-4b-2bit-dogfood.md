# Bonsai Image 4B 2-bit dogfood

Date: 2026-09-05

Owner/host: Vector, Local Mac

Model: `prism-ml/bonsai-image-ternary-4B-mlx-2bit`

Revision: `2c24c81b934a658ba5590cf39088ba929985b4a8`

## Result

Rapid-MLX generated coherent PNGs through both the direct engine and the real
OpenAI-compatible Server route. The fixed selective download contained exactly
25 reviewed data files totaling 3,888,262,196 bytes (3.62 GiB). No upstream
Python files were downloaded or executed.

| Path | Resolution | Steps | End-to-end | MLX peak | Process peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct, cold process | 512×512 | 4 | 5.392 s | 5.637 GiB | 6.475 GiB |
| Direct, cold process | 1024×1024 | 4 | 13.949 s | 5.758 GiB | 6.522 GiB |
| Server, warm model | 512×512 | 4 | 3.922 s HTTP | — | — |

The 1024² process peak supports a conservative 7.0 GiB resident admission
charge. The published alias remains at a 12 GB minimum-memory tier to retain OS,
Desktop, and request headroom on entry Apple Silicon machines.

A same-process 256² lifecycle probe also passed three consecutive generations:
first prompt 3.467 s, same-prompt cache hit 1.325 s, and different-prompt
text-encoder reload 1.627 s. Active MLX memory after each generation was
approximately 1.408 GiB, confirming that the one-entry embedding cache and
text-encoder eviction work across both hit and miss paths.

## Reproduction

Environment: Apple Silicon macOS, Python 3.12.14, MLX 0.32.2, mlx-lm 0.31.3,
mflux 0.19.1.

The Desktop sidecar's exact `mflux==0.19.0` pin was then installed in place of
0.19.1; a fresh 256² four-step generation also completed successfully (3.892 s,
110,243-byte PNG).

```bash
RAPID_MLX_AUTO_PULL=1 .venv/bin/rapid-mlx pull bonsai-image-4b-2bit

/usr/bin/time -l .venv/bin/python - <<'PY'
import time
import mlx.core as mx
from vllm_mlx.image.engine import ImageGenerationEngine

mx.reset_peak_memory()
engine = ImageGenerationEngine(
    "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
)
started = time.perf_counter()
png = engine.generate(
    prompt="A red fox curled beside an alpine lake at sunrise",
    width=1024,
    height=1024,
    num_inference_steps=4,
    seed=73,
)
print(len(png), time.perf_counter() - started, mx.get_peak_memory())
PY
```

Server route:

```bash
.venv/bin/rapid-mlx serve bonsai-image-4b-2bit --port 18090
curl http://127.0.0.1:18090/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-image-4b-2bit","prompt":"A glass greenhouse in rain","size":"512x512","n":1,"response_format":"b64_json","seed":101}'
```
