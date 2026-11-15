# MiniMax-H3-Image measurements (2026-11-14)

Environment: Mac Studio M3 Ultra, 256 GB unified memory, macOS 26.5.2,
Python 3.12.13, MLX 0.32.4, mlx-vlm 0.7.2, Rapid-MLX branch
`feat/minimax-h3-image` (runtime commit `f357424a3`). Weights:
`MiniMaxAI/MiniMax-H3-Image@5f90473023975e6271b0ed1dd84c4e88e4bbc6d3`
(`transformer_diffusers/`, `vae_diffusers/`) and the text encoder from
`MiniMaxAI/MiniMax-H3@b3c7290e66afdf293bef3b9077b7a266ef421f34`.
Concurrency 1, no competing GPU workload, one repetition per cell.

Request: 1024x1024 output (rendered on the default 2048x2048 canvas, then
resized), 6 steps, guidance 1.0, seed 42, the upstream example t2i prompt.
Edit uses one 1024x1024 reference and a "replace the wooden table with white
marble" instruction.

| quant | load s (warm cache) | t2i s/img | edit s/img (1 ref) | MLX peak GiB load / t2i / edit | process peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4-bit | 18–20 | 77.1 | 114.2 | 35.2 / 36.2 / 39.7 | 47.0 |
| 8-bit | 20.3 | 76.0 | 114.2 | 61.9 / 63.0 / 66.5 | 75.7 |

16 MP (4096x4096 canvas, 4-bit): t2i 385.6 s (about 62 s/step), edit 442.1 s (1 ref); MLX peak 38.0 / 39.7 GiB, process peak 47.1 GB, unchanged from 1 MP because the VAE decode is tiled. The 16 MP edit keeps the subjects but follows the reference layout less tightly than at 1 MP.

Derived catalog values: `min_memory_gb` 64 (4-bit) and 96 (8-bit); residency
charge 44/71 GiB; download 126.1 GiB (primary 68.6 GB + text encoder 66.7 GB).

Not measured: cold-cache load, multi-reference edit memory, run-to-run
variance.

Numerical checks (real weights, MLX FP32 vs a float64 reference
transliteration): VAE encoder max abs 7.0e-5, VAE decoder 2.3e-5, tiled decode
blend 2.0e-5, DiT (2 blocks + output head) 2.6e-4; relative L2 at most 1.7e-6.
Quantized DiT drift vs FP32 (relative L2): BF16 0.55%, 8-bit 0.16%, 4-bit 2.8%.
The full-depth DiT was not compared.

Reproduce (process peak via `/usr/bin/time -l`):

```sh
RAPID_MLX_TELEMETRY=0 DO_NOT_TRACK=1 /usr/bin/time -l python -I -c '
import time, mlx.core as mx
from huggingface_hub import snapshot_download
from rapid_mlx.image.minimax_h3_image import MiniMaxH3Image
m = MiniMaxH3Image(
    snapshot_download("MiniMaxAI/MiniMax-H3-Image", revision="5f90473023975e6271b0ed1dd84c4e88e4bbc6d3"),
    snapshot_download("MiniMaxAI/MiniMax-H3", revision="b3c7290e66afdf293bef3b9077b7a266ef421f34"),
    quantize=4)
t = time.time(); m.load(); print("load", time.time() - t, mx.get_peak_memory() / 2**30)
t = time.time(); img = m.generate_image("PROMPT", width=1024, height=1024, seed=42)
print("t2i", time.time() - t, mx.get_peak_memory() / 2**30)
img.save("t2i.png")
'
```
