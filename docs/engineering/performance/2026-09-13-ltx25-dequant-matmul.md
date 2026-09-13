# LTX-2.5 large-token dequantized matmul experiment

Date: 2026-09-13

Owner: Vector

Host: MZR-3, Mac mini `Mac16,11`, M4 Pro 12-core CPU / 16-core GPU,
48 GiB unified memory, macOS 26.5.1, MLX 0.32.0.

## Result

For large quantized transformer linears, explicitly dequantizing the affine
Q8 weight and using BF16 matmul was faster than MLX `quantized_matmul` on this
host. The implementation is opt-in through
`LTX2_DEQUANT_MATMUL_MIN_TOKENS=1024`; the upstream runtime default is
unchanged because the crossover depends on the chip and MLX version.

Runtime change: [MrMoferFRAN/ltx-2-mlx#1](https://github.com/MrMoferFRAN/ltx-2-mlx/pull/1)

Fixed workload: `MrMofer/ltx-2.5-mlx-q8` revision
`f1b56e7dc89f71a9af2cddac787b89ed22a8b7fc`, pinned LTX runtime parent
`57952288076766abe27dda3a774b2c24f7346977`, distilled two-stage, low-RAM,
768x512, 24 fps, seed 42. Each result is the median of three fresh processes
with 120 seconds of cooling between runs.

| Frames | Metric | Baseline | Experimental | Improvement |
|---:|---|---:|---:|---:|
| 121 | End to end | 266.13 s | 255.11 s | 4.14% |
| 121 | Stage 1 step | 11.16 s | 10.73 s | 3.85% |
| 121 | Stage 2 step | 46.16 s | 42.97 s | 6.91% |
| 241 | End to end | 539.37 s | 515.18 s | 4.49% |
| 241 | Stage 1 step | 21.51 s | 20.45 s | 4.93% |
| 241 | Stage 2 step | 100.61 s | 95.12 s | 5.46% |

All six experimental runs completed without swap growth. Peak MLX allocator
memory was unchanged from baseline: 15.11 GiB for 121 frames and 27.07 GiB
for 241 frames. Peak process-tree RSS was 12.92 GiB in both experimental
matrices.

Output contracts were unchanged: 768x512 H.264, exact requested frame count,
24 fps, and 48 kHz stereo AAC. For 121 frames, experimental versus baseline
video SSIM was 0.973 and audio APSNR was about 168 dB. At 241 frames video
SSIM was 0.859 while a five-timepoint visual comparison preserved subject,
motion, composition, and detail; audio APSNR was about 171 dB. Diffusion is
sensitive to floating-point accumulation order, so a multi-prompt blind test
is still required before default enablement.

## Crossover evidence

Exact-shape microbenchmarks included dequantization on every measured call.
At 1024 and 1536 tokens, dequantized BF16 matmul was 6-9% faster for both
4096-wide video and 2048-wide audio projections. Several 128-512-token audio
shapes were slower, which is why the implementation uses a configurable
minimum-token threshold instead of replacing every quantized linear.

Q4 and Q8 `quantized_matmul` were effectively equal (less than 1% apart) for
the large LTX projection and FFN shapes on this software/hardware pair. Q4 is
therefore a capacity optimization here, not a speed claim. Recomputing the
largest 241-frame video RoPE grid took about 4.9 ms versus a 100.6-second
stage-2 step, so RoPE caching was rejected. Fully resident Q8 weights also
failed to improve the real workload: 269.4 seconds for 121 frames versus the
266.13-second low-RAM baseline.

## Verification

- Focused linear, transformer-shape, and block-streaming tests: 38 passed.
- Inference/pipeline non-slow suite: 546 passed, 22 skipped.
- Ruff lint and format checks: passed.
- The repository's trainer test subset was excluded because the existing
  checkout cannot build/import `ltx_trainer_mlx` due to its unrelated package
  readme/dependency setup; no trainer code changed.

## Decision boundary

Keep the feature opt-in until it is benchmarked on at least the target M3
Ultra configuration and one additional MLX version. After the upstream PR is
merged, Rapid may update `LTX25_RUNTIME_COMMIT`. Atlas owns the decision to
enable the environment variable automatically for qualified hardware.
