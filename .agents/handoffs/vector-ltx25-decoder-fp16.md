# Vector handoff: LTX-2.5 FP16 video VAE decoder

Receiving role: Atlas

Date: 2026-09-13

Runtime branch: `vector/ltx25-decoder-hotspot` in the `ltx-2-mlx` repository,
based on upstream `ltx25` commit `57952288076766abe27dda3a774b2c24f7346977`.
PR: [MrMoferFRAN/ltx-2-mlx#2](https://github.com/MrMoferFRAN/ltx-2-mlx/pull/2).

## Verified facts

- On MZR-3, video VAE accounts for 24.61 of roughly 27.1 seconds in the
  121-frame decode tail. Audio VAE, vocoder, output conversion, and muxing are
  not the dominant path.
- `LTX2_VAE_DECODER_DTYPE=fp16` improves production-shape VAE decode by 13.1%
  at 121 frames and 14.3% at 241 frames.
- Three full 121-frame runs had a 266.22-second median versus the 266.13-second
  BF16 baseline; transformer variance hides the roughly 3.3-second VAE saving,
  so there is no supported end-to-end speedup claim.
- VAE peak MLX allocation falls 41.7% at 121 frames and 45.7% at 241 frames.
- The pipeline retains BF16 latent/output tensors; only VAE weights and its
  internal compute use FP16.
- Same-seed 121-frame output comparison: 44.53 dB video PSNR, 0.98554 video
  SSIM, and sample-identical audio.
- Focused pipeline/VAE tests: 49 passed, 2 skipped. The non-slow non-trainer
  suite passed 544 tests with 22 skips. The excluded trainer tests cannot run
  because the existing task environment cannot import the unrelated
  `ltx_trainer_mlx` package.
- No measured run grew swap or reported a thermal warning.

## Decision and risk

BF16 remains the default. The end-to-end latency gain is only about 1.2-1.3%
because transformer denoising dominates total time. The memory reduction is
material, so FP16 is useful as an explicit speed/capacity option. Do not make
it automatic until multi-prompt visual evaluation is complete; FP16 changes
decoded pixels and the single-prompt SSIM is 0.98554.

## Next action

Atlas should decide whether Rapid exposes the environment variable directly
or adds a user-facing decoder precision setting after the upstream runtime
change is accepted. Vector's next high-leverage performance task is profiling
stage-2 transformer attention and large-token linear dispatch, not further
audio/ffmpeg work.
