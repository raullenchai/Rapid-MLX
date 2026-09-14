# LTX-2.5 decoder hotspot and FP16 VAE experiment

Date: 2026-09-13

Owner: Vector

Host: MZR-3, Mac mini `Mac16,11`, M4 Pro 12-core CPU / 16-core GPU,
48 GiB unified memory, macOS 26.5.1, MLX 0.32.0.

## Result

The post-diffusion tail for the fixed 121-frame workload is almost entirely
video VAE work. Audio VAE, vocoder, WAV output, frame conversion, ffmpeg pipe
writes, and mux drain are too small to justify parallel audio/ANE or pipe
optimizations.

| Component | Time | Share of measured decode tail |
|---|---:|---:|
| Video VAE graph | 24.61 s | 90.8% |
| Audio VAE | 0.05 s | 0.2% |
| Vocoder + BWE | 1.36 s | 5.0% |
| WAV save | 0.04 s | 0.1% |
| Frame conversion | 0.14 s | 0.5% |
| ffmpeg pipe write + drain | 0.68 s | 2.5% |

Within the video VAE, up-blocks 4, 6, and 8 take 19.80 of 24.72 seconds
(80.1%). These are the highest-resolution residual stages.

Casting only VAE decoder weights to FP16 is a useful opt-in capacity mode.
The pipeline latent remains BF16 and the decoder restores its output to the
caller's BF16 dtype.

| Frames | BF16 median | FP16 median | Speedup | BF16 peak MLX | FP16 peak MLX | Peak reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 121 | 24.46 s | 21.27 s | 13.1% | 13.76 GiB | 8.01 GiB | 41.7% |
| 241 | 48.99 s | 42.00 s | 14.3% | 26.53 GiB | 14.40 GiB | 45.7% |

The 121-frame values use three decoder-only runs for the FP16 candidate and
two prior BF16 runs; the 241-frame values are medians of three runs per
configuration. All use the production latent shape for 768x512 output.

Three full 121-frame generations measured VAE graph times of 21.31, 21.30,
and 21.32 seconds, versus 24.61 seconds in the profiled BF16 baseline. Full
process times were 267.21, 266.22, and 264.19 seconds: a 266.22-second median
versus the 266.13-second BF16 median. The roughly 3.3-second decoder saving is
smaller than normal transformer-step variance, so this experiment does not
support an end-to-end speedup claim. Repeated full-process measurements are
reported separately from isolated decoder measurements for that reason.

## Quality and contract

Against the same-seed BF16 output, the full FP16-decoder output measured:

- Video PSNR: 44.53 dB
- Video SSIM: 0.98554
- Audio APSNR: infinite in both channels (sample-identical decoded audio)
- Output: 768x512, 121 frames, 24 fps, synchronized stereo audio

This is appropriate for an explicit speed/capacity option, but not sufficient
evidence to change the default precision. A multi-prompt visual evaluation is
required before default enablement.

## Rejected candidates

- Compiling the decoder was latency-neutral and increased peak MLX allocation
  by about 3 GiB.
- Folding explicit spatial zero padding into `nn.Conv3d` improved decoder-only
  latency by 1.8%, only about 0.16% end to end, below the 3% acceptance gate.
- FP16 only on the three hot residual stages improved decoder latency by 10.6%
  and reduced peak allocation by about 3.5 GiB, but quality metrics were no
  better than full FP16, so the extra mixed-precision complexity was rejected.
- Audio offload or concurrent muxing can recover at most roughly 1.5 seconds
  from this workload and does not address the dominant path.

## Reproduction

Model: `MrMofer/ltx-2.5-mlx-q8` revision
`f1b56e7dc89f71a9af2cddac787b89ed22a8b7fc`; LTX runtime parent
`57952288076766abe27dda3a774b2c24f7346977`; distilled two-stage low-RAM;
768x512, 24 fps, seed 42.

Enable the candidate with:

```sh
LTX2_VAE_DECODER_DTYPE=fp16 ltx-2-mlx generate ...
```

Use `LTX2_DECODE_PROFILE=1` for audio/video tail timing and
`LTX2_DECODE_STAGE_PROFILE=1` for per-stage VAE barriers. The standalone
production-shape runner is `scripts/benchmark_video_decoder.py` in the runtime
change branch.

Verification: 49 focused pipeline/VAE tests passed with 2 skips; the complete
non-slow non-trainer suite passed 544 tests with 22 skips. Ruff lint, targeted
format checks, and `git diff --check` passed. The trainer package is excluded
because the existing task environment cannot build/import it due to its
unrelated package readme/dependency setup; no trainer code changed.

## Decision boundary and next hotspot

Keep BF16 as default. FP16 VAE is worth exposing as an opt-in because its
41-46% decoder peak-memory reduction is material. Its theoretical end-to-end
latency effect is only about 1.2-1.3% at the measured lengths and was not
distinguishable in the three-run full-process matrix.

Further decoder kernel work should stop unless it can improve the VAE by at
least another 20%, because even that is only about 1% end to end. The next
high-leverage profile target is stage-2 transformer denoising (roughly 46
seconds per step at 121 frames and 101 seconds at 241 frames), especially its
large-token linear and attention dispatch.
