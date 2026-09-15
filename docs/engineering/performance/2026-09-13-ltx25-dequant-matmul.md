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

## Studio qualification

The target Studio is a Mac Studio `Mac15,14`, M3 Ultra (28-core CPU), 256 GiB
unified memory, macOS 26.5.2. A production-shape synthetic
`BasicAVTransformerBlock` benchmark exercised all 34 Q8 linears with BF16
activations and included dequantization on every call. It used the runtime's
real block implementation and dimensions, 1,024 text tokens, two warmups and
five measured iterations per mode. The measurement order was baseline,
experimental, baseline; the final two medians were compared to avoid shader
compilation bias. Random weights and inputs used seed 42.

| MLX | Workload shape (video/audio tokens) | Baseline block | Experimental block | Improvement |
|---:|---:|---:|---:|---:|
| 0.32.0 | 6,144 / 126 (121 frames) | 292.81 ms | 279.04 ms | 4.70% |
| 0.32.0 | 11,904 / 251 (241 frames) | 652.45 ms | 626.81 ms | 3.93% |
| 0.32.2 | 6,144 / 126 (121 frames) | 292.41 ms | 278.82 ms | 4.65% |
| 0.32.2 | 11,904 / 251 (241 frames) | 651.66 ms | 622.95 ms | 4.41% |

The same whole-block method also found regressions at smaller captured shapes
on MLX 0.32.0: 28.28 to 28.77 ms at 468/101 tokens (-1.72%) and 69.98 to
71.12 ms at 1,536/26 tokens (-1.63%). The 3,072/59-token shape improved from
136.89 to 132.97 ms (+2.87%). These results show that the 1,024-token global
threshold is not a safe universal default even though the target 5- and
10-second video shapes benefit.

This is a block-level qualification, not an end-to-end Studio claim. The
67.7 GB LTX-2.5 model snapshot was not present on Studio and the sole approved
Hugging Face cache had only 12 GiB free. The storage policy forbids a second
cache or deleting unrelated models, so a full generation could not be run on
this host. The MZR-3 table above remains the end-to-end evidence.

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
- The dispatch-specific tests pass 5/5 on both MLX 0.32.0 and 0.32.2 and now
  explicitly construct `QuantizedLinear`; this prevents a bare-`Linear`
  false positive in the test helper.
- Inference/pipeline non-slow suite: 546 passed, 22 skipped.
- Ruff lint and format checks: passed.
- The repository's trainer test subset was excluded because the existing
  checkout cannot build/import `ltx_trainer_mlx` due to its unrelated package
  readme/dependency setup; no trainer code changed.

## Decision boundary

Rapid pins the exact experimental runtime commit so the release contains the
capability, but keeps it opt-in. It is not enabled by default because the
1,024-token switch applies to every quantized linear, while crossover varies
by projection shape, Apple GPU and MLX version; Studio measured 1.6-1.7%
whole-block regressions at smaller shapes. In addition, Studio lacks an
end-to-end run and the output-changing arithmetic path has not passed a
multi-prompt blind visual/audio qualification.

A future default should use a more conservative, shape-aware boundary (for
example, initially bypassing token counts below 4,096) and must repeat MZR-3
end-to-end and quality tests before shipment. That policy is not part of this
release. The temporary pin uses the maintainer-controlled
`raullenchai/ltx-2-mlx` fork because the upstream PR is still open; switch the
repository back after upstream merge. Atlas owns any later default change.
