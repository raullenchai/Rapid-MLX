# LTX-2.5 stage-2 transformer hotspot profile

Date: 2026-09-13

Owner: Vector

Host: MZR-3, Mac mini `Mac16,11`, M4 Pro 12-core CPU / 16-core GPU,
48 GiB unified memory, macOS 26.5.1, MLX 0.32.0.

## Result

After enabling the previously qualified large-token dequantized matmul path,
one stage-2 transformer step takes roughly 43 seconds at 121 frames and 96
seconds at 241 frames. A real Q8 block at the 121-frame production shape has
6144 video tokens, 126 audio tokens, and 1024 connector tokens.

| Block component | Median | Share of component sum |
|---|---:|---:|
| Video feed-forward | 286.06 ms | 39.1% |
| Video self-attention | 257.33 ms | 35.2% |
| Video text cross-attention | 104.40 ms | 14.3% |
| Audio-to-video attention | 39.19 ms | 5.4% |
| Video-to-audio attention | 37.47 ms | 5.1% |
| All audio-local paths | 7.48 ms | 1.0% |

The component sum is 732 ms per block, or about 35.1 seconds over 48 blocks.
The remaining full-step time is block modulation, projections, streaming and
synchronization overhead. Video FFN and video self-attention are the only
kernel families large enough to materially move end-to-end latency.

## Exact candidates rejected

- FP16 activations were slower: video FFN increased from 286 to 340 ms,
  self-attention from 257 to 333 ms, and text cross-attention from 104 to
  133 ms. Peak allocation also increased.
- Combining the three self-attention Q/K/V projections changed 105.65 ms to
  107.05 ms and increased allocation, so MLX already schedules the separate
  projections efficiently.
- Query-chunking 6144-token Flash Attention changed 114.55 ms to
  114.93-114.99 ms for chunk sizes 512-3072 and increased allocation.
- Keeping dequantized FFN and QKV weights saved only 1.4% and 2.3% within
  those operations. Caching all video FFN weights would require roughly
  13 GiB, for less than 1% expected end-to-end benefit.
- Compacting Gemma's 1024-position left padding reduced Gemma evaluation from
  4.54 to 0.42 seconds for the 14-token benchmark prompt, but valid-token
  hidden states were not bit-identical and the connector intentionally
  replaces padding with learned register tokens. The maximum full-process
  upside is below the 3% gate, so this was not retained.
- Text cross-attention K/V cannot be cached across denoising steps: its input
  is modified by timestep-dependent AdaLN before projection.

No exact candidate beyond dequantized matmul met the 3% end-to-end gate.

## Approximate residual-reuse experiment

Reusing the first stage-2 transformer's block residual for the middle of three
steps reduced a full 121-frame run from the 255.11-second dequant baseline to
210.36 seconds (17.5%). The skipped block stack itself fell from roughly 43
seconds to less than one millisecond.

Quality was not acceptable against the same-seed dequant output:

| Candidate | Total | PSNR | SSIM | Audio APSNR |
|---|---:|---:|---:|---:|
| Raw previous residual | 210.36 s | 30.07 dB | 0.880923 | ~170 dB |
| Least-squares scaled residual | 213.46 s | 30.08 dB | 0.880918 | ~170 dB |

Adjacent video residuals had cosine similarity 0.9908, but even the optimal
scalar left 52.9% relative L1 error. Scaling therefore does not recover the
missing direction change. The experiment was removed from production code.

## Reproduction and durable artifact

Fixed workload: `MrMofer/ltx-2.5-mlx-q8` revision
`f1b56e7dc89f71a9af2cddac787b89ed22a8b7fc`, runtime parent `5795228`,
`LTX2_DEQUANT_MATMUL_MIN_TOKENS=1024`, distilled two-stage low-RAM,
768x512, 24 fps, seed 42.

The stacked runtime branch `vector/ltx25-stage2-hotspot` contains
`scripts/profile_transformer_block.py`, which loads a real streamed block and
measures all eight attention/FFN families at production token counts. No
approximate inference behavior is present in that branch.

## Roadmap

1. Treat the existing dequantized matmul optimization as the current exact
   fast path; it remains the only measured change above the acceptance gate.
2. For an exact next phase, prototype a native fused dequant-matmul-activation
   kernel for the two video FFN projections. This is higher effort but targets
   39% of block time without changing model semantics.
3. If a separate quality/speed tier is acceptable, calibrate a real distilled
   stage-2 residual predictor or TeaCache controller over multiple prompts and
   seeds. Simple previous-residual replay is not sufficient, though the 17.5%
   latency ceiling justifies a properly trained/calibrated investigation.
4. Do not spend further time on audio-local kernels, QKV concatenation,
   query chunking, or large dequantized-weight caches on this hardware/MLX
   version.
