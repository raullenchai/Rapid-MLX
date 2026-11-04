# Pika H3 Flex Ref2VA reference-video memory (2026-11-03)

## Result

A Ref2VA request with a video reference used to get the server killed for
memory on a 256 GB Mac (process memory reached about 197 to 203 GiB). With the
generation buffer cache capped at 1 GiB and the video VAE encoder materialized
per tile, the same Desktop request peaked at about 124 GiB of process
footprint and completed. The encoded reference latents are bit-identical to the
previous lazy path, so output quality is unchanged.

## Environment

- Apple M3 Ultra, 256 GB unified memory, macOS 26.5.2
- MLX 0.32.4, pinned H3 pipeline from mlx-vlm 0.7.2
- `pika-h3-flex` default checkpoint, Ref2VA partition
- Signed-layout Desktop build from this branch (bundled sidecar), driven through
  the Video tab: Reference-to-video, 512×512, 5 s, references = one PNG image,
  one 4 s 640×360 H.264 video, one 4 s AAC audio clip
- Run under the host large-model lock with a 160 GiB working-set claim

## Root cause

Measured stage by stage on the real checkpoint (stand-alone probes, same inputs):

| Stage | Transient above resident weights |
| --- | --- |
| Reference preparation (resize of 96 frames to 544×960) | ~12.5 GiB peak |
| Text conditioning | +1.3 GiB |
| Reference-video VAE encode, lazy (previous) | ~59 GiB (peak 69.0 GiB with only the 10 GiB VAE loaded) |
| Same, materialized per 17-frame clip | peak 57.5 GiB |
| Same, materialized per encoder tile (new) | peak 26.5 GiB |
| One transformer step at 29.5k tokens | +5 GiB |

Resident weights during Ref2VA are about 121 GiB (conditioner 48, reference
transformer 62, video VAE 10, audio VAE 0.6). The second contributor was MLX's
freed-buffer cache, which had no limit during H3 generation: after the reference
encode it held about 170 GiB of reusable buffers, which count toward process
memory. Lazy encode transients plus the unbounded cache on top of the weights
exceeded what the host could keep resident.

The lazy, per-clip and per-tile encodes produced identical latents
(`array_equal` true, max absolute difference 0.0) in the same wall time
(188 to 193 s for the encode alone).

## Fix

- `MiniMaxH3VideoEngine._generate` runs generation inside
  `bounded_cache_limit(1 GiB)`, the same helper the image engine uses.
- The Pika engine swaps the video VAE encoder instance to a subclass whose
  call evaluates its output before returning. The parameter tree is unchanged
  and no tiling, resolution, frame count, or dtype changes.

## End-to-end measurement

Process footprint sampled every 5 s with `footprint -p <server pid>`:

| Point | Footprint | Host available (free + inactive + speculative + purgeable) |
| --- | --- | --- |
| Server ready (lazy load) | 0.3 GiB | 207 GiB |
| Peak (late in denoising) | 124 GiB | 110 GiB |
| Minimum host availability during run | | 107.7 GiB |

Generation took about 21 minutes. The output was H.264 512×512 at 24 fps with
an AAC 32 kHz stereo track, 5.17 s long, and the frames matched the references.

Limitations: one run; the 5 s sampling interval can miss short spikes, and the
footprint tool reports whole GiB at this size. The probes ran the stages in
isolation, not the full pipeline.
