# Vector → Atlas: FLUX.2 Klein explicit bf16 path

- Branch: `perf/3058-image-weight-precision`
- PR: `#3065`
- Owner/host: Vector / Studio (M3 Ultra)
- Issue: `raullenchai/Rapid-MLX#3058`

## Verified facts

- The existing `flux2-klein-4b` alias remains on its pinned q4 checkpoint.
- The new `flux2-klein-4b-bf16` alias uses pinned revision
  `4d8e1bae8eb47c7766705de2cda7dabd6cc4ba67` of the 15.98 GB mflux-layout
  bf16 checkpoint. Its first transformer shard declares only BF16 tensors and
  no affine-quantization auxiliaries.
- `--image-weight-precision bf16` resolves to that concrete alias before the
  download confirmation and disk checks. No automatic hardware policy exists.
- The packaged bf16 source loads via `model_path` with `quantize=None`.
- Cold prefetch carries the registered image revision through the mirror,
  metadata, and Hub download layers, preventing a moving `main` snapshot from
  being downloaded before the pinned 16 GB snapshot.
- Relevant alias/image/download/CLI/residency tests pass; exact commands are in
  the eventual PR summary and the performance note.
- Real M2 Pro dogfood through `rapid-mlx serve` and the image endpoint measured
  52.703 s q4 versus 41.820 s bf16 median (1.26× throughput), with deterministic
  repeats and visually coherent output. The exact pinned bf16 package passed
  the repository completeness guard before loading.
- A DiffusionGemma-shape M3 Ultra microbenchmark found q4 faster than bf16 at
  both M=256 and M=4096. No Gemma precision policy was changed.
- `pr_validate` additionally exposed a pre-existing DiffusionGemma boot failure:
  the alias routes to DiffusionEngine while `diffusion_generation_family`
  returns `diffusion`. The exact failure reproduces on clean `origin/main` and
  remains a separate follow-up, not part of this image-weight PR.

## Unresolved questions and risks

- The CLI flag is a new public surface and needs Atlas compatibility approval.
- The 18 GiB residency estimate is conservative; direct-engine dogfood measured
  about 12.49 GB peak process RSS, and the alias independently enforces a 32 GB
  physical-memory floor.
- Text-to-image is qualified end to end. FLUX.2 image editing with the bf16
  package has not been separately timed in this pass.
- M1/M2-family automatic selection remains deliberately out of scope.

## Next action

Atlas should confirm the explicit CLI/alias surface and decide whether a
separate image-edit smoke is required before merge.
