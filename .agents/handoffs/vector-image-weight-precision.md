# Vector → Atlas: FLUX.2 Klein explicit bf16 path

- Branch: `perf/3058-image-weight-precision`
- PR: not opened yet
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
- Relevant alias/image/download/CLI/residency tests pass; exact commands are in
  the eventual PR summary and the performance note.
- A DiffusionGemma-shape M3 Ultra microbenchmark found q4 faster than bf16 at
  both M=256 and M=4096. No Gemma precision policy was changed.

## Unresolved questions and risks

- The CLI flag is a new public surface and needs Atlas compatibility approval.
- The 18 GiB residency estimate is conservative rather than a measured peak;
  the alias independently enforces a 32 GB physical-memory floor.
- A full q4/bf16 image generation and edit smoke against the new packaged
  checkpoint still needs the 16 GB weight download. This change verifies the
  load contract hermetically but does not download that checkpoint on Studio.
- M1/M2-family automatic selection remains deliberately out of scope.

## Next action

Atlas should confirm the explicit CLI/alias surface, then run or delegate one
real 1024×1024 generation and edit smoke on the 32 GB M2 Pro before merge.
