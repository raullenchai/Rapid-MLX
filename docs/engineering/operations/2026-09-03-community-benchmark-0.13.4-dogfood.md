# Community Benchmark 0.13.4 dogfood

## Scope

Studio host, released `Rapid-MLX Desktop.app` 0.13.4 (build 170), Community
Benchmark only. The embedded 0.13.4 CLI was used instead of an older CLI on
`PATH`.

## Reproduction

```console
/Applications/Rapid-MLX\ Desktop.app/Contents/Resources/rapid-mlx/bin/rapid-mlx benchmark catalog --json
/Applications/Rapid-MLX\ Desktop.app/Contents/Resources/rapid-mlx/bin/rapid-mlx benchmark run qwen3.5-9b-4bit --json
/Applications/Rapid-MLX\ Desktop.app/Contents/Resources/rapid-mlx/bin/rapid-mlx benchmark run qwen3.8-27b-4bit --json
/Applications/Rapid-MLX\ Desktop.app/Contents/Resources/rapid-mlx/bin/rapid-mlx benchmark run flux2-klein-4b --json
/Applications/Rapid-MLX\ Desktop.app/Contents/Resources/rapid-mlx/bin/rapid-mlx benchmark run wan2.2-ti2v-5b-q8 --json
```

Upload preview, explicit consent, receipt permissions, and idempotent retry were
checked separately. The Desktop tab loaded CLI-created runs, cancelled a live
run without leaving a child server, and shared an image result only after the
confirmation sheet.

## Findings

- Text (9B and 27B) and 1024-square image runs completed and produced valid
  atomic records.
- Upload accepted the record, stored a mode-0600 receipt, and returned the same
  submission on retry.
- The registered Wan workload requested 832x480, while the route admitted only
  64-aligned dimensions and two 720p compatibility sizes. The request therefore
  failed before generation.
- After admitting Wan's 832x480 output target, Desktop generated the video but
  result validation failed because the deliberately minimal Desktop sidecar has
  no imageio/OpenCV. It does ship a constrained FFmpeg executable.

## Resolution and regression boundary

Wan explicitly admits 832x480 while retaining existing alignment rules for
other sizes. Video artifact validation falls back to the bundled FFmpeg stream
probe when imageio is absent. The fallback reads dimensions, frame count, and
frame rate without adding another media stack to the signed sidecar. Tests pin
the route exception, capability advertisement, server error detail, and the
imageio-free probe.

Website publication is intentionally handled in a separate website change: raw
atomic uploads contain private identity material and must not be exposed
directly.
