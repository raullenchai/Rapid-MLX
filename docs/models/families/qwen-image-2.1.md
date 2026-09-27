# Qwen-Image 2.1

`qwen-image-2.1` is the low-memory default. It uses Rapid-MLX's pinned mflux
checkpoint with both the denoiser and Qwen3-VL text encoder stored as native
MLX 4-bit weights. The download is 8.9 GiB, the catalog floor is 8 GB of
unified memory, and the runtime releases the text encoder before denoising and
the denoiser after each request. It defaults to 40 denoising steps and guidance
1.0.

```sh
rapid-mlx serve qwen-image-2.1
```

Users with more memory can select `qwen-image-2.1-bf16`. That alias retains the
previous pinned `Qwen/Qwen-Image-2.1` path: mflux quantizes its transformer to
8-bit at load while the Qwen3-VL text encoder remains bf16. Its canonical
download is 30.9 GiB and its catalog floor is 32 GB.

Both aliases appear in the Mac app and support `/v1/images/generations` and
`/v1/images/edits`. The edit endpoint uses mflux img2img conditioning with one
source image and the upstream default image strength 0.4. It is distinct from
the Qwen-Image instruction-edit variant. Rapid-MLX derives an approximately
1024-square canvas from the source aspect ratio. `negative_prompt` is accepted;
true CFG runs only when guidance exceeds 1.0.

The low-memory pack is pinned to commit
`746a58556820933a2df5c75887a2570f1ad200c0`. Rapid-MLX checks every component
and requires native q4 weight, scale, and bias tensors in the encoder before it
loads. The source checkpoint is pinned to
`790c92633540aa0cb11d9abf19eb46d861714758`. Community `MLX-Serve` packs use a
different format and remain unsupported.

Measured on an M3 Ultra, the prequantized pack peaked at 4.68 GiB for a
512-square, 40-step generation and 5.14 GiB for a 1024-square smoke run. These
are MLX allocator peaks rather than total macOS process memory. The 8 GB floor
must therefore be verified on physical 8 GB and 16 GB Macs before release. See
the [low-memory qualification](../../engineering/performance/2026-09-25-qwen-image-2.1-low-memory-spike.md)
and [image release matrix](../../engineering/operations/image-release-dogfood-matrix.md).

Upstream model and implementation:
<https://huggingface.co/Qwen/Qwen-Image-2.1> and
<https://github.com/mflux-community/mflux/pull/736>. Tracking issue:
<https://github.com/raullenchai/Rapid-MLX/issues/3642>.
