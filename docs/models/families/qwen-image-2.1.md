# Qwen-Image 2.1

`qwen-image-2.1` uses the pinned `Qwen/Qwen-Image-2.1` checkpoint through mflux 0.20.0. It is a separate runtime family from Qwen-Image 1.x. Rapid-MLX quantizes the transformer to 8-bit at load, retains the Qwen3-VL text encoder in bf16, and defaults to 40 denoising steps with guidance 1.0.

The alias supports `/v1/images/generations` and `/v1/images/edits`. The latter is mflux img2img conditioning with one source image and the upstream default image strength 0.4. It is not the separate Qwen-Image instruction-edit variant. Rapid-MLX derives an approximately 1024² canvas from the source aspect ratio when editing through the OpenAI-compatible endpoint. `negative_prompt` is accepted; true CFG runs only when guidance exceeds 1.0.

```sh
rapid-mlx serve qwen-image-2.1
```

The canonical bf16 download is 30.9 GiB and is pinned to commit `790c92633540aa0cb11d9abf19eb46d861714758`. The runtime checks checkpoint completeness and the Qwen3-VL text-encoder layout before loading. Community `MLX-Serve` 4-bit/8-bit packs have a different format and are rejected; a compatible prequantized pack must use the mflux layout.

The image lane registers mflux's `MemorySaver` and tiled VAE decoding for this family. After an encoder is evicted, a new prompt causes a clean model reload; a cached prompt can reuse the resident transformer. The 32 GB minimum follows the issue reporter's M2 Max CLI run and still needs exact-candidate Server and Mac app dogfood at both 512² and 1024² before release. See the [image release matrix](../../engineering/operations/image-release-dogfood-matrix.md) for the required checks.

Upstream model and implementation: <https://huggingface.co/Qwen/Qwen-Image-2.1>, <https://github.com/mflux-community/mflux/pull/736>. Tracking issue: <https://github.com/raullenchai/Rapid-MLX/issues/3642>.
