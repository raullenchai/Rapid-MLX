# Qwen-Image 2.1

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start the catalog image-generation alias:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve qwen-image-2.1
```

The server prints its local address and may choose a free port if 8000 is busy.
Send a prompt to that address's `/v1/images/generations` route or select the
model in the Mac app. Run `rapid-mlx info qwen-image-2.1` to inspect the current
checkpoint and download size before starting it; `rapid-mlx recipe` helps choose
a model for your Mac.

## Other catalog choices

- `qwen-image-2.1-bf16` uses the larger checkpoint.
- `qwen-image-2.1-turbo` is another image-generation alias.
- `qwen-image-edit` is the instruction-edit variant.

## Reference images

`qwen-image` and `qwen-image-2.1` accept **one reference image** for editing.
Their low-memory/full-Q4 pack lacks the component configs required by the
multi-reference editing backend. Sending two or more images to
`/v1/images/edits` returns HTTP 400 before model loading. Single-image editing
keeps its existing img2img behavior.

`qwen-image-2.1-bf16` accepts up to **10 reference images** with a runtime that
provides `QwenImage21Edit`; older runtimes retain a limit of one. Turbo remains
generation-only. `/v1/models` and `rapid-mlx models --json` publish the effective
`max_reference_images`; Desktop reads the CLI catalog value to cap attachments.
The checkpoint cap is explicit alias metadata: local cached files do not
override it. Unreviewed Qwen 2.1 checkpoints default to one reference.

The Hugging Face file listings at the pinned revisions were checked using
`HfApi.model_info(repo, revision=pin).siblings` (metadata only; no weights):

| Aliases | Pinned checkpoint | `vae/config.json`, `transformer/config.json`, `text_encoder/config.json` | Checkpoint reference cap |
| --- | --- | --- | --- |
| `qwen-image`, `qwen-image-2.1` | [Q4 @ 746a585](https://huggingface.co/mlx-community/Qwen-Image-2.1-mflux-q4/tree/746a58556820933a2df5c75887a2570f1ad200c0) | All absent | 1 |
| `qwen-image-2.1-bf16` | [BF16 @ 790c926](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/790c92633540aa0cb11d9abf19eb46d861714758) | All present | 10 |
| `qwen-image-2.1-turbo` | [Turbo @ fc13ba1](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/fc13ba1986fcb905854dc8de2f52eee395f49b5b) | All present | 0 (generation-only) |

Run `rapid-mlx models --modality image-gen` for the current catalog list. The [server guide](../../guides/server.md) lists the image API routes, and `rapid-mlx serve --help` lists current runtime options.
