<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

- **Cached first reply**: `rapid-mlx serve`, `rapid-mlx chat`, and the bare command prefer complete local chat models before a download. Models in another MLX app's default model folder are discovered automatically and served in place; uncataloged models carry a not catalog-tested note. Fixes #4969.
- **Qwen-Image-2.1-Turbo**: new `qwen-image-2.1-turbo` image alias for the official distilled Qwen-Image 2.1 checkpoint. It samples the checkpoint's published eight-step schedule without classifier-free guidance, so text-rich images render in a fraction of the 40-step `qwen-image-2.1-bf16` time. Text-to-image only; needs a 32 GB Mac.
- **FLUX.1 schnell and Qwen-Image downloads work again**: a first `rapid-mlx pull flux-schnell` or `rapid-mlx pull qwen-image` (and the same download in the Desktop app) failed with a revision-not-found error after the upstream repositories were re-created. `flux-schnell` now pins the re-created build (installs that already had it cached download it once on first use). `qwen-image` now serves the latest Qwen Image model, Qwen-Image-2.1: it resolves to the same pinned checkpoint as `qwen-image-2.1` (8.9 GiB, 8 GB minimum memory, generate + edit, 40 default steps), so an install that already has `qwen-image-2.1` cached downloads nothing, and an earlier `qwen-image` download is no longer used. Fixes #4992.

**<Named thing>** — what changed, and why a user should care. Link the merged PR.
