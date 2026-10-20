<!-- Scratch space for the next release's notes. Rename to vX.Y.Z.md in the
     version-bump PR. See README.md in this directory. -->

<!-- One or two sentences: what is this release about? -->

## Highlights

- **Cached first reply**: `rapid-mlx serve`, `rapid-mlx chat`, and the bare command prefer complete local chat models before a download. Models in another MLX app's default model folder are discovered automatically and served in place; uncataloged models carry a not catalog-tested note. Fixes #4969.
- **FLUX.1 schnell and Qwen-Image downloads work again**: a first `rapid-mlx pull flux-schnell` or `rapid-mlx pull qwen-image` (and the same download in the Desktop app) failed with a revision-not-found error after the upstream repositories were re-created. Both aliases now pin the re-created builds. Installs that already had either model cached download the new build once on first use; the `qwen-image` transformer now matches the original Qwen-Image release. Fixes #4992.

**<Named thing>** — what changed, and why a user should care. Link the merged PR.
