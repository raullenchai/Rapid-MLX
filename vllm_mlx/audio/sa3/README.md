# Vendored: MLX Stable Audio 3 (SA3)

This directory is **third-party code**, vendored from
[`Stability-AI/stable-audio-3`](https://github.com/Stability-AI/stable-audio-3)
@ `124e8a799f57a1f665495ecb72e547d0a62867f1`, the `optimized/mlx` tree.

## License

The vendored code is **MIT** — Copyright (c) 2026 Stability AI. The verbatim
license text is in [`LICENSE`](./LICENSE) next to this file, and provenance is
recorded in [`NOTICE`](./NOTICE). The top-level MIT license of the upstream
repository covers `optimized/mlx` in full; there is no sub-license. Each kept
`.py` carries an `# SPDX-License-Identifier: MIT` header pointing back to the
upstream source.

The `stable-audio-tools` / `underfit` names in a few generate-path comments are
Stability's own ported components and a weight-provenance note (comments, not
foreign non-MIT code).

`vllm_mlx/audio/music.py` (`MusicEngine`) is *rapid-mlx* code under Apache-2.0;
it only drives `scripts/sa3_mlx.py` as a subprocess.

## Generate path only — dev toolchain removed

Only the text-to-music/SFX generation path is vendored. When vendoring, the
upstream developer toolchain was removed (not on `MusicEngine.generate`):

- `models/defs/`: `lora.py`, `lora_merge.py`, `latent_dataset.py`,
  `training.py`, `demo_mlx.py`
- `scripts/`: `sa3_gradio.py`, `lora_train_mlx.py`, `test_lora_merge.py`,
  `pre_encode_mlx.py`, `test_all_configs.py`, `benchmark.py`, `install.py`,
  `examples.py`, `parity_forward_mlx.py`, `parity_forward_torch.py`,
  `export_base_npz.py`, `spec.py`

References to the removed LoRA-merge module (and the `examples` help block) were
minimally severed from the kept generate-path files (`scripts/sa3_mlx.py`,
`models/defs/dit_mlx.py`, `models/defs/dit_mlx_medium.py`).

**Kept (generate path):** `scripts/sa3_mlx.py`, `scripts/weights.py`,
`models/defs/{dit_mlx.py, dit_mlx_medium.py, same_l_decoder.py,
same_s_decoder.py, same_l_encoder.py, same_s_encoder.py, t5gemma_mlx.py,
sa3_pipeline.py}`, and the `__init__.py` files.

## No torch dependency

Removing `parity_forward_torch.py` (its only top-level `import torch`) leaves
no install- or runtime-time torch dependency: importing `music.py` and running
generation never imports torch. Two **lazy** `import torch` statements remain
inside off-path raw-checkpoint conversion helpers
(`dit_mlx.py::convert_weights_from_torch_ckpt`,
`sa3_pipeline.py::load_conditioner_from_sa3_ckpt`) — these run only if someone
loads a raw upstream torch checkpoint, which the generate path never does (it
loads pre-converted MLX `.npz` weights fetched from HuggingFace).

## Weights are not in git

`models/mlx/*.npz` are **not** committed as tensors.
`MusicEngine._ensure_weights()` (and the runner's `weights.ensure_local()`)
fetch them from
[`stabilityai/stable-audio-3-optimized`](https://huggingface.co/stabilityai/stable-audio-3-optimized)
(Stability AI Community License; free commercial use under $1M revenue) on first
use **into the writable HuggingFace cache** (`~/.cache/huggingface`) and load
them straight from there. They are deliberately **not** copied/symlinked into
this vendored `models/mlx/` directory, which is read-only under a pip/brew
`site-packages` install (writing there would `PermissionError` on first
generation). A real file already vendored at `models/mlx/<name>` (a dev
checkout) is still used in place. Nothing under `models/mlx/` ships in the wheel
— the weight download is the user's own runtime retrieval, not redistribution by
rapid-mlx.

## Ruff policy

`pyproject.toml` adds `vllm_mlx/audio/sa3/**/*.py` to `[tool.ruff.format].exclude`
and to `[tool.ruff.lint.per-file-ignores]` so upstream syncs stay a clean diff —
the same policy as `vllm_mlx/models/{deepseek_v4.py,gemma4_vendored,hy_v3.py}`.
