# Vendored: MLX Stable Audio 3 (SA3)

This directory is **third-party code**, vendored verbatim. It is **not**
covered by the Apache-2.0 licence in the repository root `LICENSE`, and no
`SPDX-License-Identifier` header may be added to files under this directory.

`vllm_mlx/audio/music.py` (`MusicEngine`) is *our* code and is Apache-2.0; it
only drives `scripts/sa3_mlx.py` as a subprocess.

## Provenance

| Component | Upstream | Licence |
|---|---|---|
| `models/defs/*`, `scripts/sa3_mlx.py`, `scripts/weights.py`, `scripts/spec.py`, `scripts/sa3_gradio.py`, `scripts/benchmark.py`, `scripts/examples.py`, `scripts/export_base_npz.py`, `scripts/install.py`, `scripts/parity_forward_*.py`, `scripts/test_all_configs.py` | Stability AI — Stable Audio 3 MLX reference implementation (weights: [`stabilityai/stable-audio-3-optimized`](https://huggingface.co/stabilityai/stable-audio-3-optimized)) | **Stability AI Community License** |
| `models/defs/sa3_pipeline.py` (four components ported from `stable-audio-tools`) | Stability AI — [`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools) | Stability AI licence (see upstream) |
| `models/defs/lora.py`, `models/defs/lora_merge.py`, `scripts/lora_train_mlx.py`, `scripts/pre_encode_mlx.py`, `scripts/test_lora_merge.py`, `models/defs/latent_dataset.py`, `models/defs/training.py` | Derived from / interoperating with [`dada-bots/underfit`](https://github.com/dada-bots/underfit) (see `lora_merge.py:20`, `lora_train_mlx.py:4`) | **UNRESOLVED — see below** |

> **TODO (blocking redistribution):** the exact upstream revision each file was
> copied from is **not recorded**, and the upstream licence texts are **not
> included**. Before this tree ships in a published wheel:
>
> 1. Pin the upstream commit(s) here (the repo convention — see
>    `vllm_mlx/models/hy_v3.py` "Vendored from:").
> 2. Add the verbatim upstream licence text(s) to this directory.
> 3. Confirm the Stability AI Community License permits redistribution of the
>    source inside `rapid-mlx` (revenue threshold + attribution/notice terms),
>    and resolve the `dada-bots/underfit` licence.
>
> `[tool.setuptools.packages.find]` in `pyproject.toml` matches `vllm_mlx*`, so
> **every `.py` file here is included in the `rapid-mlx` wheel** (verified: 30
> entries). This is redistribution, not merely local vendoring.

## Weights are not in git

`models/mlx/*.npz` are **not** committed as tensors. `MusicEngine._ensure_weights()`
fetches them from the HuggingFace repo above on first use and links them into
`models/mlx/`. Nothing under `models/mlx/` is shipped in the wheel.

## Ruff policy

`pyproject.toml` adds `vllm_mlx/audio/sa3/**/*.py` to `[tool.ruff.format].exclude`
and to `[tool.ruff.lint.per-file-ignores]` so upstream syncs stay a clean diff —
the same policy as `vllm_mlx/models/{deepseek_v4.py,gemma4_vendored,hy_v3.py}`.

## Not on the `MusicEngine` runtime path

Only `scripts/sa3_mlx.py`, `scripts/weights.py`, `scripts/spec.py`,
`models/defs/sa3_pipeline.py`, `models/defs/t5gemma_mlx.py`,
`models/defs/dit_mlx*.py` and `models/defs/same_*_decoder.py` are used by
`MusicEngine.generate`. Everything else (LoRA training, the gradio UI, the
encoders, parity/benchmark/pre-encode/export/install scripts) is upstream
developer tooling — candidates for removal if we do not intend to maintain and
security-review it. Note `scripts/install.py:88` runs
`pip install -r requirements.txt`, and that `requirements.txt` was not vendored,
so the script is non-functional here.
