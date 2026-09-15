# SPDX-License-Identifier: Apache-2.0
"""Rapid-MLX adapter for the pinned SD3.5 Large 4-bit MLX checkpoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ._vendor import model_io
from ._vendor.pipeline import DiffusionPipeline

MODEL_REPO = "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized"
SHARED_REPO = "argmaxinc/stable-diffusion"
T5_TOKENIZER_REPO = "google/t5-v1_1-xxl"
MODEL_FILENAME = "sd3.5_large_4bit_quantized.safetensors"

MODEL_FILES = ("config.json", MODEL_FILENAME)
SHARED_FILES = (
    "clip_l/config.json",
    "clip_l/model.fp16.safetensors",
    "clip_g/config.json",
    "clip_g/model.fp16.safetensors",
    "tokenizer_l/vocab.json",
    "tokenizer_l/merges.txt",
    "tokenizer_g/vocab.json",
    "tokenizer_g/merges.txt",
    "t5/t5xxl.safetensors",
)
T5_TOKENIZER_FILES = (
    "config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class GeneratedImage:
    """Image-engine-compatible result container."""

    image: Image.Image


def _require_files(root: Path, names: tuple[str, ...]) -> None:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ValueError(f"Required SD3.5 asset directory is missing: {root}")
    # Hugging Face snapshots are symlink farms into the same repository's
    # sibling ``blobs/`` directory. Permit that standard layout while still
    # refusing traversal or a symlink into another repo/arbitrary filesystem.
    allowed_root = (
        resolved.parents[1].resolve()
        if resolved.parent.name == "snapshots"
        else resolved
    )
    for name in names:
        candidate = resolved.joinpath(*name.split("/"))
        try:
            real = candidate.resolve(strict=True)
        except OSError:
            raise ValueError(f"Required SD3.5 asset is missing: {name}") from None
        if (
            not candidate.is_relative_to(resolved)
            or not real.is_relative_to(allowed_root)
            or not real.is_file()
        ):
            raise ValueError(f"Required SD3.5 asset is missing: {name}")


class SD35Large:
    """Full three-encoder SD3.5 Large text-to-image runtime."""

    def __init__(
        self,
        model_path: str | Path,
        shared_path: str | Path,
        t5_tokenizer_path: str | Path,
        *,
        on_step: Callable[[int, int], None] | None = None,
    ) -> None:
        model_root = Path(model_path).resolve()
        shared_root = Path(shared_path).resolve()
        t5_root = Path(t5_tokenizer_path).resolve()
        _require_files(model_root, MODEL_FILES)
        _require_files(shared_root, SHARED_FILES)
        _require_files(t5_root, T5_TOKENIZER_FILES)

        with (model_root / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        if config != {"name": "stable-diffusion-3.5-large-4bit-quantized"}:
            raise ValueError("Unsupported SD3.5 checkpoint configuration.")

        model_io.configure_asset_roots(
            {MODEL_REPO: model_root, SHARED_REPO: shared_root},
            t5_tokenizer_root=t5_root,
        )
        self.pipeline = DiffusionPipeline(
            w16=True,
            a16=True,
            shift=3.0,
            use_t5=True,
            model_version=MODEL_REPO,
            low_memory_mode=True,
            local_ckpt=model_root / MODEL_FILENAME,
            on_step=on_step,
        )

    def generate_image(
        self,
        *,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        seed: int,
        guidance: float | None = None,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        image, _log = self.pipeline.generate_image(
            text=prompt,
            num_steps=num_inference_steps,
            cfg_weight=3.5 if guidance is None else guidance,
            negative_text=negative_prompt or "",
            latent_size=(height // 8, width // 8),
            seed=seed,
            verbose=False,
        )
        return GeneratedImage(image=image)
