"""Rapid-MLX adapter around the vendored SDXL pipeline."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from PIL import Image

from .pipelines import StableDiffusionXLPipeline
from .utils import to_pil


@dataclass(frozen=True)
class GeneratedImage:
    """Image-engine-compatible result container."""

    image: Image.Image


class SDXL:
    """Reusable text-to-image runtime for an official SDXL snapshot."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        on_step: Callable[[int, int], None] | None = None,
    ) -> None:
        self.pipeline = StableDiffusionXLPipeline.from_diffusers(
            model_path,
            quantize_unet=8,
        )
        self._on_step = on_step

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
        image = self.pipeline(
            prompt,
            negative_prompt=negative_prompt or "",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=5.0 if guidance is None else guidance,
            seed=seed,
            # Recompute the full UNet every other step and reuse its deep
            # features between adjacent scheduler steps. The shallow path and
            # scheduler still run on every step, so progress/cancel semantics
            # remain exact while 1024² dogfood falls from ~58s to ~16s.
            cache_interval=2,
            tile_vae=True,
            progress=False,
            on_step=self._on_step,
        )
        mx.eval(image)
        return GeneratedImage(image=to_pil(image[0]))
