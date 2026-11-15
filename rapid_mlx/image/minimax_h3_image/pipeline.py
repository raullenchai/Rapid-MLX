"""Local-only MiniMax-H3-Image inference, matching the official block order.

Image-specific differences from mlx-vlm video presentation: verbatim prompt,
4096-token truncation per text segment, and ``Picture i: `` labels. Reference
posterior draws round through FP16, then get 0.001 noise augmentation (t=0.999).
Noise uses MLX keys derived from seed; seeds are not bitwise-identical to the
official torch CPU generator. The scheduler uses the fixed velocity Euler form,
without sigma shifting or CFG. uint8 conversion truncates before PIL LANCZOS.
Vision preprocessing reads the local processor config with the installed
PIL/NumPy image processor instead of mlx-vlm's video preprocessing adapter.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_vlm.models.minimax_h3.conditioner import MiniMaxH3Conditioner
from mlx_vlm.models.minimax_h3.packing import patchify_video_latents
from mlx_vlm.models.minimax_h3.weights import (
    _conditioner_key_selected,
    _prepare_conditioner_weights,
    _trim_conditioner_config,
)
from mlx_vlm.models.qwen3_vl.config import ModelConfig as QwenConfig
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as QwenModel
from PIL import Image, ImageOps

from .layout import (
    MAX_PROMPT_TOKENS,
    MIN_GENERATION_AREA,
    PIXEL_MEAN,
    PIXEL_STD,
    REFERENCE_TIMESTEP,
    SIGMAS,
    bound_short_edge,
    build_image_packed_sequence,
    check_reference_size,
    get_image_canvas_size,
    row_timestep_plan,
    validate_canvas,
    validate_request,
)
from .quantize import stream_load, validate_bits
from .transformer import load_transformer
from .vae import load_vae


def velocity_euler_step(
    sample: mx.array, velocity: mx.array, sigma: float, sigma_next: float
) -> mx.array:
    delta = mx.array(sigma, mx.float32) - mx.array(sigma_next, mx.float32)
    return (sample.astype(mx.float32) + delta * velocity.astype(mx.float32)).astype(
        sample.dtype
    )


def _load_conditioner(
    model_path: Path, text_encoder_path: Path, bits: int | None
) -> MiniMaxH3Conditioner:
    from transformers import AutoTokenizer

    directory = text_encoder_path / "text_encoder"
    config = _trim_conditioner_config(
        json.loads((directory / "config.json").read_text()), text_only=False
    )
    model = QwenModel(QwenConfig.from_dict(config))
    stream_load(
        model,
        directory,
        bits=bits,
        select=lambda key: _conditioner_key_selected(key, text_only=False),
        sanitize=lambda weights: _prepare_conditioner_weights(weights, text_only=False),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    return MiniMaxH3Conditioner(model, tokenizer, has_vision=True)


class MiniMaxH3Image:
    def __init__(
        self,
        model_path: str | Path,
        text_encoder_path: str | Path,
        *,
        quantize: int | None,
        on_step: Callable[[int, int], None] | None = None,
    ) -> None:
        validate_bits(quantize)
        self.model_path = Path(model_path).expanduser()
        self.text_encoder_path = Path(text_encoder_path).expanduser()
        self.quantize = quantize
        self.on_step = on_step
        # Lazily loaded in ``load()``; typed loosely because the MLX modules
        # come from optional runtime dependencies.
        self.transformer: Any = None
        self.vae: Any = None
        self.conditioner: Any = None

    def load(self) -> None:
        """Explicitly warm lazy components, useful for operator memory measurements."""
        if self.vae is None:
            self.vae = load_vae(self.model_path)
        if self.conditioner is None:
            self.conditioner = _load_conditioner(
                self.model_path, self.text_encoder_path, self.quantize
            )
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path, self.quantize)
        pdd = json.loads((self.model_path / "pdd_config.json").read_text())
        if pdd["num_inference_steps"] != 6 or tuple(pdd["sigmas"]) != SIGMAS:
            raise ValueError(
                "MiniMax-H3-Image requires the released six-step PDD schedule"
            )
        mx.clear_cache()

    def _encode_prompt(
        self, prompt: str, images: list[Image.Image]
    ) -> tuple[mx.array, np.ndarray]:
        conditioner = self.conditioner
        tokenizer = conditioner.tokenizer
        token_ids: list[int] = []
        tags: list[int] = []
        pixels = grid = None
        if images:
            # The slow processor exposes the official PIL/NumPy path, without
            # importing the downloaded reference or executing a tensor runtime.
            from transformers import Qwen2VLImageProcessor

            processor = Qwen2VLImageProcessor.from_pretrained(
                self.model_path / "processor", local_files_only=True
            )
            vision = processor(images=images, return_tensors="np")
            pixels = mx.array(vision["pixel_values"])
            grid = mx.array(vision["image_grid_thw"], dtype=mx.int32)
            for i, shape in enumerate(vision["image_grid_thw"]):
                label = tokenizer(f"Picture {i + 1}: ", add_special_tokens=False)[
                    "input_ids"
                ][:MAX_PROMPT_TOKENS]
                block = (
                    [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [tokenizer.convert_tokens_to_ids("<|image_pad|>")]
                    * (int(np.prod(shape)) // processor.merge_size**2)
                    + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids.extend(label + block)
                tags.extend([1] * len(label) + [0] * len(block))
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][
            :MAX_PROMPT_TOKENS
        ]
        token_ids.extend(prompt_ids)
        tags.extend([1] * len(prompt_ids))
        output = conditioner._encode(
            token_ids, tags, pixel_values=pixels, image_grid_thw=grid
        )
        hidden = output.hidden_states
        mx.eval(hidden)
        del output, pixels, grid
        gc.collect()
        mx.clear_cache()
        return hidden, np.array(tags, dtype=np.int64)

    def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        seed: int,
        references: list[Image.Image] | None = None,
        min_generation_area: int = MIN_GENERATION_AREA,
        num_inference_steps: int = 6,
        guidance_scale: float = 1.0,
        cancel_check: Callable[[], None] | None = None,
    ) -> Image.Image:
        references = references or []
        validate_request(
            width,
            height,
            min_generation_area,
            num_inference_steps,
            guidance_scale,
            num_references=len(references),
        )
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("MiniMax-H3-Image prompt must be a non-empty string")
        canvas_h, canvas_w = get_image_canvas_size(height, width, min_generation_area)
        validate_canvas(canvas_h, canvas_w)
        if cancel_check:
            cancel_check()
        self.load()
        key = mx.random.key(seed)

        def noise(shape: tuple[int, ...]) -> mx.array:
            nonlocal key
            key, draw_key = mx.random.split(key)
            return mx.random.normal(shape, key=draw_key, dtype=mx.float32)

        vae = self.vae
        mean = mx.array(PIXEL_MEAN).reshape(1, 3, 1, 1)
        std = mx.array(PIXEL_STD).reshape(1, 3, 1, 1)
        reference_latents = []
        vision_images = []
        for image in references:
            image = ImageOps.exif_transpose(image).convert("RGB")
            ref_w, ref_h = image.size
            check_reference_size(ref_w, ref_h)
            ref_size = bound_short_edge(ref_w, ref_h, 1024, 2048)
            vision_size = bound_short_edge(ref_w, ref_h, 1024, 1024)
            vision_images.append(
                image.resize(vision_size, Image.Resampling.LANCZOS)
                if vision_size != image.size
                else image
            )
            ref = (
                image.resize(ref_size, Image.Resampling.LANCZOS)
                if ref_size != image.size
                else image
            )
            pixels = (
                mx.array(np.array(ref)).transpose(2, 0, 1)[None].astype(mx.float32)
                / 255.0
            )
            pixels = pixels[
                ..., : pixels.shape[-2] // 16 * 16, : pixels.shape[-1] // 16 * 16
            ]
            posterior = vae.encode((pixels - mean) / std)
            latents = (
                (
                    posterior.mean
                    + mx.exp(0.5 * posterior.logvar) * noise(posterior.mean.shape)
                )
                .astype(mx.float16)
                .astype(mx.float32)
            )
            latents = latents[
                ..., : latents.shape[-2] // 2 * 2, : latents.shape[-1] // 2 * 2
            ]
            normalized = vae.normalize(latents)
            mx.eval(normalized)
            reference_latents.append(normalized)
            del pixels, posterior, latents, normalized
        mx.clear_cache()
        text, tags = self._encode_prompt(prompt, vision_images)
        del vision_images
        lh, lw = canvas_h // 16, canvas_w // 16
        positions, tags, image_indices, text_indices, nref = (
            build_image_packed_sequence(
                tags, [z.shape[-2:] for z in reference_latents], lh, lw
            )
        )
        plan = row_timestep_plan(len(tags), image_indices, nref)
        channels = vae.config.latent_channels
        # Official generated noise is drawn in NCHW before patchification.
        target = patchify_video_latents(noise((1, channels, 1, lh, lw)), (1, 2, 2))
        condition = []
        t = mx.array(REFERENCE_TIMESTEP, mx.float32)
        for z in reference_latents:
            noised = (1.0 - t) * noise(z.shape) + t * z
            condition.append(patchify_video_latents(noised[:, :, None], (1, 2, 2)))
        rows = mx.concatenate([*condition, target], axis=0)
        mx.eval(rows)
        del condition, target, reference_latents
        mx.clear_cache()
        # Float64 positions must be created/materialized on the CPU stream.
        with mx.stream(mx.cpu):
            pos = mx.array(positions, dtype=mx.float64)
            mx.eval(pos)
        tag_array = mx.array(tags, mx.int32)
        ii, ti = mx.array(image_indices, mx.int32), mx.array(text_indices, mx.int32)
        times = np.asarray([1.0 - s for s in SIGMAS[:-1]], dtype=np.float32)
        sigmas = np.concatenate([np.float32(1.0) - times, np.zeros(1, np.float32)])
        if self.on_step:
            self.on_step(0, num_inference_steps)
        for i, (values, indices) in enumerate(plan):
            if cancel_check:
                cancel_check()
            prediction = self.transformer(
                rows[None],
                text,
                mx.array(values),
                mx.array(indices),
                pos,
                tag_array,
                ii,
                ti,
                i,
            )[0]
            generated = velocity_euler_step(
                rows[nref:], prediction[nref:], float(sigmas[i]), float(sigmas[i + 1])
            )
            rows = mx.concatenate([rows[:nref], generated], axis=0)
            mx.eval(rows)
            del prediction, generated
            if self.on_step:
                self.on_step(i + 1, num_inference_steps)
        del text
        mx.clear_cache()
        latents = (
            rows[nref:]
            .reshape(1, lh // 2, lw // 2, channels, 2, 2)
            .transpose(0, 3, 1, 4, 2, 5)
            .reshape(1, channels, lh, lw)
        )
        decoded = vae.decode(vae.denormalize(latents))
        pixels = (
            (mx.clip(decoded * std + mean, 0, 1) * 255)
            .astype(mx.uint8)[0]
            .transpose(1, 2, 0)
        )
        mx.eval(pixels)
        result = Image.fromarray(np.array(pixels))
        if result.size != (width, height):
            result = result.resize((width, height), Image.Resampling.LANCZOS)
        del decoded, pixels, latents, rows
        mx.clear_cache()
        return result
