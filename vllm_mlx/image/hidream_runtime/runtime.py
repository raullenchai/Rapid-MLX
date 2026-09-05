# SPDX-License-Identifier: MIT
"""HiDream-O1-Image-Dev inference adapted from the upstream MLX port.

Vendored from ``mlx-community/HiDream-O1-Image-Dev-mlx-bf16`` commit
``33c7a00bce8e3410304f83ec408a15a1eb6782df``.  Rapid keeps only the
text-to-image path and wraps it as an in-process backend; the original CLI,
edit/multi-reference path, diagnostics, and file-output concerns are omitted.
See the adjacent LICENSE and NOTICE for provenance and terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

PATCH_SIZE = 32
TIMESTEP_TOKEN_NUM = 1
NOISE_SCALE = 7.5
NOISE_CLIP_STD = 2.5
T_EPS = 0.001
MAX_PROMPT_TOKENS = 1024

# Dev's published 28-step schedule, from the upstream pipeline.
DEFAULT_TIMESTEPS = (
    999,
    987,
    974,
    960,
    945,
    929,
    913,
    895,
    877,
    857,
    836,
    814,
    790,
    764,
    737,
    707,
    675,
    640,
    602,
    560,
    515,
    464,
    409,
    347,
    278,
    199,
    110,
    8,
)


class FlashFlowMatchScheduler:
    """The Dev flow-matching Euler scheduler, implemented with MLX arrays."""

    def __init__(self) -> None:
        self.timesteps = np.asarray(DEFAULT_TIMESTEPS, dtype=np.float32)
        self.sigmas = np.append(self.timesteps / 1000.0, 0.0).astype(np.float32)
        self.step_index = 0

    def step(self, model_output: mx.array, sample: mx.array, *, seed: int) -> mx.array:
        index = self.step_index
        sigma = float(self.sigmas[index])
        sigma_next = float(self.sigmas[index + 1])
        sample_f = sample.astype(mx.float32)
        output_f = model_output.astype(mx.float32)
        denoised = sample_f - output_f * sigma
        key = mx.random.key(seed + index)
        noise = mx.random.normal(output_f.shape, key=key)
        std = float(mx.std(noise))
        noise = mx.clip(noise, -NOISE_CLIP_STD * std, NOISE_CLIP_STD * std)
        result = sigma_next * noise * NOISE_SCALE + (1.0 - sigma_next) * denoised
        self.step_index += 1
        return result.astype(sample.dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.fc1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)

    @staticmethod
    def embedding(t: mx.array, dim: int) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None]
        result = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return (
            mx.concatenate([result, mx.zeros_like(result[:, :1])], axis=-1)
            if dim % 2
            else result
        )

    def __call__(self, t: mx.array) -> mx.array:
        freq = self.embedding(t * 1000.0, self.frequency_embedding_size)
        return self.fc2(nn.silu(self.fc1(freq.astype(self.fc1.weight.dtype))))


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.proj1 = nn.Linear(PATCH_SIZE * PATCH_SIZE * 3, 1024, bias=False)
        self.proj2 = nn.Linear(1024, hidden_size, bias=True)

    def __call__(self, value: mx.array) -> mx.array:
        return self.proj2(self.proj1(value))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.linear = nn.Linear(hidden_size, PATCH_SIZE * PATCH_SIZE * 3, bias=True)

    def __call__(self, value: mx.array) -> mx.array:
        return self.linear(value)


@dataclass(frozen=True)
class HiDreamConfig:
    hidden_size: int = 4096
    tms_token_id: int = 151673


def _validate_custom_head_weights(expected: dict, actual: dict) -> None:
    """Fail closed before installing incompatible diffusion-head tensors."""

    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    mismatched = sorted(
        key
        for key in expected_keys & actual_keys
        if tuple(actual[key].shape) != tuple(expected[key].shape)
    )
    if missing or extra or mismatched:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"unexpected={extra}")
        if mismatched:
            details = {
                key: (tuple(actual[key].shape), tuple(expected[key].shape))
                for key in mismatched
            }
            parts.append(f"shape_mismatch={details}")
        raise ValueError("incompatible HiDream custom heads: " + "; ".join(parts))


def _patchify(image: np.ndarray) -> np.ndarray:
    channels, height, width = image.shape
    if height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError(f"HiDream dimensions must be multiples of {PATCH_SIZE}")
    value = image.reshape(
        channels,
        height // PATCH_SIZE,
        PATCH_SIZE,
        width // PATCH_SIZE,
        PATCH_SIZE,
    )
    value = np.transpose(value, (1, 3, 0, 2, 4))
    return value.reshape(
        (height // PATCH_SIZE) * (width // PATCH_SIZE),
        channels * PATCH_SIZE * PATCH_SIZE,
    )


def _unpatchify(patches: np.ndarray, height: int, width: int) -> np.ndarray:
    value = patches.reshape(
        height // PATCH_SIZE,
        width // PATCH_SIZE,
        3,
        PATCH_SIZE,
        PATCH_SIZE,
    )
    value = np.transpose(value, (2, 0, 3, 1, 4))
    return value.reshape(3, height, width)


def _rope_positions(
    *,
    input_ids: np.ndarray,
    image_grid: np.ndarray,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> np.ndarray:
    """Port of HiDream's fixed-point Qwen3-VL mRoPE index builder (T2I)."""

    batch, sequence = input_ids.shape
    positions = np.ones((3, batch, sequence), dtype=input_ids.dtype)
    for batch_index in range(batch):
        ids = input_ids[batch_index]
        starts = np.argwhere(ids == vision_start_token_id).reshape(-1)
        following = ids[starts + 1] if len(starts) else np.array([], dtype=ids.dtype)
        image_count = int((following == image_token_id).sum())
        video_count = int((following == video_token_id).sum())
        tokens = ids.tolist()
        pieces: list[np.ndarray] = []
        cursor = 0
        image_index = 0
        fix_point = 4096
        remaining_images = image_count
        remaining_videos = video_count
        for _ in range(image_count + video_count):
            image_end = (
                tokens.index(image_token_id, cursor)
                if image_token_id in tokens[cursor:] and remaining_images
                else len(tokens) + 1
            )
            video_end = (
                tokens.index(video_token_id, cursor)
                if video_token_id in tokens[cursor:] and remaining_videos
                else len(tokens) + 1
            )
            if image_end >= video_end:
                raise ValueError(
                    "HiDream text-to-image sample unexpectedly contains video tokens"
                )
            grid_t, grid_h, grid_w = (int(v) for v in image_grid[image_index])
            image_index += 1
            remaining_images -= 1
            text_length = max(0, image_end - cursor - 1)
            start_index = int(pieces[-1].max() + 1) if pieces else 0
            pieces.append(
                np.broadcast_to(
                    np.arange(text_length) + start_index, (3, text_length)
                ).copy()
            )
            t_index = np.repeat(np.arange(grid_t), grid_h * grid_w)
            h_index = np.tile(np.repeat(np.arange(grid_h), grid_w), grid_t)
            w_index = np.tile(np.arange(grid_w), grid_t * grid_h)
            fix_point -= start_index
            pieces.append(
                np.stack([t_index, h_index, w_index]) + fix_point + start_index
            )
            fix_point = 0
            cursor = image_end + grid_t * grid_h * grid_w
        if cursor < len(tokens):
            start_index = int(pieces[-1].max() + 1) if pieces else 0
            text_length = len(tokens) - cursor
            pieces.append(
                np.broadcast_to(
                    np.arange(text_length) + start_index, (3, text_length)
                ).copy()
            )
        positions[..., batch_index, :] = np.concatenate(pieces, axis=1).reshape(3, -1)
    return positions


def _encode_prompt_ids(prompt: str, processor) -> np.ndarray:
    """Apply the published chat wrapper and return bounded-model token ids."""

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    for name in ("boi", "tms"):
        attribute = f"{name}_token"
        if not getattr(tokenizer, attribute, None):
            setattr(tokenizer, attribute, f"<|{name}_token|>")
    caption = (
        processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        + tokenizer.boi_token
        + tokenizer.tms_token * TIMESTEP_TOKEN_NUM
    )
    return np.asarray(
        tokenizer.encode(caption, add_special_tokens=False), dtype=np.int64
    ).reshape(1, -1)


def _build_sample(prompt: str, height: int, width: int, processor, config) -> dict:
    input_ids = _encode_prompt_ids(prompt, processor)
    if input_ids.shape[-1] > MAX_PROMPT_TOKENS:
        raise ValueError(
            f"HiDream-O1 prompts are limited to {MAX_PROMPT_TOKENS} tokens"
        )
    image_length = (height // PATCH_SIZE) * (width // PATCH_SIZE)
    image_grid = np.asarray(
        [[1, height // PATCH_SIZE, width // PATCH_SIZE]], dtype=np.int64
    )
    image_tokens = np.full((1, image_length), config.image_token_id, dtype=np.int64)
    image_tokens[0, 0] = config.vision_start_token_id
    padded = np.concatenate([input_ids, image_tokens], axis=-1)
    position_ids = _rope_positions(
        input_ids=padded,
        image_grid=image_grid,
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id,
        vision_start_token_id=config.vision_start_token_id,
    )
    text_length = input_ids.shape[-1]
    token_types = np.zeros((1, position_ids.shape[-1]), dtype=np.int64)
    begin = text_length - TIMESTEP_TOKEN_NUM
    token_types[0, begin : begin + image_length + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, text_length - TIMESTEP_TOKEN_NUM : text_length] = 3
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": (token_types > 0).astype(np.int64),
        "vinput_mask": token_types == 1,
    }


def _attention_mask(token_types: np.ndarray) -> np.ndarray:
    batch, sequence = token_types.shape
    floor = -1e4
    result = np.full((batch, 1, sequence, sequence), floor, dtype=np.float32)
    causal = np.triu(np.full((sequence, sequence), floor, dtype=np.float32), k=1)
    for index in range(batch):
        mask = causal.copy()
        mask[token_types[index].astype(bool), :] = 0.0
        result[index, 0] = mask
    return result


class HiDreamO1(nn.Module):
    """In-process, text-to-image-only adapter over the published MLX weights."""

    def __init__(
        self,
        model_path: str,
        *,
        on_step: Callable[[int, int], None] | None = None,
    ) -> None:
        from mlx_vlm import load as mlx_vlm_load

        super().__init__()
        backbone, self.processor = mlx_vlm_load(model_path)
        self.backbone_config = backbone.config
        self.config = HiDreamConfig()
        self.visual = backbone.vision_tower
        self.language_model = backbone.language_model
        self.t_embedder1 = TimestepEmbedder(self.config.hidden_size)
        self.x_embedder = BottleneckPatchEmbed(self.config.hidden_size)
        self.final_layer2 = FinalLayer(self.config.hidden_size)
        custom_path = Path(model_path) / "extras" / "custom_heads.safetensors"
        if not custom_path.is_file():
            raise FileNotFoundError(f"missing HiDream custom heads: {custom_path}")
        custom_weights = mx.load(str(custom_path))
        from mlx.utils import tree_flatten

        expected_weights = {}
        for prefix, module in (
            ("t_embedder1", self.t_embedder1),
            ("x_embedder", self.x_embedder),
            ("final_layer2", self.final_layer2),
        ):
            expected_weights.update(
                {
                    f"{prefix}.{name}": value
                    for name, value in tree_flatten(module.parameters())
                }
            )
        _validate_custom_head_weights(expected_weights, custom_weights)
        # Strict loading on ``self`` would also demand the separately loaded
        # backbone. Exact key/shape validation above makes this scoped load
        # fail-closed while intentionally omitting those backbone tensors.
        self.load_weights(list(custom_weights.items()), strict=False)
        self.on_step = on_step

    def _text_embeddings(self, input_ids: mx.array) -> mx.array:
        return self.language_model.model.embed_tokens(input_ids)

    def _forward(
        self,
        embeddings: mx.array,
        input_ids: mx.array,
        position_ids: mx.array,
        vinputs: mx.array,
        timestep: mx.array,
        mask: mx.array,
    ) -> mx.array:
        timestep_embedding = self.t_embedder1(timestep)
        timestep_mask = mx.broadcast_to(
            (input_ids == self.config.tms_token_id)[..., None], embeddings.shape
        )
        expanded = mx.broadcast_to(timestep_embedding[:, None, :], embeddings.shape)
        embeddings = mx.where(timestep_mask, expanded, embeddings)
        image_embeddings = self.x_embedder(vinputs).astype(embeddings.dtype)
        embeddings = mx.concatenate([embeddings, image_embeddings], axis=1)
        hidden = self.language_model.model(
            mx.zeros(embeddings.shape[:2], dtype=mx.int32),
            inputs_embeds=embeddings,
            mask=mask,
            cache=None,
            position_ids=position_ids,
        )
        return self.final_layer2(hidden)

    def generate_image(
        self,
        *,
        seed: int,
        prompt: str,
        num_inference_steps: int = 28,
        height: int = 1024,
        width: int = 1024,
    ) -> SimpleNamespace:
        if num_inference_steps != 28:
            raise ValueError(
                "HiDream-O1 Dev supports its published 28-step schedule only"
            )
        if height % PATCH_SIZE or width % PATCH_SIZE:
            raise ValueError(f"HiDream dimensions must be multiples of {PATCH_SIZE}")
        sample = _build_sample(
            prompt, height, width, self.processor, self.backbone_config
        )
        input_ids = mx.array(sample["input_ids"])
        position_ids = mx.array(sample["position_ids"])
        token_types = mx.array(sample["token_types"])
        mask = mx.array(_attention_mask(sample["token_types"])).astype(mx.bfloat16)
        image_indices = mx.array(np.where(sample["vinput_mask"][0])[0].astype(np.int32))
        key = mx.random.key(seed + 1)
        noise = NOISE_SCALE * mx.random.normal((1, 3, height, width), key=key)
        patches = mx.array(_patchify(np.asarray(noise)[0])[None]).astype(mx.bfloat16)
        embeddings = self._text_embeddings(input_ids)
        mx.eval(embeddings)
        scheduler = FlashFlowMatchScheduler()
        total = len(scheduler.timesteps)
        for index, step in enumerate(scheduler.timesteps):
            if self.on_step is not None:
                # Report the number of already-completed steps. This also
                # checks cancellation before starting the next expensive
                # forward pass without claiming that pass has finished.
                self.on_step(index, total)
            timestep = mx.full([1], 1.0 - float(step) / 1000.0, dtype=mx.float32)
            sigma = max(float(step) / 1000.0, T_EPS)
            prediction = self._forward(
                embeddings,
                input_ids,
                position_ids,
                patches,
                timestep,
                mask,
            )
            generated = mx.take(prediction, image_indices, axis=1).astype(mx.float32)
            velocity = (generated - patches.astype(mx.float32)) / sigma
            patches = scheduler.step(-velocity, patches, seed=seed)
            mx.eval(patches)
            if self.on_step is not None:
                self.on_step(index + 1, total)
        image = (patches + 1) / 2
        rgb = _unpatchify(np.asarray(image[0].astype(mx.float32)), height, width)
        pixels = np.clip(rgb.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        return SimpleNamespace(image=Image.fromarray(pixels, mode="RGB"))
