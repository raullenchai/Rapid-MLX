"""Image DiT composed from mlx-vlm's MiniMax-H3 mathematical primitives.

Divergences: no audio modules/stream or shared output head; six PDD heads,
selected by step. The official reference implementation pins proj_in,
time_embedder, pdd_proj_out and rope to FP32; the remaining modules use BF16.
The final AdaLN is never quantized, but retains its checkpoint compute dtype.
Block AdaLN subclasses the shared projection solely to read the floating-point
scales dtype after quantization instead of the packed integer weight dtype.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.minimax_h3.config import MiniMaxH3TransformerConfig
from mlx_vlm.models.minimax_h3.constants import MINIMAX_H3_MODALITY_NUM
from mlx_vlm.models.minimax_h3.transformer import (
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TokenRefiner,
    MiniMaxH3TransformerBlock,
    TimestepEmbedding,
    timestep_embedding,
)

from .quantize import stream_load


class ImageAdaLNModulation(MiniMaxH3AdaLayerNormModulation):
    """Preserve activation dtype when the Linear stores packed integer weights."""

    def __call__(self, temb: mx.array) -> tuple[mx.array, ...]:
        dtype = (
            self.linear.scales.dtype
            if isinstance(self.linear, nn.QuantizedLinear)
            else self.linear.weight.dtype
        )
        values = self.linear(nn.silu(temb).astype(dtype))
        return tuple(mx.split(values.reshape(-1, 6 * self.hidden_size), 6, axis=-1))


class MiniMaxH3ImageTransformer(nn.Module):
    def __init__(
        self, config: MiniMaxH3TransformerConfig, num_pdd_heads: int = 6
    ) -> None:
        super().__init__()
        self.config = config
        self.proj_in = nn.Linear(config.video_patch_dim, config.hidden_size)
        self.context_embedder = nn.Linear(config.text_dim, config.hidden_size)
        self.time_embedder = TimestepEmbedding(
            config.freq_dim, config.time_embed_hidden_dim, config.time_embed_dim
        )
        self.rope = MiniMaxH3RotaryPosEmbed(config.rope_freq_dim, config.rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(config)
        self.transformer_blocks = [
            MiniMaxH3TransformerBlock(config) for _ in range(config.num_layers)
        ]
        for block in self.transformer_blocks:
            block.adaln_proj = ImageAdaLNModulation(
                config.time_embed_dim, config.hidden_size
            )
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            config.hidden_size, config.time_embed_dim, config.final_norm_eps
        )
        self.pdd_proj_out = [
            nn.Linear(config.hidden_size, config.video_patch_dim)
            for _ in range(num_pdd_heads)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        timestep_indices: mx.array,
        position_ids: mx.array,
        token_tags: mx.array,
        image_indices: mx.array,
        text_indices: mx.array,
        pdd_step_index: int,
    ) -> mx.array:
        if not 0 <= pdd_step_index < len(self.pdd_proj_out):
            raise ValueError("pdd_step_index is outside the PDD head range")
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError("position_ids must have shape (sequence_length, 3)")
        length = position_ids.shape[0]
        if token_tags.shape != (length,) or timestep_indices.shape != (length,):
            raise ValueError(
                "token tags and timestep indices must match the sequence length"
            )
        rotary = self.rope(position_ids)
        image = self.proj_in(hidden_states.astype(self.proj_in.weight.dtype))
        text = self.token_refiner(
            self.context_embedder(
                encoder_hidden_states.astype(self.context_embedder.weight.dtype)
            )
        )
        packed = mx.zeros((text.shape[0], length, text.shape[-1]), dtype=text.dtype)
        packed = packed.at[:, text_indices, :].add(text)
        packed = packed.at[:, image_indices, :].add(image.astype(text.dtype))
        temb = self.time_embedder(timestep_embedding(timestep, self.config.freq_dim))
        indices = (timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags).astype(
            mx.int32
        )
        for block in self.transformer_blocks:
            packed = block(packed, block.adaln_proj(temb), indices, rotary)
        packed = self.norm_out(packed, temb, timestep_indices)
        head = self.pdd_proj_out[pdd_step_index]
        return mx.take(head(packed.astype(head.weight.dtype)), image_indices, axis=1)


def transformer_dtype(key: str, value: mx.array) -> mx.array:
    return value.astype(
        mx.float32
        if key.startswith(("proj_in.", "time_embedder.", "pdd_proj_out.", "rope."))
        else mx.bfloat16
    )


def load_transformer(
    model_path: str | Path, bits: int | None = None
) -> MiniMaxH3ImageTransformer:
    directory = Path(model_path) / "transformer_diffusers"
    config = json.loads((directory / "config.json").read_text())
    model = MiniMaxH3ImageTransformer(
        MiniMaxH3TransformerConfig.from_dict(config), config.get("num_pdd_heads", 6)
    )
    stream_load(model, directory, bits=bits, cast=transformer_dtype)
    return model
