"""MiniMax-H3-Image VAE using mlx-vlm's CNN and ViT block primitives.

The 2D decoder forward is adapted minimally from mlx-vlm visual_vae.py (MIT).
Algorithm and image tiling semantics follow the official reference implementation.
Divergences from mlx-vlm video: independent config permits rotary dimension
32 (divisible by 4); two spatial axes, no temporal expansion, Conv2d post-quant,
and a single encoder frame. Default tiling is 256 pixels with >=64 overlap;
blending reads original neighbors, not previously blended/cropped tiles.
All VAE weights retain the released FP32 precision and are never quantized.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.minimax_h3.config import MiniMaxH3VideoVAEConfig
from mlx_vlm.models.minimax_h3.visual_vae import (
    MiniMaxH3DiagonalGaussianDistribution,
    MiniMaxH3VideoCausalConv3d,
    MiniMaxH3VideoEncoder3d,
    MiniMaxH3VideoTransformerBlock,
    MiniMaxH3VideoVAE,
)

from .quantize import stream_load


@dataclass(frozen=True, slots=True)
class MiniMaxH3ImageVAEConfig(MiniMaxH3VideoVAEConfig):
    decoder_num_layers: int = 24
    decoder_num_attention_heads: int = 16
    decoder_rope_dim_ratio: float = 0.5

    def __post_init__(self) -> None:
        levels = len(self.block_out_channels)
        if (
            not levels
            or len(self.spatial_downsample_factors) != levels
            or len(self.temporal_downsample_factors) != levels
        ):
            raise ValueError("downsample factors must match encoder levels")
        if (
            len(self.latents_mean) != self.latent_channels
            or len(self.latents_std) != self.latent_channels
        ):
            raise ValueError("latent statistics must match latent_channels")
        if any(std <= 0 for std in self.latents_std):
            raise ValueError("latent standard deviations must be positive")
        dim = int(self.decoder_attention_head_dim * self.decoder_rope_dim_ratio)
        if dim <= 0 or dim % 4 or dim > self.decoder_attention_head_dim:
            raise ValueError(
                "image decoder rotary dimension must be positive, divisible by 4, and fit a head"
            )


class MiniMaxH3ImageRotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, theta: float = 100.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 4:
            raise ValueError("two-axis rotary dimension must be divisible by 4")
        self.dim = dim
        self.theta = theta

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        inv_freq = 1.0 / mx.power(
            self.theta, mx.arange(self.dim // 4, dtype=mx.float32) * (4.0 / self.dim)
        )
        angles = 2.0 * math.pi * position_ids[..., None] * inv_freq
        angles = angles.reshape(*position_ids.shape[:-1], -1)
        angles = mx.tile(angles, (1, 1, 2))[:, :, None]
        return mx.cos(angles), mx.sin(angles)


class MiniMaxH3ImageViTDecoder2d(nn.Module):
    def __init__(self, config: MiniMaxH3ImageVAEConfig) -> None:
        super().__init__()
        dim = config.decoder_num_attention_heads * config.decoder_attention_head_dim
        self.patch_size = config.spatial_compression_ratio
        self.out_channels = config.out_channels
        self.num_register_tokens = config.decoder_num_register_tokens
        self.rope = MiniMaxH3ImageRotaryPosEmbed(
            int(config.decoder_attention_head_dim * config.decoder_rope_dim_ratio),
            config.decoder_rope_theta,
        )
        self.proj_in = nn.Linear(config.latent_channels, dim)
        self.register_tokens = mx.zeros(
            (1, self.num_register_tokens, dim), dtype=mx.float32
        )
        self.transformer_blocks = [
            MiniMaxH3VideoTransformerBlock(
                dim,
                config.decoder_num_attention_heads,
                config.decoder_attention_head_dim,
                config.decoder_ffn_mult,
                config.decoder_norm_eps,
            )
            for _ in range(config.decoder_num_layers)
        ]
        self.norm_out = nn.LayerNorm(dim, eps=config.decoder_norm_eps)
        self.proj_out = nn.Linear(dim, config.out_channels * self.patch_size**2)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.transpose(0, 2, 3, 1).reshape(
            batch, height * width, channels
        )
        hidden_states = self.proj_in(hidden_states)
        count = height * width
        registers = mx.broadcast_to(
            self.register_tokens.astype(hidden_states.dtype),
            (batch, self.num_register_tokens, hidden_states.shape[-1]),
        )
        hidden_states = mx.concatenate(
            [hidden_states, registers, mx.zeros_like(hidden_states[:, :1])], axis=1
        )
        grids = [
            2.0 * ((mx.arange(size, dtype=mx.float32) + 0.5) / size) - 1.0
            for size in (height, width)
        ]
        positions = mx.stack(mx.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 2)
        positions = mx.broadcast_to(positions[None], (batch, count, 2))
        positions = mx.concatenate(
            [positions, mx.zeros((batch, self.num_register_tokens + 1, 2))], axis=1
        )
        rotary = self.rope(positions)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary)
        hidden_states = self.proj_out(self.norm_out(hidden_states))[:, :count]
        patch = self.patch_size
        hidden_states = hidden_states.reshape(
            batch, height, width, self.out_channels, patch, patch
        )
        return hidden_states.transpose(0, 3, 1, 4, 2, 5).reshape(
            batch, self.out_channels, height * patch, width * patch
        )


class MiniMaxH3ImageVAE(nn.Module):
    _split_tiles = MiniMaxH3VideoVAE._split_tiles
    _blend = staticmethod(MiniMaxH3VideoVAE._blend)

    def __init__(self, config: MiniMaxH3ImageVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MiniMaxH3VideoEncoder3d(config)
        self.quant_conv = MiniMaxH3VideoCausalConv3d(
            2 * config.latent_channels, 2 * config.latent_channels, 1
        )
        self.post_quant_conv = nn.Conv2d(
            config.latent_channels, config.latent_channels, 1
        )
        self.decoder = MiniMaxH3ImageViTDecoder2d(config)
        self.use_tiling = True
        self.tile_sample_min_height = self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = self.tile_sample_min_overlap_width = 64

    def _stitch_tiles(
        self,
        tiles: list[list[mx.array]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> mx.array:
        # Inserting the singleton frame lets the shared stitcher use its video axes.
        return MiniMaxH3VideoVAE._stitch_tiles(
            self,
            [[tile[:, :, None] for tile in row] for row in tiles],
            height_overlaps,
            width_overlaps,
        )[:, :, 0]

    def _encode_tile(self, pixels: mx.array) -> mx.array:
        return self.quant_conv(self.encoder(pixels[:, :, None]))[:, :, -1]

    def _decode_tile(self, latents: mx.array) -> mx.array:
        values = self.post_quant_conv(latents.transpose(0, 2, 3, 1))
        return self.decoder(values.transpose(0, 3, 1, 2))

    def _tiled(self, values: mx.array, *, decode: bool) -> mx.array:
        operation = self._decode_tile if decode else self._encode_tile
        if not self.use_tiling:
            return operation(values)
        ratio = self.config.spatial_compression_ratio
        height, width = values.shape[-2:]
        if decode:
            height, width = height * ratio, width * ratio
        ys, yl, yo = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        xs, xl, xo = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        divisor = ratio if decode else 1
        tiles = []
        for y, h in zip(ys, yl, strict=True):
            row = []
            for x, w in zip(xs, xl, strict=True):
                tile = operation(
                    values[
                        ...,
                        y // divisor : (y + h) // divisor,
                        x // divisor : (x + w) // divisor,
                    ]
                )
                mx.eval(tile)
                row.append(tile)
            tiles.append(row)
        return self._stitch_tiles(
            tiles,
            yo if decode else [o // ratio for o in yo],
            xo if decode else [o // ratio for o in xo],
        )

    def encode(self, pixels: mx.array) -> MiniMaxH3DiagonalGaussianDistribution:
        return MiniMaxH3DiagonalGaussianDistribution(
            self._tiled(pixels.astype(mx.float32), decode=False)
        )

    def decode(self, latents: mx.array) -> mx.array:
        return self._tiled(latents.astype(mx.float32), decode=True)

    def normalize(self, latents: mx.array) -> mx.array:
        mean = mx.array(self.config.latents_mean).reshape(1, -1, 1, 1)
        std = mx.array(self.config.latents_std).reshape(1, -1, 1, 1)
        return (latents - mean) / std

    def denormalize(self, latents: mx.array) -> mx.array:
        mean = mx.array(self.config.latents_mean).reshape(1, -1, 1, 1)
        std = mx.array(self.config.latents_std).reshape(1, -1, 1, 1)
        return latents * std + mean

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return {
            key: value.transpose(0, 2, 3, 4, 1)
            if value.ndim == 5
            else value.transpose(0, 2, 3, 1)
            if value.ndim == 4
            else value
            for key, value in weights.items()
        }


def load_vae(model_path: str | Path) -> MiniMaxH3ImageVAE:
    directory = Path(model_path) / "vae_diffusers"
    config = MiniMaxH3ImageVAEConfig.from_dict(
        json.loads((directory / "config.json").read_text())
    )
    model = MiniMaxH3ImageVAE(config)
    stream_load(
        model,
        directory,
        sanitize=model.sanitize,
        cast=lambda key, value: value.astype(mx.float32),
    )
    return model
