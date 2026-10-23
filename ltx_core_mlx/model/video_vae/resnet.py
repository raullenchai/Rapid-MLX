"""Residual blocks for Video VAE.

Ported from ltx-core/src/ltx_core/model/video_vae/resnet.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.video_vae.convolution import Conv3dBlock
from ltx_core_mlx.model.video_vae.normalization import pixel_norm


class ResBlock3d(nn.Module):
    """Pre-activation residual block: PixelNorm -> SiLU -> Conv.

    Reference: ltx-core ResnetBlock3D with NormLayerType.PIXEL_NORM.
    PixelNorm is parameterless so no norm weights in safetensors.

    Forward: norm1 -> silu -> conv1 -> norm2 -> silu -> conv2 + skip
    Produces keys: ``conv1.conv.{weight,bias}``, ``conv2.conv.{weight,bias}``
    """

    def __init__(self, channels: int, causal: bool = True, spatial_padding_mode: str = "zeros"):
        super().__init__()
        self.conv1 = Conv3dBlock(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv2 = Conv3dBlock(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.conv1(nn.silu(pixel_norm(x)))
        x = self.conv2(nn.silu(pixel_norm(x)))
        return x + residual


class ResBlockStage(nn.Module):
    """A stage of N residual blocks at a fixed channel count.

    Produces keys: ``res_blocks.{i}.conv{1,2}.conv.{weight,bias}``
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.res_blocks = [
            ResBlock3d(channels, causal=causal, spatial_padding_mode=spatial_padding_mode) for _ in range(num_blocks)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.res_blocks:
            x = block(x)
        return x
