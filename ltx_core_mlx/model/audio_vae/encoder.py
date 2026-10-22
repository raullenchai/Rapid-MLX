"""Audio VAE Encoder — mel spectrogram to latent.

Ported from ltx-core audio VAE encoder. Architecture derived from
embedded_config.json: ch=128, ch_mult=[1,2,4], num_res_blocks=2,
double_z=True, causality_axis=height, norm_type=pixel.

Weight structure (after stripping "audio_vae.encoder." prefix):
    conv_in.conv.{weight,bias}             — Conv2d(2, 128, 3)
    down.0.block.{0,1}.*                   — 2 ResBlocks(128→128)
    down.0.downsample.conv.{weight,bias}   — Downsample(128)
    down.1.block.{0,1}.*                   — 2 ResBlocks(128→256, nin_shortcut on block 0)
    down.1.downsample.conv.{weight,bias}   — Downsample(256)
    down.2.block.{0,1}.*                   — 2 ResBlocks(256→512, nin_shortcut on block 0)
    mid.block_{1,2}.*                      — 2 ResBlocks(512)
    conv_out.conv.{weight,bias}            — Conv2d(512, 16, 3)  [double_z: 8 mean + 8 logvar]
    per_channel_statistics.*               — loaded separately
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.audio_vae.audio_vae import (
    AudioMidBlock,
    AudioResBlock,
    PerChannelStatistics,
    WrappedConv2d,
    pixel_norm,
)


class AudioDownsample(nn.Module):
    """2x spatial downsample via stride-2 Conv2d with causal padding.

    Key: ``downsample.conv.{weight,bias}`` — direct Conv2d, NOT WrappedConv2d.
    """

    def __init__(self, channels: int, causal: bool = False):
        super().__init__()
        self._causal = causal
        if causal:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)
        else:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        if self._causal:
            x = mx.pad(x, [(0, 0), (2, 0), (0, 1), (0, 0)])
        return self.conv(x)


class AudioDownBlock(nn.Module):
    """One encoder down-stage: N resblocks + optional downsample.

    Key prefix: ``down.<idx>.``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        add_downsample: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.block = [
            AudioResBlock(in_channels if i == 0 else out_channels, out_channels, causal=causal)
            for i in range(num_blocks)
        ]
        self.downsample = AudioDownsample(out_channels, causal=causal) if add_downsample else None

    def __call__(self, x: mx.array) -> mx.array:
        for blk in self.block:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class AudioVAEEncoder(nn.Module):
    """Audio VAE encoder: mel (B, 2, T', 64) -> latent (B, 8, T, 16).

    Architecture from embedded_config: ch=128, ch_mult=[1,2,4],
    num_res_blocks=2, double_z=True (conv_out outputs 16 = 2*z_channels).

    Weight key structure:
        conv_in  : Conv2d(2, 128, 3)
        down.0   : 2x ResBlock(128→128) + Downsample — freq 64→32
        down.1   : 2x ResBlock(128→256, shortcut) + Downsample — freq 32→16
        down.2   : 2x ResBlock(256→512, shortcut), no downsample
        mid      : 2x ResBlock(512)
        conv_out : Conv2d(512, 16, 3)  [8 mean + 8 logvar]
    """

    def __init__(self):
        super().__init__()

        self.conv_in = WrappedConv2d(2, 128, 3, padding=1, causal=True)

        # Down blocks: ch_mult=[1,2,4] → channels 128, 256, 512
        # num_res_blocks=2
        self.down = [
            AudioDownBlock(128, 128, num_blocks=2, add_downsample=True, causal=True),
            AudioDownBlock(128, 256, num_blocks=2, add_downsample=True, causal=True),
            AudioDownBlock(256, 512, num_blocks=2, add_downsample=False, causal=True),
        ]

        self.mid = AudioMidBlock(512, causal=True, add_attention=False)

        # double_z=True: output 16 channels (8 mean + 8 logvar)
        self.conv_out = WrappedConv2d(512, 16, 3, padding=1, causal=True)

        self.per_channel_statistics = PerChannelStatistics(128)

    def encode(self, mel: mx.array) -> mx.array:
        """Encode mel spectrogram to audio latent.

        Args:
            mel: (B, 2, T', 64) stereo mel spectrogram.

        Returns:
            Latent (B, 8, T, 16).
        """
        B, C, T_mel, M = mel.shape

        # Convert to NHWC: (B, T', 64, 2)
        x = mel.transpose(0, 2, 3, 1)

        x = self.conv_in(x)

        for blk in self.down:
            x = blk(x)

        x = self.mid(x)

        x = pixel_norm(x)
        x = nn.silu(x)
        x = self.conv_out(x)  # (B, T, 16, 16) — double_z

        # Take first 8 channels (mean), discard logvar
        # NHWC layout: last dim is channels, first 8 = mean
        B2, T, W, C_out = x.shape
        x = x[:, :, :, :8]  # (B, T, 16, 8) — mean only

        # Transpose to (c, f) order then flatten to (B, T, 128) for normalization
        x = x.transpose(0, 1, 3, 2)  # (B, T, 8, 16) — c first, f second
        x_flat = x.reshape(B2, T, 8 * 16)  # (B, T, 128) — c*16 + f order

        # Normalize using per-channel statistics
        mean = self.per_channel_statistics.mean_of_means.reshape(1, 1, -1)
        std = self.per_channel_statistics.std_of_means.reshape(1, 1, -1)
        x_flat = (x_flat - mean) / (std + 1e-8)

        # Reshape: (B, T, 128) -> (B, T, 8, 16) -> (B, 8, T, 16)
        x = x_flat.reshape(B2, T, 8, 16)  # (B, T, C=8, F=16) in (c, f) order
        return x.transpose(0, 2, 1, 3)  # (B, 8, T, 16) — BCTHW output
