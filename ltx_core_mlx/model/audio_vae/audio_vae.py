"""Audio VAE Decoder — latent to mel spectrogram.

Ported from ltx-core/src/ltx_core/model/audio_vae/audio_vae.py

The audio VAE uses Conv2d (not Conv1d). The latent (B, 8, T, 16) is treated
as a 2D spatial tensor (B, T, 16, 8) in MLX NHWC layout, and the network
upsamples the frequency (16) dimension while keeping time (T) unchanged,
finally producing a (B, T, 64, 2) output reshaped to (B, 2, T, 64) mel.

Weight key structure (after stripping "audio_vae." prefix):
    conv_in.conv.{weight,bias}             — Conv2d(8, 512, 3x3)
    mid.block_1.conv{1,2}.conv.{w,b}       — ResBlock(512)
    mid.attn_1.{norm,q,k,v,proj_out}.*     — AttnBlock(512)
    mid.block_2.conv{1,2}.conv.{w,b}       — ResBlock(512)
    up.2.block.{0,1,2}.*                   — 3 ResBlocks(512), upsample 512
    up.2.attn.{0,1,2}.*                    — 3 AttnBlocks(512)
    up.2.upsample.conv.conv.{w,b}          — Upsample(512)
    up.1.block.{0,1,2}.*                   — 3 ResBlocks(512→256), upsample 256
    up.1.attn.{0,1,2}.*                    — 3 AttnBlocks(256)
    up.1.upsample.conv.conv.{w,b}          — Upsample(256)
    up.0.block.{0,1,2}.*                   — 3 ResBlocks(256→128), no upsample
    conv_out.conv.{weight,bias}            — Conv2d(128, 2, 3x3)
    per_channel_statistics._mean_of_means  — (128,)
    per_channel_statistics._std_of_means   — (128,)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def pixel_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """RMS normalization over the channel dimension (PixelNorm)."""
    return mx.fast.rms_norm(x, weight=None, eps=eps)


class WrappedConv2d(nn.Module):
    """Conv2d with optional causal (height-axis) padding.

    Causal mode pads height asymmetrically (all on top, none on bottom)
    matching reference CausalConv2d with causality_axis=HEIGHT.

    Weight keys become: ``<name>.conv.weight`` / ``<name>.conv.bias``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        causal: bool = False,
    ):
        super().__init__()
        self._causal = causal
        ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]

        if causal and ks > 1:
            # Causal: manual asymmetric height padding, symmetric width padding
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            )
            self._pad_h_top = ks - 1  # Full causal pad on top
            self._pad_h_bottom = 0
            pw = padding if isinstance(padding, int) else padding[1]
            self._pad_w = pw
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self._pad_h_top = 0
            self._pad_h_bottom = 0
            self._pad_w = 0

    def __call__(self, x: mx.array) -> mx.array:
        if self._causal and (self._pad_h_top > 0 or self._pad_w > 0):
            # NHWC layout: axis 0=B, 1=H, 2=W, 3=C
            x = mx.pad(x, [(0, 0), (self._pad_h_top, self._pad_h_bottom), (self._pad_w, self._pad_w), (0, 0)])
        return self.conv(x)


class AudioResBlock(nn.Module):
    """Residual block matching ``up.*.block.*.`` or ``mid.block_*`` weight keys.

    Keys produced:
        conv1.conv.{weight,bias}
        conv2.conv.{weight,bias}
        nin_shortcut.conv.{weight,bias}   (only when in != out channels)
    """

    def __init__(self, in_channels: int, out_channels: int | None = None, causal: bool = False):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = WrappedConv2d(in_channels, out_channels, 3, padding=1, causal=causal)
        self.conv2 = WrappedConv2d(out_channels, out_channels, 3, padding=1, causal=causal)
        if in_channels != out_channels:
            self.nin_shortcut = WrappedConv2d(in_channels, out_channels, 1, padding=0)
        else:
            self.nin_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C) in MLX NHWC layout."""
        residual = x
        x = pixel_norm(x)
        x = nn.silu(x)
        x = self.conv1(x)
        x = pixel_norm(x)
        x = nn.silu(x)
        x = self.conv2(x)
        if self.nin_shortcut is not None:
            residual = self.nin_shortcut(residual)
        return x + residual


class AudioAttnBlock(nn.Module):
    """Self-attention block for audio VAE.

    Weight keys: norm.{weight,bias}, q.conv.{w,b}, k.conv.{w,b}, v.conv.{w,b}, proj_out.conv.{w,b}
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, pytorch_compatible=True)
        self.q = WrappedConv2d(channels, channels, 1, padding=0)
        self.k = WrappedConv2d(channels, channels, 1, padding=0)
        self.v = WrappedConv2d(channels, channels, 1, padding=0)
        self.proj_out = WrappedConv2d(channels, channels, 1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        residual = x
        h = self.norm(x)

        q = self.q(h).reshape(B, H * W, C)
        k = self.k(h).reshape(B, H * W, C)
        v = self.v(h).reshape(B, H * W, C)

        scale = C**-0.5
        attn = (q @ k.transpose(0, 2, 1)) * scale
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).reshape(B, H, W, C)
        out = self.proj_out(out)
        return residual + out


class AudioUpsample(nn.Module):
    """2x spatial upsample via nearest interpolation + Conv2d.

    In causal mode, drops first row after conv to maintain temporal alignment.

    Key: ``upsample.conv.conv.{weight,bias}``
    """

    def __init__(self, channels: int, causal: bool = False):
        super().__init__()
        self.conv = WrappedConv2d(channels, channels, 3, padding=1, causal=causal)
        self._causal = causal

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C)"""
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        x = self.conv(x)
        if self._causal:
            x = x[:, 1:, :, :]  # Drop first row for causal alignment
        return x


class AudioUpBlock(nn.Module):
    """One decoder up-stage: N resblocks (with optional per-block attention) + optional upsample.

    Key prefix: ``up.<idx>.``
    Children:
        block.{0,1,...} — AudioResBlock
        attn.{0,1,...}  — AudioAttnBlock (optional)
        upsample        — AudioUpsample (optional)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        add_upsample: bool = False,
        add_attention: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.block = [
            AudioResBlock(in_channels if i == 0 else out_channels, out_channels, causal=causal)
            for i in range(num_blocks)
        ]
        if add_attention:
            self.attn = [AudioAttnBlock(out_channels) for _ in range(num_blocks)]
        else:
            self.attn = None
        self.upsample = AudioUpsample(out_channels, causal=causal) if add_upsample else None

    def __call__(self, x: mx.array) -> mx.array:
        for i, blk in enumerate(self.block):
            x = blk(x)
            if self.attn is not None:
                x = self.attn[i](x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class AudioMidBlock(nn.Module):
    """Mid block: resblock, optional attention, resblock.

    Keys: mid.block_1, mid.attn_1 (optional), mid.block_2.
    """

    def __init__(self, channels: int, causal: bool = False, add_attention: bool = False):
        super().__init__()
        self.block_1 = AudioResBlock(channels, causal=causal)
        self.attn_1 = AudioAttnBlock(channels) if add_attention else None
        self.block_2 = AudioResBlock(channels, causal=causal)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.block_1(x)
        if self.attn_1 is not None:
            x = self.attn_1(x)
        x = self.block_2(x)
        return x


class PerChannelStatistics(nn.Module):
    """Per-channel normalization statistics loaded from weights.

    Safetensors keys have underscore prefix: ``_mean_of_means``, ``_std_of_means``.
    MLX treats underscore-prefixed attrs as private, so we use public names
    and remap during weight loading.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mean_of_means = mx.zeros((channels,))
        self.std_of_means = mx.ones((channels,))


class AudioVAEDecoder(nn.Module):
    """Audio VAE decoder: latent (B, 8, T, 16) -> mel (B, 2, T', 64).

    Architecture (reverse order of ``up`` indices — up.2 runs first):
        conv_in  : Conv2d(8, 512, 3)
        mid      : ResBlock(512) + AttnBlock(512) + ResBlock(512)
        up.2     : 3x ResBlock(512) + Upsample  → freq 16→32
        up.1     : 3x ResBlock(512→256) + Upsample → freq 32→64
        up.0     : 3x ResBlock(256→128), no upsample
        conv_out : Conv2d(128, 2, 3)
    """

    def __init__(self):
        super().__init__()
        # conv_in: 8 input channels (latent C1 dim)
        self.conv_in = WrappedConv2d(8, 512, 3, padding=1, causal=True)

        # Mid — config: mid_block_add_attention=False for distilled model
        self.mid = AudioMidBlock(512, causal=True, add_attention=False)

        # Up blocks — stored in list indexed [0, 1, 2] but run in REVERSE order
        # Config: attn_resolutions=[] → no attention in any up block
        # up.0: 256→128, no upsample
        # up.1: 512→256, upsample
        # up.2: 512→512, upsample
        self.up = [
            AudioUpBlock(256, 128, num_blocks=3, add_upsample=False, add_attention=False, causal=True),
            AudioUpBlock(512, 256, num_blocks=3, add_upsample=True, add_attention=False, causal=True),
            AudioUpBlock(512, 512, num_blocks=3, add_upsample=True, add_attention=False, causal=True),
        ]

        # Output
        self.conv_out = WrappedConv2d(128, 2, 3, padding=1, causal=True)

        # Per-channel normalization for latent denormalization
        self.per_channel_statistics = PerChannelStatistics(128)

    def decode(self, latent: mx.array) -> mx.array:
        """Decode audio latent to mel spectrogram.

        Args:
            latent: (B, 8, T, 16) audio latent.

        Returns:
            Mel spectrogram (B, 2, T', 64).
        """
        B, C1, T, C2 = latent.shape  # (B, 8, T, 16)

        # Flatten to (B, T, 128) for denormalization
        x_flat = latent.transpose(0, 2, 1, 3).reshape(B, T, C1 * C2)

        # Denormalize using per-channel statistics
        mean = self.per_channel_statistics.mean_of_means.reshape(1, 1, -1)
        std = self.per_channel_statistics.std_of_means.reshape(1, 1, -1)
        x_flat = x_flat * std + mean

        # Reshape back to 2D spatial: (B, T, 16, 8) — NHWC for Conv2d
        # T = height, 16 = width (frequency), 8 = channels
        x = x_flat.reshape(B, T, C1, C2).transpose(0, 1, 3, 2)  # (B, T, 8, 16) → (B, T, 16, 8) NHWC

        # Encoder/Decoder
        x = self.conv_in(x)  # (B, T, 16, 512)
        x = self.mid(x)

        # Up blocks run in reverse index order: up.2, up.1, up.0
        for i in reversed(range(len(self.up))):
            x = self.up[i](x)

        x = pixel_norm(x)  # norm_out
        x = nn.silu(x)
        x = self.conv_out(x)  # (B, T', 64, 2) in NHWC

        # Convert to (B, 2, T', 64)
        x = x.transpose(0, 3, 1, 2)
        return x
