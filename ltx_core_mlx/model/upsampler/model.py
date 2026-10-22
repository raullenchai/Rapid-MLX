"""LatentUpsampler — Conv3d/Conv2d ResBlocks with spatial or temporal upscale.

Neural upscaler for the two-stage pipeline: generates at lower resolution,
then upscales latents before refinement pass.

Supports three variants:
  - spatial_x2:   2x spatial via Conv2d + PixelShuffle2D(2)
  - spatial_x1_5: 1.5x spatial via Conv2d + PixelShuffle2D(3) + BlurDownsample(stride=2)
  - temporal_x2:  2x temporal via Conv3d + PixelShuffle3D(temporal=2)

Ported from ltx-core model/upsampler/.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Pixel shuffle helpers
# ---------------------------------------------------------------------------


def _pixel_shuffle_2d(x: mx.array, factor: int) -> mx.array:
    """2D pixel shuffle in BHWC layout.

    Matches PyTorch: rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)")
    MLX layout: (B, H, W, C*p1*p2) -> (B, H*p1, W*p2, C)
    """
    B, H, W, C_total = x.shape
    C = C_total // (factor * factor)
    # C is outermost (varies slowest), matching PyTorch (c, p1, p2) ordering
    x = x.reshape(B, H, W, C, factor, factor)
    # Interleave: (B, H, p1, W, p2, C)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    x = x.reshape(B, H * factor, W * factor, C)
    return x


def _pixel_shuffle_3d(
    x: mx.array,
    spatial_factor: int,
    temporal_factor: int,
) -> mx.array:
    """3D pixel shuffle in BDHWC layout.

    Matches PyTorch: rearrange(x, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)")
    MLX layout: (B, D, H, W, C*tf*sf*sf) -> (B, D*tf, H*sf, W*sf, C)
    """
    B, D, H, W, C_total = x.shape
    C = C_total // (spatial_factor * spatial_factor * temporal_factor)
    x = x.reshape(B, D, H, W, C, temporal_factor, spatial_factor, spatial_factor)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, D * temporal_factor, H * spatial_factor, W * spatial_factor, C)
    return x


# ---------------------------------------------------------------------------
# BlurDownsample — depthwise Conv2d with binomial kernel
# ---------------------------------------------------------------------------


def _blur_downsample(x: mx.array, kernel: mx.array, stride: int) -> mx.array:
    """Apply depthwise blur-then-downsample on BHWC tensor.

    Args:
        x: (B, H, W, C) input.
        kernel: (1, K, K, 1) binomial kernel in MLX OHWI format.
        stride: Downsampling stride.

    Returns:
        (B, H', W', C) downsampled output.
    """
    if stride == 1:
        return x

    B, H, W, C = x.shape
    K = kernel.shape[1]
    pad = K // 2

    # Depthwise convolution: convolve each channel with the same kernel
    # Reshape to (B*C, H, W, 1) to process channels independently
    x = x.transpose(0, 3, 1, 2)  # (B, C, H, W)
    x = x.reshape(B * C, H, W, 1)  # (B*C, H, W, 1)

    # Pad spatially
    x = mx.pad(x, [(0, 0), (pad, pad), (pad, pad), (0, 0)])

    # Apply conv2d with stride — kernel is (1, K, K, 1) = (O=1, K, K, I=1)
    x = mx.conv2d(x, kernel, stride=(stride, stride))  # (B*C, H', W', 1)

    _, H2, W2, _ = x.shape
    x = x.reshape(B, C, H2, W2)
    x = x.transpose(0, 2, 3, 1)  # (B, H', W', C)
    return x


# ---------------------------------------------------------------------------
# ResBlock — matches reference ltx-core res_block.py
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Residual block: conv1 -> norm1 -> SiLU -> conv2 -> norm2 -> SiLU(x + residual).

    Weight keys: conv1.weight/bias, conv2.weight/bias, norm1.weight/bias, norm2.weight/bias
    """

    def __init__(self, channels: int, dims: int = 3):
        super().__init__()
        if dims == 2:
            conv_cls = nn.Conv2d
        else:
            conv_cls = nn.Conv3d

        self.conv1 = conv_cls(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels, pytorch_compatible=True)
        self.conv2 = conv_cls(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = nn.silu(x + residual)
        return x


# ---------------------------------------------------------------------------
# Upsampler sub-module for rational resampling (spatial_x1_5)
# ---------------------------------------------------------------------------


class SpatialRationalResampler(nn.Module):
    """Rational spatial resampling: Conv2d -> PixelShuffle2D(num) -> BlurDownsample(den).

    For scale=1.5: num=3, den=2 (upsample 3x then downsample 2x = 1.5x).

    Weight keys:
        conv.weight, conv.bias — the Conv2d
        blur_down.kernel — the binomial kernel (1, K, K, 1)
    """

    def __init__(self, mid_channels: int, scale: float = 1.5):
        super().__init__()
        mapping: dict[float, tuple[int, int]] = {
            0.75: (3, 4),
            1.5: (3, 2),
            2.0: (2, 1),
            4.0: (4, 1),
        }
        if scale not in mapping:
            raise ValueError(f"Unsupported scale {scale}. Choose from {list(mapping.keys())}")
        self.num, self.den = mapping[scale]
        self.conv = nn.Conv2d(mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1)

        # BlurDownsample stores just the kernel — loaded from weights
        self.blur_down = BlurDownsampleModule(stride=self.den)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, D, H, W, C) in BDHWC layout — applied per-frame."""
        B, D, H, W, C = x.shape
        x = x.reshape(B * D, H, W, C)
        x = self.conv(x)
        x = _pixel_shuffle_2d(x, self.num)
        x = self.blur_down(x)
        _, H2, W2, C2 = x.shape
        x = x.reshape(B, D, H2, W2, C2)
        return x


class BlurDownsampleModule(nn.Module):
    """Learnable-kernel blur downsample. Kernel loaded from weights.

    Weight key: kernel (shape 1, K, K, 1 in MLX OHWI format).
    """

    def __init__(self, stride: int = 2, kernel_size: int = 5):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        # Compute deterministic binomial kernel (non-learnable buffer).
        # Reference registers this as a buffer; we compute it here so the
        # module works even when weights don't include the kernel entry.
        k = [math.comb(kernel_size - 1, i) for i in range(kernel_size)]
        k_arr = mx.array(k, dtype=mx.float32)
        k2d = k_arr[:, None] * k_arr[None, :]  # outer product
        k2d = k2d / mx.sum(k2d)
        # MLX OHWI format (1, K, K, 1)
        self.kernel = k2d.reshape(1, kernel_size, kernel_size, 1)

    def __call__(self, x: mx.array) -> mx.array:
        return _blur_downsample(x, self.kernel, self.stride)


# ---------------------------------------------------------------------------
# Main LatentUpsampler
# ---------------------------------------------------------------------------


class LatentUpsampler(nn.Module):
    """Neural latent upsampler supporting spatial and temporal variants.

    Architecture:
        initial_conv -> initial_norm -> SiLU
        -> 4x ResBlock (res_blocks)
        -> variant-specific upsampler
        -> 4x ResBlock (post_upsample_res_blocks)
        -> final_conv

    Weight key structure:
        initial_conv.weight/bias
        initial_norm.weight/bias
        res_blocks.{0-3}.conv1/conv2/norm1/norm2.weight/bias
        upsampler.{variant-specific keys}
        post_upsample_res_blocks.{0-3}.conv1/conv2/norm1/norm2.weight/bias
        final_conv.weight/bias

    Upsampler weight keys by variant:
        spatial_x2:   upsampler.0.weight, upsampler.0.bias
        spatial_x1_5: upsampler.conv.weight/bias, upsampler.blur_down.kernel
        temporal_x2:  upsampler.0.weight, upsampler.0.bias

    Args:
        in_channels: Input/output latent channels (128).
        mid_channels: Hidden channel dimension.
        num_blocks_per_stage: Number of ResBlocks per stage.
        spatial_upsample: Whether to spatially upsample.
        temporal_upsample: Whether to temporally upsample.
        spatial_scale: Spatial scale factor (2.0 or 1.5).
        rational_resampler: Use rational resampler for non-integer scales.
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        self.spatial_scale = float(spatial_scale)
        self.rational_resampler = rational_resampler

        self.initial_conv = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels, pytorch_compatible=True)

        self.res_blocks = [ResBlock(mid_channels, dims=3) for _ in range(num_blocks_per_stage)]

        # Variant-specific upsampler
        # For spatial_x2 and temporal_x2, the reference uses nn.Sequential
        # which produces keys like upsampler.0.weight. We use a list to match.
        # For rational_resampler, the reference uses a named sub-module with
        # keys upsampler.conv.weight and upsampler.blur_down.kernel.
        self._upsampler_type: str
        if spatial_upsample and temporal_upsample:
            raise NotImplementedError("Combined spatial+temporal upsample not yet supported")
        elif spatial_upsample:
            if rational_resampler:
                self._upsampler_type = "rational"
                self.upsampler = SpatialRationalResampler(mid_channels, scale=spatial_scale)
            else:
                self._upsampler_type = "spatial_sequential"
                # List produces keys upsampler.0.weight/bias (matching nn.Sequential)
                self.upsampler = [nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1)]
        elif temporal_upsample:
            self._upsampler_type = "temporal_sequential"
            # List produces keys upsampler.0.weight/bias
            self.upsampler = [nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1)]
        else:
            raise ValueError("Either spatial_upsample or temporal_upsample must be True")

        self.post_upsample_res_blocks = [ResBlock(mid_channels, dims=3) for _ in range(num_blocks_per_stage)]

        self.final_conv = nn.Conv3d(mid_channels, in_channels, kernel_size=3, padding=1)

    def _apply_upsampler(self, x: mx.array) -> mx.array:
        """Apply variant-specific upsampler.

        Args:
            x: (B, D, H, W, C) in BDHWC layout.

        Returns:
            Upsampled tensor in BDHWC layout.
        """
        if self._upsampler_type == "spatial_sequential":
            # Conv2d per-frame + PixelShuffle2D(2)
            B, D, H, W, C = x.shape
            x = x.reshape(B * D, H, W, C)
            x = self.upsampler[0](x)
            x = _pixel_shuffle_2d(x, 2)
            _, H2, W2, C2 = x.shape
            x = x.reshape(B, D, H2, W2, C2)
            return x

        elif self._upsampler_type == "rational":
            # SpatialRationalResampler handles everything
            return self.upsampler(x)

        elif self._upsampler_type == "temporal_sequential":
            # Conv3d + PixelShuffle3D(temporal=2)
            x = self.upsampler[0](x)
            x = _pixel_shuffle_3d(x, spatial_factor=1, temporal_factor=2)
            return x

        else:
            raise ValueError(f"Unknown upsampler type: {self._upsampler_type}")

    def __call__(self, latent: mx.array) -> mx.array:
        """Upsample latent.

        Args:
            latent: (B, C, F, H, W) in PyTorch channel-first layout.

        Returns:
            Upsampled latent in (B, C, F, H', W') layout.
        """
        # BCFHW -> BFHWC for MLX
        x = latent.transpose(0, 2, 3, 4, 1)

        # Initial conv + norm + activation
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = nn.silu(x)

        # Pre-upsample residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Upsampler (variant-specific)
        x = self._apply_upsampler(x)

        # Remove first frame after temporal upsample
        # (first frame encodes one pixel frame in the VAE)
        if self.temporal_upsample:
            x = x[:, 1:, :, :, :]

        # Post-upsample residual blocks
        for block in self.post_upsample_res_blocks:
            x = block(x)

        # Final conv
        x = self.final_conv(x)

        # BFHWC -> BCFHW
        return x.transpose(0, 4, 1, 2, 3)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LatentUpsampler:
        """Create the right upsampler variant from an embedded config dict.

        Expected config keys (from embedded_config.json):
            in_channels, mid_channels, num_blocks_per_stage,
            spatial_upsample, temporal_upsample, spatial_scale,
            rational_resampler
        """
        return cls(
            in_channels=config.get("in_channels", 128),
            mid_channels=config.get("mid_channels", 512),
            num_blocks_per_stage=config.get("num_blocks_per_stage", 4),
            spatial_upsample=config.get("spatial_upsample", True),
            temporal_upsample=config.get("temporal_upsample", False),
            spatial_scale=config.get("spatial_scale", 2.0),
            rational_resampler=config.get("rational_resampler", False),
        )
