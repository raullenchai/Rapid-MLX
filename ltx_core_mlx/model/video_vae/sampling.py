"""Up/downsampling operations for Video VAE.

Ported from ltx-core/src/ltx_core/model/video_vae/sampling.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.video_vae.convolution import Conv3dBlock

# ---------------------------------------------------------------------------
# Pixel-shuffle / space-to-depth helpers
# ---------------------------------------------------------------------------


def pixel_shuffle_3d(
    x: mx.array,
    spatial_factor: int,
    temporal_factor: int,
) -> mx.array:
    """Rearrange channels into spatial/temporal dimensions (depth-to-space).

    Matches: ``"b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)"``
    Channel split order: (c, p1=temporal, p2=height, p3=width) — c outermost.
    In BDHWC layout: C_total = C * tf * sf * sf where C varies slowest.

    Input:  (B, D, H, W, C * sf^2 * tf)
    Output: (B, D*tf, H*sf, W*sf, C)
    """
    B, D, H, W, C_total = x.shape
    C = C_total // (spatial_factor * spatial_factor * temporal_factor)
    x = x.reshape(B, D, H, W, C, temporal_factor, spatial_factor, spatial_factor)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, D * temporal_factor, H * spatial_factor, W * spatial_factor, C)
    return x


def unpatchify_spatial(
    x: mx.array,
    patch_size: int,
) -> mx.array:
    """Reverse spatial patchification: depth-to-space for the final VAE output.

    Matches the reference ``unpatchify`` from ltx-core ops.py:
        ``"b (c p r q) f h w -> b c (f p) (h q) (w r)"``
    with ``p=1, q=patch_size, r=patch_size``.

    Channel split order: (c, p=1, r=width, q=height) — note r (width) comes
    BEFORE q (height). This differs from ``pixel_shuffle_3d`` which uses
    (c, temporal, height, width). Using ``pixel_shuffle_3d`` for unpatchify
    swaps H/W sub-pixels and causes checkerboard artifacts.

    Input:  (B, F, H, W, C * patch_size^2)   (BFHWC, temporal patch=1)
    Output: (B, F, H*patch_size, W*patch_size, C)
    """
    B, F, H, W, C_total = x.shape
    ps = patch_size
    C = C_total // (ps * ps)
    # Split channels as (C, r_width, q_height) matching reference (c, p=1, r, q)
    x = x.reshape(B, F, H, W, C, ps, ps)
    # Indices: B=0, F=1, H=2, W=3, C=4, r_W=5, q_H=6
    # Target:  (B, F, H, q_H, W, r_W, C) -> (B, F, H*ps, W*ps, C)
    x = x.transpose(0, 1, 2, 6, 3, 5, 4)
    x = x.reshape(B, F, H * ps, W * ps, C)
    return x


def patchify_spatial(x: mx.array, patch_size: int = 4) -> mx.array:
    """Spatial patchification: space-to-depth rearrangement.

    Reference: ltx-core ops.py patchify with patch_size_hw=4, patch_size_t=1.
    einops: ``"b c (f p) (h q) (w r) -> b (c p r q) f h w"`` with p=1, q=4, r=4.

    Channel ordering: (c, p=1, r=patch_W, q=patch_H) -- c outermost.
    In BFHWC layout: splits H and W into patches, packs into channels.

    Args:
        x: (B, F, H, W, C) in BFHWC layout, C=3.
        patch_size: Spatial patch size (default 4).

    Returns:
        (B, F, H/ps, W/ps, C * ps * ps) in BFHWC layout.
    """
    B, F, H, W, C = x.shape
    ps = patch_size
    # Split spatial dims: (B, F, H//ps, ps_h, W//ps, ps_w, C)
    x = x.reshape(B, F, H // ps, ps, W // ps, ps, C)
    # Rearrange to channel order (C, r_W, q_H) matching reference (c, p, r, q)
    # From indices: 0=B, 1=F, 2=H//ps, 3=q_H, 4=W//ps, 5=r_W, 6=C
    # To: (B, F, H//ps, W//ps, C, r_W, q_H)
    x = x.transpose(0, 1, 2, 4, 6, 5, 3)
    return x.reshape(B, F, H // ps, W // ps, C * ps * ps)


def space_to_depth(
    x: mx.array,
    stride: tuple[int, int, int],
) -> mx.array:
    """Space-to-depth rearrangement for downsampling.

    Reference einops: ``"b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w"``
    Channel ordering: (c, p1=temporal, p2=H, p3=W) -- c outermost, p3 innermost.

    In BDHWC layout, produces (B, D, H, W, C * prod(stride)).

    Args:
        x: (B, D_full, H_full, W_full, C) in BDHWC layout.
        stride: (temporal_stride, height_stride, width_stride).

    Returns:
        (B, D, H, W, C * prod(stride)) with reference channel ordering.
    """
    B, D_full, H_full, W_full, C = x.shape
    st, sh, sw = stride
    D = D_full // st
    H = H_full // sh
    W = W_full // sw
    # Split dims: (B, D, st, H, sh, W, sw, C)
    x = x.reshape(B, D, st, H, sh, W, sw, C)
    # Rearrange to (B, D, H, W, C, st, sh, sw) -- C outermost in channel group
    x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
    return x.reshape(B, D, H, W, C * st * sh * sw)


# ---------------------------------------------------------------------------
# Upsample / downsample modules
# ---------------------------------------------------------------------------


class DepthToSpaceUpsample(nn.Module):
    """Convolution used as an upsample layer (depth-to-space).

    The conv may output more channels than the input (for pixel-shuffle
    spatial/temporal upsampling). The caller handles the rearrangement.

    Produces key: ``conv.conv.{weight,bias}``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.conv = Conv3dBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class SpaceToDepthDownsample(nn.Module):
    """Downsampling via space-to-depth with group-mean skip connection.

    Reference: ltx-core sampling.py SpaceToDepthDownsample.

    Two-branch architecture:
      - Skip: space-to-depth rearrange -> group channels -> mean -> out_channels
      - Conv: conv3d -> space-to-depth rearrange -> out_channels
      - Output = conv + skip

    Produces key: ``conv.conv.{weight,bias}``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
    ):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * int.__mul__(*stride[1:]) * stride[0] // out_channels
        conv_out_ch = out_channels // (stride[0] * stride[1] * stride[2])
        self.conv = Conv3dBlock(
            in_channels,
            conv_out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,  # Encoder always uses causal convolutions
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: x is (B, D, H, W, C) in MLX convention."""
        # Prepend first frame for temporal downsampling (causal padding)
        if self.stride[0] == 2:
            x = mx.concatenate([x[:, :1, :, :, :], x], axis=1)

        # Skip connection: space-to-depth -> group-mean
        x_in = space_to_depth(x, self.stride)
        if self.group_size > 1:
            B, D, H, W, C_total = x_in.shape
            C_out = C_total // self.group_size
            x_in = x_in.reshape(B, D, H, W, C_out, self.group_size)
            x_in = x_in.mean(axis=-1)

        # Conv branch: conv -> space-to-depth
        x_conv = self.conv(x)
        x_conv = space_to_depth(x_conv, self.stride)

        return x_conv + x_in
