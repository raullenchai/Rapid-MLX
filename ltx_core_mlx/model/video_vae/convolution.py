"""3D convolution with causal or non-causal temporal padding.

Ported from ltx-core/src/ltx_core/model/video_vae/convolution.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class Conv3dBlock(nn.Module):
    """3D convolution with causal or non-causal temporal padding.

    When causal=True: replicates first frame for temporal padding (front only).
    When causal=False: standard symmetric zero-padding on all dimensions
    (matching reference ``make_conv_nd`` with ``causal=False``).

    MLX Conv3d weight layout: (O, D, H, W, I)
    Produces key: ``conv.{weight,bias}``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 1,
        causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.spatial_padding = (padding[1], padding[2])
        self.spatial_padding_mode = spatial_padding_mode

        # Reference CausalConv3d: always created the same way, with causal
        # flag controlling runtime padding behavior. We handle padding manually
        # so Conv3d always gets padding=0.
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: x is (B, D, H, W, C) in MLX convention."""
        tk = self.kernel_size[0]
        if self.causal:
            # Causal: replicate first frame (kernel_size-1) times at front only
            if tk > 1:
                first_frame = mx.repeat(x[:, :1, :, :, :], tk - 1, axis=1)
                x = mx.concatenate([first_frame, x], axis=1)
        else:
            # Non-causal: symmetric replicate padding (first AND last frame)
            # Reference: first_frame_pad repeats (kernel_size-1)//2 times at front,
            #            last_frame_pad repeats (kernel_size-1)//2 times at back.
            pad_size = (tk - 1) // 2
            if pad_size > 0:
                first_pad = mx.repeat(x[:, :1, :, :, :], pad_size, axis=1)
                last_pad = mx.repeat(x[:, -1:, :, :, :], pad_size, axis=1)
                x = mx.concatenate([first_pad, x, last_pad], axis=1)

        # Spatial padding (always symmetric)
        sp_h, sp_w = self.spatial_padding
        if sp_h > 0 or sp_w > 0:
            if self.spatial_padding_mode == "reflect":
                # Manual reflect padding for BDHWC layout.
                # For pad size p, reflect takes pixels [1..p] and [-(1+p)..-1].
                if sp_h > 0:
                    x = mx.concatenate(
                        [x[:, :, 1 : 1 + sp_h, :, :], x, x[:, :, -(1 + sp_h) : -1, :, :]],
                        axis=2,
                    )
                if sp_w > 0:
                    x = mx.concatenate(
                        [x[:, :, :, 1 : 1 + sp_w, :], x, x[:, :, :, -(1 + sp_w) : -1, :]],
                        axis=3,
                    )
            else:
                x = mx.pad(x, [(0, 0), (0, 0), (sp_h, sp_h), (sp_w, sp_w), (0, 0)])

        return self.conv(x)
