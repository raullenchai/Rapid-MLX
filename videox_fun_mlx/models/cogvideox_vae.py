# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VAE foundation layers for CogVideoX-Fun, ported to MLX.

All tensors use channels-last layout: (B, D, H, W, C).
Original PyTorch code uses channels-first: (B, C, D, H, W).
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_arsenal.spatial import upsample_nearest, interpolate_3d, avg_pool1d, replicate_pad


class CogVideoXCausalConv3d(nn.Module):
    """A 3D causal convolution layer that pads the input tensor to ensure causality.

    Args:
        in_channels: Number of channels in the input tensor.
        out_channels: Number of output channels produced by the convolution.
        kernel_size: Kernel size of the convolutional kernel.
        stride: Stride of the convolution.
        dilation: Dilation rate of the convolution.
        pad_mode: Padding mode ("constant" or "replicate").
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        self.height_pad = (height_kernel_size - 1) // 2
        self.width_pad = (width_kernel_size - 1) // 2
        self.time_pad = time_kernel_size - 1
        self.time_kernel_size = time_kernel_size

        stride = stride if isinstance(stride, tuple) else (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=True,
        )

    def __call__(self, inputs: mx.array, conv_cache: Optional[mx.array] = None) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass.

        Args:
            inputs: (B, D, H, W, C) channels-last tensor.
            conv_cache: Optional cached temporal frames from previous call.

        Returns:
            Tuple of (output, conv_cache). conv_cache is None for replicate mode.
        """
        new_cache = None

        if self.pad_mode == "replicate":
            inputs = replicate_pad(
                inputs,
                [
                    (0, 0),
                    (self.time_pad, 0),
                    (self.height_pad, self.height_pad),
                    (self.width_pad, self.width_pad),
                    (0, 0),
                ],
            )
        else:
            # Constant pad mode with cache support
            if self.time_kernel_size > 1:
                if conv_cache is not None:
                    cached = [conv_cache]
                else:
                    cached = [mx.repeat(inputs[:, :1], self.time_pad, axis=1)]
                inputs = mx.concatenate(cached + [inputs], axis=1)

            new_cache = inputs[:, -self.time_kernel_size + 1 :] if self.time_kernel_size > 1 else None

            # Spatial padding
            if self.height_pad > 0 or self.width_pad > 0:
                inputs = mx.pad(
                    inputs,
                    [
                        (0, 0),
                        (0, 0),
                        (self.height_pad, self.height_pad),
                        (self.width_pad, self.width_pad),
                        (0, 0),
                    ],
                )

        output = self.conv(inputs)
        return output, new_cache


class CogVideoXSpatialNorm3D(nn.Module):
    """Spatially conditioned normalization for 3D video data.

    See https://arxiv.org/abs/2209.09002.

    Args:
        f_channels: Number of channels for input to group norm and output.
        zq_channels: Number of channels for the quantized vector.
        groups: Number of groups for group normalization.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_groups=groups, dims=f_channels, pytorch_compatible=True)
        self.conv_y = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)
        self.conv_b = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)

    def __call__(
        self,
        f: mx.array,
        zq: mx.array,
        conv_cache: Optional[Dict[str, mx.array]] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Forward pass.

        Args:
            f: Feature tensor (B, D, H, W, C).
            zq: Quantized tensor (B, D', H', W', C_zq).
            conv_cache: Optional dict of conv caches.

        Returns:
            Tuple of (output, new_conv_cache).
        """
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        # Resize zq to match f's spatial/temporal dimensions
        # In NDHWC layout, spatial dims are indices 1,2,3
        f_d, f_h, f_w = f.shape[1], f.shape[2], f.shape[3]

        if f_d > 1 and f_d % 2 == 1:
            # Split first frame and rest, resize separately
            f_first_shape = (1, f_h, f_w)
            f_rest_shape = (f_d - 1, f_h, f_w)

            zq_first = zq[:, :1]
            zq_rest = zq[:, 1:]

            zq_first = interpolate_3d(zq_first, f_first_shape)
            zq_rest = interpolate_3d(zq_rest, f_rest_shape)
            zq = mx.concatenate([zq_first, zq_rest], axis=1)
        else:
            zq = interpolate_3d(zq, (f_d, f_h, f_w))

        conv_y, new_conv_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_conv_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))

        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        return new_f, new_conv_cache


class CogVideoXUpsample3D(nn.Module):
    """A 3D upsampling layer for CogVideoX.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to input.
        compress_time: Whether to upsample the time dimension as well.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time: bool = False,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.compress_time = compress_time

    def __call__(self, inputs: mx.array) -> mx.array:
        """Forward pass.

        Args:
            inputs: (B, D, H, W, C) channels-last tensor.

        Returns:
            Upsampled tensor (B, D', H', W', C_out).
        """
        if self.compress_time:
            if inputs.shape[1] > 1 and inputs.shape[1] % 2 == 1:
                # Split first frame, upsample separately
                x_first = inputs[:, 0]  # (B, H, W, C)
                x_rest = inputs[:, 1:]  # (B, D-1, H, W, C)

                x_first = upsample_nearest(x_first, scale_factor=2)  # (B, 2H, 2W, C)
                B, D_rest, H_rest, W_rest, C = x_rest.shape
                x_rest = x_rest.reshape(B * D_rest, H_rest, W_rest, C)
                x_rest = upsample_nearest(x_rest, scale_factor=2)
                x_rest = x_rest.reshape(B, D_rest, x_rest.shape[1], x_rest.shape[2], C)

                # Temporal upsample the rest by 2x
                x_rest = mx.repeat(x_rest, 2, axis=1)

                x_first = mx.expand_dims(x_first, axis=1)  # (B, 1, 2H, 2W, C)
                inputs = mx.concatenate([x_first, x_rest], axis=1)
            elif inputs.shape[1] > 1:
                # Full 3D upsample (spatial + temporal)
                inputs = upsample_nearest(inputs, scale_factor=2)
            else:
                # Single frame: spatial-only upsample
                x = inputs[:, 0]  # (B, H, W, C)
                x = upsample_nearest(x, scale_factor=2)
                inputs = mx.expand_dims(x, axis=1)
        else:
            # Spatial-only 2x upsample
            B, D, H, W, C = inputs.shape
            inputs = inputs.reshape(B * D, H, W, C)
            inputs = upsample_nearest(inputs, scale_factor=2)
            inputs = inputs.reshape(B, D, inputs.shape[1], inputs.shape[2], C)

        # Apply 2D conv to each frame
        B, D, H, W, C = inputs.shape
        inputs = inputs.reshape(B * D, H, W, C)
        inputs = self.conv(inputs)
        inputs = inputs.reshape(B, D, inputs.shape[1], inputs.shape[2], inputs.shape[3])

        return inputs


def _get_activation(name: str):
    """Get activation function by name."""
    if name in ("swish", "silu"):
        return nn.silu
    elif name == "mish":
        return nn.mish
    elif name == "gelu":
        return nn.gelu
    elif name == "relu":
        return nn.relu
    else:
        raise ValueError(f"Unknown activation: {name}")


class CogVideoXResnetBlock3D(nn.Module):
    """A 3D ResNet block used in the CogVideoX model.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (defaults to in_channels).
        dropout: Dropout rate.
        temb_channels: Number of time embedding channels.
        groups: Number of groups for group normalization.
        eps: Epsilon for normalization layers.
        non_linearity: Activation function name.
        conv_shortcut: Whether to use a convolution shortcut.
        spatial_norm_dim: Dimension for spatial norm (if used instead of group norm).
        pad_mode: Padding mode for causal convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        conv_shortcut: bool = False,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = _get_activation(non_linearity)
        self.use_conv_shortcut = conv_shortcut
        self.spatial_norm_dim = spatial_norm_dim

        if spatial_norm_dim is None:
            self.norm1 = nn.GroupNorm(num_groups=groups, dims=in_channels, pytorch_compatible=True)
            self.norm2 = nn.GroupNorm(num_groups=groups, dims=out_channels, pytorch_compatible=True)
        else:
            self.norm1 = CogVideoXSpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )
            self.norm2 = CogVideoXSpatialNorm3D(
                f_channels=out_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
            )

        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
        )

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CogVideoXCausalConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    pad_mode=pad_mode,
                )
            else:
                self.conv_shortcut = nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def __call__(
        self,
        inputs: mx.array,
        temb: Optional[mx.array] = None,
        zq: Optional[mx.array] = None,
        conv_cache: Optional[Dict[str, mx.array]] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Forward pass.

        Args:
            inputs: (B, D, H, W, C) tensor.
            temb: Optional time embedding (B, temb_channels).
            zq: Optional spatial norm conditioning tensor.
            conv_cache: Optional dict of conv caches.

        Returns:
            Tuple of (output, new_conv_cache).
        """
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states = inputs

        if zq is not None:
            hidden_states, new_conv_cache["norm1"] = self.norm1(hidden_states, zq, conv_cache=conv_cache.get("norm1"))
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states, new_conv_cache["conv1"] = self.conv1(hidden_states, conv_cache=conv_cache.get("conv1"))

        if temb is not None:
            # temb is (B, temb_channels), project and broadcast to (B, 1, 1, 1, C)
            hidden_states = hidden_states + self.temb_proj(self.nonlinearity(temb)).reshape(temb.shape[0], 1, 1, 1, -1)

        if zq is not None:
            hidden_states, new_conv_cache["norm2"] = self.norm2(hidden_states, zq, conv_cache=conv_cache.get("norm2"))
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states, new_conv_cache["conv2"] = self.conv2(hidden_states, conv_cache=conv_cache.get("conv2"))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                inputs, new_conv_cache["conv_shortcut"] = self.conv_shortcut(
                    inputs, conv_cache=conv_cache.get("conv_shortcut")
                )
            else:
                inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states, new_conv_cache


class CogVideoXDownsample3D(nn.Module):
    """A 3D downsampling layer using Conv2d per frame + optional temporal avg pool.

    Mirrors diffusers ``CogVideoXDownsample3D`` exactly: uses Conv2d for spatial
    downsampling and avg_pool1d for temporal compression.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel (default 3).
        stride: Spatial stride (default 2).
        padding: Padding (default 0).
        compress_time: Whether to also halve the temporal dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ) -> None:
        super().__init__()
        self.compress_time = compress_time
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        """Forward pass.

        Args:
            inputs: (B, D, H, W, C) channels-last tensor.

        Returns:
            Downsampled tensor.
        """
        B, D, H, W, C = inputs.shape

        if self.compress_time:
            # Temporal compression via avg_pool1d
            # (B, D, H, W, C) -> (B*H*W, D, C) -> avg_pool -> reshape back
            x = inputs.transpose(0, 2, 3, 1, 4)  # (B, H, W, D, C)
            x = x.reshape(B * H * W, D, C)

            if D % 2 == 1:
                # Keep first frame, pool the rest
                x_first = x[:, :1]
                x_rest = x[:, 1:]
                if x_rest.shape[1] > 0:
                    x_rest = avg_pool1d(x_rest, kernel_size=2, stride=2)
                x = mx.concatenate([x_first, x_rest], axis=1)
            else:
                x = avg_pool1d(x, kernel_size=2, stride=2)

            new_D = x.shape[1]
            x = x.reshape(B, H, W, new_D, C).transpose(0, 3, 1, 2, 4)  # (B, D', H, W, C)
            inputs = x
            B, D, H, W, C = inputs.shape

        # Spatial pad: (0, 1, 0, 1) on H and W — matching diffusers F.pad(x, (0,1,0,1))
        inputs = mx.pad(inputs, [(0, 0), (0, 0), (0, 1), (0, 1), (0, 0)])

        # Apply Conv2d per frame: (B, D, H, W, C) -> (B*D, H, W, C) -> Conv2d -> reshape
        B, D, H, W, C = inputs.shape
        inputs = inputs.reshape(B * D, H, W, C)
        inputs = self.conv(inputs)
        _, H2, W2, C2 = inputs.shape
        inputs = inputs.reshape(B, D, H2, W2, C2)

        return inputs


class CogVideoXDownBlock3D(nn.Module):
    """A downsampling block used in the CogVideoX model.

    Contains a sequence of ResnetBlock3D layers followed by an optional
    CogVideoXDownsample3D.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        temb_channels: Number of time embedding channels.
        dropout: Dropout rate.
        num_layers: Number of resnet layers.
        resnet_eps: Epsilon for normalization layers.
        resnet_act_fn: Activation function name.
        resnet_groups: Number of groups for group normalization.
        add_downsample: Whether to add a downsampling layer.
        downsample_padding: Padding for the downsampler.
        compress_time: Whether to downsample the temporal dimension.
        pad_mode: Padding mode for causal convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
        compress_time: bool = False,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = resnets
        self.downsamplers = None

        if add_downsample:
            self.downsamplers = [
                CogVideoXDownsample3D(
                    out_channels,
                    out_channels,
                    padding=downsample_padding,
                    compress_time=compress_time,
                )
            ]

    def __call__(
        self,
        hidden_states: mx.array,
        temb: Optional[mx.array] = None,
        zq: Optional[mx.array] = None,
        conv_cache: Optional[Dict[str, mx.array]] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Forward pass.

        Args:
            hidden_states: (B, D, H, W, C) tensor.
            temb: Optional time embedding.
            zq: Optional spatial norm conditioning tensor.
            conv_cache: Optional dict of conv caches.

        Returns:
            Tuple of (output, new_conv_cache).
        """
        new_conv_cache: Dict[str, mx.array] = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
            )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states, new_conv_cache


class CogVideoXMidBlock3D(nn.Module):
    """A middle block used in the CogVideoX model.

    Contains a sequence of ResnetBlock3D layers with no up/downsampling.

    Args:
        in_channels: Number of input channels.
        temb_channels: Number of time embedding channels.
        dropout: Dropout rate.
        num_layers: Number of resnet layers.
        resnet_eps: Epsilon for normalization layers.
        resnet_act_fn: Activation function name.
        resnet_groups: Number of groups for group normalization.
        spatial_norm_dim: Dimension for spatial norm (if used).
        pad_mode: Padding mode for causal convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    spatial_norm_dim=spatial_norm_dim,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )
        self.resnets = resnets

    def __call__(
        self,
        hidden_states: mx.array,
        temb: Optional[mx.array] = None,
        zq: Optional[mx.array] = None,
        conv_cache: Optional[Dict[str, mx.array]] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Forward pass.

        Args:
            hidden_states: (B, D, H, W, C) tensor.
            temb: Optional time embedding.
            zq: Optional spatial norm conditioning tensor.
            conv_cache: Optional dict of conv caches.

        Returns:
            Tuple of (output, new_conv_cache).
        """
        new_conv_cache: Dict[str, mx.array] = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
            )

        return hidden_states, new_conv_cache


class CogVideoXUpBlock3D(nn.Module):
    """An upsampling block used in the CogVideoX model.

    Contains a sequence of ResnetBlock3D layers followed by an optional
    CogVideoXUpsample3D.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        temb_channels: Number of time embedding channels.
        dropout: Dropout rate.
        num_layers: Number of resnet layers.
        resnet_eps: Epsilon for normalization layers.
        resnet_act_fn: Activation function name.
        resnet_groups: Number of groups for group normalization.
        spatial_norm_dim: Dimension for spatial norm (if used).
        add_upsample: Whether to add an upsampling layer.
        upsample_padding: Padding for the upsampler conv.
        compress_time: Whether to upsample the temporal dimension.
        pad_mode: Padding mode for causal convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: int = 16,
        add_upsample: bool = True,
        upsample_padding: int = 1,
        compress_time: bool = False,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_norm_dim=spatial_norm_dim,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = resnets
        self.upsamplers = None

        if add_upsample:
            self.upsamplers = [
                CogVideoXUpsample3D(
                    out_channels,
                    out_channels,
                    padding=upsample_padding,
                    compress_time=compress_time,
                )
            ]

    def __call__(
        self,
        hidden_states: mx.array,
        temb: Optional[mx.array] = None,
        zq: Optional[mx.array] = None,
        conv_cache: Optional[Dict[str, mx.array]] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Forward pass.

        Args:
            hidden_states: (B, D, H, W, C) tensor.
            temb: Optional time embedding.
            zq: Optional spatial norm conditioning tensor.
            conv_cache: Optional dict of conv caches.

        Returns:
            Tuple of (output, new_conv_cache).
        """
        new_conv_cache: Dict[str, mx.array] = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states, new_conv_cache


class CogVideoXEncoder3D(nn.Module):
    """The encoder of a CogVideoX variational autoencoder.

    Architecture: conv_in -> down_blocks -> mid_block -> norm_out -> conv_out.
    Outputs ``2 * out_channels`` (mean + logvar).

    Args:
        in_channels: Number of input channels (e.g. 3 for RGB).
        out_channels: Number of latent channels.
        down_block_types: Tuple of down block type strings.
        block_out_channels: Tuple of output channels for each block.
        layers_per_block: Number of resnet layers per block.
        act_fn: Activation function name.
        norm_eps: Epsilon for normalization.
        norm_num_groups: Number of groups for group normalization.
        dropout: Dropout rate.
        pad_mode: Padding mode for causal convolutions.
        temporal_compression_ratio: Ratio of temporal compression.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()

        temporal_compress_level = int(math.log2(temporal_compression_ratio))

        self.conv_in = CogVideoXCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)
        self.down_blocks: List[CogVideoXDownBlock3D] = []

        # Down blocks
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if down_block_type != "CogVideoXDownBlock3D":
                raise ValueError("Invalid `down_block_type`. Must be `CogVideoXDownBlock3D`")

            down_block = CogVideoXDownBlock3D(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=0,
                dropout=dropout,
                num_layers=layers_per_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                add_downsample=not is_final_block,
                compress_time=compress_time,
            )
            self.down_blocks.append(down_block)

        # Mid block
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=block_out_channels[-1],
            temb_channels=0,
            dropout=dropout,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            pad_mode=pad_mode,
        )

        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], pytorch_compatible=True)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            block_out_channels[-1], 2 * out_channels, kernel_size=3, pad_mode=pad_mode
        )

    def __call__(
        self,
        sample: mx.array,
        temb: Optional[mx.array] = None,
        conv_cache: Optional[Dict[str, mx.array]] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Forward pass.

        Args:
            sample: (B, D, H, W, C) input tensor.
            temb: Optional time embedding.
            conv_cache: Optional dict of conv caches.

        Returns:
            Tuple of (output, new_conv_cache). Output has ``2 * out_channels``
            channels (mean + logvar).
        """
        new_conv_cache: Dict[str, mx.array] = {}
        conv_cache = conv_cache or {}

        hidden_states, new_conv_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        # 1. Down
        for i, down_block in enumerate(self.down_blocks):
            conv_cache_key = f"down_block_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = down_block(
                hidden_states, temb, None, conv_cache=conv_cache.get(conv_cache_key)
            )

        # 2. Mid
        hidden_states, new_conv_cache["mid_block"] = self.mid_block(
            hidden_states, temb, None, conv_cache=conv_cache.get("mid_block")
        )

        # 3. Post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states, new_conv_cache["conv_out"] = self.conv_out(hidden_states, conv_cache=conv_cache.get("conv_out"))

        return hidden_states, new_conv_cache


class CogVideoXDecoder3D(nn.Module):
    """The decoder of a CogVideoX variational autoencoder.

    Architecture: conv_in -> mid_block -> up_blocks -> norm_out -> conv_out.
    Uses spatial norm conditioning from the latent input throughout.

    Args:
        in_channels: Number of latent channels.
        out_channels: Number of output channels (e.g. 3 for RGB).
        up_block_types: Tuple of up block type strings.
        block_out_channels: Tuple of output channels for each block.
        layers_per_block: Number of resnet layers per block.
        act_fn: Activation function name.
        norm_eps: Epsilon for normalization.
        norm_num_groups: Number of groups for group normalization.
        dropout: Dropout rate.
        pad_mode: Padding mode for causal convolutions.
        temporal_compression_ratio: Ratio of temporal compression.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()

        reversed_block_out_channels = list(reversed(block_out_channels))

        self.conv_in = CogVideoXCausalConv3d(
            in_channels, reversed_block_out_channels[0], kernel_size=3, pad_mode=pad_mode
        )

        # Mid block (with spatial norm conditioned on latent)
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=reversed_block_out_channels[0],
            temb_channels=0,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            spatial_norm_dim=in_channels,
            pad_mode=pad_mode,
        )

        # Up blocks
        self.up_blocks: List[CogVideoXUpBlock3D] = []

        output_channel = reversed_block_out_channels[0]
        temporal_compress_level = int(math.log2(temporal_compression_ratio))

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if up_block_type != "CogVideoXUpBlock3D":
                raise ValueError("Invalid `up_block_type`. Must be `CogVideoXUpBlock3D`")

            up_block = CogVideoXUpBlock3D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=0,
                dropout=dropout,
                num_layers=layers_per_block + 1,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                spatial_norm_dim=in_channels,
                add_upsample=not is_final_block,
                compress_time=compress_time,
                pad_mode=pad_mode,
            )
            self.up_blocks.append(up_block)

        self.norm_out = CogVideoXSpatialNorm3D(reversed_block_out_channels[-1], in_channels, groups=norm_num_groups)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            reversed_block_out_channels[-1], out_channels, kernel_size=3, pad_mode=pad_mode
        )

    def __call__(
        self,
        sample: mx.array,
        temb: Optional[mx.array] = None,
        conv_cache: Optional[Dict[str, mx.array]] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Forward pass.

        Args:
            sample: (B, D, H, W, C) latent tensor.
            temb: Optional time embedding.
            conv_cache: Optional dict of conv caches.

        Returns:
            Tuple of (output, new_conv_cache).
        """
        new_conv_cache: Dict[str, mx.array] = {}
        conv_cache = conv_cache or {}

        hidden_states, new_conv_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        # 1. Mid
        hidden_states, new_conv_cache["mid_block"] = self.mid_block(
            hidden_states, temb, sample, conv_cache=conv_cache.get("mid_block")
        )

        # 2. Up
        for i, up_block in enumerate(self.up_blocks):
            conv_cache_key = f"up_block_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = up_block(
                hidden_states, temb, sample, conv_cache=conv_cache.get(conv_cache_key)
            )

        # 3. Post-process
        hidden_states, new_conv_cache["norm_out"] = self.norm_out(
            hidden_states, sample, conv_cache=conv_cache.get("norm_out")
        )
        hidden_states = self.conv_act(hidden_states)
        hidden_states, new_conv_cache["conv_out"] = self.conv_out(hidden_states, conv_cache=conv_cache.get("conv_out"))

        return hidden_states, new_conv_cache


class AutoencoderKLCogVideoX(nn.Module):
    """VAE model with KL loss for CogVideoX.

    Encodes video into latents and decodes latent representations back.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        scaling_factor: float = 1.15258426,
        shift_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        self.encoder = CogVideoXEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.decoder = CogVideoXDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )

    def encode(self, x: mx.array):
        from videox_fun_mlx.utils import DiagonalGaussianDistribution

        h, _ = self.encoder(x)
        return DiagonalGaussianDistribution(h)

    def decode(self, z: mx.array) -> mx.array:
        dec, _ = self.decoder(z)
        return dec

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, subfolder: str = None):
        """Load from mlx-forge converted dir or PyTorch HuggingFace dir."""
        import os
        from videox_fun_mlx.utils import load_config, load_mlx_weights

        # If subfolder specified, it's HuggingFace style (path/vae/)
        if subfolder:
            config_path = os.path.join(pretrained_model_path, subfolder)
        else:
            config_path = pretrained_model_path

        # Priority: vae_config.json > config.json["vae"] > config.json
        import json as _json

        vae_config_file = os.path.join(pretrained_model_path, "vae_config.json")
        if os.path.exists(vae_config_file):
            with open(vae_config_file) as f:
                config = _json.load(f)
        else:
            config = load_config(config_path)
            if "vae" in config and "latent_channels" not in config:
                config = config["vae"]

        init_keys = {
            "in_channels",
            "out_channels",
            "down_block_types",
            "up_block_types",
            "block_out_channels",
            "latent_channels",
            "layers_per_block",
            "act_fn",
            "norm_eps",
            "norm_num_groups",
            "temporal_compression_ratio",
            "scaling_factor",
            "shift_factor",
        }
        filtered_config = {k: v for k, v in config.items() if k in init_keys}
        for k in ("down_block_types", "up_block_types", "block_out_channels"):
            if k in filtered_config and isinstance(filtered_config[k], list):
                filtered_config[k] = tuple(filtered_config[k])

        model = cls(**filtered_config)
        weights = load_mlx_weights(pretrained_model_path, "vae")
        model.load_weights(list(weights.items()))
        return model
