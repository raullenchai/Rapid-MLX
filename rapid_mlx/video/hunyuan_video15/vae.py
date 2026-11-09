# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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

# Modified for Rapid-MLX: native MLX inference port (decoder only for VAE).
"""Causal HunyuanVideo 1.5 VAE decoder and upstream spatial tiling/blending."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class Conv3d(nn.Module):
    """BCFHW interface, MLX OTHWI checkpoint layout."""

    def __init__(self, inputs, outputs, kernel=1):
        super().__init__()
        self.weight = mx.zeros((outputs, kernel, kernel, kernel, inputs))
        self.bias = mx.zeros((outputs,))

    def __call__(self, x):
        return (
            mx.conv3d(x.transpose(0, 2, 3, 4, 1), self.weight) + self.bias
        ).transpose(0, 4, 1, 2, 3)


class CausalConv(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv = Conv3d(inputs, outputs, 3)

    def __call__(self, x):
        return self.conv(
            mx.pad(x, [(0, 0), (0, 0), (2, 0), (1, 1), (1, 1)], mode="edge")
        )


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = mx.ones((dim, 1, 1, 1))

    def __call__(self, x):
        y = x.astype(mx.float32)
        y = y / mx.maximum(mx.sqrt(mx.sum(y * y, axis=1, keepdims=True)), 1e-12)
        return y.astype(x.dtype) * math.sqrt(x.shape[1]) * self.gamma


class ResnetBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(inputs), RMSNorm(outputs)
        self.conv1, self.conv2 = (
            CausalConv(inputs, outputs),
            CausalConv(outputs, outputs),
        )
        self.conv_shortcut = Conv3d(inputs, outputs) if inputs != outputs else None

    def __call__(self, x):
        h = self.conv1(nn.silu(self.norm1(x)))
        mx.eval(h)
        h = self.conv2(nn.silu(self.norm2(h)))
        result = h + (self.conv_shortcut(x) if self.conv_shortcut is not None else x)
        mx.eval(result)
        return result


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_q, self.to_k, self.to_v, self.proj_out = (
            Conv3d(dim, dim) for _ in range(4)
        )

    def __call__(self, x):
        h = self.norm(x)
        b, c, t, height, width = x.shape

        def flatten(y):
            return y.reshape(b, c, -1).transpose(0, 2, 1)[:, None]

        q, k, v = flatten(self.to_q(h)), flatten(self.to_k(h)), flatten(self.to_v(h))
        frames = mx.arange(t * height * width) // (height * width)
        # Every pixel can attend to the whole current frame and all prior frames.
        mask = frames[:, None] >= frames[None, :]
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5, mask=mask)
        return x + self.proj_out(h[:, 0].transpose(0, 2, 1).reshape(x.shape))


class MidBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.resnets = [ResnetBlock(dim, dim), ResnetBlock(dim, dim)]
        self.attentions = [AttentionBlock(dim)]

    def __call__(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


def rearrange(x, temporal):
    b, packed, t, h, w = x.shape
    c = packed // (temporal * 4)
    return (
        x.reshape(b, temporal, 2, 2, c, t, h, w)
        .transpose(0, 4, 5, 1, 6, 2, 7, 3)
        .reshape(b, c, t * temporal, h * 2, w * 2)
    )


class Upsample(nn.Module):
    def __init__(self, inputs, outputs, temporal):
        super().__init__()
        self.conv = CausalConv(inputs, outputs * (8 if temporal else 4))
        self.temporal = temporal
        self.repeats = outputs * (8 if temporal else 4) // inputs

    def __call__(self, x):
        h = self.conv(x)
        if self.temporal:
            first = rearrange(h[:, :, :1], 1)
            first = first[:, : first.shape[1] // 2]
            rest = rearrange(h[:, :, 1:], 2)
            h = mx.concatenate([first, rest], axis=2)
            shortcut = mx.concatenate(
                [
                    mx.repeat(rearrange(x[:, :, :1], 1), self.repeats // 2, axis=1),
                    mx.repeat(rearrange(x[:, :, 1:], 2), self.repeats, axis=1),
                ],
                axis=2,
            )
        else:
            h = rearrange(h, 1)
            shortcut = rearrange(mx.repeat(x, self.repeats, axis=1), 1)
        result = h + shortcut
        mx.eval(result)
        return result


class UpBlock(nn.Module):
    def __init__(self, inputs, outputs, layers, upsample_outputs, temporal):
        super().__init__()
        self.resnets = [
            ResnetBlock(inputs if i == 0 else outputs, outputs) for i in range(layers)
        ]
        self.upsamplers = (
            [Upsample(outputs, upsample_outputs, temporal)]
            if upsample_outputs is not None
            else []
        )

    def __call__(self, x):
        for layer in [*self.resnets, *self.upsamplers]:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = list(reversed(config["block_out_channels"]))
        inputs = config["latent_channels"]
        self.repeat = channels[0] // inputs
        self.conv_in = CausalConv(inputs, channels[0])
        self.mid_block = MidBlock(channels[0])
        self.up_blocks = []
        previous = channels[0]
        for i, outputs in enumerate(channels):
            up = i < math.log2(config["spatial_compression_ratio"])
            up_outputs = (
                (channels[i + 1] if config["upsample_match_channel"] else outputs)
                if up
                else None
            )
            self.up_blocks.append(
                UpBlock(
                    previous,
                    outputs,
                    config["layers_per_block"] + 1,
                    up_outputs,
                    i < math.log2(config["temporal_compression_ratio"]),
                )
            )
            previous = up_outputs or outputs
        self.norm_out = RMSNorm(channels[-1])
        self.conv_out = CausalConv(channels[-1], config["out_channels"])

    def __call__(self, x, cancel_check=lambda: None):
        x = self.conv_in(x) + mx.repeat(x, self.repeat, axis=1)
        x = self.mid_block(x)
        mx.eval(x)
        for block in self.up_blocks:
            cancel_check()
            x = block(x)
            mx.eval(x)
        result = self.conv_out(nn.silu(self.norm_out(x)))
        mx.eval(result)
        return result


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = Decoder(config)

    def decode(self, z, cancel_check=lambda: None):
        return self.decoder(z, cancel_check)

    def tiled_decode(
        self, z, cancel_check=lambda: None, on_progress=None, tile_sample_size=256
    ):
        """Diffusers 25% overlap tiles; materialize each tile on the host.

        Temporal attention sees the entire clip. Independently decoding time
        windows would change the causal attention and first-frame upsampling.
        """
        scale = self.config["spatial_compression_ratio"]
        tile = tile_sample_size // scale
        stride = int(tile * 0.75)
        blend = tile_sample_size // 4
        limit = tile_sample_size - blend
        height, width = z.shape[-2:]
        output = np.empty(
            (
                z.shape[0],
                self.config["out_channels"],
                (z.shape[2] - 1) * self.config["temporal_compression_ratio"] + 1,
                height * scale,
                width * scale,
            ),
            dtype=np.float32,
        )
        previous_row = None
        total = math.ceil(height / stride) * math.ceil(width / stride)
        done = 0
        for row, i in enumerate(range(0, height, stride)):
            current = []
            for col, j in enumerate(range(0, width, stride)):
                cancel_check()
                decoded = self.decoder(
                    z[:, :, :, i : i + tile, j : j + tile], cancel_check
                )
                mx.eval(decoded)
                tile_pixels = np.array(decoded.astype(mx.float32))
                del decoded
                mx.clear_cache()
                if previous_row is not None:
                    extent = min(
                        blend, previous_row[col].shape[-2], tile_pixels.shape[-2]
                    )
                    alpha = (np.arange(extent, dtype=np.float32) / extent).reshape(
                        1, 1, 1, -1, 1
                    )
                    tile_pixels[..., :extent, :] = (
                        previous_row[col][..., -extent:, :] * (1 - alpha)
                        + tile_pixels[..., :extent, :] * alpha
                    )
                if col:
                    extent = min(blend, current[-1].shape[-1], tile_pixels.shape[-1])
                    alpha = np.arange(extent, dtype=np.float32) / extent
                    tile_pixels[..., :extent] = (
                        current[-1][..., -extent:] * (1 - alpha)
                        + tile_pixels[..., :extent] * alpha
                    )
                current.append(tile_pixels)
                h, w = (
                    min(limit, tile_pixels.shape[-2]),
                    min(limit, tile_pixels.shape[-1]),
                )
                output[
                    ..., row * limit : row * limit + h, col * limit : col * limit + w
                ] = tile_pixels[..., :h, :w]
                done += 1
                if on_progress:
                    on_progress("decoding", done, total)
            previous_row = current
        return output
