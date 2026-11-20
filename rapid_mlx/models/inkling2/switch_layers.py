# Copyright © 2023 Apple Inc.
# SPDX-License-Identifier: MIT

import math

import mlx.core as mx
import mlx.nn as nn


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


def _quantize_experts(w, *, group_size, bits, mode):
    """Quantize each expert with its own nvfp4 tensor scale."""
    if mode != "nvfp4":
        raise ValueError(f"A global scale needs 'nvfp4' mode, got '{mode}'.")
    gs = mx.max(mx.abs(w), axis=(-2, -1)).astype(mx.float32)
    qs = [
        mx.quantize(w[e], group_size, bits, mode=mode, global_scale=gs[e])
        for e in range(w.shape[0])
    ]
    return mx.stack([q for q, _ in qs]), mx.stack([s for _, s in qs]), gs


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        global_scale: bool = False,
    ):
        super().__init__()

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        scale = math.sqrt(1 / input_dims)
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )
        self._quantize(weight, global_scale)
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))
        # Freeze this model's parameters
        self.freeze()

    def _quantize(self, weight, global_scale=False):
        if global_scale:
            self.weight, self.scales, self.global_scale = _quantize_experts(
                weight, group_size=self.group_size, bits=self.bits, mode=self.mode
            )
            self.biases = None
        else:
            self.weight, self.scales, *biases = mx.quantize(
                weight, self.group_size, self.bits, mode=self.mode
            )
            self.biases = biases[0] if biases else None
            self.global_scale = None

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            global_scale=self.global_scale,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        global_scale: bool = False,
    ):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql._quantize(self.weight, global_scale)

        if "bias" in self:
            ql.bias = self.bias
        return ql
