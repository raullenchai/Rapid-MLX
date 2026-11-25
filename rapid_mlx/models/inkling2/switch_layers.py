# Copyright © 2023 Apple Inc.
# SPDX-License-Identifier: MIT

import math
import os

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


# MLX 0.32.3 added ``global_scale`` to ``gather_qmm``. Without it, nvfp4 is
# decoded with this fixed tensor scale (6 * 448).
_GATHER_QMM_HAS_GLOBAL_SCALE = "global_scale" in (mx.gather_qmm.__doc__ or "")
_NVFP4_DEFAULT_GLOBAL_SCALE = 2688.0
_NVFP4_DECODE_ENABLED = True


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
        self._nvfp4_decode = None
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
        if (
            _NVFP4_DECODE_ENABLED
            and os.environ.get("RAPID_MLX_INKLING2_FAST_DECODE", "1") != "0"
            and mx.default_device() == mx.gpu
            and sorted_indices is False
        ):
            launch = self._nvfp4_decode
            if (
                launch is None
                or x.shape != launch.x_shape
                or indices.shape != launch.indices_shape
                or indices.dtype != launch.index_dtype
                or x.dtype != mx.bfloat16
            ):
                launch = None
                if (
                    (self.mode, self.bits, self.group_size) == ("nvfp4", 4, 16)
                    and "bias" not in self
                    and self.get("biases") is None
                    and x.dtype == mx.bfloat16
                    and x.ndim >= 3
                    and x.shape[-2] == 1
                    and math.prod(x.shape[:-3]) == 1
                    and indices.ndim >= 1
                    and mx.issubdtype(indices.dtype, mx.integer)
                    and 1 <= indices.size <= 16
                    and indices.shape[-1] == indices.size
                    and x.shape[-3] in (1, indices.size)
                    and x.shape[-1] == self.input_dims
                    and x.shape[-1] % 16 == 0
                ):
                    from .nvfp4_decode import prepare

                    launch = self._nvfp4_decode = prepare(
                        x, self.weight, indices, self.global_scale
                    )
            if launch is not None and launch.kernel is not None:
                try:
                    return launch.kernel(
                        inputs=[
                            x,
                            self.weight,
                            self.scales,
                            indices,
                            self.global_scale if launch.has_scale else self.scales,
                        ],
                        **launch.kwargs,
                    )[0]
                except Exception:
                    # Keep invocation failures on the stock path on later tokens too.
                    launch.kernel = None
        global_scale = self.global_scale
        native_scale = global_scale is not None and _GATHER_QMM_HAS_GLOBAL_SCALE
        fallback_scale = global_scale is not None and not native_scale
        dtype = x.dtype
        if fallback_scale:
            # The unscaled product can exceed the half-precision range.
            x = x.astype(mx.float32)
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
            sorted_indices=sorted_indices,
            **({"global_scale": global_scale} if native_scale else {}),
        )
        if fallback_scale:
            # MLX before 0.32.3: gather_qmm has no global_scale argument and
            # decodes nvfp4 with the default tensor scale. The result is linear
            # in that scale, so rescaling each expert's output is exact.
            scale = global_scale[indices] / _NVFP4_DEFAULT_GLOBAL_SCALE
            scale = mx.reshape(scale, scale.shape + (1,) * (x.ndim - scale.ndim))
            x = (x * scale.astype(mx.float32)).astype(dtype)
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
