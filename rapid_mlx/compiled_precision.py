# SPDX-License-Identifier: Apache-2.0
"""Precision-preserving primitives used while tracing decode graphs.

MLX's runtime compiler can fuse ``mx.sigmoid`` with a fast exponential that
does not reproduce the eager Metal library's BF16/FP32 bytes.  The custom
primitive below spells out the precise exponential and remains opaque to the
outer ``mx.compile`` graph.  Normal eager execution is unchanged.

The implementation follows the MLX sigmoid operator's public formula.  The
precision-routing design was adapted from MIT-licensed work Copyright © 2023
Apple Inc.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any

import mlx.core as mx

_PRECISE_DTYPES = (mx.float32, mx.bfloat16)
_IN_COMPILED_DECODE = ContextVar("rapid_mlx_compiled_decode", default=False)
_KERNEL = None
_GATED_PRODUCT_KERNEL = None
_PROBE_LOCK = Lock()
_PROBE_RESULT = None

_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    T x = inp[i];
    T y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    out[i] = (x < 0) ? y : 1 - y;
"""

_GATED_PRODUCT_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    T x = gate[i];
    T y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    T sigmoid = (x < 0) ? y : 1 - y;
    out[i] = value[i] * sigmoid;
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="rapid_precise_sigmoid",
            input_names=["inp"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _KERNEL


def precise_sigmoid(x: mx.array) -> mx.array:
    """Return eager-``mx.sigmoid`` bytes from inside a compiled span."""
    if x.dtype not in _PRECISE_DTYPES or not mx.metal.is_available():
        return mx.sigmoid(x)
    return _kernel()(
        inputs=[x],
        template=[("T", x.dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(min(256, x.size), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def precise_gated_product(gate: mx.array, value: mx.array) -> mx.array:
    """Fuse equal-shaped ``value * eager_sigmoid(gate)`` without losing bits."""
    global _GATED_PRODUCT_KERNEL
    if (
        gate.shape != value.shape
        or gate.dtype != value.dtype
        or gate.dtype not in _PRECISE_DTYPES
        or not mx.metal.is_available()
    ):
        return value * precise_sigmoid(gate)
    if _GATED_PRODUCT_KERNEL is None:
        _GATED_PRODUCT_KERNEL = mx.fast.metal_kernel(
            name="rapid_precise_sigmoid_mul",
            input_names=["gate", "value"],
            output_names=["out"],
            source=_GATED_PRODUCT_SOURCE,
        )
    return _GATED_PRODUCT_KERNEL(
        inputs=[gate, value],
        template=[("T", gate.dtype)],
        grid=(gate.size, 1, 1),
        threadgroup=(min(256, gate.size), 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
    )[0]


def gated_product(gate: mx.array, value: mx.array) -> mx.array:
    """Preserve eager bytes, including from inside a compiled decode trace."""
    if _IN_COMPILED_DECODE.get():
        return precise_gated_product(gate, value)
    return value * mx.sigmoid(gate)


def probe_compiled_precision() -> bool:
    """Verify the opaque primitives against eager bytes once per process."""
    global _PROBE_RESULT
    with _PROBE_LOCK:
        if _PROBE_RESULT is not None:
            return _PROBE_RESULT
        if not mx.metal.is_available():
            _PROBE_RESULT = False
            return False
        base = mx.arange(4096, dtype=mx.float32)
        for dtype in _PRECISE_DTYPES:
            gate = mx.sin(base * 0.731).astype(dtype)
            # Include the known BF16 fast/precise exponential edge explicitly.
            gate = mx.concatenate([mx.array([-6.84375], dtype=dtype), gate[1:]])
            value = mx.cos(base * 0.193).astype(dtype)
            expected_sigmoid = mx.sigmoid(gate)
            expected_product = value * expected_sigmoid
            actual_sigmoid = precise_sigmoid(gate)
            actual_product = precise_gated_product(gate, value)
            mx.eval(
                expected_sigmoid,
                expected_product,
                actual_sigmoid,
                actual_product,
            )
            if not (
                bool(mx.array_equal(expected_sigmoid, actual_sigmoid).item())
                and bool(mx.array_equal(expected_product, actual_product).item())
            ):
                _PROBE_RESULT = False
                return False
        _PROBE_RESULT = True
        return True


@contextmanager
def compiled_decode_precision():
    """Route participating gates through exact primitives while tracing."""
    token = _IN_COMPILED_DECODE.set(True)
    try:
        yield
    finally:
        _IN_COMPILED_DECODE.reset(token)


def gate_sigmoid(x: mx.array) -> mx.array:
    """Use the stock primitive eagerly and its byte-exact form when tracing."""
    if _IN_COMPILED_DECODE.get():
        return precise_sigmoid(x)
    return mx.sigmoid(x)


def in_compiled_decode() -> bool:
    """Return whether the current Python trace is building a decode replay."""
    return _IN_COMPILED_DECODE.get()


def install_qwen35_attention_gate_precision(model: Any) -> int:
    """Route qualified full-attention gates through :func:`gate_sigmoid`.

    Outside a compiled trace this remains the dependency's exact eager
    operation.  The class patch is needed because replacing ``mlx.core``'s
    module-global sigmoid would affect unrelated models in the process.
    """
    if not probe_compiled_precision():
        return 0
    try:
        from mlx_lm.models import qwen3_next as q
    except ImportError:  # pragma: no cover - mlx-lm is a core dependency
        return 0
    attention_class = getattr(q, "Qwen3NextAttention", None)
    if not isinstance(attention_class, type):
        return 0

    modules = [
        module for _, module in model.named_modules() if type(module) is attention_class
    ]
    if not modules:
        return 0
    dynamic_class = attention_class  # type: Any
    if not getattr(dynamic_class, "_rapid_compiled_gate_precision", False):

        def patched(self, x, mask=None, cache=None):
            batch, length, _ = x.shape
            projected = self.q_proj(x)
            queries, gate = mx.split(
                projected.reshape(batch, length, self.num_attention_heads, -1),
                2,
                axis=-1,
            )
            gate = gate.reshape(batch, length, -1)
            keys, values = self.k_proj(x), self.v_proj(x)
            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                keys.reshape(batch, length, self.num_key_value_heads, -1)
            ).transpose(0, 2, 1, 3)
            values = values.reshape(
                batch, length, self.num_key_value_heads, -1
            ).transpose(0, 2, 1, 3)
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)
            output = q.scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=mask,
            )
            output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
            return self.o_proj(gated_product(gate, output))

        dynamic_class.__call__ = patched
        dynamic_class._rapid_compiled_gate_precision = True
    for module in modules:
        dynamic_module = module  # type: Any
        dynamic_module._rapid_compiled_gate_precision = True
    return len(modules)
