# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT
"""Cached single-token visit normalization with the stock depth reduction.

Fallback covers construction, the one-time evaluated probe per kernel
variant, and guarded invocation. Evaluation of real lazy outputs is
intentionally not wrapped because decode evaluates asynchronously.
"""

import threading
from typing import Any

import mlx.core as mx

from .compat import row_threads, scalar
from .kernels import _DEPTH_ATTENTION_SOURCE

# RMSNorm owns four consecutive values per virtual thread. Depth attention
# retains its own thread count and strided accumulation, including both FMAs.
_NORM = """
    threadgroup float norm_buf[32];
    threadgroup float norm_inv;
    if (sg == 0) norm_buf[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0; base < NT; base += TG) {
        uint nt = base + tid;
        float sum = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            uint d = nt * 4 + j;
            float v = d < D ? float(partial[d]) : 0.0f;
            sum = metal::fma(v, v, sum);
        }
        sum = simd_sum(sum);
        if (lane == 0 && nt < NT) norm_buf[nt / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float sum = simd_sum(norm_buf[lane]);
        if (lane == 0) norm_inv = metal::precise::rsqrt(sum / float(D) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < D; d += TG) {
        residual[d] = float(partial[d]);
        normed[(APPEND ? K * D : 0) + d] = float(partial[d]) * norm_inv;
        if (APPEND) for (uint i = 0; i < K; ++i) normed[i * D + d] = blocks[i * D + d];
    }
"""
_SOURCE = (
    _DEPTH_ATTENTION_SOURCE.replace(
        "    float acc[N * Q];", _NORM + "    float acc[N * Q];"
    )
    .replace("partial[size_t(row) * D + d]", "float(partial[d]) * norm_inv")
    .replace(
        "acc[i * Q + j] += x[i] * qv;",
        "acc[i * Q + j] = metal::fma(x[i], qv, acc[i * Q + j]);",
    )
    .replace("o += w[i * Q + j] * x[i];", "o = metal::fma(w[i * Q + j], x[i], o);")
)
_LOCK = threading.Lock()


class _Variant:
    def __init__(self):
        self.lock = threading.Lock()
        self.kernel: Any = None
        self.ready = False


_KERNELS: dict[tuple, _Variant] = {}


def _ready(key, launch):
    if key not in _KERNELS:
        with _LOCK:
            if key not in _KERNELS:
                _KERNELS[key] = _Variant()
    state = _KERNELS[key]
    if not state.ready:
        with state.lock:
            if not state.ready:
                try:
                    kernel = mx.fast.metal_kernel(
                        name="inkling2_visit_norm_decode",
                        input_names=["blocks", "partial", "queries", "scale"],
                        output_names=["out", "normed", "residual"],
                        source=_SOURCE,
                        header="#pragma clang fp contract(off)\n",
                        ensure_row_contiguous=True,
                        compile_options={"math_mode": "safe"},
                    )
                    # Force compilation and execution on synthetic inputs of
                    # this exact variant before success. Cache failures too.
                    probe = dict(launch)
                    probe["inputs"] = [
                        mx.zeros(x.shape, x.dtype) for x in launch["inputs"]
                    ]
                    mx.eval(*kernel(**probe))
                except Exception:
                    kernel = None
                state.kernel = kernel
                state.ready = True
    return state


def visit_norm_decode(blocks, partial, queries, *, append_block=False):
    """Fuse visit RMSNorm, depth attention, residual cast and optional bank append.

    Fallback covers construction, the one-time evaluated probe per kernel
    variant, and guarded invocation. Evaluation of real lazy outputs is
    intentionally not wrapped because decode evaluates asynchronously.
    A failed variant is never retried.
    Row-contiguous copies match stock RMSNorm's layout; widths above 4096
    use a different stock reduction and fall back.
    """
    if any(x is None for x in (blocks, partial, queries)) or partial.ndim != 3:
        return None
    D = partial.shape[-1]
    if not (
        partial.shape == (1, 1, D)
        and 0 < D <= 4096
        and partial.dtype in (mx.float32, mx.float16, mx.bfloat16)
        and blocks.ndim == 4
        and blocks.shape[1:] == partial.shape
        and 0 < blocks.shape[0] <= 32
        and queries.shape in ((1, D), (3, D))
        and blocks.dtype == queries.dtype == mx.float32
    ):
        return None
    K, Q, TG = blocks.shape[0], queries.shape[0], row_threads(D)
    key = (D, K, Q, partial.dtype, append_block)
    launch = dict(
        inputs=[blocks, partial, queries, scalar(D**-0.5)],
        template=[
            ("D", D),
            ("K", K),
            ("Q", Q),
            ("TG", TG),
            ("NT", ((D + 127) // 128) * 32),
            ("APPEND", append_block),
        ],
        grid=(TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[
            (1, 1, Q, D),
            (K + 1, 1, 1, D) if append_block else partial.shape,
            partial.shape,
        ],
        output_dtypes=[mx.float32] * 3,
    )
    state = _ready(key, launch)
    with state.lock:
        if state.kernel is None:
            return None
        try:
            # The call under this lock only constructs the lazy graph node.
            # Evaluation of real lazy outputs is intentionally not wrapped
            # because decode evaluates asynchronously.
            return state.kernel(**launch)
        except Exception:
            state.kernel = None
            return None
