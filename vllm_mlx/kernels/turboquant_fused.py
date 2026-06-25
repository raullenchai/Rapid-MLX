# SPDX-License-Identifier: Apache-2.0
"""Python binding for the vendored TurboQuant fused Metal kernels.

The Metal source lives in ``turboquant_fused.metal`` (vendored from
arozanov/turboquant-mlx, Apache-2.0). This module loads the source at
first call, dispatches via ``mx.fast.metal_kernel`` and caches the
compiled handle at module scope.

The two public helpers — :func:`fused_quantize_v4` and
:func:`fused_dequant_v4_fp16` — match the V-side TurboQuant encode /
decode contract; :func:`fused_quantize_k8` adds the rapid-mlx K-side
symmetric uniform path for the K8V4 mix.

All three return ``None`` when Metal compilation fails so the caller
can fall back to the pure-MLX reference path without an unhandled
exception (R15 Phase 4 deliverable: "Falls back to unfused path if
Metal compilation fails").
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metal source loading
# ---------------------------------------------------------------------------

_METAL_SOURCE_PATH = Path(__file__).with_name("turboquant_fused.metal")
_KERNEL_SENTINEL = "// >>> kernel: "

# Module-level cache for compiled kernels. ``False`` is the
# "tried-and-failed" sentinel so we don't keep recompiling on the hot
# path after the first failure.
_KERNEL_CACHE: dict[str, object] = {}

# Mirrors arozanov's packing table for uint32 word density.
_VALS_PER_WORD = {1: 32, 2: 16, 3: 10, 4: 8}


def _load_kernel_sources() -> dict[str, str]:
    """Split the .metal file into one ``{name: source}`` mapping.

    Sentinel format is ``// >>> kernel: <name>`` on a line of its own.
    Everything between two sentinels (or from a sentinel to EOF) is the
    body shipped to ``mx.fast.metal_kernel``.
    """
    text = _METAL_SOURCE_PATH.read_text(encoding="utf-8")
    parts: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(_KERNEL_SENTINEL):
            if current_name is not None:
                parts[current_name] = "\n".join(current_lines)
            current_name = line[len(_KERNEL_SENTINEL) :].strip()
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        parts[current_name] = "\n".join(current_lines)
    return parts


def _compile_kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
) -> object | None:
    """Compile a kernel from the .metal file. Returns ``None`` on failure.

    Failure paths (any of which is benign — the caller falls back to the
    pure-MLX reference path):
      * Metal is unavailable on this host (CPU-only build, or running
        under a remote test runner that does not have a GPU).
      * The Metal source has a syntax error after a future upstream
        port.
      * ``mx.fast.metal_kernel`` raises because of an MLX version
        mismatch (the API moved between 0.20 and 0.22).
    """
    cached = _KERNEL_CACHE.get(name)
    if cached is False:
        return None
    if cached is not None:
        return cached
    try:
        sources = _load_kernel_sources()
        if name not in sources:
            logger.warning(
                "turboquant_fused: kernel %r missing from %s; falling back",
                name,
                _METAL_SOURCE_PATH.name,
            )
            _KERNEL_CACHE[name] = False
            return None
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=sources[name],
        )
    except Exception as exc:  # pragma: no cover - exercised only on Metal-less hosts
        logger.warning(
            "turboquant_fused: failed to compile %r (%s); falling back", name, exc
        )
        _KERNEL_CACHE[name] = False
        return None
    _KERNEL_CACHE[name] = kernel
    return kernel


def _packed_dim_v4(dim: int, bits: int) -> int:
    vpw = _VALS_PER_WORD[bits]
    return (dim + vpw - 1) // vpw


# ---------------------------------------------------------------------------
# Public bindings
# ---------------------------------------------------------------------------


def is_metal_available() -> bool:
    """Return True when Metal compilation will work on this host.

    Used by the metrics route to surface the
    ``rapid_mlx_turboquant_fused_kernel{status}`` gauge at serve boot.
    """
    try:
        return mx.default_device() == mx.gpu and mx.metal.is_available()
    except Exception:
        return False


def fused_quantize_v4(
    vectors: mx.array,
    signs: mx.array,
    boundaries: mx.array,
    dim: int,
    bits: int,
) -> tuple[mx.array, mx.array] | None:
    """Encode V vectors with the V-side TurboQuant fused kernel.

    Returns ``(packed_uint32, norms_fp32)`` on success, ``None`` on
    compile failure (caller falls back to :mod:`vllm_mlx.turboquant`'s
    unfused path).
    """
    if not is_metal_available():
        return None
    kernel = _compile_kernel(
        "tq_fused_quantize_v4",
        input_names=["inp", "signs", "boundaries", "dims"],
        output_names=["packed_out", "norms_out"],
    )
    if kernel is None:
        return None

    n_vecs = vectors.shape[0]
    vpw = _VALS_PER_WORD[bits]
    p_dim = _packed_dim_v4(dim, bits)
    n_centroids = boundaries.shape[0] + 1
    dims_arr = mx.array([dim, bits, vpw, p_dim, n_centroids], dtype=mx.uint32)

    try:
        outputs = kernel(
            inputs=[
                vectors.reshape(n_vecs * dim).astype(mx.float32),
                signs.astype(mx.float32),
                boundaries.astype(mx.float32),
                dims_arr,
            ],
            template=[],
            grid=(n_vecs * dim, 1, 1),
            threadgroup=(dim, 1, 1),
            output_shapes=[(n_vecs * p_dim,), (n_vecs,)],
            output_dtypes=[mx.uint32, mx.float32],
        )
    except Exception as exc:  # pragma: no cover - mlx-version edge cases
        logger.warning("turboquant_fused: V4 dispatch failed (%s); falling back", exc)
        return None

    packed = outputs[0].reshape(n_vecs, p_dim)
    return packed, outputs[1]


def fused_dequant_v4_fp16(
    packed: mx.array,
    norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    dim: int,
    bits: int,
) -> mx.array | None:
    """Decode V vectors with the V-side TurboQuant fused kernel.

    Returns the dequantized FP16 array on success, ``None`` on compile
    failure.
    """
    if not is_metal_available():
        return None
    kernel = _compile_kernel(
        "tq_fused_dequant_v4_fp16",
        input_names=["packed", "norms", "centroids", "signs", "scale", "dims"],
        output_names=["out"],
    )
    if kernel is None:
        return None

    seq_len = norms.shape[0]
    vpw = _VALS_PER_WORD[bits]
    p_dim = _packed_dim_v4(dim, bits)
    scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    dims_arr = mx.array([dim, bits, vpw, p_dim], dtype=mx.uint32)

    try:
        outputs = kernel(
            inputs=[
                packed.astype(mx.uint32).reshape(-1),
                norms.astype(mx.float32),
                centroids.astype(mx.float32),
                signs.astype(mx.float32),
                scale,
                dims_arr,
            ],
            template=[],
            grid=(seq_len * dim, 1, 1),
            threadgroup=(dim, 1, 1),
            output_shapes=[(seq_len, dim)],
            output_dtypes=[mx.float16],
        )
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "turboquant_fused: V4 dequant dispatch failed (%s); falling back", exc
        )
        return None
    return outputs[0]


def fused_quantize_k8(
    vectors: mx.array,
    signs: mx.array,
    dim: int,
) -> tuple[mx.array, mx.array, mx.array] | None:
    """Encode K vectors with the K-side symmetric uniform fused kernel.

    Returns ``(packed_uint8, norms_fp32, k_scales_fp32)`` on success,
    ``None`` on compile failure (caller falls back to the pure-MLX
    K-side reference path).
    """
    if not is_metal_available():
        return None
    kernel = _compile_kernel(
        "tq_fused_quantize_k8",
        input_names=["inp", "signs", "dims"],
        output_names=["packed_k_out", "norms_out", "k_scales_out"],
    )
    if kernel is None:
        return None

    n_vecs = vectors.shape[0]
    dims_arr = mx.array([dim, 8], dtype=mx.uint32)

    try:
        outputs = kernel(
            inputs=[
                vectors.reshape(n_vecs * dim).astype(mx.float32),
                signs.astype(mx.float32),
                dims_arr,
            ],
            template=[],
            grid=(n_vecs * dim, 1, 1),
            threadgroup=(dim, 1, 1),
            output_shapes=[(n_vecs * dim,), (n_vecs,), (n_vecs,)],
            output_dtypes=[mx.uint8, mx.float32, mx.float32],
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("turboquant_fused: K8 dispatch failed (%s); falling back", exc)
        return None

    packed = outputs[0].reshape(n_vecs, dim)
    return packed, outputs[1], outputs[2]


def reset_kernel_cache_for_tests() -> None:
    """Test-only hook: drop the module cache so the next call recompiles."""
    _KERNEL_CACHE.clear()
