"""Qualified long-context affine8 attention; return None for stock fallback.

The fused implementation is numerically approximate (like dense Metal SDPA).
It does not change the cache representation or the default KV dtype policy.
"""

import logging
import os
import platform
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

_LOG = logging.getLogger(__name__)
_DISABLED = False
_WARMED_VARIANTS: set[tuple[object, int, int, int, bool]] = set()


@lru_cache(maxsize=1)
def _qualified_runtime():
    return (
        mx.metal.is_available()
        and mx.__version__ == "0.32.3"
        and platform.mac_ver()[0] == "26.5.2"
        and mx.device_info().get("device_name") == "Apple M3 Ultra"
    )


@lru_cache(maxsize=1)
def _kernel():
    source = Path(__file__).with_suffix(".metal").read_text()
    header, source = source.split("using namespace mpp::tensor_ops;", 1)
    return mx.fast.metal_kernel(
        name="rapid_affine8_attention",
        input_names=[
            "q",
            "kq",
            "ksc",
            "kbi",
            "vq",
            "vsc",
            "vbi",
            "scl",
            "tk",
            "attn_mask",
            "row_valid",
        ],
        output_names=["m_out", "l_out", "o_out"],
        header=header,
        source="using namespace mpp::tensor_ops;" + source,
        ensure_row_contiguous=False,
    )


def try_affine8_attention(q, k, v, *, scale, mask=None, group_size=64, bits=8):
    """Dispatch only the measured B=1, D=256, affine8/group64 envelope.

    Boolean masks must broadcast identically across heads. Additive masks,
    batches, other runtimes and short contexts retain stock attention.
    RAPID_MLX_AFFINE8_ATTENTION=0 disables the optimization for A/B testing.
    """
    global _DISABLED
    if (
        _DISABLED
        or os.environ.get("RAPID_MLX_AFFINE8_ATTENTION", "1") == "0"
        or bits != 8
        or group_size != 64
        or q.ndim != 4
        or q.shape[0] != 1
        or q.shape[-1] != 256
        or q.dtype not in (mx.float16, mx.bfloat16)
        or mx.default_device() != mx.gpu
        or not _qualified_runtime()
    ):
        return None
    if not all(isinstance(t, (tuple, list)) and len(t) == 3 for t in (k, v)):
        return None
    if any(not isinstance(a, mx.array) or a.ndim != 4 for a in (*k, *v)):
        return None
    _, hq, tq, d = q.shape
    _, hkv, tk, _ = k[0].shape
    if hkv == 0 or hq == 0 or hq % hkv or not 1 <= tq <= 8:
        return None
    mr = hq // hkv * tq
    mpad = (mr + 7) // 8 * 8
    if mpad > 56 or tk < (2048 if tq >= 4 else 8192 if tq >= 2 else 16384):
        return None
    for packed, scales, biases in (k, v):
        if (
            packed.shape != (1, hkv, tk, 64)
            or packed.dtype != mx.uint32
            or scales.shape != (1, hkv, tk, 4)
            or biases.shape != scales.shape
            or scales.dtype != q.dtype
            or biases.dtype != q.dtype
        ):
            return None
    if isinstance(mask, mx.array):
        if (
            mask.dtype != mx.bool_
            or not 2 <= mask.ndim <= (5 if hq > hkv else 4)
            or mask.shape[-2:] != (tq, tk)
            or any(n != 1 for n in mask.shape[:-2])
        ):
            return None
        mask = mx.contiguous(mask.reshape(tq, tk))
        row_valid = mx.any(mask, axis=-1)
    elif (isinstance(mask, str) and mask == "causal") or (mask is None and tq == 1):
        mask = None
        row_valid = mx.array([True])
    else:
        return None
    ns = max(1, min(128, ((tk + 31) // 32) // 8))
    qp = mx.contiguous(mx.pad(q.reshape(hkv, mr, d), ((0, 0), (0, mpad - mr), (0, 0))))
    try:
        m, denom, o = _kernel()(
            inputs=[
                qp,
                *k,
                *v,
                mx.array([scale], mx.float32),
                mx.array([tk], mx.int32),
                mask if mask is not None else mx.array([True]),
                row_valid,
            ],
            template=[
                ("T", q.dtype),
                ("BITS", 8),
                ("GS", 64),
                ("DK", d),
                ("GQA", hq // hkv),
                ("TQ", tq),
                ("MPAD", mpad),
                ("HAS_MASK", mask is not None),
            ],
            grid=(256, hkv, ns),
            threadgroup=(256, 1, 1),
            output_shapes=[(hkv, ns, mpad), (hkv, ns, mpad), (hkv, ns, mpad, d)],
            output_dtypes=[mx.float32] * 3,
        )
        variant = (q.dtype, hq // hkv, tq, mpad, mask is not None)
        if variant not in _WARMED_VARIANTS:
            # MLX compiles Metal lazily during evaluation. Warm each template
            # once inside the fallback boundary; subsequent calls remain lazy.
            mx.eval(m, denom, o)
            _WARMED_VARIANTS.add(variant)
    except (RuntimeError, ValueError):
        _DISABLED = True
        _LOG.warning(
            "Affine8 attention unavailable; retaining stock attention", exc_info=True
        )
        return None
    m, denom, o = m[:, :, :mr], denom[:, :, :mr], o[:, :, :mr, :]
    w = mx.exp(m - mx.max(m, axis=1, keepdims=True))
    return (
        (mx.sum(o * w[..., None], axis=1) / mx.sum(denom * w, axis=1)[..., None])
        .astype(q.dtype)
        .reshape(q.shape)
    )
