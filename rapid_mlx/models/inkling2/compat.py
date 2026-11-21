# Copyright © 2023-2026 Apple Inc.
# SPDX-License-Identifier: MIT
"""Text runtime helpers; provenance and compatibility scope are in NOTICE."""

from functools import lru_cache
from inspect import signature
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx import nn

from ... import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.switch_layers import SwitchLinear as _NativeSwitchLinear

if TYPE_CHECKING:
    from .switch_layers import SwitchLinear, _gather_sort, _scatter_unsort
elif "global_scale" in signature(_NativeSwitchLinear.to_quantized).parameters:
    from mlx_lm.models.switch_layers import SwitchLinear, _gather_sort, _scatter_unsort
else:
    from .switch_layers import SwitchLinear, _gather_sort, _scatter_unsort

try:
    from mlx_lm.kernels.expert_routing import expert_routing
except ImportError:
    expert_routing = None

__all__ = [
    "SwitchLinear",
    "_gather_sort",
    "_scatter_unsort",
    "expert_routing",
    "gather_last_axis",
    "shard_experts_inplace",
    "scalar",
    "row_threads",
]


@lru_cache
def scalar(value: float, dtype: mx.Dtype = mx.float32) -> mx.array:
    """A cached 0-d array, which a kernel gets as a constant."""
    return mx.array(value, dtype=dtype)


def row_threads(dim: int) -> int:
    """Threads for a threadgroup that handles one row of dim values."""
    return min(1024, max(32, dim // 8 // 32 * 32))


def gather_last_axis(x: mx.array, group: mx.distributed.Group) -> mx.array:
    """Gather the last axis of ``x``, which is split across ``group``."""
    n, size = group.size(), x.shape[-1]
    parts = mx.distributed.all_gather(x.reshape(-1, size), group=group)
    # (ranks * rows, size) to (rows, ranks * size), in rank order.
    parts = parts.reshape(n, -1, size).transpose(1, 0, 2)
    return parts.reshape(*x.shape[:-1], n * size)


def shard_experts_inplace(
    module: nn.Module, sharding: str, group: mx.distributed.Group
) -> None:
    N, r = group.size(), group.rank()
    updates = {}
    for name, w in module.parameters().items():
        if not isinstance(w, mx.array):
            continue
        if w.ndim < 3:
            # we don't shard scales for the nvfp4 weights
            continue
        axis = w.ndim - 2 if sharding == "all-to-sharded" else w.ndim - 1
        n = w.shape[axis] // N
        lo, hi = r * n, (r + 1) * n
        per_expert = w.size // w.shape[0]
        step = max(1, (1 << 31) // (2 * per_expert))
        parts = []
        for i in range(0, w.shape[0], step):
            chunk = w[i : i + step]
            chunk = chunk[:, lo:hi] if axis == 1 else chunk[..., lo:hi]
            chunk = mx.contiguous(chunk)
            mx.eval(chunk)
            parts.append(chunk)
        updates[name] = mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    module.update(updates)
