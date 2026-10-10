# SPDX-License-Identifier: Apache-2.0
"""Bound Darwin GLM shard reads and final weight materialization.

The pinned vision loader has no weight-reader callback. Bind its existing
functions to per-call namespaces (as the video runtime does), rather than
patching module globals or duplicating architecture/quantization logic.
"""

from __future__ import annotations

import logging
from types import FunctionType

logger = logging.getLogger(__name__)
_EVAL_BATCH_BYTES = 1024**3


class _BoundedEval:
    """Keep the final sanitizer graph from scheduling a checkpoint-sized copy."""

    def __init__(self, mx):
        self._mx = mx

    def __getattr__(self, name):
        return getattr(self._mx, name)

    def eval(self, *trees):
        from mlx.utils import tree_flatten

        batch = []
        size = 0
        for _, tensor in tree_flatten(trees):
            if batch and size + tensor.nbytes > _EVAL_BATCH_BYTES:
                self._mx.eval(batch)
                self._mx.clear_cache()
                batch = []
                size = 0
            batch.append(tensor)
            size += tensor.nbytes
        if batch:
            self._mx.eval(batch)
            self._mx.clear_cache()


def _bind(function: FunctionType, name: str, replacement) -> FunctionType:
    if not isinstance(function, FunctionType) or name not in function.__code__.co_names:
        raise RuntimeError(
            "Installed mlx-vlm loader is incompatible with GLM shardwise loading: "
            f"missing {name} binding. Install the repository-pinned vision runtime."
        )
    namespace = dict(function.__globals__)
    namespace[name] = replacement
    scoped = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    scoped.__kwdefaults__ = function.__kwdefaults__
    return scoped


def load_glm5_shardwise(model_name: str, **kwargs):
    """Eagerly evaluate and evict each shard before opening the next one.

    Used by Darwin GLM MLLM and native-MTP target loading. The pinned loader
    still owns sanitization, strict key validation, quantization, and processors.
    Eviction runs only after successful evaluation: lazy arrays must never lose
    their source before materialization. No process-global functions change.
    """
    import mlx.core as mx
    from mlx_vlm import utils

    from ..runtime.ubc_evict import ubc_evict

    reader = getattr(utils, "_load_safetensors", None)
    if not callable(reader):
        raise RuntimeError(
            "Installed mlx-vlm loader lacks the GLM shard reader. "
            "Install the repository-pinned vision runtime."
        )

    def read_shard(path):
        weights = reader(path)
        mx.eval(weights)
        evicted = ubc_evict(str(path))
        logger.info(
            "GLM shard materialized: %s (file-cache invalidation requested: %.1f MiB)",
            path,
            evicted / 2**20,
        )
        return weights

    model_loader = _bind(utils.load_model, "_load_safetensors", read_shard)
    # GLM sanitization stacks per-expert tensors. Even with eager shard reads,
    # a single eval of that entire graph can schedule every destination before
    # releasing its inputs. Bound that second materialization phase too.
    model_loader = _bind(model_loader, "mx", _BoundedEval(mx))
    loader = _bind(utils.load, "load_model", model_loader)
    return loader(model_name, **kwargs)
