# SPDX-License-Identifier: Apache-2.0
"""Overlap qualified Qwen3.5-family layer execution with graph construction.

MLX is lazy: without an explicit submission point, Python builds the complete
decoder graph before Metal begins executing it.  Rapid's Qwen3.5/3.6 serving
path wraps each decoder layer so narrow decode and verification slabs are
submitted as soon as they are available.  The returned array is unchanged;
only command submission timing moves.

Set ``RAPID_MLX_QWEN35_EAGER_LAYER_DISPATCH=0`` before process start to keep
the dependency's stock scheduling.  Wide prefill slabs are left untouched.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Any

from .._env import env_falsey
from ..lattice import ModelProfile, resolve, runtime_context

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_INSTALLED = False


def _not_compiling() -> bool:
    return False


_in_compiled_decode = _not_compiling
_ENABLED = not env_falsey("RAPID_MLX_QWEN35_EAGER_LAYER_DISPATCH")


def _layer_geometry(layer):
    """Read installed facts, retaining the original rank-one norm contract."""
    norm = getattr(layer, "input_layernorm", None)
    weight = getattr(norm, "weight", None)
    shape = tuple(getattr(weight, "shape", ()))
    mlp = getattr(layer, "mlp", None)
    return (
        shape[0] if len(shape) == 1 else None,
        getattr(mlp, "num_experts", 0),
        getattr(mlp, "top_k", 0),
    )


def install_qwen3_5_eager_dispatch() -> None:
    """Install the inference-only Qwen3.5 decoder-layer submission hook."""
    global _INSTALLED, _in_compiled_decode

    with _LOCK:
        if _INSTALLED:
            return
        try:
            import mlx.core as mx
            from mlx_lm.models import qwen3_5 as upstream

            q: Any = upstream

            from ..compiled_precision import in_compiled_decode
        except ImportError:  # pragma: no cover - optional on non-Mac installs
            logger.debug("Qwen3.5 eager dispatch unavailable; skipping install")
            return
        _in_compiled_decode = in_compiled_decode

        if getattr(q, "_RAPID_MLX_EAGER_LAYER_DISPATCH_INSTALLED", False):
            _INSTALLED = True
            return
        if not hasattr(q, "DecoderLayer"):
            logger.debug("mlx-lm Qwen3.5 DecoderLayer unavailable; skipping install")
            return

        original = q.DecoderLayer.__call__
        q._RAPID_MLX_ORIG_DECODER_LAYER_CALL = original

        # Freeze candidate overrides for this installation. The legacy switch
        # remains the independent import-time master gate above. Profiles already
        # load once per process; cache hits never enter resolution/profile code.
        environment = dict(os.environ)
        # Real device and software facts let chip-specific cells apply; the
        # builtin cell is universal, so unknown hardware keeps today's policy.
        try:
            device = dict(mx.device_info())
        except Exception:  # pragma: no cover - defensive: facts stay unknown
            device = {}

        @lru_cache(maxsize=4096)
        def dispatch_policy(hidden, experts, top_k, rows):
            # Exact-row buckets preserve arbitrary overlay row boundaries.
            ctx = runtime_context(
                device,
                model=ModelProfile(
                    family="qwen3_5_decoder_layer",
                    hidden_size=hidden,
                    num_experts=experts,
                    top_k=top_k,
                ),
                flattened_rows=rows,
            )
            return bool(
                resolve("qwen35_eager_layer_dispatch", ctx, environ=environment).value
            )

        def _eager_layer_call(self, *args, **kwargs):
            output = original(self, *args, **kwargs)
            shape = getattr(output, "shape", ())
            rows = int(shape[0]) * int(shape[1]) if len(shape) >= 2 else None
            if (
                _ENABLED
                and not _in_compiled_decode()
                and rows is not None
                and dispatch_policy(*_layer_geometry(self), rows)
            ):
                mx.async_eval(output)
            return output

        q.DecoderLayer.__call__ = _eager_layer_call
        q._RAPID_MLX_EAGER_LAYER_DISPATCH_INSTALLED = True
        _INSTALLED = True
        logger.debug("Qwen3.5 lattice eager layer dispatch installed")


def uninstall_qwen3_5_eager_dispatch() -> None:
    """Undo the process-wide hook. Test-only."""
    global _INSTALLED

    with _LOCK:
        if not _INSTALLED:
            return
        try:
            from mlx_lm.models import qwen3_5 as upstream

            q: Any = upstream
        except ImportError:  # pragma: no cover
            return
        original = getattr(q, "_RAPID_MLX_ORIG_DECODER_LAYER_CALL", None)
        if original is not None:
            q.DecoderLayer.__call__ = original
        q._RAPID_MLX_EAGER_LAYER_DISPATCH_INSTALLED = False
        _INSTALLED = False


def is_installed() -> bool:
    return _INSTALLED
