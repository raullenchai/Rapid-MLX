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

logger = logging.getLogger(__name__)

_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_MAX_ROWS = 64
_QUALIFIED_HIDDEN_SIZE = 2048
_QUALIFIED_NUM_EXPERTS = 256
_QUALIFIED_TOP_K = 8
_LOCK = threading.Lock()
_INSTALLED = False
_ENABLED = (
    os.environ.get("RAPID_MLX_QWEN35_EAGER_LAYER_DISPATCH", "1").strip().lower()
    not in _FALSE_VALUES
)


def _is_qualified_layer(layer) -> bool:
    """Match the measured 35B-A3B layer shape, not every family member."""
    norm = getattr(layer, "input_layernorm", None)
    weight = getattr(norm, "weight", None)
    shape = tuple(getattr(weight, "shape", ()))
    mlp = getattr(layer, "mlp", None)
    return (
        shape == (_QUALIFIED_HIDDEN_SIZE,)
        and getattr(mlp, "num_experts", None) == _QUALIFIED_NUM_EXPERTS
        and getattr(mlp, "top_k", None) == _QUALIFIED_TOP_K
    )


def install_qwen3_5_eager_dispatch() -> None:
    """Install the inference-only Qwen3.5 decoder-layer submission hook."""
    global _INSTALLED

    with _LOCK:
        if _INSTALLED:
            return
        try:
            import mlx.core as mx
            from mlx_lm.models import qwen3_5 as q
        except ImportError:  # pragma: no cover - optional on non-Mac installs
            logger.debug("Qwen3.5 eager dispatch unavailable; skipping install")
            return

        if getattr(q, "_RAPID_MLX_EAGER_LAYER_DISPATCH_INSTALLED", False):
            _INSTALLED = True
            return
        if not hasattr(q, "DecoderLayer"):
            logger.debug("mlx-lm Qwen3.5 DecoderLayer unavailable; skipping install")
            return

        original = q.DecoderLayer.__call__
        q._RAPID_MLX_ORIG_DECODER_LAYER_CALL = original

        def _eager_layer_call(self, *args, **kwargs):
            output = original(self, *args, **kwargs)
            shape = getattr(output, "shape", ())
            rows = int(shape[0]) * int(shape[1]) if len(shape) >= 2 else None
            if (
                _ENABLED
                and rows is not None
                and rows <= _MAX_ROWS
                and _is_qualified_layer(self)
            ):
                mx.async_eval(output)
            return output

        q.DecoderLayer.__call__ = _eager_layer_call
        q._RAPID_MLX_EAGER_LAYER_DISPATCH_INSTALLED = True
        _INSTALLED = True
        logger.debug(
            "Qwen3.5 35B-A3B eager layer dispatch installed (max rows: %d)",
            _MAX_ROWS,
        )


def uninstall_qwen3_5_eager_dispatch() -> None:
    """Undo the process-wide hook. Test-only."""
    global _INSTALLED

    with _LOCK:
        if not _INSTALLED:
            return
        try:
            from mlx_lm.models import qwen3_5 as q
        except ImportError:  # pragma: no cover
            return
        original = getattr(q, "_RAPID_MLX_ORIG_DECODER_LAYER_CALL", None)
        if original is not None:
            q.DecoderLayer.__call__ = original
        q._RAPID_MLX_EAGER_LAYER_DISPATCH_INSTALLED = False
        _INSTALLED = False


def is_installed() -> bool:
    return _INSTALLED
