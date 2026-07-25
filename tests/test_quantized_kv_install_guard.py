# SPDX-License-Identifier: Apache-2.0
"""The QuantizedBatchKVCache install must degrade gracefully when the running
``mlx_lm`` predates the ``BatchGenerator._make_new_cache`` hook.

Regression: a too-old mlx-lm (< 0.31.3, whose ``BatchGenerator`` has no
``_make_new_cache``) made the previously-unguarded ``batch_gen._make_new_cache``
access raise ``AttributeError`` on EVERY scheduler step. The
generation-error-recovery path swallowed it, turning a stale-dependency install
into a silent, permanent serve hang (zero output, no surfaced error). The
install now returns ``False`` (skip) so the caller keeps the bf16 live cache and
warns, instead of the server wedging.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")  # the module under test imports mlx.core at import time

from vllm_mlx.quantized_batch_cache import (  # noqa: E402
    _QuantizableKVCache,
    install_quantized_batch_cache,
)


class _NoHookBatchGen:
    """A BatchGenerator-like object WITHOUT ``_make_new_cache`` (old mlx-lm)."""


class _HookedBatchGen:
    """A BatchGenerator-like object WITH ``_make_new_cache`` (mlx-lm >= 0.31.3)."""

    def __init__(self):
        from mlx_lm.models.cache import KVCache

        self._make_new_cache = lambda: [KVCache()]


def test_install_skips_and_leaves_no_hook_when_make_new_cache_missing():
    bg = _NoHookBatchGen()
    installed = install_quantized_batch_cache(bg, group_size=64, bits=4)
    # Must report "not installed" (so the caller falls back to bf16) ...
    assert installed is False
    # ... and must NOT have raised or fabricated a hook on the way out.
    assert not hasattr(bg, "_make_new_cache")


def test_install_wraps_and_returns_true_when_hook_present():
    bg = _HookedBatchGen()
    orig = bg._make_new_cache
    installed = install_quantized_batch_cache(bg, group_size=64, bits=4)
    assert installed is True
    # The hook was replaced (wrapped), not left as-is.
    assert bg._make_new_cache is not orig
    # The wrapped hook swaps plain KVCache layers for the quantizable type.
    caches = bg._make_new_cache()
    assert caches and all(isinstance(c, _QuantizableKVCache) for c in caches)
