# SPDX-License-Identifier: Apache-2.0
"""#4108: the HTTP 503 names the admission gate that actually rejected.

The Metal-memory gate and the concurrency gate both raise
``BackpressureError``. The 503 used to say "max concurrent requests reached"
for both, so an idle server at its memory limit told clients to wait for
requests that did not exist.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from rapid_mlx.errors import BackpressureError, MetalMemoryBackpressureError
from rapid_mlx.service.helpers import _raise_backpressure_503


def _detail(exc: Exception) -> HTTPException:
    with pytest.raises(HTTPException) as info:
        _raise_backpressure_503(exc)
    assert info.value.status_code == 503
    assert info.value.headers == {"Retry-After": "1"}
    return info.value


def test_metal_memory_rejection_says_memory_limit():
    message = (
        "Metal memory in use is 36.3 GB with no request running (reserved KV "
        "0.0 GB + projected KV 0.0 GB for this request), but the current "
        "limit is 36.2 GB (D-METAL-CAP)."
    )
    detail = _detail(MetalMemoryBackpressureError(message)).detail
    assert detail.startswith("Server is at its Metal memory limit.")
    assert "max concurrent" not in detail
    # The engine's own explanation (and its D-METAL-CAP grep token) rides along.
    assert message in detail


def test_concurrency_rejection_keeps_busy_wording():
    detail = _detail(
        BackpressureError("max_concurrent_requests=4 reached (currently 4 in-flight)")
    ).detail
    assert detail.startswith("Server is busy (max concurrent requests reached).")
    assert "max_concurrent_requests=4" in detail


def test_other_backpressure_is_plain_busy():
    detail = _detail(
        BackpressureError("generation is paused for a model lifecycle operation")
    ).detail
    assert detail.startswith("Server is busy. Retry after the Retry-After delay.")
    assert "max concurrent" not in detail


def test_metal_memory_error_is_still_backpressure():
    # Every existing ``except BackpressureError`` keeps catching it.
    assert issubclass(MetalMemoryBackpressureError, BackpressureError)
