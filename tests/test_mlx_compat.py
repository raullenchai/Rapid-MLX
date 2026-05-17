# SPDX-License-Identifier: Apache-2.0
"""
Tests for the MLX hardware-compat shim (#404 M5 single-stream).

We can't test on actual M5 from CI, but we can:
1. Verify the shim is installed by importing vllm_mlx.
2. Mock the probe failure and assert the fallback path returns
   ``mx.default_stream(...)``.
3. Verify idempotency — install() can be called multiple times safely.
4. Verify the shim is transparent on hardware that works (this runs on
   the dev's actual hardware in the test-apple-silicon CI job).
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("mlx.core")


def test_shim_installed_after_vllm_mlx_import():
    """Importing vllm_mlx must install the compat shim before any submodule
    that loads mlx_lm.generate gets a chance to capture the original API."""
    import mlx.core as mx

    # Drop the flag if a previous test left it set, then re-import.
    if hasattr(mx, "_rapid_mlx_compat_installed"):
        delattr(mx, "_rapid_mlx_compat_installed")

    import vllm_mlx  # noqa: F401

    assert getattr(mx, "_rapid_mlx_compat_installed", False) is True


def test_install_is_idempotent():
    import mlx.core as mx

    from vllm_mlx import _mlx_compat

    _mlx_compat.install()
    first = mx.new_thread_local_stream
    _mlx_compat.install()
    second = mx.new_thread_local_stream
    assert first is second, "second install() must not re-wrap the function"


def test_fallback_engages_when_probe_raises(monkeypatch):
    """Simulate M5: probe raises 'no Stream(gpu, 1)' → patched function must
    return mx.default_stream(device) instead of the unusable stream."""
    import mlx.core as mx

    from vllm_mlx import _mlx_compat

    # Make `_probe` always fail with the M5-shaped error. We poke the
    # ``mx`` namespace because the patch wires `with mx.stream(stream)`
    # → `mx.array(...) + mx.array(...)` — substituting `mx.stream`
    # itself is the cleanest interception point.
    class _BoomStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            raise RuntimeError("There is no Stream(gpu, 1) in current thread.")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mx, "stream", _BoomStream)

    # Force a fresh install with our broken probe environment.
    monkeypatch.setattr(mx, "_rapid_mlx_compat_installed", False, raising=False)
    importlib.reload(_mlx_compat)
    _mlx_compat.install()

    device = mx.default_device()
    fallback = mx.new_thread_local_stream(device)
    expected = mx.default_stream(device)
    # mx.default_stream is comparable by repr; compare structurally.
    assert repr(fallback) == repr(expected), (
        f"M5 fallback should return mx.default_stream({device!r}); got {fallback!r}"
    )


def test_fallback_does_not_engage_on_unrelated_runtime_error(monkeypatch):
    """If `with mx.stream(stream)` raises a RuntimeError that doesn't look
    like the M5 single-stream signature, the shim must NOT swallow it —
    we want unexpected failures to surface, not get silently degraded."""
    import mlx.core as mx

    from vllm_mlx import _mlx_compat

    class _BoomStream:
        def __init__(self, stream):
            pass

        def __enter__(self):
            raise RuntimeError("Some completely unrelated MLX error")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mx, "stream", _BoomStream)
    monkeypatch.setattr(mx, "_rapid_mlx_compat_installed", False, raising=False)
    importlib.reload(_mlx_compat)
    _mlx_compat.install()

    with pytest.raises(RuntimeError, match="completely unrelated"):
        mx.new_thread_local_stream(mx.default_device())


def test_happy_path_unchanged_on_real_hardware():
    """On hardware where the original API works (M1–M4), the patched
    function must return a usable stream — and `with mx.stream(stream)`
    must run a trivial op. This is the test that confirms the shim is
    transparent for users who don't need it."""
    import mlx.core as mx

    from vllm_mlx import _mlx_compat

    # Cleanup from prior monkeypatched tests
    if hasattr(mx, "_rapid_mlx_compat_installed"):
        delattr(mx, "_rapid_mlx_compat_installed")
    importlib.reload(_mlx_compat)
    _mlx_compat.install()

    stream = mx.new_thread_local_stream(mx.default_device())
    with mx.stream(stream):
        result = (mx.array([1.0]) + mx.array([2.0])).item()
    assert result == 3.0
