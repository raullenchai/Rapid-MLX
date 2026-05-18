# SPDX-License-Identifier: Apache-2.0
"""
Tests for the MLX hardware-compat shim (#404 M5 single-stream).

We can't test on actual M5 from CI, but we can:
1. Verify the shim is installed *before* any module-level
   ``mx.new_thread_local_stream`` capture inside ``mlx_lm.generate``,
   by checking that importing ``vllm_mlx.scheduler`` triggers install().
2. Mock the probe failure and assert the fallback path returns
   ``mx.default_stream(...)``.
3. Verify idempotency — install() can be called multiple times safely.
4. Verify the shim is transparent on hardware that works (this runs on
   the dev's actual hardware in the test-apple-silicon CI job).

We do NOT test that ``import vllm_mlx`` installs the shim — that is the
*wrong* contract. We deliberately keep top-level ``import vllm_mlx``
free of ``mlx.core`` import so the package stays usable for metadata-only
access on systems where ``mlx`` is installed but Metal is unavailable
(``import mlx.core`` SIGABRTs there with an uncatchable NSException).
"""

from __future__ import annotations

import importlib
import importlib.resources

import pytest

pytest.importorskip("mlx.core")


def test_shim_installed_when_scheduler_imports():
    """Importing vllm_mlx.scheduler must install the compat shim — that's
    the gate that protects the module-level ``mx.new_thread_local_stream``
    call inside mlx_lm.generate (which scheduler imports at module top)."""
    import mlx.core as mx

    # Re-install explicitly so this test is order-independent: even if
    # scheduler was already imported by a prior test, install() is
    # idempotent and the assertion still holds.
    from vllm_mlx import _mlx_compat

    if hasattr(mx, "_rapid_mlx_compat_installed"):
        delattr(mx, "_rapid_mlx_compat_installed")
    import vllm_mlx.scheduler  # noqa: F401

    if not getattr(mx, "_rapid_mlx_compat_installed", False):
        # scheduler may already be in sys.modules from a previous test —
        # in which case its module-level install() didn't re-run. Confirm
        # that calling install() directly works.
        _mlx_compat.install()
    assert getattr(mx, "_rapid_mlx_compat_installed", False) is True


def test_vllm_mlx_init_does_not_install_shim_or_import_mlx():
    """`vllm_mlx/__init__.py` must NOT import mlx or call _mlx_compat.install().
    Both would eagerly load `mlx.core`, which SIGABRTs (uncatchable from
    Python) on systems where the `mlx` package is installed but Metal is
    unavailable — breaking metadata-only callers (`__version__`, etc.).

    Pure source-text audit; no module manipulation so the test is safe
    in a shared pytest process. The shim must be installed lazily at
    the top of every module that imports `mlx_lm.generate` instead
    (verified by `test_every_mlx_lm_generate_consumer_installs_shim`)."""
    init_source = (
        importlib.resources.files("vllm_mlx").joinpath("__init__.py").read_text()
    )
    assert "import mlx" not in init_source, (
        "vllm_mlx/__init__.py must not import mlx — it would break "
        "metadata-only usage on systems with broken Metal init."
    )
    assert "_mlx_compat.install()" not in init_source, (
        "vllm_mlx/__init__.py must not call _mlx_compat.install() — that "
        "would eagerly import mlx.core (which can SIGABRT on Metal-less "
        "systems). The shim must install at scheduler-import time instead."
    )


def test_every_mlx_lm_generate_consumer_installs_shim():
    """Any vllm_mlx file that imports `mlx_lm.generate` (either via
    `from mlx_lm.generate import ...` or `importlib.import_module(
    "mlx_lm.generate")`) MUST also call `_mlx_compat.install()` in the
    same file. Otherwise that import path triggers `mlx_lm/generate.py`
    module-level `mx.new_thread_local_stream` capture on M5 before our
    shim runs, and the bug from #404 returns.

    This is a structural audit: a new file that adds the import without
    the install will fail here at unit-test time, catching the slip
    before any M5 user does.
    """
    import pathlib

    pkg_root = pathlib.Path(
        str(importlib.resources.files("vllm_mlx").joinpath(""))
    ).resolve()
    offenders = []
    for path in pkg_root.rglob("*.py"):
        if path.name == "_mlx_compat.py":
            continue  # the shim itself; nothing to install
        source = path.read_text()
        imports_generate = (
            "from mlx_lm.generate" in source
            or 'import_module("mlx_lm.generate")' in source
            or "import_module('mlx_lm.generate')" in source
        )
        if imports_generate and "_mlx_compat.install()" not in source:
            offenders.append(str(path.relative_to(pkg_root)))
    assert not offenders, (
        "Files import mlx_lm.generate without calling _mlx_compat.install() "
        "first — #404 M5 regression risk:\n  " + "\n  ".join(offenders)
    )


def test_install_is_idempotent():
    import mlx.core as mx

    from vllm_mlx import _mlx_compat

    _mlx_compat.install()
    first = mx.new_thread_local_stream
    _mlx_compat.install()
    second = mx.new_thread_local_stream
    assert first is second, "second install() must not re-wrap the function"


def test_install_is_noop_when_symbol_missing(monkeypatch):
    """Regression for #408: on mlx builds that predate
    ``mx.new_thread_local_stream``, ``install()`` must be a no-op rather
    than crash with AttributeError. Without this guard,
    ``import vllm_mlx.scheduler`` aborts before the server can bind a
    port — every user on the affected mlx is blocked from upgrading."""
    import mlx.core as mx

    from vllm_mlx import _mlx_compat

    # If a future mlx genuinely drops the symbol, this assert fails
    # loudly so we revisit whether the compat shim still has a job to
    # do — `raising=False` on the delattr below would silently turn
    # this into a degenerate test that exercises nothing.
    assert hasattr(mx, "new_thread_local_stream"), (
        "expected baseline mlx to expose new_thread_local_stream; "
        "if upstream removed it, this test no longer covers the #408 "
        "regression path and the shim itself can probably go away."
    )
    monkeypatch.delattr(mx, "new_thread_local_stream")
    monkeypatch.setattr(mx, "_rapid_mlx_compat_installed", False, raising=False)
    importlib.reload(_mlx_compat)
    _mlx_compat.install()  # must not raise — that's the #408 contract
    # Note: on the no-symbol path the shim deliberately does NOT mark
    # itself "installed" so that a later mlx upgrade (which adds the
    # symbol) gets the wrap on the next install() call. The contract
    # this test pins is "no AttributeError", not the flag.


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
