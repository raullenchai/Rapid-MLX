# SPDX-License-Identifier: Apache-2.0
"""Memory estimates and early warnings for image workloads."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.model_aliases import resolve_profile
from rapid_mlx.runtime.resident_models import estimate_model_bytes


@pytest.mark.parametrize(
    "alias,peak", [("flux2-klein-4b", 13.15), ("z-image-turbo", 12.02)]
)
def test_image_memory_metadata(alias, peak):
    profile = resolve_profile(alias)
    assert profile.min_memory_gb == 16
    assert estimate_model_bytes(alias) >= (peak + 1) * 1024**3
    assert estimate_model_bytes(profile.hf_path) >= (peak + 1) * 1024**3


@pytest.mark.parametrize(
    "alias,floor,estimate", [("flux-schnell", 32, 21.4), ("sdxl-base", 16, 14.0)]
)
def test_image_memory_metadata_for_measured_footprints(alias, floor, estimate):
    assert resolve_profile(alias).min_memory_gb == floor
    assert estimate_model_bytes(alias) == int(estimate * 1024**3)


def test_serve_image_memory_warning(monkeypatch, capsys):
    from rapid_mlx import cli

    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(total=12 * 1024**3)
    )
    cli._check_alias_min_memory("flux2-klein-4b")
    output = capsys.readouterr().out
    assert "'flux2-klein-4b' needs a Mac with at least 16 GB" in output
    assert "this Mac has 12.0 GB" in output
    assert "stopped by macOS" in output


def test_image_benchmark_warns_before_server_start(monkeypatch, capsys):
    from rapid_mlx.bench import _server
    from rapid_mlx.community_bench import local_runner

    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(total=12 * 1024**3)
    )

    def start(*args, **kwargs):
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "'flux2-klein-4b' needs a Mac with at least 16 GB" in captured.err
        raise RuntimeError("server start reached")

    monkeypatch.setattr(_server, "serve", start)
    with pytest.raises(RuntimeError, match="server start reached"):
        local_runner._run_image("flux2-klein-4b")


def test_generation_cache_limit_stays_bounded_until_last_context_exits(monkeypatch):
    from rapid_mlx.image.engine import _bounded_generation_cache

    original = 4 * 1024**3
    limits = []
    clears = []
    current = original

    def set_cache_limit(limit):
        nonlocal current
        previous = current
        current = limit
        limits.append(limit)
        return previous

    mx = ModuleType("mlx.core")
    mx.set_cache_limit = set_cache_limit
    mx.clear_cache = lambda: clears.append(True)
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    first = _bounded_generation_cache()
    second = _bounded_generation_cache()
    first.__enter__()
    second.__enter__()
    try:
        first.__exit__(None, None, None)
        assert current == 1024**3
        assert limits == [0, 1024**3]
        assert clears == []
    finally:
        second.__exit__(None, None, None)
    assert current == original
    assert limits == [0, 1024**3, original]
    assert clears == [True]


def test_generation_cache_restores_limit_when_bound_fails(monkeypatch):
    from rapid_mlx.image.engine import _bounded_generation_cache

    original = 4 * 1024**3
    current = original
    limits = []

    def set_cache_limit(limit):
        nonlocal current
        limits.append(limit)
        if len(limits) == 2:
            raise RuntimeError("bound failed")
        previous = current
        current = limit
        return previous

    mx = ModuleType("mlx.core")
    mx.set_cache_limit = set_cache_limit
    mx.clear_cache = lambda: None
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    with _bounded_generation_cache():
        assert current == original
        assert limits == [0, 1024**3, original]

    assert limits == [0, 1024**3, original]


def test_generation_cache_enters_when_bound_and_restore_fail(monkeypatch):
    from rapid_mlx.image import engine

    original = 4 * 1024**3
    limits = []

    def set_cache_limit(limit):
        limits.append(limit)
        if len(limits) > 1:
            raise RuntimeError("set cache limit failed")
        return original

    mx = ModuleType("mlx.core")
    mx.set_cache_limit = set_cache_limit
    mx.clear_cache = lambda: None
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    with engine._bounded_generation_cache():
        assert limits == [0, 1024**3, original]

    assert limits == [0, 1024**3, original]
    assert engine._GENERATION_CACHE_DEPTH == 0


def test_generation_cache_preserves_outside_limit_change(monkeypatch):
    from rapid_mlx.image.engine import _bounded_generation_cache

    current = 4 * 1024**3
    outside_limit = 2 * 1024**3

    def set_cache_limit(limit):
        nonlocal current
        previous = current
        current = limit
        return previous

    mx = ModuleType("mlx.core")
    mx.set_cache_limit = set_cache_limit
    mx.clear_cache = lambda: None
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    with _bounded_generation_cache():
        mx.set_cache_limit(outside_limit)

    assert current == outside_limit


def test_generation_cache_ignores_restore_failure(monkeypatch):
    from rapid_mlx.image import engine

    original = 4 * 1024**3
    limits = []

    def set_cache_limit(limit):
        limits.append(limit)
        if len(limits) == 3:
            raise RuntimeError("restore failed")
        return original

    mx = ModuleType("mlx.core")
    mx.set_cache_limit = set_cache_limit
    mx.clear_cache = lambda: None
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    with engine._bounded_generation_cache():
        pass

    assert limits == [0, 1024**3, original]
    from rapid_mlx import _mlx_cache_limit

    assert _mlx_cache_limit._entry_limit is None
    assert engine._GENERATION_CACHE_DEPTH == 0


def test_generation_cache_ignores_clear_failure(monkeypatch):
    from rapid_mlx.image.engine import _bounded_generation_cache

    calls = []

    def clear_cache():
        calls.append(True)
        raise RuntimeError("clear failed")

    mx = ModuleType("mlx.core")
    mx.set_cache_limit = lambda limit: 4 * 1024**3
    mx.clear_cache = clear_cache
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    with _bounded_generation_cache():
        pass

    assert calls == [True]


@pytest.mark.parametrize("nested_request", [False, True])
def test_managed_cache_requests_respect_generation_bound(monkeypatch, nested_request):
    from pathlib import Path

    from rapid_mlx._mlx_cache_limit import request_cache_limit
    from rapid_mlx.image.engine import _bounded_generation_cache

    current = 4 * 1024**3
    clears = []

    def set_cache_limit(limit):
        nonlocal current
        previous = current
        current = limit
        return previous

    mx = ModuleType("mlx.core")
    mx.set_cache_limit = set_cache_limit
    mx.clear_cache = lambda: clears.append(True)
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    batched_source = (
        Path(__file__).parents[1] / "rapid_mlx/engine/batched.py"
    ).read_text()
    assert "import request_cache_limit" in batched_source
    assert "request_cache_limit(cache_limit)" in batched_source
    assert request_cache_limit(5 * 1024**3) == 4 * 1024**3
    assert current == 5 * 1024**3
    assert request_cache_limit(4 * 1024**3) == 5 * 1024**3
    with _bounded_generation_cache():
        assert request_cache_limit(3 * 1024**3) == 4 * 1024**3
        assert current == 1024**3
        if nested_request:
            with _bounded_generation_cache():
                assert request_cache_limit(2 * 1024**3) == 3 * 1024**3
                assert current == 1024**3
        assert current == 1024**3
        assert clears == []
    assert current == (2 if nested_request else 3) * 1024**3
    assert clears == [True]
