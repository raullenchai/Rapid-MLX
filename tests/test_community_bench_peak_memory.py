# SPDX-License-Identifier: Apache-2.0
"""``runner`` peak-memory helpers — MLX-free (a fake ``mlx.core`` is
installed per test), so the hosted Linux lane covers both API spellings."""

from __future__ import annotations

from vllm_mlx.community_bench import runner


def test_peak_memory_prefers_the_undeprecated_mlx_api(monkeypatch) -> None:
    """mlx >= 0.22 spells the counters ``mx.get_peak_memory`` /
    ``mx.reset_peak_memory``; the ``mx.metal`` spellings print a deprecation
    line on every call, which used to land inside ``benchmark run`` output."""
    import sys
    import types

    calls: list[str] = []

    def install(core) -> None:
        # ``import mlx.core as mx`` resolves the ``core`` attribute of the
        # ``mlx`` package first, so both entries must point at the fake.
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=core))
        monkeypatch.setitem(sys.modules, "mlx.core", core)

    metal = types.SimpleNamespace(
        get_peak_memory=lambda: calls.append("metal.get") or 5 * 1024 * 1024,
        reset_peak_memory=lambda: calls.append("metal.reset"),
    )
    modern = types.SimpleNamespace(
        get_peak_memory=lambda: calls.append("get") or 3 * 1024 * 1024,
        reset_peak_memory=lambda: calls.append("reset"),
        metal=metal,
    )
    install(modern)
    assert runner._read_peak_ram_mb() == 3
    runner._reset_peak_ram()
    assert calls == ["get", "reset"]

    legacy = types.SimpleNamespace(metal=metal)
    install(legacy)
    calls.clear()
    assert runner._read_peak_ram_mb() == 5
    runner._reset_peak_ram()
    assert calls == ["metal.get", "metal.reset"]

    bare = types.SimpleNamespace()
    install(bare)
    assert runner._read_peak_ram_mb() is None
    runner._reset_peak_ram()


def test_peak_memory_helpers_swallow_runtime_errors(monkeypatch) -> None:
    """A counter that raises (no Metal device, driver error) degrades to
    ``None`` / a no-op instead of aborting the bench."""
    import sys
    import types

    def boom():
        raise ValueError("no metal device")

    def boom_os():
        raise OSError("driver")

    core = types.SimpleNamespace(get_peak_memory=boom, reset_peak_memory=boom_os)
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    assert runner._read_peak_ram_mb() is None
    runner._reset_peak_ram()  # must not raise
