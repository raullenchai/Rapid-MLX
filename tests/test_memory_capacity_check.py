# SPDX-License-Identifier: Apache-2.0
"""Tests for the pre-flight memory check (issue #324).

On low-memory Apple Silicon (e.g. Mac mini M4 24 GB), loading a model that
forces unified memory past ~85% of total can trip the iBoot AMCC async-abort
firmware path and **kernel-panic the entire machine** rather than raise a
userspace OOM. ``_check_memory_capacity`` warns the user before this
happens. It is best-effort — never aborts, falls through silently when it
can't read sizes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vllm_mlx.cli import _check_memory_capacity


def _fake_psutil(total_gb: float, used_gb: float = 0.0):
    """Build a minimal psutil mock with both ``total`` and ``available``.

    The pre-flight check now uses *projected* pressure (already-used + working
    set), not just (working / total), so tests must specify how much RAM is
    already in use. Default ``used_gb=0`` means a freshly-booted machine.
    """
    fake = MagicMock()
    total_bytes = int(total_gb * (1024**3))
    used_bytes = int(used_gb * (1024**3))
    fake.virtual_memory.return_value = MagicMock(
        total=total_bytes,
        available=max(0, total_bytes - used_bytes),
    )
    return fake


def _patch_size_bytes(monkeypatch, size_gb: float):
    """Stub the local-path branch of ``_check_memory_capacity`` to report
    a fixed model size without doing real I/O or HF lookups."""

    def _fake_isdir(p):
        return True

    def _fake_walk(p):
        # One file of the requested size at the model root.
        yield p, [], ["weights.safetensors"]

    def _fake_getsize(p):
        return int(size_gb * (1024**3))

    monkeypatch.setattr("os.path.isdir", _fake_isdir)
    monkeypatch.setattr("os.walk", _fake_walk)
    monkeypatch.setattr("os.path.getsize", _fake_getsize)


def test_hard_warning_fires_on_24gb_mac_with_14gb_model_realistic_load(
    monkeypatch, capsys
):
    """The exact issue #324 scenario: 14 GB Gemma-4-26B-4bit on a 24 GB
    Mac mini M4. Realistic 6 GB already used by macOS + browser before
    serve starts → working set 14 * 1.5 = 21 GB → projected (6 + 21) / 24
    = 113% → HARD warning naming the kernel panic risk. The reporter who
    actually hit this gets the strongest message, not a soft hint."""
    _patch_size_bytes(monkeypatch, size_gb=14.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(24.0, used_gb=6.0)}):
        _check_memory_capacity("/local/path/to/gemma-4-26b-4bit")
    out = capsys.readouterr().out
    assert "kernel panic" in out, (
        f"the very case that filed the issue must hit the hard tier: {out!r}"
    )
    assert "issue #324" in out


def test_hard_warning_still_fires_at_fresh_boot_on_24gb_mac(monkeypatch, capsys):
    """Same 14 GB model on the same 24 GB Mac, but at boot with ~0 used:
    projected pressure (0 + 21) / 24 = 87.5% → still HARD tier (>= 85%).
    Confirms the formula doesn't silently slide into "soft" at fresh boot.
    """
    _patch_size_bytes(monkeypatch, size_gb=14.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(24.0, used_gb=0.0)}):
        _check_memory_capacity("/local/path/to/gemma-4-26b-4bit")
    out = capsys.readouterr().out
    assert "Memory pressure" in out, f"expected warning, got: {out!r}"
    # 0 + 21 = 21 / 24 = 87.5% → HARD tier
    assert "kernel panic" in out


def test_soft_warning_fires_on_borderline_pressure(monkeypatch, capsys):
    """Soft tier (65% ≤ ratio < 85%): 10 GB model on 32 GB Mac with
    8 GB used → projected (8 + 15) / 32 = 71.9% → soft warning.
    Verify the soft message (note, not warning) and the recommended
    --gpu-memory-utilization 0.85."""
    _patch_size_bytes(monkeypatch, size_gb=10.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(32.0, used_gb=8.0)}):
        _check_memory_capacity("/local/path/to/medium")
    out = capsys.readouterr().out
    assert "Memory pressure note" in out
    # Soft-tier markers: no "issue #324" callout, no 0.75 recommendation,
    # no "iBoot AMCC threshold" language. (Both tiers mention "kernel
    # panics" as a hint, so we don't gate on that string.)
    assert "issue #324" not in out
    assert "iBoot AMCC" not in out
    assert "--gpu-memory-utilization 0.75" not in out
    assert "--gpu-memory-utilization 0.85" in out


def test_hard_warning_fires_on_catastrophic_mismatch(monkeypatch, capsys):
    """At ratio ≥ 0.85, the warning escalates to red + names the kernel
    panic risk + suggests --gpu-memory-utilization 0.75. Pin the message
    surface so a future refactor doesn't silently weaken it."""
    # 18 GB on 24 GB Mac → 1.5x = 27 GB working set → 112% (catastrophic).
    _patch_size_bytes(monkeypatch, size_gb=18.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(24.0, used_gb=0.0)}):
        _check_memory_capacity("/local/path/to/large")
    out = capsys.readouterr().out
    assert "kernel panic" in out, f"hard warning must name the risk: {out!r}"
    assert "issue #324" in out
    assert "--gpu-memory-utilization 0.75" in out


def test_already_loaded_pressure_triggers_warning(monkeypatch, capsys):
    """Codex r1 finding: a 10 GB model on a 24 GB Mac that already has
    8 GB used by macOS + Chrome lands at projected (8 + 15) / 24 = 95.8%
    — kernel-panic territory — but the OLD formula (working / total) gave
    only 62.5% and stayed silent. This test pins the new behavior so a
    refactor can't regress to the absolute formula."""
    _patch_size_bytes(monkeypatch, size_gb=10.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(24.0, used_gb=8.0)}):
        _check_memory_capacity("/local/path/to/midsize")
    out = capsys.readouterr().out
    assert "kernel panic" in out, (
        f"must catch the realistic 'machine already in use' case: {out!r}"
    )


def test_no_warning_with_comfortable_headroom(monkeypatch, capsys):
    """A small model on a big machine must produce zero output. The check
    is best-effort and should be invisible when it has nothing to add."""
    # 4 GB model on 96 GB Mac Studio with 8 GB used → projected (8 + 6) / 96
    # = 14.6% → silent.
    _patch_size_bytes(monkeypatch, size_gb=4.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(96.0, used_gb=8.0)}):
        _check_memory_capacity("/local/path/to/small")
    out = capsys.readouterr().out
    assert out == "", f"comfortable model must not warn; got: {out!r}"


def test_silent_when_psutil_unavailable(monkeypatch, capsys):
    """Best-effort: if psutil can't be imported, fall through silently
    rather than blocking startup.

    Round-1 review noted that the previous version of this test was
    vacuous — patching ``builtins.__import__`` doesn't intercept an
    import that's already in ``sys.modules`` (psutil is installed in
    our test env). The documented "force ImportError" idiom is
    ``sys.modules[<name>] = None``, which causes a fresh ``import``
    statement to raise ``ModuleNotFoundError``. ``monkeypatch.setitem``
    auto-restores after the test so we don't poison the rest of the
    test session.
    """
    import sys as _sys

    _patch_size_bytes(monkeypatch, size_gb=20.0)
    monkeypatch.setitem(_sys.modules, "psutil", None)
    _check_memory_capacity("/local/path/to/anything")
    out = capsys.readouterr().out
    assert out == "", f"missing psutil must be silent; got: {out!r}"


def test_silent_when_size_lookup_fails(monkeypatch, capsys):
    """If neither local path nor HF cache nor HF API can resolve the size,
    skip the check — the loader's error paths handle real failures."""

    def _fake_isdir(p):
        return False

    monkeypatch.setattr("os.path.isdir", _fake_isdir)

    # HF lookups all return 0 / raise.
    def _no_cache(*a, **kw):
        return None

    def _api_fail(*a, **kw):
        raise RuntimeError("offline")

    with (
        patch("huggingface_hub.try_to_load_from_cache", _no_cache),
        patch("huggingface_hub.model_info", _api_fail),
        patch.dict("sys.modules", {"psutil": _fake_psutil(24.0)}),
    ):
        _check_memory_capacity("mlx-community/Some-Unreachable-Model")
    out = capsys.readouterr().out
    assert out == "", f"unresolvable size must be silent; got: {out!r}"


def test_never_calls_sys_exit(monkeypatch):
    """Defensive: the memory check is advisory, must never abort startup
    even on a catastrophic mismatch (212 GB model on 8 GB machine)."""
    _patch_size_bytes(monkeypatch, size_gb=212.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(8.0)}):
        # Must NOT raise SystemExit. capsys not asserted — we just need
        # the call to complete.
        _check_memory_capacity("/local/path/to/huge")


def test_warning_includes_actionable_recommendations(monkeypatch, capsys):
    """The warning must give the user a concrete next step (rapid-mlx
    models / --gpu-memory-utilization), not just describe the problem.
    Pins the actionability of the message."""
    _patch_size_bytes(monkeypatch, size_gb=14.0)
    with patch.dict("sys.modules", {"psutil": _fake_psutil(24.0, used_gb=0.0)}):
        _check_memory_capacity("/local/path/to/gemma-4-26b-4bit")
    out = capsys.readouterr().out
    assert "--gpu-memory-utilization" in out


def _function_loads_global(func, name: str) -> bool:
    """Bytecode-level check: does ``func`` reference ``name`` as a
    global / name lookup?

    More robust than ``inspect.getsource`` + string grep:
    - won't false-positive on a stale comment or docstring containing
      the literal symbol (those are stripped at compile time)
    - catches the symbol regardless of how surrounding code formats
    - misses only the case where the call moves through a helper, in
      which case the helper itself needs a wiring test of its own
    """
    import dis

    return any(
        ins.opname in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF") and ins.argval == name
        for ins in dis.get_instructions(func)
    )


def test_check_is_wired_into_serve_and_bench():
    """The pre-flight is useless if we forget to call it. Bytecode
    inspection asserts both serve_command and bench_command actually
    reference ``_check_memory_capacity`` so a future refactor that
    silently drops the call is caught.

    Round-2 review noted that the prior source-grep version passed
    on stale comments containing the literal symbol. Bytecode loads
    don't include comments — only real references survive compilation.
    """
    from vllm_mlx import cli

    assert _function_loads_global(cli.serve_command, "_check_memory_capacity"), (
        "serve_command must call _check_memory_capacity"
    )
    assert _function_loads_global(cli.bench_command, "_check_memory_capacity"), (
        "bench_command must call _check_memory_capacity"
    )
