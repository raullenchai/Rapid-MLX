# SPDX-License-Identifier: Apache-2.0
"""CPU-only workgroup contracts, plus an isolated macOS thread-exit regression."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx.runtime import perf_workgroup as perf


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(perf.sys, "platform", "darwin")
    monkeypatch.delenv("RAPID_MLX_STEP_WORKGROUP", raising=False)
    audio = SimpleNamespace(AudioWorkIntervalCreate=Mock(return_value=123))
    system = SimpleNamespace(
        **{
            name: Mock(return_value=0)
            for name in (
                "os_workgroup_join",
                "os_workgroup_leave",
                "os_workgroup_interval_start",
                "os_workgroup_interval_finish",
            )
        },
        mach_absolute_time=Mock(return_value=100),
    )

    def timebase(ptr):
        ptr._obj.numer = 125
        ptr._obj.denom = 3
        return 0

    system.mach_timebase_info = Mock(side_effect=timebase)
    loader = Mock(side_effect=[audio, system])
    monkeypatch.setattr(perf.ctypes, "CDLL", loader)
    return SimpleNamespace(audio=audio, system=system, loader=loader)


@pytest.mark.parametrize("reason", ["platform", "environment", "symbol", "library"])
def test_unavailable_is_lazy_noop(native, monkeypatch, reason):
    if reason == "platform":
        monkeypatch.setattr(perf.sys, "platform", "linux")
    elif reason == "environment":
        monkeypatch.setenv("RAPID_MLX_STEP_WORKGROUP", "0")
    elif reason == "symbol":
        del native.system.os_workgroup_interval_finish
    else:
        native.loader.side_effect = OSError("unavailable")
    wg = perf.StepWorkgroup()
    native.loader.assert_not_called()
    for _ in range(2):
        with wg.interval():
            pass
    native.audio.AudioWorkIntervalCreate.assert_not_called()
    if reason in ("platform", "environment"):
        native.loader.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "raises"),
    [
        (name, raises)
        for name in ("create", "join", "timebase", "start", "finish")
        for raises in (False, True)
    ]
    + [("leave", True), ("clock", True)],
)
def test_native_failure_disables_without_raising(native, failure, raises, caplog):
    caplog.set_level("DEBUG", logger=perf.__name__)
    target = {
        "create": native.audio.AudioWorkIntervalCreate,
        "join": native.system.os_workgroup_join,
        "timebase": native.system.mach_timebase_info,
        "start": native.system.os_workgroup_interval_start,
        "finish": native.system.os_workgroup_interval_finish,
        "leave": native.system.os_workgroup_leave,
        "clock": native.system.mach_absolute_time,
    }[failure]
    target.side_effect = RuntimeError("native failure") if raises else None
    target.return_value = 0 if failure == "create" else 1
    wg = perf.StepWorkgroup()
    for _ in range(2):
        with wg.interval():
            pass
    assert target.call_count == 1
    assert sum("Step workgroup unavailable" in r.message for r in caplog.records) == 1
    assert native.system.os_workgroup_interval_finish.call_count == (
        1 if failure in ("finish", "leave") else 0
    )
    assert native.system.os_workgroup_leave.call_count == (
        1 if failure in ("start", "finish", "leave", "clock") else 0
    )
    if native.system.os_workgroup_leave.called:
        assert (
            native.system.os_workgroup_leave.call_args.args[1]
            is native.system.os_workgroup_join.call_args.args[1]
        )


def test_pairing_deadline_and_exception(native):
    wg = perf.StepWorkgroup()
    for fail in (False, True):
        try:
            with wg.interval():
                if fail:
                    raise ValueError("step failed")
        except ValueError as exc:
            assert str(exc) == "step failed"
    native.audio.AudioWorkIntervalCreate.assert_called_once()
    assert native.system.os_workgroup_join.call_count == 2
    assert native.system.os_workgroup_interval_start.call_count == 2
    assert native.system.os_workgroup_interval_finish.call_count == 2
    native.system.os_workgroup_interval_start.assert_called_with(
        123, 100, 100 + 50_000_000 * 3 // 125, None
    )
    joins = native.system.os_workgroup_join.call_args_list
    leaves = native.system.os_workgroup_leave.call_args_list
    assert len(leaves) == len(joins) == 2
    for join, leave in zip(joins, leaves, strict=True):
        assert join.args[0] == leave.args[0] == 123
        assert join.args[1] is leave.args[1]
    assert joins[0].args[1] is not joins[1].args[1]


@pytest.mark.parametrize("start_fails", [False, True])
@pytest.mark.parametrize("error", [ValueError, SystemExit, KeyboardInterrupt])
def test_step_exception_always_leaves(native, start_fails, error):
    wg = perf.StepWorkgroup()
    native.system.os_workgroup_interval_start.return_value = int(start_fails)
    with pytest.raises(error), wg.interval():
        raise error("step failed")
    native.system.os_workgroup_leave.assert_called_once()
    assert native.system.os_workgroup_interval_finish.call_count == int(not start_fails)
    assert not wg._lock.locked()


def test_idle_does_not_join(native):
    with perf.StepWorkgroup().interval(has_work=False):
        pass
    native.loader.assert_not_called()


def test_concurrent_use_skips_without_blocking(native):
    wg = perf.StepWorkgroup()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with wg.interval():

            def worker():
                with wg.interval():
                    return threading.get_ident()

            assert pool.submit(worker).result(timeout=5) != threading.get_ident()
            native.system.os_workgroup_join.assert_called_once()
            native.system.os_workgroup_leave.assert_not_called()
        pool.submit(worker).result(timeout=5)
    native.audio.AudioWorkIntervalCreate.assert_called_once()
    assert native.system.os_workgroup_join.call_count == 2
    assert native.system.os_workgroup_leave.call_count == 2


def test_initialization_contention_skips_and_failure_disables_all_threads(native):
    wg = perf.StepWorkgroup()
    creating = threading.Event()
    resume = threading.Event()

    def create(*_):
        creating.set()
        assert resume.wait(timeout=5)
        return 0

    native.audio.AudioWorkIntervalCreate.side_effect = create

    def worker():
        with wg.interval():
            pass

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        try:
            assert creating.wait(timeout=5)
            worker()  # Must skip immediately, even while initialization is underway.
            native.audio.AudioWorkIntervalCreate.assert_called_once()
            native.system.os_workgroup_join.assert_not_called()
        finally:
            resume.set()
        future.result(timeout=5)
        worker()  # A different thread sees the process-wide disabled state.
    native.audio.AudioWorkIntervalCreate.assert_called_once()
    native.system.os_workgroup_join.assert_not_called()


def test_nested_use_skips(native):
    wg = perf.StepWorkgroup()
    with wg.interval(), wg.interval():
        native.system.os_workgroup_join.assert_called_once()
    native.system.os_workgroup_leave.assert_called_once()


@pytest.fixture
def engine(native):
    # Extract only the real tensor-free step methods for CPU-only CI.
    source = Path(__file__).parents[1] / "rapid_mlx" / "engine_core.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_step_once", "_step_coalesced", "generate_batch_sync"}
    ]
    # ``__package__`` resolves generate_batch_sync's ``from .request import``.
    namespace = {"step_workgroup": perf.StepWorkgroup(), "__package__": "rapid_mlx"}
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace["EngineCore"]()


@pytest.mark.parametrize("method", ["_step_once", "_step_coalesced"])
@pytest.mark.parametrize("fail", [False, True])
def test_engine_brackets_each_step(native, engine, method, fail):
    events = []
    for name, event in (
        ("os_workgroup_join", "join"),
        ("os_workgroup_interval_start", "start"),
        ("os_workgroup_interval_finish", "finish"),
        ("os_workgroup_leave", "leave"),
    ):
        getattr(native.system, name).side_effect = lambda *_, event=event: (
            events.append(event) or 0
        )
    calls = 0

    def step():
        nonlocal calls
        calls += 1
        events.append("step")
        if fail and calls == (2 if method == "_step_coalesced" else 1):
            raise ValueError("step failed")
        return SimpleNamespace(finished_request_ids=[], has_work=True)

    engine.scheduler = SimpleNamespace(get_num_waiting=lambda: 0, step=step)
    if method == "_step_once":
        if fail:
            with pytest.raises(ValueError, match="step failed"):
                engine._step_once()
        else:
            assert engine._step_once().has_work
        count = 1
    else:
        outputs, error = engine._step_coalesced(3)
        count = 2 if fail else 3
        assert len(outputs) == (1 if fail else 3)
        assert isinstance(error, ValueError) if fail else error is None
    assert events == ["join", "start", "step", "finish", "leave"] * count


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS workgroup ABI")
def test_native_short_lived_thread_exits_in_subprocess():
    code = """
import ctypes
import threading
from rapid_mlx.runtime.perf_workgroup import StepWorkgroup
try:
    audio = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    getattr(audio, "AudioWorkIntervalCreate")
    for name in ("os_workgroup_join", "os_workgroup_leave",
                 "os_workgroup_interval_start", "os_workgroup_interval_finish",
                 "mach_absolute_time", "mach_timebase_info"):
        getattr(system, name)
except (OSError, AttributeError):
    raise SystemExit(77)
wg = StepWorkgroup()
completed = []
def worker():
    for _ in range(3):
        with wg.interval():
            assert not wg._disabled
            completed.append(True)
thread = threading.Thread(target=worker)
thread.start()
thread.join(timeout=10)
assert not thread.is_alive()
assert len(completed) == 3
assert wg._wg and not wg._disabled
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "RAPID_MLX_STEP_WORKGROUP": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 77:
        pytest.skip("native workgroup library or symbol unavailable")
    assert result.returncode == 0, result.stdout + result.stderr


def test_generate_batch_sync_brackets_each_step(native, engine):
    from rapid_mlx.request import SamplingParams

    events = []
    native.system.os_workgroup_join.side_effect = lambda *_: events.append("join") or 0
    native.system.os_workgroup_leave.side_effect = lambda *_: events.append("leave")
    pending = []

    def add_request(request):
        pending.append(request.request_id)

    def step():
        events.append("step")
        rid = pending.pop(0)
        done = SimpleNamespace(request_id=rid, finished=True)
        return SimpleNamespace(outputs=[done])

    engine.scheduler = SimpleNamespace(
        add_request=add_request,
        has_requests=lambda: bool(pending),
        step=step,
        remove_finished_request=lambda rid: None,
    )
    outputs = engine.generate_batch_sync([[1], [2]], SamplingParams(max_tokens=1))
    assert [o.finished for o in outputs] == [True, True]
    assert events == ["join", "step", "leave"] * 2
