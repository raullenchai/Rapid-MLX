# SPDX-License-Identifier: Apache-2.0
"""Cancellation cleans up benchmark servers and reports a conventional exit."""

import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx.bench import _server
from rapid_mlx.community_bench import cli
from rapid_mlx.community_bench.acceleration import AccelerationRequest


def test_serve_interrupt_kills_unresponsive_group(monkeypatch, tmp_path):
    real_popen = subprocess.Popen
    child = None
    script = """
import http.server, signal, sys
signal.signal(signal.SIGTERM, signal.SIG_IGN)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""

    def popen(cmd, **kwargs):
        nonlocal child
        child = real_popen(
            [sys.executable, "-c", script, cmd[cmd.index("--port") + 1]], **kwargs
        )
        return child

    monkeypatch.setattr(_server.subprocess, "Popen", popen)
    try:
        with (
            pytest.raises(KeyboardInterrupt),
            _server.serve("fake", log_path=tmp_path / "server.log"),
        ):
            start = time.monotonic()
            raise KeyboardInterrupt
        assert time.monotonic() - start < 4
        with pytest.raises(ProcessLookupError):
            os.killpg(child.pid, 0)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def test_serve_normal_exit_uses_full_grace(monkeypatch, tmp_path):
    calls = []
    real_terminate = _server._terminate

    def spy(proc, **kwargs):
        calls.append(kwargs["grace_s"])
        real_terminate(proc, **kwargs)

    monkeypatch.setattr(_server, "_wait_for_health", lambda *args: None)
    monkeypatch.setattr(_server, "_terminate", spy)
    real_popen = subprocess.Popen
    monkeypatch.setattr(
        _server.subprocess,
        "Popen",
        lambda cmd, **kwargs: real_popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
        ),
    )
    with _server.serve("fake", log_path=tmp_path / "server.log"):
        pass
    assert calls == [10]


def test_serve_removes_temporary_log(monkeypatch):
    logs = []

    def popen(cmd, **kwargs):
        logs.append(Path(kwargs["stdout"].name))
        return SimpleNamespace()

    monkeypatch.setattr(_server.subprocess, "Popen", popen)
    monkeypatch.setattr(_server, "_wait_for_health", lambda *args: None)
    monkeypatch.setattr(_server, "_terminate", lambda *args, **kwargs: None)
    with _server.serve("fake"):
        assert len(logs) == 1
        assert logs[0].exists()
    assert not logs[0].exists()


@pytest.mark.parametrize("isolated", [True, False])
def test_second_interrupt_kills_and_propagates(monkeypatch, isolated):
    signals = []

    class Proc:
        pid = 1234
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            signals.append(signal.SIGTERM)

        def kill(self):
            signals.append(signal.SIGKILL)

        def wait(self, timeout):
            if len(signals) == 1:
                raise KeyboardInterrupt
            self.returncode = 0

    monkeypatch.setattr(_server.os, "killpg", lambda pid, sig: signals.append(sig))
    with pytest.raises(KeyboardInterrupt):
        _server._terminate(Proc(), isolated_process_group=isolated)
    assert signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX process groups")
def test_interrupt_during_escalation_kill(monkeypatch):
    script = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(30)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, start_new_session=True
    )
    real_killpg = os.killpg
    kill_attempts = 0

    def killpg(pid, sig):
        nonlocal kill_attempts
        if sig == signal.SIGKILL:
            kill_attempts += 1
            if kill_attempts == 1:
                raise KeyboardInterrupt
        return real_killpg(pid, sig)

    try:
        assert proc.stdout.readline() == b"ready\n"
        monkeypatch.setattr(_server.os, "killpg", killpg)
        with pytest.raises(KeyboardInterrupt):
            _server._terminate(proc, grace_s=0)
        assert kill_attempts == 2
        assert proc.returncode == -signal.SIGKILL
        with pytest.raises(ProcessLookupError):
            real_killpg(proc.pid, 0)
    finally:
        if proc.poll() is None:
            real_killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX process groups")
def test_terminate_kills_child_after_leader_exits():
    script = """
import subprocess, sys, time
subprocess.Popen([sys.executable, '-c',
    'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
    'print("ready", flush=True); time.sleep(30)'])
time.sleep(30)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert proc.stdout.readline() == b"ready\n"
        start = time.monotonic()
        _server._terminate(proc, grace_s=2)
        assert time.monotonic() - start >= 1.8
        # The child inherits stdout, so EOF proves it has stopped too.
        assert select.select([proc.stdout], [], [], 5)[0]
        assert proc.stdout.read() == b""
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX process groups")
def test_terminate_allows_child_to_exit_during_grace(tmp_path):
    marker = tmp_path / "child-exited"
    script = """
import subprocess, sys, time
child = '''
import signal, sys, time
def stop(signum, frame):
    time.sleep(0.3)
    open(sys.argv[1], "w").write("exited")
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
print("ready", flush=True)
time.sleep(30)
'''
subprocess.Popen([sys.executable, '-c', child, sys.argv[1]])
time.sleep(30)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(marker)],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert proc.stdout.readline() == b"ready\n"
        start = time.monotonic()
        _server._terminate(proc, grace_s=2)
        assert time.monotonic() - start < 1.8
        assert marker.read_text() == "exited"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX process groups")
def test_terminate_already_reaped_leader_kills_child():
    script = """
import subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
print(child.pid, flush=True)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, start_new_session=True
    )
    try:
        child_pid = int(proc.stdout.readline())
        proc.wait(timeout=5)
        _server._terminate(proc, grace_s=2)
        deadline = time.monotonic() + 5
        while True:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            assert time.monotonic() < deadline, "child was not reaped"
            time.sleep(0.05)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        proc.stdout.close()


def test_terminate_already_reaped_leader_empty_group(monkeypatch):
    def missing(*args):
        raise ProcessLookupError

    monkeypatch.setattr(_server.os, "killpg", missing)
    started = time.monotonic()
    _server._terminate(SimpleNamespace(pid=1234, returncode=0))
    assert time.monotonic() - started < 1


def test_terminate_reaps_leader_when_initial_group_signal_fails(monkeypatch):
    waits = []

    class Proc:
        pid = 1234
        returncode = None

        def wait(self, timeout):
            waits.append(timeout)
            self.returncode = 0
            return self.returncode

    def killpg(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(_server.os, "killpg", killpg)
    proc = Proc()
    _server._terminate(proc)
    assert waits == [5]
    assert proc.returncode == 0


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX process groups")
def test_terminate_handles_missing_group_during_escalation(monkeypatch):
    signals = []

    class Proc:
        pid = 1234
        returncode = None

        def wait(self, timeout):
            if timeout != 5:
                raise subprocess.TimeoutExpired("server", timeout)
            self.returncode = 0

    def killpg(pid, sig):
        signals.append(sig)
        if sig == signal.SIGKILL:
            raise ProcessLookupError

    monkeypatch.setattr(_server.os, "killpg", killpg)
    _server._terminate(Proc(), grace_s=0)
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_terminate_handles_timeout_in_final_reap():
    waits = []

    class Proc:
        pid = 1234
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout):
            waits.append(timeout)
            raise subprocess.TimeoutExpired("server", timeout)

    _server._terminate(Proc(), isolated_process_group=False, grace_s=0)
    assert waits == [0, 5]


@pytest.mark.parametrize("error", [PermissionError, KeyboardInterrupt])
def test_terminate_group_poll_errors(monkeypatch, error):
    signals = []

    class Proc:
        pid = 1234
        returncode = None

        def wait(self, timeout):
            self.returncode = 0

    def killpg(pid, sig):
        signals.append(sig)
        if sig == 0:
            raise error

    monkeypatch.setattr(_server.os, "killpg", killpg)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            _server._terminate(Proc())
        assert signals == [signal.SIGTERM, 0, signal.SIGKILL]
    else:
        _server._terminate(Proc())
        assert signals == [signal.SIGTERM, 0]


@pytest.mark.parametrize("isolated", [True, False])
def test_interrupt_during_escalation_wait(monkeypatch, isolated):
    signals = []
    waits = []

    class Proc:
        pid = 1234
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            signals.append(signal.SIGTERM)

        def kill(self):
            signals.append(signal.SIGKILL)

        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("server", timeout)
            if len(waits) == 2:
                assert signals[-1] == signal.SIGKILL
                raise KeyboardInterrupt
            self.returncode = 0

    monkeypatch.setattr(_server.os, "killpg", lambda pid, sig: signals.append(sig))
    with pytest.raises(KeyboardInterrupt):
        _server._terminate(Proc(), isolated_process_group=isolated)
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert len(waits) == 3


def args(json_mode=False):
    return SimpleNamespace(
        benchmark_action="run", benchmark_model="fake", json=json_mode
    )


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("flush_fails", [False, True])
def test_hard_exit_cancelled_runs_cleanup_and_flushes_before_exit(
    monkeypatch, cleanup_fails, flush_fails
):
    calls = []

    def cleanup():
        calls.append("cleanup")
        if cleanup_fails:
            raise SystemExit("cleanup interrupted")

    def flush(name):
        calls.append(name)
        if flush_fails:
            raise BrokenPipeError("stream closed")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(cli.atexit, "_run_exitfuncs", cleanup)
    monkeypatch.setattr(
        cli.sys, "stdout", SimpleNamespace(flush=lambda: flush("stdout"))
    )
    monkeypatch.setattr(
        cli.sys, "stderr", SimpleNamespace(flush=lambda: flush("stderr"))
    )
    monkeypatch.setattr(cli.os, "_exit", lambda code: calls.append(("exit", code)))

    cli._hard_exit_cancelled(143)

    assert calls == ["cleanup", "stdout", "stderr", ("exit", 143)]


def test_cancel_backstop_exits_without_waiting_for_cleanup(monkeypatch):
    calls = []
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(cli.atexit, "_run_exitfuncs", lambda: calls.append("cleanup"))
    monkeypatch.setattr(cli.os, "_exit", lambda code: calls.append(("exit", code)))

    cli._cancel_backstop(130)

    assert calls == [("exit", 130)]


@pytest.mark.parametrize("helper", [cli._hard_exit_cancelled, cli._cancel_backstop])
def test_cancel_exit_helpers_are_inert_under_pytest(monkeypatch, helper):
    calls = []
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "cancellation guard")
    monkeypatch.setattr(cli.atexit, "_run_exitfuncs", lambda: calls.append("cleanup"))
    monkeypatch.setattr(cli.os, "_exit", calls.append)
    helper(130)
    assert calls == []


def test_cli_cancels_backstop_when_runner_recovers_from_signal(monkeypatch, capsys):
    calls = []
    handlers = {}

    def install(sig, handler):
        previous = handlers.get(sig)
        handlers[sig] = handler
        return previous

    class Timer:
        def __init__(self, interval, callback, arguments):
            assert interval == cli._CANCEL_BACKSTOP_SECONDS
            assert callback is cli._cancel_backstop
            assert arguments == [143]

        def start(self):
            calls.append("start")

        def cancel(self):
            calls.append("cancel")

    def recover(*args, **kwargs):
        with pytest.raises(KeyboardInterrupt):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return {"recovered": True}

    monkeypatch.setattr(cli.signal, "signal", install)
    monkeypatch.setattr(cli.threading, "Timer", Timer)
    monkeypatch.setattr(cli, "run_local", recover)
    monkeypatch.setattr(cli, "request_from_args", lambda args: AccelerationRequest())
    assert cli.benchmark_command(args(True)) == 0
    assert calls == ["start", "cancel"]
    assert json.loads(capsys.readouterr().out) == {"recovered": True}
    assert handlers == {signal.SIGINT: None, signal.SIGTERM: None}


def test_cli_interrupt_with_abandoned_worker_hard_exits(monkeypatch, capsys):
    calls = []

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_local", interrupt)
    monkeypatch.setattr(cli, "request_from_args", lambda args: AccelerationRequest())
    monkeypatch.setattr(cli, "abandoned_native_worker", lambda: True)
    monkeypatch.setattr(cli, "_hard_exit_cancelled", calls.append)

    assert cli.benchmark_command(args()) == 130
    assert calls == [130]
    assert "Benchmark cancelled" in capsys.readouterr().err


def test_cli_interrupt(monkeypatch, capsys):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_local", interrupt)
    monkeypatch.setattr(cli, "request_from_args", lambda args: AccelerationRequest())
    assert cli.benchmark_command(args()) == 130
    assert (
        capsys.readouterr().err
        == "Benchmark cancelled; benchmark processes were stopped.\n"
    )


def test_cli_sigterm_restores_handler(monkeypatch, capsys):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def terminate(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1)

    monkeypatch.setattr(cli, "run_local", terminate)
    monkeypatch.setattr(cli, "request_from_args", lambda args: AccelerationRequest())
    assert cli.benchmark_command(args()) == 143
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())
    assert "Traceback" not in capsys.readouterr().err


def test_cli_interrupt_json(monkeypatch, capsys):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_local", interrupt)
    monkeypatch.setattr(cli, "request_from_args", lambda args: AccelerationRequest())
    assert cli.benchmark_command(args(True)) == 130
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {
        "error": "benchmark cancelled",
        "cancelled": True,
    }


@pytest.mark.parametrize(
    ("sig", "baseline"),
    [(signal.SIGINT, True), (signal.SIGTERM, True), (signal.SIGINT, False)],
)
def test_cli_cancels_blocked_text_load(tmp_path, sig, baseline):
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core"], capture_output=True, check=False
    )
    if probe.returncode:
        pytest.skip("MLX cannot load in this environment")
    script = r"""
import os, sys, threading
from rapid_mlx.community_bench import local_runner, workspace
from rapid_mlx import cli
import rapid_mlx.utils.tokenizer as tokenizer

def block(*args, **kwargs):
    print("LOADING", file=sys.__stdout__, flush=True)
    threading.Event().wait()

local_runner.collect = lambda: (None, None)
local_runner.run_conditions = lambda: None
local_runner.resolve_provenance = lambda: None
local_runner.model_is_cached = lambda *args: True
local_runner._ensure_model_downloaded = lambda *args: None
workspace.benchmark_runtime_readiness = lambda *args: {"status": "ready"}
tokenizer.load_model_with_fallback = block
if os.environ["TEST_BASELINE"] == "0":
    import rapid_mlx.server as server
    server.load_model = block
sys.argv = ["rapid-mlx", "benchmark", "run", "lfm2.5-2.6b-4bit"]
if os.environ["TEST_BASELINE"] == "1":
    sys.argv.append("--baseline")
cli.main()
"""
    env = os.environ.copy()
    env.update(
        RAPID_MLX_HOME=str(tmp_path),
        XDG_STATE_HOME=str(tmp_path),
        RAPID_MLX_TELEMETRY="0",
        DO_NOT_TRACK="1",
        TEST_BASELINE="1" if baseline else "0",
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        ready, _, _ = select.select([proc.stdout], [], [], 30)
        assert ready, "loader did not start"
        assert proc.stdout.readline().strip() == "LOADING"
        proc.send_signal(sig)
        assert proc.wait(timeout=10) == 128 + sig
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()


def test_outstanding_futures_include_worker_blocked_in_initializer():
    from rapid_mlx.community_bench import local_runner

    release = threading.Event()
    executor = local_runner._TrackingExecutor(
        max_workers=1, initializer=lambda: release.wait(5)
    )
    try:
        executor.submit(lambda: None)
        # The worker is still inside the initializer: nothing is "running",
        # yet joining the executor would block.
        assert executor.futures.any_outstanding()
    finally:
        release.set()
        executor.shutdown(wait=True)
    assert not executor.futures.any_outstanding()


def test_finished_futures_are_released():
    from rapid_mlx.community_bench import local_runner

    executor = local_runner._TrackingExecutor(max_workers=1)
    try:
        for _ in range(50):
            executor.submit(lambda: object()).result()
        # result() can return just before the done callback discards the
        # future, so wait briefly for the last discard.
        deadline = time.monotonic() + 5
        while executor.futures._pending:
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        executor.shutdown(wait=True)


def test_cancelled_queued_future_with_worker_in_initializer_is_outstanding():
    from rapid_mlx.community_bench import local_runner

    entered = threading.Event()
    release = threading.Event()

    def initializer():
        entered.set()
        release.wait(5)

    executor = local_runner._TrackingExecutor(max_workers=1, initializer=initializer)
    try:
        future = executor.submit(lambda: None)
        assert entered.wait(5)
        # Cancellation of the queued work (as asyncio does for a cancelled
        # run_in_executor await) must not hide the worker that is still
        # initializing; joining it now would block.
        future.cancel()
        assert executor.futures.any_outstanding()
        executor.shutdown(wait=False, cancel_futures=True)
    finally:
        release.set()
    deadline = time.monotonic() + 5
    while executor.futures.any_outstanding():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_join_workers_reports_worker_still_blocked():
    from rapid_mlx.community_bench import local_runner

    entered = threading.Event()
    release = threading.Event()

    def initializer():
        entered.set()
        release.wait(5)

    executor = local_runner._TrackingExecutor(max_workers=1, initializer=initializer)
    try:
        future = executor.submit(lambda: None)
        assert entered.wait(5)
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        started = time.monotonic()
        assert executor.join_workers(timeout=0.2) is False
        assert time.monotonic() - started < 1
    finally:
        release.set()
    assert executor.join_workers(timeout=5) is True
