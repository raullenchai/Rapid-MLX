# SPDX-License-Identifier: Apache-2.0
"""Cancellation cleans up benchmark servers and reports a conventional exit."""

import fcntl
import io
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


@pytest.mark.parametrize("echo", [True, False])
def test_serve_echoes_real_child_log(monkeypatch, echo, capsys):
    real_popen = subprocess.Popen
    script = """
import http.server, sys
print('ENGINE-DEBUG-LINE', flush=True)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""
    monkeypatch.setattr(
        _server.subprocess,
        "Popen",
        lambda cmd, **kwargs: real_popen(
            [sys.executable, "-c", script, cmd[cmd.index("--port") + 1]], **kwargs
        ),
    )
    output = io.StringIO()
    with _server.serve("fake", echo_log=output if echo else None):
        pass
    assert ("ENGINE-DEBUG-LINE" in output.getvalue()) is echo
    assert "ENGINE-DEBUG-LINE" not in capsys.readouterr().err


def test_serve_echoes_child_log_to_pipe(monkeypatch):
    real_popen = subprocess.Popen
    real_write = os.write
    writes = []
    script = """
import http.server, sys
print('ENGINE-DEBUG-LINE', flush=True)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""
    monkeypatch.setattr(
        _server.subprocess,
        "Popen",
        lambda cmd, **kwargs: real_popen(
            [sys.executable, "-c", script, cmd[cmd.index("--port") + 1]], **kwargs
        ),
    )
    read_fd, write_fd = os.pipe()

    def write(fd, data):
        if fd == write_fd:
            writes.append(data)
        return real_write(fd, data)

    monkeypatch.setattr(_server.os, "write", write)
    try:
        with (
            os.fdopen(write_fd, "w") as output,
            _server.serve("fake", echo_log=output),
        ):
            assert select.select([read_fd], [], [], 3)[0]
            assert b"ENGINE-DEBUG-LINE\n" in os.read(read_fd, 4096)
        assert any(b"ENGINE-DEBUG-LINE\n" in data for data in writes)
    finally:
        os.close(read_fd)


def test_serve_stops_echo_when_pipe_is_full(monkeypatch):
    real_popen = subprocess.Popen
    script = """
import http.server, sys
print('ENGINE-DEBUG-LINE', flush=True)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""
    monkeypatch.setattr(
        _server.subprocess,
        "Popen",
        lambda cmd, **kwargs: real_popen(
            [sys.executable, "-c", script, cmd[cmd.index("--port") + 1]], **kwargs
        ),
    )
    read_fd, write_fd = os.pipe()
    output = None
    try:
        flags = fcntl.fcntl(write_fd, fcntl.F_GETFL)
        fcntl.fcntl(write_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while True:
                os.write(write_fd, b"x" * 4096)
        except BlockingIOError:
            pass
        finally:
            fcntl.fcntl(write_fd, fcntl.F_SETFL, flags)

        output = os.fdopen(write_fd, "w")
        with _server.serve("fake", echo_log=output):
            echo_thread = next(
                thread
                for thread in threading.enumerate()
                if getattr(getattr(thread, "_target", None), "__name__", "")
                == "echo_output"
            )
            time.sleep(0.3)  # Let the echo attempt to write to the full pipe.
            start = time.monotonic()
        elapsed = time.monotonic() - start
        echo_thread.join(timeout=3)
        assert not echo_thread.is_alive(), "echo thread survived teardown"
        assert not any(
            thread.name == "rapid-mlx-bench-log-echo"
            for thread in threading.enumerate()
        )
        assert elapsed < 6
    finally:
        os.close(read_fd)
        if output is not None:
            output.close()
        else:
            os.close(write_fd)


def test_serve_teardown_with_blocked_echo(monkeypatch):
    real_popen = subprocess.Popen
    script = """
import http.server, sys
print('ENGINE-DEBUG-LINE', flush=True)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""
    monkeypatch.setattr(
        _server.subprocess,
        "Popen",
        lambda cmd, **kwargs: real_popen(
            [sys.executable, "-c", script, cmd[cmd.index("--port") + 1]], **kwargs
        ),
    )
    monkeypatch.setattr(_server, "_ECHO_JOIN_TIMEOUT_S", 0.1, raising=False)
    entered = threading.Event()
    release = threading.Event()
    errors = []

    class BlockingStream:
        def write(self, text):
            entered.set()
            release.wait()

        def flush(self):
            pass

    def run():
        try:
            with _server.serve("fake", echo_log=BlockingStream()):
                assert entered.wait(3), "echo did not start"
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        worker.join(timeout=6)
        assert not worker.is_alive(), "server teardown blocked on echo"
        assert not errors
    finally:
        release.set()
        worker.join(timeout=3)


@pytest.mark.parametrize(
    ("flag", "json_mode", "existing_level", "expected_level", "expected_echo"),
    [
        ("DEBUG", False, None, "DEBUG", "1"),
        ("DEBUG", True, None, "DEBUG", None),
        (None, False, None, "WARNING", None),
        (None, False, "error", "ERROR", "1"),
    ],
)
def test_cli_run_log_environment(
    monkeypatch, flag, json_mode, existing_level, expected_level, expected_echo
):
    level_var = "RAPID_MLX_LOG_LEVEL"
    echo_var = "RAPID_MLX_BENCH_SERVER_LOG_ECHO"
    monkeypatch.delenv(level_var, raising=False)
    if existing_level is not None:
        monkeypatch.setenv(level_var, existing_level)
    monkeypatch.setenv(echo_var, "previous")
    observed = []

    def run(*args, **kwargs):
        observed.append((os.environ.get(level_var), os.environ.get(echo_var)))
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_local", run)
    monkeypatch.setattr(cli, "request_from_args", lambda args: AccelerationRequest())
    command_args = args(json_mode)
    command_args.log_level = flag
    assert cli.benchmark_command(command_args) == 130
    assert observed == [(expected_level, expected_echo)]
    assert os.environ.get(level_var) == existing_level
    assert os.environ.get(echo_var) == "previous"


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


def test_terminate_already_reaped_leader_does_not_signal(monkeypatch):
    signals = []
    monkeypatch.setattr(_server.os, "killpg", lambda *args: signals.append(args))
    _server._terminate(SimpleNamespace(pid=1234, returncode=0))
    assert signals == []


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
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1)

    monkeypatch.setattr(cli, "run_local", terminate)
    monkeypatch.setattr(cli, "request_from_args", lambda args: AccelerationRequest())
    assert cli.benchmark_command(args()) == 143
    assert signal.getsignal(signal.SIGTERM) == previous
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
