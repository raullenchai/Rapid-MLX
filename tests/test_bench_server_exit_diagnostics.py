# SPDX-License-Identifier: Apache-2.0
"""Diagnostics for benchmark server exits during requests."""

import contextlib
import os
import signal
import subprocess
import sys
import time
from http.client import RemoteDisconnected

import pytest
import requests

from rapid_mlx.bench import _server
from rapid_mlx.community_bench import local_runner


@pytest.mark.parametrize(
    "ending, expected",
    [("os.kill(os.getpid(), signal.SIGKILL)", "SIGKILL"), ("sys.exit(3)", None)],
)
def test_describe_exited_server(tmp_path, ending, expected):
    log = tmp_path / "server.log"
    with log.open("w") as output:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os, signal, sys; print('first', flush=True); "
                f"print('last line', flush=True); {ending}",
            ],
            stdout=output,
        )
        try:
            error = _server.describe_server_exit(proc, log, wait_s=5)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    assert error is not None
    assert error.signal_name == expected
    assert "last line" in str(error)
    if expected:
        assert error.returncode == -signal.SIGKILL
        assert "killed by SIGKILL" in str(error)
        assert "short on memory" in str(error)
        assert local_runner._failure_code(error) == "runtime_error"
    else:
        assert error.returncode == 3
        assert "exit code 3" in str(error)
        assert "short on memory" not in str(error)


def test_describe_running_server(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert _server.describe_server_exit(proc, tmp_path / "log", wait_s=0.01) is None
    finally:
        proc.kill()
        proc.wait()


def test_serve_exit_error_reports_killed_child(monkeypatch, tmp_path):
    real_popen = subprocess.Popen
    proc = None
    log = tmp_path / "server.log"

    def popen(_cmd, **kwargs):
        nonlocal proc
        proc = real_popen(
            [
                sys.executable,
                "-c",
                "import time; print('server booted', flush=True); time.sleep(30)",
            ],
            **kwargs,
        )
        return proc

    monkeypatch.setattr(_server, "_wait_for_health", lambda *args: None)
    monkeypatch.setattr(_server.subprocess, "Popen", popen)
    try:
        with _server.serve("example", log_path=log) as info:
            deadline = time.monotonic() + 5
            while (
                "server booted" not in log.read_text() and time.monotonic() < deadline
            ):
                assert info["exit_error"](wait_s=0) is None
                time.sleep(0.01)
            assert "server booted" in log.read_text()
            assert info["exit_error"](wait_s=0) is None
            os.kill(proc.pid, signal.SIGKILL)
            err = info["exit_error"](wait_s=5)
            assert err.signal_name == "SIGKILL"
            assert "server booted" in str(err)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()


def test_describe_unknown_signal_and_empty_log(tmp_path):
    class FakeProc:
        def poll(self):
            return -999

    err = _server.describe_server_exit(FakeProc(), tmp_path / "missing.log", wait_s=0)
    assert err.signal_name == "signal 999"
    assert "killed by signal 999" in str(err)
    assert "wrote no log output" in str(err)
    assert err.log_tail == ""


@pytest.mark.parametrize(
    "exception_type",
    [requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout],
)
def test_request_preserves_timeouts(exception_type):
    def request(*args, **kwargs):
        raise exception_type("timed out")

    with pytest.raises(exception_type, match="timed out"):
        local_runner._bench_server_request({}, request, "http://local")


@pytest.mark.parametrize("exited", [True, False])
@pytest.mark.parametrize(
    "failure",
    [
        requests.exceptions.ConnectionError(
            (
                "Connection aborted.",
                RemoteDisconnected("Remote end closed connection without response"),
            )
        ),
        requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead"),
    ],
)
def test_image_connection_failure_reports_server_exit(monkeypatch, exited, failure):
    error = _server.ServerExitedError(
        "The local model server stopped unexpectedly (killed by SIGKILL).\n"
        "--- server log tail ---\nlast log line",
        -signal.SIGKILL,
        "SIGKILL",
        "last log line",
    )

    @contextlib.contextmanager
    def serve(*args, **kwargs):
        yield {
            "base_url": "http://local/v1",
            "exit_error": lambda: error if exited else None,
        }

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    calls = 0

    def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Response()
        raise failure

    monkeypatch.setattr(_server, "serve", serve)
    monkeypatch.setattr(local_runner.requests, "post", request)
    with pytest.raises(_server.ServerExitedError if exited else RuntimeError) as caught:
        local_runner._run_image("example")
    assert calls == 2
    if exited:
        assert "killed by SIGKILL" in str(caught.value)
        assert "last log line" in str(caught.value)
        assert local_runner._failure_code(caught.value) == "runtime_error"
    else:
        assert (
            "failed without a response, but the server process is still running"
            in str(caught.value)
        )
        assert type(caught.value) is RuntimeError
