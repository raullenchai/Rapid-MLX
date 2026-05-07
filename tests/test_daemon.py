# SPDX-License-Identifier: Apache-2.0
import signal
from argparse import Namespace
from pathlib import Path

import pytest

from vllm_mlx import daemon


@pytest.fixture()
def daemon_dirs(tmp_path, monkeypatch):
    daemon_dir = tmp_path / "daemons"
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(daemon, "DAEMON_DIR", daemon_dir)
    monkeypatch.setattr(daemon, "LOG_DIR", log_dir)
    return daemon_dir, log_dir


def _record(
    daemon_id: str,
    *,
    supervisor_pid: int,
    child_pid: int | None = None,
    model: str = "mlx-community/Qwen3.5-4B-MLX-4bit",
    original_alias: str | None = None,
    served_model_name: str | None = None,
) -> dict:
    return {
        "id": daemon_id,
        "model": model,
        "original_alias": original_alias,
        "served_model_name": served_model_name or model,
        "host": "127.0.0.1",
        "port": 8123,
        "base_url": "http://127.0.0.1:8123",
        "v1_url": "http://127.0.0.1:8123/v1",
        "supervisor_pid": supervisor_pid,
        "child_pid": child_pid,
        "command": ["python", "-m", "vllm_mlx.cli", "serve", model],
        "created_at": 1.0,
        "updated_at": 1.0,
        "restarts": 0,
        "state": "running",
        "log_path": "/tmp/test.log",
        "stop_path": "/tmp/test.stop",
        "record_path": "",
    }


def _write_record(record: dict) -> Path:
    path = daemon.DAEMON_DIR / f"{record['id']}.json"
    record["record_path"] = str(path)
    daemon._atomic_write_json(path, record)
    return path


def test_start_daemon_filters_daemon_flag_and_records_metadata(daemon_dirs, monkeypatch):
    class FakePopen:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.kwargs = kwargs
            self.pid = 4321

    started = {}

    def fake_popen(cmd, **kwargs):
        proc = FakePopen(cmd, **kwargs)
        started["cmd"] = cmd
        started["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    args = Namespace(
        model="mlx-community/Qwen3.5-4B-MLX-4bit",
        _original_alias=None,
        served_model_name=None,
        host="0.0.0.0",
        port=8000,
    )

    record = daemon.start_daemon(
        args,
        ["serve", "mlx-community/Qwen3.5-4B-MLX-4bit", "--daemon", "--port", "8000"],
    )

    assert record["supervisor_pid"] == 4321
    assert record["base_url"] == "http://127.0.0.1:8000"
    assert "--daemon" not in record["command"]
    assert record["command"][-3:] == [
        "mlx-community/Qwen3.5-4B-MLX-4bit",
        "--port",
        "8000",
    ]
    assert started["cmd"][:3] == [daemon.sys.executable, "-m", "vllm_mlx.daemon"]
    assert started["kwargs"]["start_new_session"] is True
    assert Path(record["record_path"]).exists()


def test_resolve_daemon_by_supervisor_child_model_and_alias(daemon_dirs, monkeypatch):
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: True)
    _write_record(
        _record(
            "one",
            supervisor_pid=100,
            child_pid=101,
            original_alias="qwen3.5-4b",
            served_model_name="local",
        )
    )

    assert daemon.resolve_daemon("100").record["id"] == "one"
    assert daemon.resolve_daemon("101").record["id"] == "one"
    assert daemon.resolve_daemon("qwen3.5-4b").record["id"] == "one"
    assert daemon.resolve_daemon("local").record["id"] == "one"


def test_resolve_daemon_rejects_ambiguous_model(daemon_dirs, monkeypatch):
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: True)
    _write_record(_record("one", supervisor_pid=100, model="same"))
    _write_record(_record("two", supervisor_pid=200, model="same"))

    with pytest.raises(daemon.DaemonError, match="Multiple daemons"):
        daemon.resolve_daemon("same")


def test_stop_daemon_writes_stop_marker_and_marks_stopped(
    daemon_dirs, tmp_path, monkeypatch
):
    alive = {100}
    stop_path = tmp_path / "daemon.stop"
    record = _record("one", supervisor_pid=100)
    record["stop_path"] = str(stop_path)
    path = _write_record(record)

    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: int(pid or 0) in alive)

    def fake_signal(pid, sig):
        assert sig in {signal.SIGTERM, signal.SIGKILL}
        alive.discard(int(pid))
        return True

    monkeypatch.setattr(daemon, "_signal_pid", fake_signal)

    stopped = daemon.stop_daemon("100")

    assert stopped["state"] == "stopped"
    assert stop_path.exists()
    assert daemon._read_json(path)["state"] == "stopped"


def test_list_daemons_marks_dead_records_stale(daemon_dirs, monkeypatch):
    path = _write_record(_record("one", supervisor_pid=100))
    monkeypatch.setattr(daemon, "_pid_alive", lambda pid: False)

    assert daemon.list_daemons() == []
    assert daemon._read_json(path)["state"] == "stale"


def test_supervisor_restarts_failed_child_until_stop_marker(daemon_dirs, tmp_path, monkeypatch):
    record = _record("one", supervisor_pid=0)
    record["log_path"] = str(tmp_path / "daemon.log")
    record["stop_path"] = str(tmp_path / "daemon.stop")
    path = _write_record(record)
    popen_calls = []

    class FakeChild:
        pid = 222

        def poll(self):
            return 2

        def wait(self, timeout=None):
            return 2

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return FakeChild()

    def fake_sleep(_delay):
        Path(record["stop_path"]).write_text("stop\n", encoding="utf-8")

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon.time, "sleep", fake_sleep)

    assert daemon.supervise(path) == 0
    final = daemon._read_json(path)
    assert len(popen_calls) == 1
    assert final["restarts"] == 1
    assert final["state"] == "stopped"


def test_cli_serve_daemon_dispatches_without_loading_server(monkeypatch, capsys):
    from vllm_mlx import cli

    captured = {}

    def fake_start_daemon(args, raw_args):
        captured["args"] = args
        captured["raw_args"] = raw_args
        return {
            "model": args.model,
            "original_alias": None,
            "supervisor_pid": 1234,
            "base_url": "http://127.0.0.1:8000",
            "log_path": "/tmp/lightning.log",
        }

    monkeypatch.setattr(daemon, "start_daemon", fake_start_daemon)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "lightning-mlx",
            "serve",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
            "--daemon",
        ],
    )

    cli.main()

    out = capsys.readouterr().out
    assert captured["args"].daemon is True
    assert captured["raw_args"][-1] == "--daemon"
    assert "Started daemon mlx-community/Qwen3.5-4B-MLX-4bit (pid 1234)." in out


def test_cli_dispatches_status_kill_and_tui(monkeypatch):
    from vllm_mlx import cli

    calls = []
    monkeypatch.setattr(cli, "daemon_status_command", lambda args: calls.append("status"))
    monkeypatch.setattr(
        cli, "daemon_kill_command", lambda args: calls.append(("kill", args.target))
    )
    monkeypatch.setattr(
        cli, "daemon_tui_command", lambda args: calls.append(("tui", args.target))
    )

    monkeypatch.setattr(cli.sys, "argv", ["lightning-mlx", "status"])
    cli.main()
    monkeypatch.setattr(cli.sys, "argv", ["lightning-mlx", "kill", "123"])
    cli.main()
    monkeypatch.setattr(cli.sys, "argv", ["lightning-mlx", "tui", "qwen3.5-4b"])
    cli.main()

    assert calls == ["status", ("kill", "123"), ("tui", "qwen3.5-4b")]
