# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import builtins
import errno
import json
import os
import socket
import subprocess
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import requests
from fastapi import HTTPException

from rapid_mlx import cli, server
from rapid_mlx._process_identity import marker_identity, process_identity
from rapid_mlx.runtime.primary_lifecycle import PrimaryModelLifecycle
from rapid_mlx.service import helpers
from rapid_mlx.telemetry import model_events, registry, server_start

REPO_ROOT = Path(__file__).resolve().parents[1]


def _marker_payload(pid: int, *, current: bool = False) -> dict[str, object]:
    identity = process_identity(pid) if current else None
    return {
        "pid": pid,
        "create_time": identity.create_time if identity is not None else 1.0,
        "boot_time": identity.boot_time if identity is not None else 1.0,
        "app_version": "0.15.1",
    }


@pytest.fixture(autouse=True)
def _reset_state():
    server_start._reset_for_tests()
    yield
    server_start._reset_for_tests()


def _capture(monkeypatch):
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track",
        lambda event, props: events.append((event, dict(props))) or True,
    )
    return events


class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.bodies.append(self.rfile.read(length))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_stale_marker_reports_previous_run_unterminated_once_to_loopback(tmp_path):
    home = tmp_path / "home"
    state_dir = home / ".rapid-mlx" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "serve-inflight-99999999.json").write_text(
        json.dumps(_marker_payload(99_999_999)),
        encoding="utf-8",
    )
    sink = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    sink.bodies = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=sink.serve_forever, daemon=True)
    thread.start()
    program = """
import rapid_mlx
from rapid_mlx.telemetry import build_gate, common_props, consent_runtime, posthog_sender, server_start, state
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from rapid_mlx.telemetry.common_props import PlatformFacts

rapid_mlx.__version__ = "0.15.1"
stamp = ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
build_gate.official_build = lambda: stamp
consent_runtime.upload_allowed = lambda: True
common_props.read_platform_facts = lambda: PlatformFacts(
    os="darwin", os_version="25.3", arch="arm64", chip="m1-pro",
    memory_gb=32, python_version="3.11"
)
state.get_or_create_client_id = lambda: "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"
state.session_id = lambda: "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
server_start.attempted("ignored", load_policy="lazy")
posthog_sender.get_sender().flush(5.0)
"""
    env = dict(os.environ, HOME=str(home))
    env["RAPID_MLX_POSTHOG_URL"] = f"http://127.0.0.1:{sink.server_port}/batch/"
    for name in (
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "CIRCLECI",
        "TRAVIS",
        "BUILDKITE",
        "JENKINS_URL",
        "TEAMCITY_VERSION",
        "RAPID_MLX_TELEMETRY",
        "DO_NOT_TRACK",
    ):
        env.pop(name, None)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    finally:
        sink.shutdown()
        thread.join(timeout=2.0)
        sink.server_close()

    assert proc.returncode == 0, proc.stderr
    attempted = [
        item
        for body in sink.bodies  # type: ignore[attr-defined]
        for item in json.loads(body)["batch"]
        if item["event"] == "server_start_state"
    ]
    assert len(attempted) == 1
    assert attempted[0]["properties"]["previous_run_unterminated"] is True


def test_attempted_then_ready_exactly_once(monkeypatch):
    events = _capture(monkeypatch)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.attempted("ignored", load_policy="lazy")
    server_start.ready()
    server_start.ready()
    server_start.failed("bind")

    assert [props["state"] for _, props in events] == ["attempted", "ready"]
    assert all(name == "server_start_state" for name, _ in events)
    assert all("failure_stage" not in props for _, props in events)
    assert events[0][1] == {
        "state": "attempted",
        "model_type": "llm",
        "load_policy": "eager",
    }
    assert events[1][1] == {
        "state": "ready",
        "model_type": "llm",
        "load_policy": "eager",
    }
    assert all(registry.validate(name, props) == props for name, props in events)


@pytest.mark.parametrize(
    ("terminal", "marker_survives"),
    [(server_start.ready, True), (lambda: server_start.failed("bind"), False)],
)
def test_marker_lifetime_at_terminal_with_telemetry_disabled(
    monkeypatch, tmp_path, terminal, marker_survives
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: False)

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    marker = tmp_path / ".rapid-mlx" / "state" / f"serve-inflight-{os.getpid()}.json"
    assert marker.is_file()
    assert marker.stat().st_mode & 0o777 == 0o600
    assert marker.parent.stat().st_mode & 0o777 == 0o700
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert isinstance(payload["create_time"], float)
    assert isinstance(payload["boot_time"], float)
    assert isinstance(payload["app_version"], str)

    terminal()
    assert marker.exists() is marker_survives


def test_live_marker_is_not_reported_overwritten_or_removed(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    live_pid = os.getppid()
    marker = tmp_path / ".rapid-mlx" / "state" / f"serve-inflight-{live_pid}.json"
    marker.parent.mkdir(parents=True)
    original = _marker_payload(live_pid, current=True)
    marker.write_text(json.dumps(original), encoding="utf-8")
    events = _capture(monkeypatch)

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    own_marker = marker.parent / f"serve-inflight-{os.getpid()}.json"

    assert own_marker.is_file()
    server_start.ready()

    assert all("previous_run_unterminated" not in props for _, props in events)
    assert json.loads(marker.read_text(encoding="utf-8")) == original
    assert own_marker.exists()


def test_stale_marker_adds_attempted_property_in_process(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = tmp_path / ".rapid-mlx" / "state" / "serve-inflight-99999999.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(_marker_payload(99_999_999)),
        encoding="utf-8",
    )
    events = _capture(monkeypatch)

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    assert events[0][1]["previous_run_unterminated"] is True


def test_terminal_stale_marker_does_not_report_unterminated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = tmp_path / ".rapid-mlx" / "state" / "serve-inflight-99999999.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps({**_marker_payload(99_999_999), "startup_terminal": True}),
        encoding="utf-8",
    )
    events = _capture(monkeypatch)

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    assert "previous_run_unterminated" not in events[0][1]


def test_any_unterminated_stale_marker_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    state_dir = tmp_path / ".rapid-mlx" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "serve-inflight-99999998.json").write_text(
        json.dumps(_marker_payload(99_999_998)), encoding="utf-8"
    )
    (state_dir / "serve-inflight-99999999.json").write_text(
        json.dumps({**_marker_payload(99_999_999), "startup_terminal": True}),
        encoding="utf-8",
    )
    events = _capture(monkeypatch)

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    assert events[0][1]["previous_run_unterminated"] is True


def test_ready_marks_live_marker_terminal(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    _capture(monkeypatch)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    marker = server_start._marker_path()

    server_start.ready()

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["startup_terminal"] is True


def test_ready_marks_live_marker_terminal_when_telemetry_disabled(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: False)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    marker = server_start._marker_path()

    server_start.ready()

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["startup_terminal"] is True


@pytest.mark.parametrize(
    "name",
    ["other-123.json", "serve-inflight-nope.json", "serve-inflight-123.txt"],
)
def test_invalid_marker_filenames_have_no_pid(name):
    assert server_start._marker_pid(Path(name)) is None


def test_atomic_marker_rejects_zero_progress_write(monkeypatch, tmp_path):
    marker = tmp_path / "state" / "serve-inflight-123.json"
    monkeypatch.setattr(os, "write", lambda *_args: 0)

    with pytest.raises(OSError, match="made no progress"):
        server_start._atomic_write_marker(marker)


def test_atomic_marker_requires_current_process_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(server_start, "process_identity", lambda _pid: None)

    with pytest.raises(OSError, match="determine current process identity"):
        server_start._atomic_write_marker(tmp_path / "serve-inflight-123.json")


def test_state_directory_uses_windows_compatible_validation(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(server_start.os, "name", "nt")
    monkeypatch.setattr(
        server_start.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Windows directory validation must not os.open a directory")
        ),
    )

    assert server_start._prepare_state_dir(state_dir) is True


def test_marker_write_and_remove_failures_are_inert(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        server_start,
        "_atomic_write_marker",
        lambda _path: (_ for _ in ()).throw(OSError("write failed")),
    )
    assert server_start._begin_inflight_marker() == (False, False)

    server_start._owns_inflight_marker = True
    monkeypatch.setattr(
        server_start,
        "_marker_path",
        lambda: (_ for _ in ()).throw(OSError("remove failed")),
    )
    server_start._remove_inflight_marker()
    assert server_start._owns_inflight_marker is False


def test_cleanup_preserves_replacement_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = server_start._marker_path()
    snapshot = server_start._atomic_write_marker(marker)
    replacement = marker.with_suffix(".replacement")
    replacement.write_text("new owner", encoding="utf-8")
    os.replace(replacement, marker)
    server_start._owns_inflight_marker = True
    server_start._owned_inflight_snapshot = snapshot

    server_start._remove_inflight_marker()

    assert marker.read_text(encoding="utf-8") == "new owner"
    assert server_start._owns_inflight_marker is False
    assert server_start._owned_inflight_snapshot is None


def test_marker_scan_failure_is_inert(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        Path,
        "glob",
        lambda *_args: (_ for _ in ()).throw(OSError("scan failed")),
    )
    monkeypatch.setattr(server_start, "_atomic_write_marker", lambda _path: (1, 2))

    assert server_start._begin_inflight_marker() == (False, True)


def test_stale_marker_unlink_failure_still_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = tmp_path / ".rapid-mlx" / "state" / "serve-inflight-99999999.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps(_marker_payload(99_999_999)), encoding="utf-8")
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unlink failed")),
    )
    monkeypatch.setattr(server_start, "_atomic_write_marker", lambda _path: (1, 2))

    assert server_start._begin_inflight_marker() == (True, True)


def test_invalid_marker_unlink_failure_is_silent(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = tmp_path / ".rapid-mlx" / "state" / "serve-inflight-99999999.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unlink failed")),
    )
    monkeypatch.setattr(server_start, "_atomic_write_marker", lambda _path: (1, 2))

    assert server_start._begin_inflight_marker() == (False, True)


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "{}",
        json.dumps({"pid": 99_999_999}),
        json.dumps(
            {
                "pid": 99_999_999,
                "create_time": "old",
                "boot_time": 1.0,
                "app_version": "0.15.1",
            }
        ),
    ],
)
def test_garbage_dead_pid_marker_is_deleted_without_telemetry(
    monkeypatch, tmp_path, payload
):
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = tmp_path / ".rapid-mlx" / "state" / "serve-inflight-99999999.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(payload, encoding="utf-8")

    previous, owns = server_start._begin_inflight_marker()

    assert previous is False
    assert owns is True
    assert not marker.exists()


def test_marker_payload_rejects_extra_keys_and_unbounded_app_version():
    payload = _marker_payload(123)
    assert marker_identity({**payload, "injected": "ignored"}) is None
    assert marker_identity({**payload, "app_version": "x" * 257}) is None


@pytest.mark.parametrize(
    "payload",
    [
        {**_marker_payload(123), "pid": True},
        {**_marker_payload(123), "pid": 0},
        {**_marker_payload(123), "pid": "1"},
        {**_marker_payload(123), "create_time": True},
        {**_marker_payload(123), "create_time": float("nan")},
        {**_marker_payload(123), "boot_time": False},
        {**_marker_payload(123), "boot_time": float("inf")},
        {**_marker_payload(123), "app_version": ""},
        {**_marker_payload(123), "app_version": 1},
    ],
)
def test_marker_payload_type_matrix_is_rejected(payload):
    assert marker_identity(payload) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pid", 2**31),
        ("create_time", -1.0),
        ("create_time", 10**1000),
        ("create_time", float(2**40 + 1)),
        ("boot_time", -1.0),
        ("boot_time", float(2**40 + 1)),
    ],
)
def test_marker_payload_bounds_are_rejected_before_process_probe(field, value):
    payload = {**_marker_payload(123), field: value}
    assert marker_identity(payload) is None


def test_marker_reader_treats_overflowing_numeric_payload_as_invalid(tmp_path):
    marker = tmp_path / "serve-inflight-123.json"
    marker.write_text(
        json.dumps({**_marker_payload(123), "create_time": 10**1000}),
        encoding="utf-8",
    )

    assert server_start._read_marker(marker) is None


def test_marker_identity_parse_failure_is_invalid(monkeypatch):
    from rapid_mlx import _process_identity as identity

    monkeypatch.setattr(identity, "_valid_time", lambda _value: True)
    payload = {**_marker_payload(123), "create_time": object()}

    assert identity.marker_identity(payload) is None


def test_huge_marker_payload_is_rejected(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    marker = server_start._marker_path().with_name("serve-inflight-99999999.json")
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {**_marker_payload(99_999_999), "app_version": "x" * (8 * 1024 * 1024)}
        ),
        encoding="utf-8",
    )

    assert marker.stat().st_size > 4096
    assert server_start._read_marker(marker) is None
    assert so._marker_for_pid(marker.parent.parent / "logs", 99_999_999) is None


def test_marker_reader_rejects_file_that_grows_during_read(monkeypatch, tmp_path):
    marker = tmp_path / "serve-inflight-123.json"
    marker.write_bytes(b"x")
    real_fstat = os.fstat
    real_read = os.read
    with monkeypatch.context() as patch:
        patch.setattr(
            server_start.os,
            "fstat",
            lambda fd: SimpleNamespace(st_mode=real_fstat(fd).st_mode, st_size=1),
        )
        patch.setattr(
            server_start.os,
            "read",
            lambda fd, size: b"x" * 4097 if size == 1024 else real_read(fd, size),
        )
        assert server_start._read_marker(marker) is None


def test_state_directory_symlink_is_not_followed(monkeypatch, tmp_path):
    home = tmp_path / "home"
    base = home / ".rapid-mlx"
    redirected = tmp_path / "redirected"
    base.mkdir(parents=True)
    redirected.mkdir(mode=0o755)
    (base / "state").symlink_to(redirected, target_is_directory=True)
    monkeypatch.setenv("HOME", str(home))

    assert server_start._begin_inflight_marker() == (False, False)
    assert list(redirected.iterdir()) == []
    assert redirected.stat().st_mode & 0o777 == 0o755

    with pytest.raises(OSError, match="state directory is unavailable"):
        server_start._atomic_write_marker(base / "state" / "serve-inflight-1.json")


def test_state_marker_symlink_does_not_delete_target(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    target = tmp_path / "external.json"
    target.write_text("not json", encoding="utf-8")
    state = tmp_path / "home" / ".rapid-mlx" / "state"
    state.mkdir(parents=True)
    (state / "serve-inflight-99999999.json").symlink_to(target)

    server_start._begin_inflight_marker()

    assert target.exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_marker_reader_rejects_fifo_without_blocking(tmp_path):
    marker = tmp_path / "serve-inflight-123.json"
    os.mkfifo(marker)

    assert server_start._read_marker(marker) is None


def test_invalid_marker_removal_preserves_concurrent_replacement(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    peer = tmp_path / "home" / ".rapid-mlx" / "state" / "serve-inflight-99999999.json"
    peer.parent.mkdir(parents=True)
    peer.write_text("broken", encoding="utf-8")
    real_read = server_start._read_marker

    def raced_read(path):
        if path == peer:
            replacement = path.with_suffix(".replacement")
            replacement.write_text(
                json.dumps(_marker_payload(99_999_999)), encoding="utf-8"
            )
            os.replace(replacement, path)
            return None
        return real_read(path)

    monkeypatch.setattr(server_start, "_read_marker", raced_read)

    server_start._begin_inflight_marker()

    assert peer.exists()


def test_quarantine_restore_does_not_overwrite_newer_replacement(
    monkeypatch, tmp_path, caplog
):
    marker = tmp_path / "serve-inflight-999.json"
    marker.write_text("original", encoding="utf-8")
    snapshot = server_start._marker_snapshot(marker)
    real_rename = os.rename
    injected = False

    def raced_rename(source, destination):
        nonlocal injected
        if Path(source) == marker and not injected:
            injected = True
            replacement = marker.with_suffix(".replacement")
            replacement.write_text("replacement-B", encoding="utf-8")
            os.replace(replacement, marker)
            result = real_rename(source, destination)
            marker.write_text("replacement-C", encoding="utf-8")
            return result
        return real_rename(source, destination)

    monkeypatch.setattr(server_start.os, "rename", raced_rename)
    server_start._remove_marker_snapshot(marker, snapshot)

    stale = marker.with_name(f".{marker.name}.stale-{os.getpid()}")
    assert injected is True
    assert marker.read_text(encoding="utf-8") == "replacement-C"
    assert stale.read_text(encoding="utf-8") == "replacement-B"
    assert "preserving quarantined serve marker" in caplog.text


def test_quarantine_restores_racing_replacement_when_path_is_free(
    monkeypatch, tmp_path
):
    marker = tmp_path / "serve-inflight-999.json"
    marker.write_text("original", encoding="utf-8")
    snapshot = server_start._marker_snapshot(marker)
    real_rename = os.rename
    injected = False

    def raced_rename(source, destination):
        nonlocal injected
        if Path(source) == marker and not injected:
            injected = True
            replacement = marker.with_suffix(".replacement")
            replacement.write_text("replacement-B", encoding="utf-8")
            os.replace(replacement, marker)
        return real_rename(source, destination)

    monkeypatch.setattr(server_start.os, "rename", raced_rename)
    server_start._remove_marker_snapshot(marker, snapshot)

    assert injected is True
    assert marker.read_text(encoding="utf-8") == "replacement-B"


def test_restore_link_unsupported_claim_succeeds_without_warning(
    tmp_path, monkeypatch, caplog
):
    marker = tmp_path / "serve-inflight-999.json"
    marker.write_text("original", encoding="utf-8")
    snapshot = server_start._marker_snapshot(marker)
    real_rename = os.rename

    def raced_rename(source, destination):
        replacement = marker.with_suffix(".replacement")
        replacement.write_text("replacement-B", encoding="utf-8")
        os.replace(replacement, marker)
        return real_rename(source, destination)

    monkeypatch.setattr(server_start.os, "rename", raced_rename)
    monkeypatch.setattr(
        server_start.os,
        "link",
        lambda *_args: (_ for _ in ()).throw(OSError(errno.ENOTSUP, "unsupported")),
    )

    server_start._remove_marker_snapshot(marker, snapshot)

    stale = marker.with_name(f".{marker.name}.stale-{os.getpid()}")
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "replacement-B"
    assert not stale.exists()
    assert not caplog.records


def test_claim_is_cleaned_if_replace_fails(tmp_path, monkeypatch, caplog):
    marker = tmp_path / "serve-inflight-999.json"
    marker.write_text("original-A", encoding="utf-8")
    snapshot = server_start._marker_snapshot(marker)
    stale = marker.with_name(f".{marker.name}.stale-{os.getpid()}")
    real_rename = os.rename
    real_replace = os.replace

    def raced_rename(source, destination):
        replacement = marker.with_suffix(".replacement-B")
        replacement.write_text("replacement-B", encoding="utf-8")
        os.replace(replacement, marker)
        monkeypatch.setattr(server_start.os, "rename", real_rename)
        return real_rename(source, destination)

    def failed_replace(source, destination):
        if Path(source) == stale and Path(destination) == marker:
            raise OSError(errno.EIO, "injected replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(server_start.os, "rename", raced_rename)
    monkeypatch.setattr(
        server_start.os,
        "link",
        lambda *_args: (_ for _ in ()).throw(OSError(errno.ENOTSUP, "unsupported")),
    )
    monkeypatch.setattr(server_start.os, "replace", failed_replace)

    server_start._remove_marker_snapshot(marker, snapshot)

    assert not marker.exists()
    assert stale.read_text(encoding="utf-8") == "replacement-B"
    assert len(caplog.records) == 1
    assert str(stale) in caplog.text
    assert str(marker) in caplog.text


def test_restore_link_unsupported_preserves_newer_marker(tmp_path, monkeypatch, caplog):
    marker = tmp_path / "serve-inflight-999.json"
    marker.write_text("original", encoding="utf-8")
    snapshot = server_start._marker_snapshot(marker)
    real_rename = os.rename

    def raced_rename(source, destination):
        replacement = marker.with_suffix(".replacement")
        replacement.write_text("replacement-B", encoding="utf-8")
        os.replace(replacement, marker)
        monkeypatch.setattr(server_start.os, "rename", real_rename)
        return real_rename(source, destination)

    def failed_link(_source, _destination):
        marker.write_text("replacement-C", encoding="utf-8")
        raise OSError(errno.EXDEV, "unsupported")

    monkeypatch.setattr(server_start.os, "rename", raced_rename)
    monkeypatch.setattr(server_start.os, "link", failed_link)

    server_start._remove_marker_snapshot(marker, snapshot)

    stale = marker.with_name(f".{marker.name}.stale-{os.getpid()}")
    assert marker.read_text(encoding="utf-8") == "replacement-C"
    assert stale.read_text(encoding="utf-8") == "replacement-B"
    assert len(caplog.records) == 1
    assert str(stale) in caplog.text
    assert str(marker) in caplog.text


def test_restore_fallback_does_not_overwrite_arrival_after_free_check(
    tmp_path, monkeypatch, caplog
):
    marker = tmp_path / "serve-inflight-999.json"
    marker.write_text("original-A", encoding="utf-8")
    snapshot = server_start._marker_snapshot(marker)
    real_open = os.open
    real_rename = os.rename
    rename_calls = 0

    def raced_rename(source, destination):
        nonlocal rename_calls
        rename_calls += 1
        replacement = marker.with_suffix(".replacement-B")
        replacement.write_text("replacement-B", encoding="utf-8")
        os.replace(replacement, marker)
        return real_rename(source, destination)

    def raced_open(path, flags, mode=0o777):
        if Path(path) == marker and flags & os.O_EXCL:
            replacement = marker.with_suffix(".replacement-C")
            replacement.write_text("replacement-C", encoding="utf-8")
            os.replace(replacement, marker)
        return real_open(path, flags, mode)

    monkeypatch.setattr(server_start.os, "rename", raced_rename)
    monkeypatch.setattr(server_start.os, "open", raced_open)
    monkeypatch.setattr(
        server_start.os,
        "link",
        lambda *_args: (_ for _ in ()).throw(OSError(errno.ENOTSUP, "unsupported")),
    )

    server_start._remove_marker_snapshot(marker, snapshot)

    stale = marker.with_name(f".{marker.name}.stale-{os.getpid()}")
    assert rename_calls == 1
    assert marker.read_text(encoding="utf-8") == "replacement-C"
    assert stale.read_text(encoding="utf-8") == "replacement-B"
    assert len(caplog.records) == 1
    assert str(stale) in caplog.text
    assert str(marker) in caplog.text


def test_state_dir_and_marker_cleanup_defensive_races(monkeypatch, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    real_fstat = os.fstat
    with monkeypatch.context() as patch:
        patch.setattr(
            server_start.os,
            "fstat",
            lambda fd: SimpleNamespace(
                st_dev=real_fstat(fd).st_dev,
                st_ino=real_fstat(fd).st_ino + 1,
            ),
        )
        assert server_start._prepare_state_dir(state) is False

    with monkeypatch.context() as patch:
        patch.setattr(
            server_start.os,
            "lstat",
            lambda _path: (_ for _ in ()).throw(PermissionError("denied")),
        )
        assert server_start._prepare_state_dir(state) is False
        assert server_start._marker_snapshot(state / "missing") is None

    marker = state / "serve-inflight-123.json"
    marker.write_text("broken", encoding="utf-8")
    snapshot = server_start._marker_snapshot(marker)
    assert snapshot is not None
    stale = marker.with_name(f".{marker.name}.stale-{os.getpid()}")

    with monkeypatch.context() as patch:
        patch.setattr(
            server_start.os,
            "lstat",
            lambda path: (
                (_ for _ in ()).throw(PermissionError("denied"))
                if path == stale
                else os.stat(path, follow_symlinks=False)
            ),
        )
        server_start._remove_marker_snapshot(marker, snapshot)
        assert marker.exists()

    stale.write_text("occupied", encoding="utf-8")
    server_start._remove_marker_snapshot(marker, snapshot)
    assert marker.exists()
    stale.unlink()

    with monkeypatch.context() as patch:
        patch.setattr(
            server_start,
            "_marker_snapshot",
            lambda path: snapshot if path == marker else (snapshot[0], snapshot[1] + 1),
        )
        patch.setattr(
            server_start.os,
            "link",
            lambda *_args: (_ for _ in ()).throw(PermissionError("restore failed")),
        )
        server_start._remove_marker_snapshot(marker, snapshot)
        assert marker.exists()


def test_windows_state_directory_rejects_identity_swap(monkeypatch, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    real = os.lstat(state)
    stats = iter(
        [
            real,
            SimpleNamespace(
                st_mode=real.st_mode,
                st_dev=real.st_dev,
                st_ino=real.st_ino + 1,
            ),
        ]
    )
    monkeypatch.setattr(server_start.os, "name", "nt")
    monkeypatch.setattr(server_start.os, "chmod", lambda *_args: None)
    monkeypatch.setattr(server_start.os, "lstat", lambda _path: next(stats))

    assert server_start._prepare_state_dir(state) is False


def test_atomic_marker_windows_return_and_directory_fsync_failure(
    monkeypatch, tmp_path
):
    marker = tmp_path / "serve-inflight-123.json"
    windows_tmp = tmp_path / ".windows.tmp"
    fd = os.open(windows_tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    monkeypatch.setattr(
        server_start.tempfile,
        "mkstemp",
        lambda **_kwargs: (fd, str(windows_tmp)),
    )
    monkeypatch.setattr(server_start, "Path", lambda _value: windows_tmp)
    monkeypatch.setattr(server_start.os, "name", "nt")
    snapshot = server_start._atomic_write_marker(marker)
    assert snapshot == server_start._marker_snapshot(marker)

    monkeypatch.undo()
    marker.unlink()
    real_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(open_fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        return real_fsync(open_fd)

    monkeypatch.setattr(server_start.os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="directory fsync failed"):
        server_start._atomic_write_marker(marker)
    assert not marker.exists()


def test_mark_terminal_ignores_replaced_marker_and_logs_write_failure(
    monkeypatch, tmp_path, caplog
):
    caplog.set_level("DEBUG")
    marker = tmp_path / "serve-inflight-123.json"
    server_start._owns_inflight_marker = False
    server_start._mark_inflight_terminal()
    server_start._owns_inflight_marker = True
    server_start._owned_inflight_snapshot = (1, 2)
    monkeypatch.setattr(server_start, "_marker_path", lambda: marker)
    monkeypatch.setattr(server_start, "_marker_snapshot", lambda _path: (1, 3))
    server_start._mark_inflight_terminal()

    monkeypatch.setattr(server_start, "_marker_snapshot", lambda _path: (1, 2))
    monkeypatch.setattr(
        server_start,
        "_atomic_write_marker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )
    server_start._mark_inflight_terminal()

    assert "could not mark serve startup terminal" in caplog.text


def test_pid_reuse_does_not_hide_pre_reboot_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = server_start._marker_path()
    marker.parent.mkdir(parents=True)
    stale = _marker_payload(os.getpid(), current=True)
    stale["boot_time"] = float(stale["boot_time"]) - 100.0
    marker.write_text(json.dumps(stale), encoding="utf-8")

    previous, owns = server_start._begin_inflight_marker()

    assert previous is True
    assert owns is True
    assert json.loads(marker.read_text(encoding="utf-8")) != stale


@pytest.mark.parametrize(
    "stage",
    ["resolve", "download", "preflight", "prepare", "engine_start", "bind"],
)
def test_each_failure_stage_is_the_only_terminal(monkeypatch, stage):
    events = _capture(monkeypatch)
    server_start.attempted("flux-schnell", load_policy="lazy")
    server_start.failed(stage)
    server_start.ready()
    server_start.failed("bind")

    assert [props["state"] for _, props in events] == ["attempted", "failed"]
    assert events[-1][1]["failure_stage"] == stage
    assert "failure_stage" not in events[0][1]


def test_invalid_values_are_omitted_or_ignored(monkeypatch):
    events = _capture(monkeypatch)
    server_start.attempted(SimpleNamespace(), load_policy="surprise")
    server_start.failed("surprise")
    server_start.ready()

    assert events == [
        (
            "server_start_state",
            {"state": "attempted", "model_type": "other"},
        ),
        (
            "server_start_state",
            {"state": "ready", "model_type": "other"},
        ),
    ]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen3.5-4b-4bit", "eager"),
        ("flux-schnell", "lazy"),
        ("ltx-2.3-mlx-q4", "lazy"),
        ("kokoro", "lazy"),
    ],
)
def test_load_policy_matches_adapter_lanes(model, expected):
    assert server_start.load_policy(model) == expected


def test_explicit_lazy_load_overrides_adapter_policy():
    assert server_start.load_policy("qwen3.5-4b-4bit") == "eager"
    assert server_start.load_policy("qwen3.5-4b-4bit", lazy_load=True) == "lazy"


@pytest.mark.asyncio
async def test_post_ready_lazy_503_does_no_server_start_telemetry_work(monkeypatch):
    events = _capture(monkeypatch)

    class FailingLazyEngine:
        _loaded = False

        async def start(self):
            raise RuntimeError("lazy load failed")

        async def stop(self):
            self._loaded = False

    engine = FailingLazyEngine()
    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)
    monkeypatch.setattr(
        helpers,
        "get_config",
        lambda: SimpleNamespace(primary_model_lifecycle=lifecycle),
    )
    server_start.attempted("qwen3.5-4b-4bit", load_policy="lazy")
    server_start.ready()

    imports: list[str] = []
    real_import = builtins.__import__

    def reject_server_start_import(name, *args, **kwargs):
        if name == "rapid_mlx.telemetry.server_start":
            imports.append(name)
            raise AssertionError("request path imported server_start telemetry")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "rapid_mlx.telemetry.server_start")
    monkeypatch.setattr(builtins, "__import__", reject_server_start_import)

    with pytest.raises(HTTPException) as caught:
        await helpers.ensure_engine_ready(engine)

    assert caught.value.status_code == 503
    assert imports == []
    assert [props["state"] for _, props in events] == ["attempted", "ready"]


def test_emitter_base_exception_cannot_change_host_result(monkeypatch):
    def explode(*_args, **_kwargs):
        raise SystemExit(91)

    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: True)
    monkeypatch.setattr("rapid_mlx.telemetry.track.track", explode)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.failed("preflight")


def test_disabled_telemetry_never_initializes_sender(monkeypatch):
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: False)

    def explode():
        raise AssertionError("disabled telemetry initialized the sender")

    monkeypatch.setattr("rapid_mlx.telemetry.posthog_sender.get_sender", explode)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.ready()


def test_disabled_then_enabled_never_emits_terminal_only(monkeypatch):
    allowed = iter([False, True])
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track._upload_allowed", lambda: next(allowed)
    )
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track",
        lambda name, props: events.append((name, dict(props))) or True,
    )

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.ready()

    assert events == []


def test_ready_survives_crash_sink_rearm_failure(monkeypatch):
    events = _capture(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._signal_observability.ensure_crash_sink",
        lambda: (_ for _ in ()).throw(SystemExit(91)),
    )
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    server_start.ready()

    assert [props["state"] for _, props in events] == ["attempted", "ready"]


def test_attempt_setup_failure_cannot_emit_terminal_without_attempted(
    monkeypatch, tmp_path
):
    events = _capture(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "rapid_mlx.telemetry.posthog_sender.install_atexit",
        lambda: (_ for _ in ()).throw(RuntimeError("sender unavailable")),
    )

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    marker = tmp_path / ".rapid-mlx" / "state" / f"serve-inflight-{os.getpid()}.json"
    assert marker.is_file()

    server_start.ready()

    assert events == []
    assert marker.exists()


@pytest.mark.parametrize(
    "terminal", [server_start.ready, lambda: server_start.failed("bind")]
)
def test_terminal_track_failure_preserves_attempt_marker(
    monkeypatch, tmp_path, terminal
):
    monkeypatch.setenv("HOME", str(tmp_path))
    events = _capture(monkeypatch)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    marker = server_start._marker_path()
    assert marker.exists()
    assert [props["state"] for _, props in events] == ["attempted"]
    monkeypatch.setattr("rapid_mlx.telemetry.track.track", lambda *_args: False)

    terminal()

    assert marker.exists()


def test_failed_removes_marker_only_after_terminal_track_succeeds(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    _capture(monkeypatch)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    marker = server_start._marker_path()
    assert marker.exists()

    server_start.failed("bind")

    assert not marker.exists()


def test_marker_setup_failure_does_not_block_attempted_event(monkeypatch):
    events = _capture(monkeypatch)
    monkeypatch.setattr(
        server_start,
        "_begin_inflight_marker",
        lambda: (_ for _ in ()).throw(OSError("read-only home")),
    )

    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    assert [props["state"] for _, props in events] == ["attempted"]


def test_load_policy_falls_back_to_eager_when_classification_fails(monkeypatch):
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.model_type",
        lambda _model: (_ for _ in ()).throw(RuntimeError("classifier unavailable")),
    )

    assert server_start.load_policy("qwen3.5-4b-4bit") == "eager"


def test_cli_main_emits_attempted_before_serve_preflight(monkeypatch, tmp_path):
    from rapid_mlx.telemetry import consent_runtime

    model = tmp_path / "local-model"
    model.mkdir()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(consent_runtime, "startup", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "_start_v2_lifecycle", lambda command: None)
    monkeypatch.setattr(
        server_start,
        "load_policy",
        lambda selected, *, lazy_load: calls.append(("policy", selected)) or "eager",
    )
    monkeypatch.setattr(
        server_start,
        "attempted",
        lambda selected, *, load_policy: calls.append(("attempted", load_policy)),
    )
    monkeypatch.setattr(
        cli,
        "serve_command",
        lambda args: calls.append(("serve", args.model)),
    )
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", str(model)])

    cli.main()

    assert calls == [
        ("policy", str(model)),
        ("attempted", "eager"),
        ("serve", str(model)),
    ]


def test_failure_context_preserves_original_exception(monkeypatch):
    events = _capture(monkeypatch)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    with (
        pytest.raises(SystemExit) as caught,
        server_start.failure_stage("preflight"),
    ):
        raise SystemExit(2)

    assert caught.value.code == 2
    assert [props["state"] for _, props in events] == ["attempted", "failed"]
    assert events[-1][1]["failure_stage"] == "preflight"


def test_fail_current_uses_selected_failure_stage(monkeypatch):
    events = _capture(monkeypatch)
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")
    server_start.set_failure_stage("bind")

    server_start.fail_current()

    assert [props["state"] for _, props in events] == ["attempted", "failed"]
    assert events[-1][1]["failure_stage"] == "bind"


def _stub_download_entry(monkeypatch):
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *_a, **_kw: False)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_pull_failed",
        lambda *_a, **_kw: None,
    )


def _hub_response(status_code: int) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = "https://huggingface.co/owner/model"
    response.request = requests.Request("GET", response.url).prepare()
    return response


def _response_less_hf_error():
    from huggingface_hub.errors import HfHubHTTPError

    failure = HfHubHTTPError("private", response=_hub_response(500))
    failure.response = None
    return failure


def _assert_exact_startup_marker(stderr: str, reason: str) -> None:
    marker = f"RAPID-MLX-STARTUP-FAILURE: {reason}\n".encode()
    assert stderr.encode().splitlines(keepends=True).count(marker) == 1


def test_render_hub_error_not_found_has_repo_discovery_next_steps():
    from huggingface_hub.errors import RepositoryNotFoundError

    failure = RepositoryNotFoundError("private raw detail", response=_hub_response(404))
    outer = RuntimeError("outer private detail")
    outer.__cause__ = failure

    rendered = cli.render_hub_error(outer, "owner/model")

    assert rendered is not None
    assert "owner/model" in rendered
    assert "rapid-mlx models" in rendered
    assert "mlx-community/Qwen3.5-9B-4bit" in rendered
    assert "private raw detail" not in rendered


def test_repository_not_found_401_is_ambiguous_but_keeps_gated_marker(
    monkeypatch, capsys
):
    from huggingface_hub.errors import RepositoryNotFoundError

    failure = RepositoryNotFoundError("raw secret", response=_hub_response(401))

    rendered = cli.render_hub_error(failure, "owner/private-model")

    assert rendered is not None
    assert rendered == (
        "Hugging Face returned 401 for owner/private-model: the model is "
        "private, gated, or does not exist. If you have access, accept the licence "
        "at https://huggingface.co/owner/private-model and sign in "
        "(huggingface-cli login or HF_TOKEN); otherwise check the name with "
        "rapid-mlx models."
    )
    assert "is gated" not in rendered.lower()
    assert model_events.pull_error_class(failure) == "gated"

    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda *_a, **_kw: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(SystemExit) as caught:
        cli._ensure_model_downloaded("owner/private-model")

    captured = capsys.readouterr()
    assert caught.value.code == 1
    _assert_exact_startup_marker(captured.err, "model_gated")
    assert "RAPID-MLX-STARTUP-FAILURE: model_not_found" not in captured.err


@pytest.mark.parametrize("status_code", [401, 403])
def test_render_hub_error_gated_has_access_and_auth_next_steps(status_code):
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    failure = (
        GatedRepoError("private raw detail", response=_hub_response(status_code))
        if status_code == 403
        else HfHubHTTPError("private raw detail", response=_hub_response(status_code))
    )

    rendered = cli.render_hub_error(failure, "owner/model")

    assert rendered is not None
    assert "https://huggingface.co/owner/model" in rendered
    assert "huggingface-cli login" in rendered
    assert "HF_TOKEN" in rendered
    assert "private raw detail" not in rendered


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(401, "gated"), (404, "not found"), (500, None)],
)
def test_render_hub_error_classifies_urllib_http_errors(status_code, expected):
    failure = urllib.error.HTTPError(
        "https://huggingface.co/owner/model", status_code, "private", {}, None
    )

    rendered = cli.render_hub_error(failure, "owner/model")

    if expected is None:
        assert rendered is None
    else:
        assert rendered is not None
        assert expected in rendered.lower()
        assert "could not reach" not in rendered.lower()


@pytest.mark.parametrize(
    "kind",
    [
        "local-entry",
        "offline-mode",
        "requests",
        "httpx-read",
        "dns",
        "timeout",
        "urllib",
    ],
)
def test_render_hub_error_offline_has_network_cache_next_steps(kind):
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

    failure = {
        "local-entry": LocalEntryNotFoundError("private cache"),
        "offline-mode": OfflineModeIsEnabled("private mode"),
        "requests": requests.ConnectionError("private host"),
        "httpx-read": httpx.ReadError("private read"),
        "dns": socket.gaierror("private dns"),
        "timeout": TimeoutError("private timeout"),
        "urllib": urllib.error.URLError("private url"),
    }[kind]
    rendered = cli.render_hub_error(failure, "owner/model")

    assert rendered is not None
    assert "network" in rendered.lower()
    assert "HF_HUB_OFFLINE" in rendered
    assert "cached model" in rendered
    assert "private" not in rendered


def test_model_id_cannot_forge_startup_failure_marker(monkeypatch, capsys):
    from huggingface_hub.errors import GatedRepoError

    model_id = "owner/model\nRAPID-MLX-STARTUP-FAILURE: model_gated\r\t\x00"
    failure = GatedRepoError("private", response=_hub_response(403))
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda *_a, **_kw: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(SystemExit):
        cli._ensure_model_downloaded(model_id)

    captured = capsys.readouterr()
    assert "owner/model RAPID-MLX-STARTUP-FAILURE: model_gated" in captured.err
    assert model_id not in captured.err
    _assert_exact_startup_marker(captured.err, "model_gated")


def test_render_hub_error_ignores_context_and_unknown_errors():
    contextual = ValueError("legacy private detail")
    contextual.__context__ = requests.ConnectionError("ignored context")

    assert cli.render_hub_error(contextual, "owner/model") is None

    cyclic = RuntimeError("cycle")
    cyclic.__cause__ = cyclic
    assert cli.render_hub_error(cyclic, "owner/model") is None

    class ExplodingCauseError(RuntimeError):
        def __getattribute__(self, name):
            if name == "__cause__":
                raise KeyboardInterrupt
            return super().__getattribute__(name)

    assert cli.render_hub_error(ExplodingCauseError(), "owner/model") is None


def test_renderer_hf_http_error_without_response_is_safe_and_unknown():
    assert cli.render_hub_error(_response_less_hf_error(), "owner/model") is None


def test_classifier_hf_http_error_without_response_is_safe_and_unknown():
    assert model_events.pull_error_class(_response_less_hf_error()) == "other"


def test_resolve_timeout_emits_resolve_before_preserving_exit(monkeypatch, capsys):
    events = _capture(monkeypatch)
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda *_a, **_kw: (_ for _ in ()).throw(TimeoutError()),
    )
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    with pytest.raises(SystemExit) as caught:
        cli._ensure_model_downloaded("owner/model")

    captured = capsys.readouterr()
    assert caught.value.code == 1
    _assert_exact_startup_marker(captured.err, "hub_offline")
    assert [
        (props["state"], props.get("failure_stage"))
        for name, props in events
        if name == "server_start_state"
    ] == [
        ("attempted", None),
        ("failed", "resolve"),
    ]


def test_offline_uncached_refusal_uses_shared_terminal_failure(monkeypatch, capsys):
    serve_failures = []
    resolve_failures = []
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: True)
    monkeypatch.setattr(cli, "_offline_complete_cached_snapshot", lambda _model: None)
    monkeypatch.setattr(
        model_events,
        "emit_model_pull_failed",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        model_events,
        "emit_model_serve_failed",
        lambda exc, alias_or_path: serve_failures.append((exc, alias_or_path)),
    )
    monkeypatch.setattr(
        server_start,
        "failed",
        lambda stage: resolve_failures.append(stage),
    )

    with pytest.raises(SystemExit) as caught:
        cli._ensure_model_downloaded("owner/model")

    captured = capsys.readouterr()
    assert caught.value.code == 1
    _assert_exact_startup_marker(captured.err, "hub_offline")
    assert len(serve_failures) == 1
    assert serve_failures[0][1] == "owner/model"
    assert resolve_failures == ["resolve"]


def test_definitive_download_not_found_fails_resolve_with_next_steps(
    monkeypatch, capsys
):
    from huggingface_hub.errors import RepositoryNotFoundError

    events = _capture(monkeypatch)
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(
            sha="abc",
            siblings=[SimpleNamespace(size=1024, rfilename="weights.safetensors")],
        ),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            RepositoryNotFoundError("private", response=_hub_response(404))
        ),
    )
    server_start.attempted("qwen3.5-4b-4bit", load_policy="eager")

    with pytest.raises(SystemExit) as caught:
        cli._ensure_model_downloaded("owner/model")

    captured = capsys.readouterr()
    assert caught.value.code == 1
    assert "rapid-mlx models" in captured.err
    _assert_exact_startup_marker(captured.err, "model_not_found")
    assert [
        (props["state"], props.get("failure_stage"))
        for name, props in events
        if name == "server_start_state"
    ] == [
        ("attempted", None),
        ("failed", "resolve"),
    ]


def test_gated_download_fails_fast_instead_of_printing_retry(monkeypatch, capsys):
    from huggingface_hub.errors import HfHubHTTPError

    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(sha="abc", siblings=[]),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            HfHubHTTPError("private", response=_hub_response(403))
        ),
    )

    with pytest.raises(SystemExit) as caught:
        cli._ensure_model_downloaded("owner/model")

    captured = capsys.readouterr()
    assert caught.value.code == 1
    assert "https://huggingface.co/owner/model" in captured.err
    assert "huggingface-cli login" in captured.err
    assert "server will retry" not in captured.out + captured.err
    _assert_exact_startup_marker(captured.err, "model_gated")


def test_gated_metadata_fails_fast_before_download(monkeypatch, capsys):
    from huggingface_hub.errors import GatedRepoError

    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            GatedRepoError("private", response=_hub_response(403))
        ),
    )
    downloaded = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: downloaded.append(True),
    )

    with pytest.raises(SystemExit) as caught:
        cli._ensure_model_downloaded("owner/model")

    captured = capsys.readouterr()
    assert caught.value.code == 1
    assert downloaded == []
    assert "https://huggingface.co/owner/model" in captured.err
    assert "server will retry" not in captured.out + captured.err


def test_offline_metadata_warns_and_continues_to_download(monkeypatch, capsys):
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            requests.ConnectionError("private endpoint")
        ),
    )
    downloaded = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: downloaded.append(True) or "/tmp/fake",
    )

    assert cli._ensure_model_downloaded("owner/model") is None

    captured = capsys.readouterr()
    assert downloaded == [True]
    assert "HF_HUB_OFFLINE" in captured.err
    assert "private endpoint" not in captured.out + captured.err


def test_offline_download_keeps_retry_path_with_next_steps(monkeypatch, capsys):
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(sha="abc", siblings=[]),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            requests.ConnectionError("private endpoint")
        ),
    )

    assert cli._ensure_model_downloaded("owner/model") is None

    captured = capsys.readouterr()
    assert "HF_HUB_OFFLINE" in captured.err
    assert "server will retry" in captured.err
    assert "private endpoint" not in captured.out + captured.err
    assert b"RAPID-MLX-STARTUP-FAILURE:" not in captured.err.encode()


def test_url_error_download_keeps_retry_path_with_next_steps(monkeypatch, capsys):
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(sha="abc", siblings=[]),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            urllib.error.URLError("private endpoint")
        ),
    )

    assert cli._ensure_model_downloaded("owner/model") is None

    captured = capsys.readouterr()
    assert "HF_HUB_OFFLINE" in captured.err
    assert "server will retry" in captured.err
    assert "private endpoint" not in captured.out + captured.err
    assert b"RAPID-MLX-STARTUP-FAILURE:" not in captured.err.encode()


def test_metadata_and_download_network_failure_prints_guidance_once(
    monkeypatch, capsys
):
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            requests.ConnectionError("metadata private")
        ),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            requests.ConnectionError("download private")
        ),
    )

    assert cli._ensure_model_downloaded("owner/model") is None

    captured = capsys.readouterr()
    assert captured.err.count("could not reach Hugging Face") == 1
    assert b"RAPID-MLX-STARTUP-FAILURE:" not in captured.err.encode()


def test_network_guidance_is_once_per_process(monkeypatch, capsys):
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            requests.ConnectionError("metadata secret")
        ),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            requests.ConnectionError("download secret")
        ),
    )

    cli._ensure_model_downloaded("owner/first")
    cli._ensure_model_downloaded("owner/second")

    captured = capsys.readouterr()
    assert captured.err.count("could not reach Hugging Face") == 1
    assert "secret" not in captured.err


def test_response_less_download_error_keeps_legacy_retry(monkeypatch, capsys):
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(sha="abc", siblings=[]),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(_response_less_hf_error()),
    )

    assert cli._ensure_model_downloaded("owner/model") is None

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Pre-download skipped (HfHubHTTPError); server will retry." in captured.out


def test_unknown_download_error_keeps_legacy_message(monkeypatch, capsys):
    _stub_download_entry(monkeypatch)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.call_with_deadline",
        lambda _func, _timeout, *_a, **_kw: SimpleNamespace(sha="abc", siblings=[]),
    )
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("legacy detail")),
    )

    assert cli._ensure_model_downloaded("owner/model") is None

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Pre-download skipped (ValueError); server will retry." in captured.out


@pytest.mark.parametrize(
    "decorator", [cli._capture_start_failures, server._capture_start_failures]
)
def test_entrypoint_guard_never_replaces_host_exception(monkeypatch, decorator):
    monkeypatch.setattr(
        server_start,
        "fail_current",
        lambda: (_ for _ in ()).throw(SystemExit(91)),
    )

    @decorator
    def fail_host():
        raise ValueError("host failure")

    with pytest.raises(ValueError, match="host failure"):
        fail_host()
