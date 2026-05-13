# SPDX-License-Identifier: Apache-2.0
"""Tests for boot-persistent daemon autostart (launchd + systemd --user)."""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path

import pytest

from vllm_mlx import persistence


@pytest.fixture()
def darwin_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "PLATFORM", "darwin")
    monkeypatch.setattr(persistence, "LAUNCHAGENTS_DIR", tmp_path / "LaunchAgents")
    monkeypatch.setattr(persistence, "SYSTEMD_USER_DIR", tmp_path / "systemd_user_unused")
    return tmp_path


@pytest.fixture()
def linux_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "PLATFORM", "linux")
    monkeypatch.setattr(persistence, "LAUNCHAGENTS_DIR", tmp_path / "LaunchAgents_unused")
    monkeypatch.setattr(persistence, "SYSTEMD_USER_DIR", tmp_path / "systemd_user")
    return tmp_path


def _record(daemon_id: str = "abc123", record_path: str = "/tmp/r.json") -> dict:
    return {
        "id": daemon_id,
        "model": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "record_path": record_path,
        "log_path": "/tmp/abc123.log",
    }


def test_label_for_id_is_stable():
    assert persistence.service_label("abc123") == "com.lightning-mlx.abc123"


def test_install_darwin_writes_plist_and_loads_it(darwin_layout, monkeypatch):
    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(cmd)

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(persistence.subprocess, "run", fake_run)

    record = _record(record_path="/tmp/abc.json")
    persistence.install_autostart(record)

    plist_path = darwin_layout / "LaunchAgents" / "com.lightning-mlx.abc123.plist"
    assert plist_path.exists()
    data = plistlib.loads(plist_path.read_bytes())
    assert data["Label"] == "com.lightning-mlx.abc123"
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is True
    assert data["ProgramArguments"][:3] == [sys.executable, "-m", "vllm_mlx.daemon"]
    assert data["ProgramArguments"][-2:] == ["supervise", "/tmp/abc.json"]

    assert any(cmd[:2] == ["launchctl", "load"] for cmd in calls)
    assert any(str(plist_path) in cmd for cmd in calls)


def test_uninstall_darwin_unloads_and_removes_plist(darwin_layout, monkeypatch):
    plist_path = darwin_layout / "LaunchAgents" / "com.lightning-mlx.abc123.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_bytes(b"<plist/>")

    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(cmd)

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(persistence.subprocess, "run", fake_run)

    persistence.uninstall_autostart("abc123")

    assert not plist_path.exists()
    assert any(cmd[:2] == ["launchctl", "unload"] for cmd in calls)


def test_uninstall_darwin_is_idempotent_when_plist_missing(darwin_layout, monkeypatch):
    monkeypatch.setattr(persistence.subprocess, "run", lambda *a, **k: None)
    persistence.uninstall_autostart("does-not-exist")


def test_install_linux_writes_unit_and_enables_it(linux_layout, monkeypatch):
    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(cmd)

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(persistence.subprocess, "run", fake_run)

    record = _record(record_path="/tmp/abc.json")
    persistence.install_autostart(record)

    unit_path = linux_layout / "systemd_user" / "lightning-mlx-abc123.service"
    assert unit_path.exists()
    text = unit_path.read_text(encoding="utf-8")
    assert "[Service]" in text
    assert f"{sys.executable} -m vllm_mlx.daemon supervise /tmp/abc.json" in text
    assert "Restart=always" in text
    assert "WantedBy=default.target" in text

    cmds = [tuple(c) for c in calls]
    assert ("systemctl", "--user", "daemon-reload") in cmds
    assert ("systemctl", "--user", "enable", "--now", "lightning-mlx-abc123.service") in cmds


def test_install_on_unsupported_platform_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "PLATFORM", "win32")
    with pytest.raises(persistence.PersistenceError, match="not supported"):
        persistence.install_autostart(_record())


def test_uninstall_linux_disables_and_removes_unit(linux_layout, monkeypatch):
    unit_path = linux_layout / "systemd_user" / "lightning-mlx-abc123.service"
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text("stub", encoding="utf-8")

    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(tuple(cmd))

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(persistence.subprocess, "run", fake_run)

    persistence.uninstall_autostart("abc123")

    assert not unit_path.exists()
    assert ("systemctl", "--user", "disable", "--now", "lightning-mlx-abc123.service") in calls
    assert ("systemctl", "--user", "daemon-reload") in calls
