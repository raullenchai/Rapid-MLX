# SPDX-License-Identifier: Apache-2.0
"""Boot-persistent autostart for lightning-mlx daemons.

macOS: per-user LaunchAgent (~/Library/LaunchAgents) loaded via launchctl.
Linux: per-user systemd unit (~/.config/systemd/user) enabled via systemctl.

Both run the same supervisor entrypoint used by the in-process daemon
(`python -m vllm_mlx.daemon supervise <record>`), so behavior at boot/login
matches behavior right after `lightning-mlx serve --daemon`.
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path
from typing import Any

PLATFORM = sys.platform

LAUNCHAGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"


class PersistenceError(RuntimeError):
    """Raised when autostart install/uninstall fails in a user-visible way."""


def service_label(daemon_id: str) -> str:
    return f"com.lightning-mlx.{daemon_id}"


def _systemd_unit_name(daemon_id: str) -> str:
    return f"lightning-mlx-{daemon_id}.service"


def _plist_path(daemon_id: str) -> Path:
    return LAUNCHAGENTS_DIR / f"{service_label(daemon_id)}.plist"


def _unit_path(daemon_id: str) -> Path:
    return SYSTEMD_USER_DIR / _systemd_unit_name(daemon_id)


def _supervise_argv(record_path: str) -> list[str]:
    return [sys.executable, "-m", "vllm_mlx.daemon", "supervise", record_path]


def install_autostart(record: dict[str, Any]) -> None:
    daemon_id = record["id"]
    record_path = str(record["record_path"])
    log_path = str(record.get("log_path") or "/dev/null")
    if PLATFORM == "darwin":
        _install_launchd(daemon_id, record_path, log_path)
    elif PLATFORM.startswith("linux"):
        _install_systemd(daemon_id, record_path)
    else:
        raise PersistenceError(
            f"Boot persistence not supported on platform {PLATFORM!r}. "
            "Use --daemon=non-persist."
        )


def uninstall_autostart(daemon_id: str) -> None:
    if PLATFORM == "darwin":
        _uninstall_launchd(daemon_id)
    elif PLATFORM.startswith("linux"):
        _uninstall_systemd(daemon_id)


def _install_launchd(daemon_id: str, record_path: str, log_path: str) -> None:
    LAUNCHAGENTS_DIR.mkdir(parents=True, exist_ok=True)
    plist_path = _plist_path(daemon_id)
    plist: dict[str, Any] = {
        "Label": service_label(daemon_id),
        "ProgramArguments": _supervise_argv(record_path),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": log_path,
        "StandardErrorPath": log_path,
        "ProcessType": "Interactive",
    }
    plist_path.write_bytes(plistlib.dumps(plist))
    # Best-effort unload first to overwrite an existing definition cleanly.
    subprocess.run(
        ["launchctl", "unload", str(plist_path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    result = subprocess.run(
        ["launchctl", "load", "-w", str(plist_path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if getattr(result, "returncode", 0) != 0:
        raise PersistenceError(
            f"launchctl load failed for {plist_path}: "
            f"{getattr(result, 'stderr', b'') or ''}"
        )


def _uninstall_launchd(daemon_id: str) -> None:
    plist_path = _plist_path(daemon_id)
    subprocess.run(
        ["launchctl", "unload", str(plist_path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        plist_path.unlink()
    except FileNotFoundError:
        pass


def _install_systemd(daemon_id: str, record_path: str) -> None:
    SYSTEMD_USER_DIR.mkdir(parents=True, exist_ok=True)
    unit_path = _unit_path(daemon_id)
    exec_start = " ".join(_supervise_argv(record_path))
    unit_text = (
        "[Unit]\n"
        f"Description=lightning-mlx daemon {daemon_id}\n"
        "After=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={exec_start}\n"
        "Restart=always\n"
        "RestartSec=2\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    unit_path.write_text(unit_text, encoding="utf-8")
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    result = subprocess.run(
        ["systemctl", "--user", "enable", "--now", _systemd_unit_name(daemon_id)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if getattr(result, "returncode", 0) != 0:
        raise PersistenceError(
            f"systemctl --user enable failed for {_systemd_unit_name(daemon_id)}: "
            f"{getattr(result, 'stderr', b'') or ''}. "
            "If you want this daemon to survive logout/reboot, also run: "
            "loginctl enable-linger $USER"
        )


def _uninstall_systemd(daemon_id: str) -> None:
    unit_name = _systemd_unit_name(daemon_id)
    subprocess.run(
        ["systemctl", "--user", "disable", "--now", unit_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _unit_path(daemon_id).unlink()
    except FileNotFoundError:
        pass
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
