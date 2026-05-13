# SPDX-License-Identifier: Apache-2.0
"""Detached daemon supervision for ``lightning-mlx serve``."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_mlx.persistence import PersistenceError, install_autostart, uninstall_autostart

APP_DIR = Path.home() / ".lightning-mlx"
DAEMON_DIR = APP_DIR / "daemons"
LOG_DIR = APP_DIR / "logs"
BACKOFF_SECONDS = (1, 2, 5, 10, 30)


class DaemonError(RuntimeError):
    """Raised for user-visible daemon control failures."""


@dataclass
class DaemonMatch:
    record: dict[str, Any]
    path: Path


def _now() -> float:
    return time.time()


def _ensure_dirs() -> None:
    DAEMON_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _record_path(daemon_id: str) -> Path:
    return DAEMON_DIR / f"{daemon_id}.json"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    _ensure_dirs()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive(pid: int | str | None) -> bool:
    try:
        pid_int = int(pid or 0)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_pid(pid: int | str | None, sig: int) -> bool:
    try:
        pid_int = int(pid or 0)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _base_url(host: str, port: int) -> str:
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    return f"http://{display_host}:{port}"


def _is_persistent(args: argparse.Namespace, raw_args: list[str]) -> bool:
    """`--daemon` defaults to persistent. `--daemon=non-persist` (or `=ephemeral`)
    opts out. Tolerates `args.daemon` being bool, str, or None."""
    value = getattr(args, "daemon", None)
    if isinstance(value, str) and value.lower() in {"non-persist", "ephemeral", "no", "false", "0"}:
        return False
    for arg in raw_args:
        if arg.startswith("--daemon="):
            val = arg.split("=", 1)[1].lower()
            if val in {"non-persist", "ephemeral", "no", "false", "0"}:
                return False
    return True


def _filtered_serve_args(raw_args: list[str]) -> list[str]:
    filtered: list[str] = []
    for arg in raw_args:
        if arg == "--daemon":
            continue
        if arg.startswith("--daemon="):
            continue
        filtered.append(arg)
    # Daemon always reclaims its port: inject --force so the child kills any
    # stale process holding the port before binding.
    if "--force" not in filtered:
        filtered.append("--force")
    return filtered


def start_daemon(args: argparse.Namespace, raw_args: list[str]) -> dict[str, Any]:
    """Start a detached supervisor for a serve command and return its record.

    Persistent mode (default for ``--daemon``) installs a launchd/systemd unit
    so the supervisor restarts at user login / system boot. Use
    ``--daemon=non-persist`` for the original in-process detached mode.
    """
    _ensure_dirs()
    daemon_id = uuid.uuid4().hex[:12]
    log_path = LOG_DIR / f"{daemon_id}.log"
    stop_path = DAEMON_DIR / f"{daemon_id}.stop"
    record_path = _record_path(daemon_id)
    persistent = _is_persistent(args, raw_args)

    command = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        *_filtered_serve_args(raw_args),
    ]
    record: dict[str, Any] = {
        "id": daemon_id,
        "model": args.model,
        "original_alias": getattr(args, "_original_alias", None),
        "served_model_name": args.served_model_name or args.model,
        "host": args.host,
        "port": int(args.port),
        "base_url": _base_url(args.host, int(args.port)),
        "v1_url": _base_url(args.host, int(args.port)) + "/v1",
        "supervisor_pid": None,
        "child_pid": None,
        "command": command,
        "created_at": _now(),
        "updated_at": _now(),
        "restarts": 0,
        "state": "starting",
        "log_path": str(log_path),
        "stop_path": str(stop_path),
        "record_path": str(record_path),
        "persistent": persistent,
    }
    # Remove any leftover stop marker from a previous lifecycle; otherwise
    # launchd/systemd would start the supervisor and it would exit immediately.
    if stop_path.exists():
        try:
            stop_path.unlink()
        except OSError:
            pass
    _atomic_write_json(record_path, record)

    if persistent:
        try:
            install_autostart(record)
        except Exception as exc:
            try:
                record_path.unlink()
            except OSError:
                pass
            raise DaemonError(f"Failed to install autostart: {exc}") from exc
        return record

    supervisor_cmd = [
        sys.executable,
        "-m",
        "vllm_mlx.daemon",
        "supervise",
        str(record_path),
    ]
    log_fh = open(log_path, "ab")
    try:
        proc = subprocess.Popen(  # noqa: S603 - args are constructed by us
            supervisor_cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_fh.close()

    current = _read_json(record_path) or record
    current["supervisor_pid"] = proc.pid
    current["updated_at"] = _now()
    _atomic_write_json(record_path, current)
    return current


def _records() -> list[DaemonMatch]:
    if not DAEMON_DIR.exists():
        return []
    matches: list[DaemonMatch] = []
    for path in sorted(DAEMON_DIR.glob("*.json")):
        record = _read_json(path)
        if record:
            matches.append(DaemonMatch(record=record, path=path))
    return matches


def list_daemons(clean_stale: bool = True) -> list[DaemonMatch]:
    live: list[DaemonMatch] = []
    for match in _records():
        record = match.record
        alive = _pid_alive(record.get("supervisor_pid"))
        if alive:
            live.append(match)
            continue
        # Persistent daemons are owned by launchd/systemd; a dead supervisor pid
        # just means a relaunch is in flight. Surface them as "pending" instead
        # of marking them stale.
        if record.get("persistent"):
            record["state"] = "pending"
            record["updated_at"] = _now()
            _atomic_write_json(match.path, record)
            live.append(DaemonMatch(record=record, path=match.path))
            continue
        if clean_stale:
            record["state"] = "stale"
            record["updated_at"] = _now()
            _atomic_write_json(match.path, record)
    return live


def _target_matches(record: dict[str, Any], target: str) -> bool:
    if target.isdigit():
        return target in {
            str(record.get("supervisor_pid") or ""),
            str(record.get("child_pid") or ""),
        }
    names = {
        str(record.get("model") or ""),
        str(record.get("original_alias") or ""),
        str(record.get("served_model_name") or ""),
    }
    return target in names


def resolve_daemon(target: str) -> DaemonMatch:
    matches = [match for match in list_daemons() if _target_matches(match.record, target)]
    if not matches:
        raise DaemonError(f"No daemon matched {target!r}.")
    if len(matches) > 1:
        labels = ", ".join(
            f"{m.record.get('supervisor_pid')}:{m.record.get('model')}" for m in matches
        )
        raise DaemonError(f"Multiple daemons matched {target!r}: {labels}")
    return matches[0]


def stop_daemon(target: str, timeout_s: float = 10.0) -> dict[str, Any]:
    match = resolve_daemon(target)
    record = match.record
    # Uninstall autostart BEFORE signaling so launchd/systemd can't restart
    # the supervisor we are about to kill.
    if record.get("persistent"):
        try:
            uninstall_autostart(str(record["id"]))
        except Exception:
            pass
        record["persistent"] = False
        record["updated_at"] = _now()
        _atomic_write_json(match.path, record)
    Path(str(record["stop_path"])).write_text("stop\n", encoding="utf-8")
    _signal_pid(record.get("supervisor_pid"), signal.SIGTERM)
    deadline = _now() + timeout_s
    while _now() < deadline:
        if not _pid_alive(record.get("supervisor_pid")):
            break
        time.sleep(0.1)
    if _pid_alive(record.get("supervisor_pid")):
        _signal_pid(record.get("supervisor_pid"), signal.SIGKILL)
    record["state"] = "stopped"
    record["updated_at"] = _now()
    _atomic_write_json(match.path, record)
    return record


def format_status(matches: list[DaemonMatch]) -> str:
    if not matches:
        return "No lightning-mlx daemons running."
    rows = [
        "PID       CHILD     STATE      RESTARTS  URL                       MODEL",
        "--------  --------  ---------  --------  ------------------------  ----------------",
    ]
    for match in matches:
        r = match.record
        rows.append(
            f"{str(r.get('supervisor_pid') or '-'):8}  "
            f"{str(r.get('child_pid') or '-'):8}  "
            f"{str(r.get('state') or '-'):9}  "
            f"{str(r.get('restarts') or 0):8}  "
            f"{str(r.get('base_url') or '-'):24}  "
            f"{str(r.get('original_alias') or r.get('model') or '-')}"
        )
    return "\n".join(rows)


def _terminate_child(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform != "win32":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    except ProcessLookupError:
        pass


def supervise(record_path: Path) -> int:
    record = _read_json(record_path)
    if not record:
        return 1

    stop_path = Path(str(record["stop_path"]))
    log_path = Path(str(record["log_path"]))
    command = list(record["command"])
    supervisor_pid = os.getpid()
    child: subprocess.Popen[Any] | None = None
    stopping = False

    def _handle_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True
        stop_path.write_text("stop\n", encoding="utf-8")
        _terminate_child(child)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    restart_index = 0
    while not stopping and not stop_path.exists():
        with open(log_path, "ab") as log_fh:
            child = subprocess.Popen(  # noqa: S603 - saved command from our CLI
                command,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=os.environ.copy(),
                start_new_session=True,
                close_fds=True,
            )
        record["supervisor_pid"] = supervisor_pid
        record["child_pid"] = child.pid
        record["state"] = "running"
        record["updated_at"] = _now()
        _atomic_write_json(record_path, record)

        return_code = child.wait()
        child = None
        if stopping or stop_path.exists():
            break

        record["child_pid"] = None
        record["state"] = f"restarting({return_code})"
        record["restarts"] = int(record.get("restarts") or 0) + 1
        record["updated_at"] = _now()
        _atomic_write_json(record_path, record)
        delay = BACKOFF_SECONDS[min(restart_index, len(BACKOFF_SECONDS) - 1)]
        restart_index += 1
        time.sleep(delay)

    record["child_pid"] = None
    record["state"] = "stopped"
    record["updated_at"] = _now()
    _atomic_write_json(record_path, record)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    supervisor = subparsers.add_parser("supervise")
    supervisor.add_argument("record_path", type=Path)
    args = parser.parse_args()
    if args.command == "supervise":
        raise SystemExit(supervise(args.record_path))


if __name__ == "__main__":
    main()
