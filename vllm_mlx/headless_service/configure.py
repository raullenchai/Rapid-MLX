# SPDX-License-Identifier: Apache-2.0
"""Stage, inspect, and transactionally apply service configuration."""

from __future__ import annotations

import json
import pwd
import subprocess
import sys
from pathlib import Path

from .common import DEFAULT_DOMAIN, DEFAULT_LABEL
from .config import (
    ServiceConfigError,
    atomic_write,
    atomic_write_definition,
    backup_config_path,
    config_bytes,
    config_digest,
    credential_path,
    ensure_credential_dir,
    load_config,
    pending_config_path,
    private_file_present,
)
from .definition import installed_identity
from .install import _plist_path, _port_busy, _wait_ready, is_root


def _identity_or_error(label: str) -> tuple[str, Path, Path]:
    identity = installed_identity(label)
    if identity is None:
        raise ServiceConfigError(
            f"service {label} has no readable installed definition; install it first"
        )
    return identity


def _account(user: str):
    try:
        return pwd.getpwnam(user)
    except KeyError:
        raise ServiceConfigError(
            f"configured service account {user!r} no longer exists"
        ) from None


def configure_command(args) -> int:
    label = getattr(args, "label", None) or DEFAULT_LABEL
    dry_run = bool(getattr(args, "dry_run", False))
    try:
        user, home, current_path = _identity_or_error(label)
        current = load_config(current_path)
        updates = {}
        for arg_name in (
            "model",
            "host",
            "port",
            "log_retention_days",
            "log_max_mb",
            "log_backup_count",
        ):
            value = getattr(args, arg_name, None)
            if value is not None:
                updates[arg_name] = value
        raw_serve_args = getattr(args, "serve_args", None)
        if getattr(args, "clear_serve_args", False):
            updates["serve_args"] = ()
        elif raw_serve_args:
            values = tuple(raw_serve_args)
            if values[:1] == ("--",):
                values = values[1:]
            updates["serve_args"] = values
        candidate = current.updated(**updates)
    except ServiceConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rendered = config_bytes(candidate)
    pending = pending_config_path(home, label)
    if dry_run:
        print("Dry run — validated candidate config (not staged):")
        print(rendered.decode(), end="")
        return 0
    if not is_root():
        print(
            "error: staging system service configuration requires root; re-run with sudo.",
            file=sys.stderr,
        )
        return 1
    try:
        account = _account(user)
        atomic_write_definition(pending, rendered)
    except (OSError, ServiceConfigError) as exc:
        print(f"error: could not stage {pending}: {exc}", file=sys.stderr)
        return 2
    print(f"staged {pending} ({config_digest(candidate)[:12]}).")
    print("Run `sudo rapid-mlx service apply` to activate it with rollback protection.")
    return 0


def _bootstrap(label: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", "bootstrap", DEFAULT_DOMAIN, str(_plist_path(label))],
        capture_output=True,
        text=True,
    )


def _bootout(label: str) -> None:
    subprocess.run(
        ["launchctl", "bootout", f"{DEFAULT_DOMAIN}/{label}"],
        check=False,
        capture_output=True,
        text=True,
    )


def apply_command(args) -> int:
    label = getattr(args, "label", None) or DEFAULT_LABEL
    dry_run = bool(getattr(args, "dry_run", False))
    try:
        user, home, current_path = _identity_or_error(label)
        current = load_config(current_path)
        pending_path = pending_config_path(home, label)
        candidate = load_config(pending_path)
        if candidate.label != current.label or candidate.service_user != user:
            raise ServiceConfigError(
                "pending config cannot change label or service account"
            )
        if candidate.executable != current.executable:
            raise ServiceConfigError(
                "pending config cannot change the service executable"
            )
    except ServiceConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if (candidate.host, candidate.port) != (current.host, current.port) and _port_busy(
        candidate.host, candidate.port
    ):
        print(
            f"error: candidate endpoint {candidate.host}:{candidate.port} is already in use.",
            file=sys.stderr,
        )
        return 1
    if dry_run:
        print(
            f"[DRY-RUN] validate {pending_path}\n"
            f"[DRY-RUN] bootout {DEFAULT_DOMAIN}/{label}\n"
            f"[DRY-RUN] promote candidate {config_digest(candidate)[:12]}\n"
            f"[DRY-RUN] bootstrap {_plist_path(label)} and require /readyz\n"
            "[DRY-RUN] restore previous config and service on any failure"
        )
        return 0
    if not is_root():
        print(
            "error: applying system service configuration requires root; re-run with sudo.",
            file=sys.stderr,
        )
        return 1

    try:
        account = _account(user)
    except ServiceConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    previous = config_bytes(current)
    candidate_bytes = config_bytes(candidate)
    backup = backup_config_path(home, label)
    try:
        atomic_write_definition(backup, previous)
        _bootout(label)
        atomic_write_definition(current_path, candidate_bytes)
        boot = _bootstrap(label)
        if boot.returncode != 0:
            raise ServiceConfigError(
                f"launchctl bootstrap failed: {boot.stderr.strip() or boot.stdout.strip()}"
            )
        if not _wait_ready(candidate.host, candidate.port):
            raise ServiceConfigError(
                f"candidate did not become ready on {candidate.host}:{candidate.port} within 120s"
            )
    except (OSError, ServiceConfigError) as exc:
        _bootout(label)
        try:
            atomic_write_definition(current_path, previous)
            rollback = _bootstrap(label)
            rollback_ok = rollback.returncode == 0 and _wait_ready(
                current.host, current.port
            )
        except OSError:
            rollback_ok = False
        state = "previous service restored" if rollback_ok else "ROLLBACK FAILED"
        print(f"error: apply failed: {exc}; {state}.", file=sys.stderr)
        return 2

    try:
        pending_path.unlink()
    except OSError:
        pass
    print(
        f"applied {config_digest(candidate)[:12]}; {label} is ready on "
        f"{candidate.host}:{candidate.port}. Previous config: {backup}"
    )
    return 0


def config_show_command(args) -> int:
    label = getattr(args, "label", None) or DEFAULT_LABEL
    try:
        _, home, current_path = _identity_or_error(label)
        current = load_config(current_path)
        pending = pending_config_path(home, label)
        payload = {
            "active": current.to_dict(),
            "active_digest": config_digest(current),
            "pending": load_config(pending).to_dict() if pending.exists() else None,
        }
    except ServiceConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def credential_command(args) -> int:
    label = getattr(args, "label", None) or DEFAULT_LABEL
    action = getattr(args, "credential_command", None)
    try:
        user, home, _ = _identity_or_error(label)
    except ServiceConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    path = credential_path(home, label)
    if action == "status":
        print(json.dumps({"configured": private_file_present(path), "path": str(path)}))
        return 0
    if not is_root():
        print(
            "error: changing a system service credential requires root; re-run with sudo.",
            file=sys.stderr,
        )
        return 1
    if action == "unset":
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"error: could not remove credential: {exc}", file=sys.stderr)
            return 2
        print("service API credential removed; restart the service to disable auth.")
        return 0
    if action != "set":
        print("error: expected credential set/status/unset", file=sys.stderr)
        return 2
    if sys.stdin.isatty():
        print(
            "error: credential set reads one line from stdin; pipe it from a secret manager "
            "or enter it with terminal echo disabled.",
            file=sys.stderr,
        )
        return 1
    secret_input = sys.stdin.read()
    secret = secret_input.removesuffix("\n").removesuffix("\r")
    if not secret or "\n" in secret or "\r" in secret:
        print("error: credential must be exactly one non-empty line", file=sys.stderr)
        return 1
    try:
        account = _account(user)
        ensure_credential_dir(home, uid=account.pw_uid, gid=account.pw_gid)
        atomic_write(
            path,
            (secret + "\n").encode(),
            uid=account.pw_uid,
            gid=account.pw_gid,
        )
    except (OSError, ServiceConfigError) as exc:
        print(f"error: could not store credential: {exc}", file=sys.stderr)
        return 2
    print("service API credential stored with mode 0600; restart to activate it.")
    return 0
