# SPDX-License-Identifier: Apache-2.0
"""Health-gated in-place service upgrade with dependency rollback."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from .common import DEFAULT_LABEL
from .config import ServiceConfigError, atomic_write, load_config
from .configure import _account, _bootout, _bootstrap, _identity_or_error
from .install import _wait_ready, is_root


def _target(version: str | None, extras: str | None) -> str:
    if version is not None and not re.fullmatch(r"[A-Za-z0-9.!+_-]+", version):
        raise ServiceConfigError("version contains unsupported characters")
    if extras is not None and not re.fullmatch(
        r"[A-Za-z0-9_-]+(?:,[A-Za-z0-9_-]+)*", extras
    ):
        raise ServiceConfigError("extras must be a comma-separated list of names")
    suffix = f"[{extras}]" if extras else ""
    pin = f"=={version}" if version else ""
    return f"rapid-mlx{suffix}{pin}"


def _as_user(
    user: str,
    argv: list[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess:
    """Run through macOS sudo so HOME/groups match the service account."""
    return subprocess.run(
        ["/usr/bin/sudo", "-u", user, "-H", *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _restore(
    *,
    user: str,
    python: Path,
    requirements: Path,
    label: str,
    host: str,
    port: int,
) -> bool:
    _bootout(label)
    restored = _as_user(
        user,
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "-r",
            str(requirements),
        ],
        timeout=1800,
    )
    if restored.returncode != 0:
        return False
    boot = _bootstrap(label)
    return boot.returncode == 0 and _wait_ready(host, port)


def upgrade_command(args) -> int:
    label = getattr(args, "label", None) or DEFAULT_LABEL
    dry_run = bool(getattr(args, "dry_run", False))
    try:
        user, home, current_path = _identity_or_error(label)
        config = load_config(current_path)
        account = _account(user)
        target = _target(getattr(args, "version", None), getattr(args, "extras", None))
    except ServiceConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    python = home / ".rapid-mlx" / "bin" / "python"
    if not python.is_file() or not os.access(python, os.X_OK):
        print(
            f"error: service runtime Python is missing or not executable: {python}",
            file=sys.stderr,
        )
        return 1
    requirements = current_path.parent / f"{label}.previous-requirements.txt"
    plan = [
        f"freeze current environment -> {requirements}",
        f"bootout system/{label}",
        f"install {target} as {user}",
        "run rapid-mlx doctor (diagnostic)",
        f"bootstrap and require {config.host}:{config.port}/readyz",
        "restore frozen environment and old service on any failure",
    ]
    if dry_run:
        print("Dry run — would perform these upgrade steps:")
        for step in plan:
            print(f"  [DRY-RUN] {step}")
        return 0
    if not is_root():
        print(
            "error: upgrading a system service requires root; re-run with sudo.",
            file=sys.stderr,
        )
        return 1

    frozen = _as_user(user, [str(python), "-m", "pip", "freeze"], timeout=120)
    if frozen.returncode != 0 or not frozen.stdout.strip():
        print(
            f"error: could not snapshot the service environment: {frozen.stderr.strip()}",
            file=sys.stderr,
        )
        return 2
    try:
        atomic_write(
            requirements,
            frozen.stdout.encode(),
            uid=account.pw_uid,
            gid=account.pw_gid,
        )
    except OSError as exc:
        print(f"error: could not save rollback snapshot: {exc}", file=sys.stderr)
        return 2

    _bootout(label)
    install_argv = [str(python), "-m", "pip", "install", "--upgrade"]
    if getattr(args, "pre", False):
        install_argv.append("--pre")
    install_argv.append(target)
    upgraded = _as_user(user, install_argv, timeout=1800)
    if upgraded.returncode == 0:
        doctor = _as_user(user, [config.executable, "doctor"], timeout=180)
        if doctor.returncode != 0:
            print(
                "warning: upgraded `rapid-mlx doctor` reported issues; "
                "the real launchd readiness gate will decide acceptance.",
                file=sys.stderr,
            )
        boot = _bootstrap(label)
        healthy = boot.returncode == 0 and _wait_ready(config.host, config.port)
    else:
        healthy = False

    if healthy:
        print(
            f"upgraded {label} to {target}; service is ready. Rollback snapshot: "
            f"{requirements}"
        )
        return 0

    reason = upgraded.stderr.strip() if upgraded.returncode else "readiness gate failed"
    rollback_ok = _restore(
        user=user,
        python=python,
        requirements=requirements,
        label=label,
        host=config.host,
        port=config.port,
    )
    state = "previous environment restored" if rollback_ok else "ROLLBACK FAILED"
    print(f"error: upgrade failed ({reason}); {state}.", file=sys.stderr)
    return 2
