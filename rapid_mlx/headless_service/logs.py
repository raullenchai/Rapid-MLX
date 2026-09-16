# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx service logs`` — tail the daemon stdout/stderr logs.

Simple and dependency-free: prints each log (stdout then stderr) up to a
line cap, or streams both with ``--follow`` (which stays attached, like
``tail -F``, so the operator sees the daemon restart across KeepAlive
rebirths). Reading is read-only — it never touches the daemon.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .common import DEFAULT_LABEL, STDERR_LOG_NAME, STDOUT_LOG_NAME, log_dir_for


def _log_paths(label: str, user: str | None) -> tuple[Path, Path] | None:
    """Resolve the two log paths. Prefers the installed plist's declared
    paths (source of truth), falling back to the service account default."""
    from .install import _plist_path
    from .plist import parse_plist

    plist_path = _plist_path(label)
    if plist_path.is_file():
        try:
            config = parse_plist(plist_path.read_bytes())
            out = config.get("StandardOutPath")
            err = config.get("StandardErrorPath")
            if out and err:
                return Path(out), Path(err)
        except Exception:
            pass
    if user:
        log_dir = log_dir_for(user)
        if log_dir is not None:
            return log_dir / STDOUT_LOG_NAME, log_dir / STDERR_LOG_NAME
    return None


def logs_command(args) -> int:
    label = getattr(args, "label", None) or DEFAULT_LABEL
    user = getattr(args, "service_user", None)
    follow = bool(getattr(args, "follow", False))
    tail_n = int(getattr(args, "tail", None) or 200)

    paths = _log_paths(label, user)
    if paths is None:
        print(
            f"error: no log paths known for {label} — is it installed? "
            "Run `rapid-mlx service status` first.",
            file=sys.stderr,
        )
        return 1

    out_path, err_path = paths
    if follow:
        try:
            subprocess.run(["tail", "-F", str(out_path), str(err_path)], check=True)
        except KeyboardInterrupt:
            pass
        except subprocess.CalledProcessError:
            print("error: tail failed", file=sys.stderr)
            return 1
        return 0

    for name, path in (("stdout", out_path), ("stderr", err_path)):
        if not path.is_file():
            print(f"({name}: not present: {path})")
            continue
        print(f"=== {name}: {path} ===")
        try:
            subprocess.run(["tail", "-n", str(tail_n), str(path)], check=False)
        except OSError as exc:
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(logs_command(sys.argv))
