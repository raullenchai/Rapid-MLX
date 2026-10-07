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


def _configured_log_file(label: str) -> str | None:
    """The ``--log-file`` in the active service config, if readable."""
    from .config import load_config
    from .install import _plist_path, log_file_from_serve_args
    from .plist import parse_plist

    try:
        program = parse_plist(_plist_path(label).read_bytes()).get(
            "ProgramArguments", []
        )
        config_path = Path(program[program.index("--config") + 1])
        return log_file_from_serve_args(load_config(config_path).serve_args)
    except Exception:
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
    target = _configured_log_file(label)
    if target == "/dev/null":
        print(
            "note: the service discards server output (--log-file /dev/null); "
            "the logs below contain only the service runtime's own messages."
        )
    elif target is not None and target not in ("-", "/dev/stderr", "/dev/stdout"):
        print(
            f"note: the service writes server output to {target} "
            "(--log-file); the logs below contain only the service runtime's "
            "own messages."
        )
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
