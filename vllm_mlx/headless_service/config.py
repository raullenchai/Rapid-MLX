# SPDX-License-Identifier: Apache-2.0
"""Versioned, atomic configuration for the always-on macOS service.

The LaunchDaemon points at a stable config path instead of embedding the
effective ``serve`` invocation in its plist.  Operators stage a candidate with
``service configure`` and promote it with ``service apply``.  The split keeps a
bad edit from putting launchd into a crash loop and gives apply a known-good
file to restore when the readiness gate fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .common import DEFAULT_LABEL

SCHEMA_VERSION = 1
SERVICE_CONFIG_ROOT = Path("/Library/Application Support/Rapid-MLX/Services")
DEFAULT_LOG_RETENTION_DAYS = 7
DEFAULT_LOG_MAX_MB = 100
DEFAULT_LOG_BACKUP_COUNT = 5


class ServiceConfigError(ValueError):
    """A persisted service definition is missing, unsafe, or invalid."""


@dataclass(frozen=True)
class ServiceConfig:
    schema_version: int
    label: str
    service_user: str
    executable: str
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    serve_args: tuple[str, ...] = ()
    credential_file: str | None = None
    log_retention_days: int = DEFAULT_LOG_RETENTION_DAYS
    log_max_mb: int = DEFAULT_LOG_MAX_MB
    log_backup_count: int = DEFAULT_LOG_BACKUP_COUNT

    def validated(self) -> ServiceConfig:
        from .common import validate_label
        from .install import refuse_secret_flags

        if self.schema_version != SCHEMA_VERSION:
            raise ServiceConfigError(
                f"unsupported service config schema {self.schema_version}; "
                f"this Rapid-MLX supports schema {SCHEMA_VERSION}"
            )
        validate_label(self.label)
        if not self.service_user or self.service_user == "root":
            raise ServiceConfigError("service_user must be a non-root account")
        executable = Path(self.executable)
        if not executable.is_absolute():
            raise ServiceConfigError("executable must be an absolute path")
        if not self.model or "\0" in self.model:
            raise ServiceConfigError("model must not be empty")
        if not self.host or "\0" in self.host or any(ch.isspace() for ch in self.host):
            raise ServiceConfigError(
                "host must be a non-empty address without whitespace"
            )
        if not 1 <= self.port <= 65535:
            raise ServiceConfigError("port must be in [1, 65535]")
        if self.log_retention_days < 1:
            raise ServiceConfigError("log_retention_days must be at least 1")
        if self.log_max_mb < 1:
            raise ServiceConfigError("log_max_mb must be at least 1")
        if self.log_backup_count < 1:
            raise ServiceConfigError("log_backup_count must be at least 1")
        try:
            refuse_secret_flags(self.serve_args)
        except Exception as exc:
            raise ServiceConfigError(str(exc)) from None
        if any("\0" in token for token in self.serve_args):
            raise ServiceConfigError("serve_args must not contain NUL bytes")
        if self.credential_file is not None:
            credential = Path(self.credential_file)
            if not credential.is_absolute():
                raise ServiceConfigError("credential_file must be an absolute path")
        return self

    def updated(self, **changes: Any) -> ServiceConfig:
        return replace(self, **changes).validated()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "service_user": self.service_user,
            "executable": self.executable,
            "model": self.model,
            "host": self.host,
            "port": self.port,
            "serve_args": list(self.serve_args),
            "credential_file": self.credential_file,
            "log_retention_days": self.log_retention_days,
            "log_max_mb": self.log_max_mb,
            "log_backup_count": self.log_backup_count,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ServiceConfig:
        allowed = {
            "schema_version",
            "label",
            "service_user",
            "executable",
            "model",
            "host",
            "port",
            "serve_args",
            "credential_file",
            "log_retention_days",
            "log_max_mb",
            "log_backup_count",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ServiceConfigError(
                f"unknown service config fields: {', '.join(unknown)}"
            )
        try:
            raw_serve_args = raw.get("serve_args", [])
            if not isinstance(raw_serve_args, list) or not all(
                isinstance(item, str) for item in raw_serve_args
            ):
                raise TypeError("serve_args must be an array of strings")
            config = cls(
                schema_version=int(raw["schema_version"]),
                label=str(raw["label"]),
                service_user=str(raw["service_user"]),
                executable=str(raw["executable"]),
                model=str(raw["model"]),
                host=str(raw.get("host", "127.0.0.1")),
                port=int(raw.get("port", 8000)),
                serve_args=tuple(raw_serve_args),
                credential_file=(
                    str(raw["credential_file"])
                    if raw.get("credential_file") is not None
                    else None
                ),
                log_retention_days=int(
                    raw.get("log_retention_days", DEFAULT_LOG_RETENTION_DAYS)
                ),
                log_max_mb=int(raw.get("log_max_mb", DEFAULT_LOG_MAX_MB)),
                log_backup_count=int(
                    raw.get("log_backup_count", DEFAULT_LOG_BACKUP_COUNT)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceConfigError(f"malformed service config: {exc}") from None
        return config.validated()


def config_dir(home: Path) -> Path:
    # Keep account context explicit in callers, but store the definition away
    # from the replaceable ~/.rapid-mlx virtual environment.
    del home
    return SERVICE_CONFIG_ROOT


def config_path(home: Path, label: str = DEFAULT_LABEL) -> Path:
    return config_dir(home) / f"{label}.json"


def pending_config_path(home: Path, label: str = DEFAULT_LABEL) -> Path:
    return config_dir(home) / f"{label}.pending.json"


def backup_config_path(home: Path, label: str = DEFAULT_LABEL) -> Path:
    return config_dir(home) / f"{label}.previous.json"


def credential_path(home: Path, label: str = DEFAULT_LABEL) -> Path:
    return home / ".rapid-mlx-secrets" / f"{label}.credential"


def load_config(path: Path) -> ServiceConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ServiceConfigError(f"service config does not exist: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ServiceConfigError(f"cannot read service config {path}: {exc}") from None
    if not isinstance(raw, dict):
        raise ServiceConfigError("service config root must be an object")
    return ServiceConfig.from_dict(raw)


def config_bytes(config: ServiceConfig) -> bytes:
    config.validated()
    return (json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n").encode()


def config_digest(config: ServiceConfig) -> str:
    return hashlib.sha256(config_bytes(config)).hexdigest()


def atomic_write(
    path: Path,
    data: bytes,
    *,
    mode: int = 0o600,
    uid: int | None = None,
    gid: int | None = None,
) -> None:
    """Durably replace ``path`` without exposing a partial definition."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        os.fchmod(fd, mode)
        if uid is not None or gid is not None:
            os.fchown(
                fd, uid if uid is not None else -1, gid if gid is not None else -1
            )
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_definition(path: Path, data: bytes) -> None:
    """Write a non-secret system definition root-owned in production."""
    owner = 0 if os.geteuid() == 0 else None
    atomic_write(path, data, mode=0o644, uid=owner, gid=owner)


def ensure_config_dir(home: Path, *, uid: int, gid: int) -> Path:
    """Create the root-owned, non-secret service definition directory."""
    del uid, gid
    target = config_dir(home)
    target.mkdir(mode=0o755, parents=True, exist_ok=True)
    os.chmod(target, 0o755)
    if os.geteuid() == 0:
        os.chown(target, 0, 0)
    return target


def ensure_credential_dir(home: Path, *, uid: int, gid: int) -> Path:
    """Create the private secret directory traversable by the service only."""
    target = credential_path(home).parent
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(target, 0o700)
    os.chown(target, uid, gid)
    return target


def assert_private_file(path: Path, *, expected_uid: int | None = None) -> None:
    """Refuse a credential that is symlinked, group/world-readable, or foreign."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise ServiceConfigError(
            f"cannot inspect credential file {path}: {exc}"
        ) from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ServiceConfigError("credential file must be a regular, non-symlink file")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ServiceConfigError(
            "credential file must not be accessible by group or others"
        )
    if expected_uid is not None and info.st_uid != expected_uid:
        raise ServiceConfigError(
            f"credential file is owned by uid {info.st_uid}, expected {expected_uid}"
        )


def private_file_present(path: Path) -> bool | None:
    """Return presence, or ``None`` when permissions prevent inspection."""
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except PermissionError:
        return None
    except OSError:
        return False
