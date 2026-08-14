# SPDX-License-Identifier: Apache-2.0
"""User-owned model aliases with a small, fail-closed persistence contract."""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
from pathlib import Path

SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_TARGET_PART_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


class UserAliasError(ValueError):
    """The user alias store or a requested mutation is invalid."""


def config_path() -> Path:
    override = os.environ.get("RAPID_MLX_USER_ALIASES_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "rapid-mlx" / "user-aliases.json"


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise UserAliasError(
            "alias names must be 1-64 ASCII letters, digits, '.', '_' or '-', "
            "and must start with a letter or digit"
        )


def _is_hf_repo_id(target: str) -> bool:
    parts = target.split("/")
    return len(parts) == 2 and all(_TARGET_PART_RE.fullmatch(part) for part in parts)


def _read_raw(path: Path) -> object:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"version": SCHEMA_VERSION, "aliases": {}}
    if stat.S_ISLNK(info.st_mode):
        raise UserAliasError(f"refusing symlinked user alias config: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise UserAliasError(f"user alias config is not a regular file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise UserAliasError(f"user alias config is not a regular file: {path}")
            with os.fdopen(fd, encoding="utf-8") as handle:
                fd = -1
                return json.load(handle)
        finally:
            if fd >= 0:
                os.close(fd)
    except UserAliasError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UserAliasError(f"cannot read user alias config {path}: {exc}") from exc


def load_user_aliases() -> dict[str, str]:
    path = config_path()
    raw = _read_raw(path)
    if not isinstance(raw, dict) or set(raw) != {"version", "aliases"}:
        raise UserAliasError(
            f"user alias config {path} must contain only 'version' and 'aliases'"
        )
    if raw["version"] != SCHEMA_VERSION:
        raise UserAliasError(
            f"unsupported user alias config version {raw['version']!r} in {path}"
        )
    aliases = raw["aliases"]
    if not isinstance(aliases, dict):
        raise UserAliasError(f"'aliases' in {path} must be an object")
    parsed: dict[str, str] = {}
    folded: set[str] = set()
    for name, target in aliases.items():
        _validate_name(name)
        if name.casefold() in folded:
            raise UserAliasError(f"user alias names collide by case: {name!r}")
        if not isinstance(target, str) or not target:
            raise UserAliasError(f"target for user alias {name!r} must be a string")
        folded.add(name.casefold())
        parsed[name] = target
    return parsed


def _validated_mapping(
    aliases: dict[str, str],
    builtins: dict[str, str],
    reserved_names: frozenset[str] = frozenset(),
) -> dict[str, str]:
    builtin_folded = {name.casefold() for name in builtins}
    reserved_folded = builtin_folded | {name.casefold() for name in reserved_names}
    folded_names = [name.casefold() for name in aliases]
    if len(set(folded_names)) != len(folded_names):
        raise UserAliasError("user alias names collide by case")
    user_folded = set(folded_names)
    for name, target in aliases.items():
        _validate_name(name)
        if name.casefold() in reserved_folded:
            raise UserAliasError(f"{name!r} is a reserved built-in alias")
        if target.casefold() in user_folded:
            raise UserAliasError(
                f"user alias {name!r} targets another user alias; chains are not allowed"
            )
        if target not in builtins and not _is_hf_repo_id(target):
            raise UserAliasError(
                f"target {target!r} must be a built-in alias or Hugging Face repo id"
            )
    return aliases


def validated_user_aliases(
    builtins: dict[str, str], reserved_names: frozenset[str] = frozenset()
) -> dict[str, str]:
    return _validated_mapping(load_user_aliases(), builtins, reserved_names)


def _ensure_safe_parent(path: Path) -> None:
    parent = path.parent
    existed = parent.exists()
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise UserAliasError(f"refusing unsafe user alias directory: {parent}")
    if not existed:
        try:
            os.chmod(parent, 0o700)
        except OSError as exc:
            raise UserAliasError(
                f"cannot secure user alias directory {parent}: {exc}"
            ) from exc


def _write_aliases(aliases: dict[str, str]) -> None:
    path = config_path()
    _ensure_safe_parent(path)
    # Never replace through an existing symlink. os.replace would replace the
    # link rather than its target, but rejecting it makes tampering explicit.
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise UserAliasError(f"refusing symlinked user alias config: {path}")
    except FileNotFoundError:
        pass
    payload = (
        json.dumps(
            {"version": SCHEMA_VERSION, "aliases": aliases},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
    except OSError as exc:
        raise UserAliasError(f"cannot write user alias config {path}: {exc}") from exc


def set_user_alias(
    name: str,
    target: str,
    builtins: dict[str, str],
    reserved_names: frozenset[str] = frozenset(),
) -> None:
    aliases = load_user_aliases()
    aliases[name] = target
    _validated_mapping(aliases, builtins, reserved_names)
    _write_aliases(aliases)


def remove_user_alias(
    name: str,
    builtins: dict[str, str],
    reserved_names: frozenset[str] = frozenset(),
) -> bool:
    aliases = validated_user_aliases(builtins, reserved_names)
    if name not in aliases:
        return False
    del aliases[name]
    _write_aliases(aliases)
    return True
