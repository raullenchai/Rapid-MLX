# SPDX-License-Identifier: Apache-2.0
"""Local model-path detection and privacy-safe missing-file diagnostics."""

from __future__ import annotations

import difflib
import errno
import json
import os
import re
from pathlib import Path, PurePosixPath

MAX_MISSING_LOCAL_MODEL_FILES = 5
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _typed_ref(model_ref: str) -> str:
    """Remove shell quotes accidentally passed literally and surrounding space."""
    if os.path.exists(os.path.expanduser(model_ref)):
        return model_ref
    value = model_ref.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        inside = value[1:-1]
        value = inside if os.path.exists(os.path.expanduser(inside)) else inside.strip()
    return value


def resolved_model_ref(model_ref: str) -> str:
    """Use an absolute path for local input, preserving Hub repo ids."""
    # A literal existing directory can legitimately end in a space. Only
    # repair whitespace/quotes when that exact spelling does not resolve.
    value = _typed_ref(model_ref)
    # An unprefixed org/name is a Hub id when the whole path is missing,
    # regardless of whether an unrelated local org directory exists.
    if _HF_REPO_ID.fullmatch(value) and not os.path.exists(value):
        return value
    if is_local_model_ref(value) or ("/" in value and Path(value).parent.is_dir()):
        if "/" not in value and not value.startswith(("~", ".")):
            from .model_aliases import local_dir_shadows_alias

            if local_dir_shadows_alias(value) or os.path.exists(value):
                return value
        return str(Path(value).expanduser().absolute())
    return value


def _exception_chain(exc: BaseException | None) -> tuple[BaseException, ...]:
    """Return the bounded explicit cause chain used for load diagnostics."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen and len(chain) < 16:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__
    return tuple(chain)


def _has_authoritative_load_failure(exc: BaseException | None) -> bool:
    """Whether a typed failure must retain its own classification/message."""
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    from .model_load_errors import (
        IncompatibleWeights,
        InvalidModelConfig,
        QuantizationMismatch,
        TokenizerLoadFailed,
    )
    from .runtime.optional_runtime import OptionalRuntimeMissing

    authoritative = (
        OptionalRuntimeMissing,
        InvalidModelConfig,
        TokenizerLoadFailed,
        IncompatibleWeights,
        QuantizationMismatch,
        HfHubHTTPError,
        RepositoryNotFoundError,
        ModuleNotFoundError,
    )
    return any(isinstance(current, authoritative) for current in _exception_chain(exc))


def is_local_model_ref(model_ref: object) -> bool:
    """Whether ``model_ref`` denotes a local path, including a missing one."""
    if not isinstance(model_ref, str) or not model_ref:
        return False
    from .telemetry.redact import normalize_model_path

    if normalize_model_path(model_ref) == "<local>":
        return True
    try:
        return os.path.exists(os.path.expanduser(model_ref))
    except (OSError, ValueError):
        return True


def _safe_relative_name(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        return None
    return path.as_posix()


def _missing_index_shards(root: Path) -> list[str]:
    missing: set[str] = set()
    try:
        indexes = sorted(root.rglob("*.safetensors.index.json"))
    except OSError:
        return []
    for index_path in indexes:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict):
            continue
        for raw_name in weight_map.values():
            name = _safe_relative_name(raw_name)
            if name is None:
                continue
            candidate = index_path.parent.joinpath(*PurePosixPath(name).parts)
            try:
                present = candidate.is_file() and candidate.stat().st_size > 0
            except OSError:
                present = False
            if not present:
                missing.add(candidate.relative_to(root).as_posix())
    return sorted(missing)


def _exception_filename_path(exc: BaseException) -> Path | None:
    """Return a text filename as a path without accepting bytes paths."""
    if not isinstance(exc, FileNotFoundError) or exc.filename is None:
        return None
    try:
        filename = os.fspath(exc.filename)
    except TypeError:
        return None
    if not isinstance(filename, str):
        return None
    return Path(filename)


def missing_local_model_files(
    model_ref: object,
    exc: BaseException | None = None,
    *,
    limit: int = MAX_MISSING_LOCAL_MODEL_FILES,
) -> tuple[str, ...]:
    """Return bounded, model-root-relative missing names for a local failure."""
    if not is_local_model_ref(model_ref) or not isinstance(model_ref, str):
        return ()
    root = Path(model_ref).expanduser().absolute()
    if not root.is_dir():
        return ()

    missing = set(_missing_index_shards(root))
    for current in _exception_chain(exc):
        candidate = _exception_filename_path(current)
        if candidate is None:
            continue
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError:
            relative = None
        if relative and relative != ".":
            missing.add(relative)
    return tuple(sorted(missing)[: max(0, limit)])


def local_model_failure_message(
    model_ref: object,
    exc: BaseException | None = None,
    *,
    include_supplied_path: bool = False,
) -> str | None:
    """Render a local-model failure without exposing paths to HTTP clients."""
    if _has_authoritative_load_failure(exc):
        return None
    if not isinstance(model_ref, str):
        return None
    typed = _typed_ref(model_ref) if include_supplied_path else model_ref
    if not is_local_model_ref(typed) and not (
        include_supplied_path and "/" in typed and Path(typed).parent.is_dir()
    ):
        return None
    expanded = Path(typed).expanduser().absolute()
    try:
        exists = expanded.exists()
    except OSError:
        return None
    shown = f" {model_ref!r}" if include_supplied_path else ""
    if not exists:
        missing_path = False
        for current in _exception_chain(exc):
            candidate = _exception_filename_path(current)
            if candidate is not None and candidate.expanduser().absolute() == expanded:
                missing_path = True
                break
        if missing_path:
            message = f"The local model path{shown} does not exist."
            if not include_supplied_path:
                return message
            message += f" Resolved path: {expanded}."
            try:
                siblings = [p.name for p in expanded.parent.iterdir() if p.is_dir()]
            except OSError:
                siblings = []
            matches = difflib.get_close_matches(
                expanded.name, siblings, n=3, cutoff=0.0
            )
            if matches:
                message += " Nearby model folders: " + ", ".join(matches) + "."
            repo_id = typed.removeprefix("./")
            if typed.startswith("./") and _HF_REPO_ID.fullmatch(repo_id):
                message += f" If you meant a Hugging Face repo, use {repo_id}."
            elif typed.startswith("~/") or typed.startswith(("./", "../", "/")):
                message += " If you meant a Hugging Face repo, use org/model without a path prefix."
            return message
        return None
    missing = missing_local_model_files(model_ref, exc)
    if missing:
        return (
            f"The local model directory{shown} is missing required files: "
            f"{', '.join(missing)}."
        )
    return None


def raise_if_missing_local_model(model_ref: object) -> None:
    """Raise before a missing explicit local path can be treated as a Hub id."""
    if not is_local_model_ref(model_ref) or not isinstance(model_ref, str):
        return
    expanded = Path(model_ref).expanduser().absolute()
    exists = expanded.exists()
    if not exists:
        raise FileNotFoundError(
            errno.ENOENT,
            "Local model path does not exist",
            model_ref,
        )
