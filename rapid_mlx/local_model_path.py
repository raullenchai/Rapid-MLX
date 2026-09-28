# SPDX-License-Identifier: Apache-2.0
"""Local model-path detection and privacy-safe missing-file diagnostics."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path, PurePosixPath

MAX_MISSING_LOCAL_MODEL_FILES = 5


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
        if isinstance(current, FileNotFoundError) and current.filename:
            candidate = Path(current.filename)
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
    if not is_local_model_ref(model_ref) or not isinstance(model_ref, str):
        return None
    expanded = Path(model_ref).expanduser().absolute()
    try:
        exists = expanded.exists()
    except OSError:
        return None
    shown = f" {model_ref!r}" if include_supplied_path else ""
    if not exists:
        missing_path = any(
            isinstance(current, FileNotFoundError)
            and current.filename is not None
            and Path(current.filename).expanduser().absolute() == expanded
            for current in _exception_chain(exc)
        )
        if missing_path:
            return f"The local model path{shown} does not exist."
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
    try:
        exists = expanded.exists()
    except OSError:
        exists = False
    if not exists:
        raise FileNotFoundError(
            errno.ENOENT,
            "Local model path does not exist",
            model_ref,
        )
