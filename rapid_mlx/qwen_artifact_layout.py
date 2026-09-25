# SPDX-License-Identifier: Apache-2.0
"""Dependency-free Qwen MTP sidecar path discovery and classification.

The production injector and the offline artifact-truth probe share this file so
they cannot silently disagree about candidate precedence.  Discovery preserves
the injector's historical behavior exactly, including returning a root
``model.safetensors``.  Classification makes that last shape explicitly
ambiguous; it is never evidence that the file is an MTP head.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MTPWeightPathState(str, Enum):
    """Closed path shapes produced by the current locator."""

    ROOT_MTP = "root_mtp"
    ROOT_MODEL_MTP = "root_model_mtp"
    NESTED_MODEL = "nested_model"
    ROOT_MODEL_AMBIGUOUS = "root_model_ambiguous"
    NOT_FOUND = "not_found"
    UNSUPPORTED = "unsupported"


class MTPWeightStorage(str, Enum):
    """Filesystem representation of a located candidate."""

    SYMLINK = "symlink"
    REGULAR_FILE = "regular_file"
    OTHER = "other"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class MTPWeightLayout:
    """A locator observation; this type deliberately makes no eligibility claim."""

    state: MTPWeightPathState
    relative_path: str | None
    storage: MTPWeightStorage
    candidate: Path | None


def find_mtp_weights_file(sidecar_dir: Path) -> Path | None:
    """Return the first candidate using the production injector's precedence.

    Keep ``exists()`` rather than ``is_file()`` here for exact compatibility
    with the historical injector.  The truth classifier reports a non-file as
    ``unsupported``; changing production acceptance belongs in a separate,
    behavior-changing fix.
    """

    candidates = (
        sidecar_dir / "mtp.safetensors",
        sidecar_dir / "model-mtp.safetensors",
        sidecar_dir / "mtp" / "model.safetensors",
        sidecar_dir / "model.safetensors",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def inspect_mtp_weights_layout(sidecar_dir: Path) -> MTPWeightLayout:
    """Classify exactly what :func:`find_mtp_weights_file` would return."""

    candidate = find_mtp_weights_file(sidecar_dir)
    if candidate is None:
        return MTPWeightLayout(
            state=MTPWeightPathState.NOT_FOUND,
            relative_path=None,
            storage=MTPWeightStorage.NONE,
            candidate=None,
        )

    try:
        relative = candidate.relative_to(sidecar_dir).as_posix()
    except ValueError:
        return MTPWeightLayout(
            state=MTPWeightPathState.UNSUPPORTED,
            relative_path=None,
            storage=MTPWeightStorage.OTHER,
            candidate=None,
        )

    if candidate.is_symlink():
        storage = MTPWeightStorage.SYMLINK
    elif candidate.is_file():
        storage = MTPWeightStorage.REGULAR_FILE
    else:
        storage = MTPWeightStorage.OTHER

    states = {
        "mtp.safetensors": MTPWeightPathState.ROOT_MTP,
        "model-mtp.safetensors": MTPWeightPathState.ROOT_MODEL_MTP,
        "mtp/model.safetensors": MTPWeightPathState.NESTED_MODEL,
        "model.safetensors": MTPWeightPathState.ROOT_MODEL_AMBIGUOUS,
    }
    state = states.get(relative, MTPWeightPathState.UNSUPPORTED)
    if storage is MTPWeightStorage.OTHER:
        state = MTPWeightPathState.UNSUPPORTED
    return MTPWeightLayout(
        state=state,
        relative_path=relative if state is not MTPWeightPathState.UNSUPPORTED else None,
        storage=storage,
        candidate=(candidate if state is not MTPWeightPathState.UNSUPPORTED else None),
    )


__all__ = [
    "MTPWeightLayout",
    "MTPWeightPathState",
    "MTPWeightStorage",
    "find_mtp_weights_file",
    "inspect_mtp_weights_layout",
]
