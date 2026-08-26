"""Metadata-backed capacity charges for shared residency roles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoleCapacity:
    requested_bytes: int | None
    source: str


def speech_input_capacity(model_id: str) -> RoleCapacity:
    """Resolve a speech-input charge without model-name inference or I/O."""

    from ..audio.registry import resolve_audio_alias
    from ..model_sizes import size_bytes

    entry = resolve_audio_alias(model_id)
    hf_id = entry.hf_id if entry is not None else model_id
    requested_bytes = size_bytes(hf_id)
    return RoleCapacity(
        requested_bytes=requested_bytes,
        source="catalog" if requested_bytes is not None else "unknown",
    )
