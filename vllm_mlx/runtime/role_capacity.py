"""Metadata-backed capacity charges for shared residency roles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoleCapacity:
    """A role's footprint charge resolved from durable metadata, not guesses."""

    requested_bytes: int | None
    source: str


def alignment_capacity(model_id: str) -> RoleCapacity:
    """Resolve an alignment-role charge from catalog metadata.

    The forced-aligner aliases (``qwen3-aligner`` / ``qwen3-forced-aligner``)
    and the underlying HF id (``mlx-community/Qwen3-ForcedAligner-0.6B-8bit``)
    all resolve to the same ``hf_id`` through the audio-alias registry, from
    which the checked-in download-size manifest (:mod:`vllm_mlx.model_sizes`)
    derives a real byte footprint BEFORE any weight loading — so admission can
    decide with knowledge of the size rather than blind.

    A missing catalog entry yields ``requested_bytes=None`` with
    ``source="unknown"`` so a configured residency ceiling fails closed
    (``role_capacity_unknown``) instead of admitting without a typed decision.
    """
    from ..audio.registry import resolve_audio_alias
    from ..model_sizes import size_bytes

    entry = resolve_audio_alias(model_id)
    hf_id = entry.hf_id if entry is not None else model_id
    requested_bytes = size_bytes(hf_id)
    return RoleCapacity(
        requested_bytes=requested_bytes,
        source="catalog" if requested_bytes is not None else "unknown",
    )
