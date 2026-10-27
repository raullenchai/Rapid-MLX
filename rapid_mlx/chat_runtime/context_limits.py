"""Shared chat context policy for owner requests and inference startup."""

from __future__ import annotations

from typing import Any

DEFAULT_CONTEXT_TOKENS = 32768
MAX_CONTEXT_TOKENS = 131072


def context_limit(
    requested: int | None = None,
    *,
    model: dict[str, Any] | None = None,
    previous: int | None = None,
) -> int:
    """Respect an explicit request, the harness ceiling and advertised server limits.

    Harness chats are capped at ``MAX_CONTEXT_TOKENS`` even when a larger window is
    requested: longer transcripts are compacted instead.
    """
    limits = [requested or DEFAULT_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS]
    advertised = [
        value
        for name in ("context_window", "max_model_len")
        if type(value := (model or {}).get(name)) is int and value > 0
    ]
    limits.extend(advertised)
    if not advertised and previous is not None:
        limits.append(previous)
    result = min(limits)
    if result < 512:
        raise ValueError(
            "The served context window is too small for chat (512 minimum)"
        )
    return result


def startup_context_limit(model: str, requested: int | None = None) -> int:
    """Choose the initial server ceiling from cached checkpoint metadata only."""
    from rapid_mlx.model_aliases import resolve_profile
    from rapid_mlx.model_metadata import read_model_metadata

    profile = resolve_profile(model)
    reference = getattr(profile, "hf_path", None) or model
    metadata = read_model_metadata(reference)
    config = metadata.config if metadata is not None else {}
    config = config or {}
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        config = text_config
    native = next(
        (
            value
            for key in (
                "max_position_embeddings",
                "n_positions",
                "max_seq_len",
                "seq_length",
            )
            if type(value := config.get(key)) is int and value > 0
        ),
        None,
    )
    return context_limit(requested, model={"context_window": native})
