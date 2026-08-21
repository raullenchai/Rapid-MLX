# SPDX-License-Identifier: Apache-2.0
"""Upgrade stale Gemma 4 checkpoint templates to Google's canonical contract.

Most Rapid-MLX Gemma 4 aliases are conversions whose ``chat_template.jinja``
predates Google's 2026-07-09 tool-loop fixes.  Model weights remain current,
but replaying OpenAI tool history through those templates can render JSON null
as Python ``None`` and omit the corrected continuation framing.

The canonical templates are bundled so serving stays offline.  We replace only
the recognizable pre-canonical Gemma 4 template; unknown/custom templates and
already-current canonical templates are left untouched.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib.resources import files

logger = logging.getLogger(__name__)

_CANONICAL_MARKER = "Google Gemma 4 Canonical Chat Template"
_STALE_SIGNATURES = (
    "macro format_argument(argument, escape_keys=True)",
    "<|tool_call>call:",
    "namespace(prev_message_type=None)",
)
_FULL_GENERATION_CUE = "if not enable_thinking | default(false)"
_GEMMA4_NAME_MARKERS = ("gemma-4", "gemma_4", "gemma4")


@lru_cache(maxsize=2)
def _canonical_template(variant: str) -> str:
    if variant not in {"compact", "full"}:
        raise ValueError(f"unknown Gemma 4 template variant: {variant}")
    return (
        files("vllm_mlx")
        .joinpath("templates", f"gemma4_{variant}.jinja")
        .read_text(encoding="utf-8")
    )


def _template_owner(applicator):
    """Return the object that owns the live ``chat_template`` attribute."""

    if isinstance(getattr(applicator, "chat_template", None), str):
        return applicator
    tokenizer = getattr(applicator, "tokenizer", None)
    if isinstance(getattr(tokenizer, "chat_template", None), str):
        return tokenizer
    return None


def upgrade_stale_gemma4_chat_template(applicator, model_name: str = "") -> bool:
    """Install the matching canonical template when ``applicator`` is stale.

    E2B/E4B use Google's compact canonical variant; 12B/26B/31B use the full
    variant.  Gemma identity must be present in the served name or tokenizer
    metadata before template signatures are considered.  The old templates
    retain the compact/full distinction in their final generation-cue block;
    the model name also protects compact checkpoints whose publisher made
    superficial edits to that block.
    """

    owner = _template_owner(applicator)
    if owner is None:
        return False
    identity = " ".join(
        str(value).lower()
        for value in (
            model_name,
            getattr(owner, "name_or_path", ""),
            (getattr(owner, "init_kwargs", None) or {}).get("name_or_path", ""),
        )
        if value
    )
    if not any(marker in identity for marker in _GEMMA4_NAME_MARKERS):
        return False
    current = owner.chat_template
    if _CANONICAL_MARKER in current:
        return False
    if not all(signature in current for signature in _STALE_SIGNATURES):
        return False

    lowered_name = model_name.lower()
    compact_name = "e2b" in lowered_name or "e4b" in lowered_name
    variant = (
        "full" if _FULL_GENERATION_CUE in current and not compact_name else "compact"
    )
    owner.chat_template = _canonical_template(variant)
    logger.info(
        "Upgraded stale Gemma 4 chat template to bundled Google canonical "
        "2026-07-09 variant=%s (model=%s)",
        variant,
        model_name or "<unknown>",
    )
    return True


__all__ = ["upgrade_stale_gemma4_chat_template"]
