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
from hashlib import sha256
from importlib.resources import files

logger = logging.getLogger(__name__)

_CANONICAL_MARKER = "Google Gemma 4 Canonical Chat Template"
_GEMMA4_NAME_MARKERS = ("gemma-4", "gemma_4", "gemma4")
_FULL_NAME_MARKERS = ("12b", "26b", "31b")
_KNOWN_STALE_TEMPLATE_VARIANTS = {
    # mlx-community E2B/E4B conversions (4/6/8-bit).
    "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082": "compact",
    # mlx-community 12B/26B/31B conversions (standard and most QAT variants).
    "36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0": "full",
    "94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413": "full",
}


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
    metadata before its exact template hash is considered.  Explicit E2B/E4B
    and 12B/26B/31B identities choose compact/full directly; the audited hash
    inventory supplies the fallback for unusually named local checkpoints.
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
    known_variant = _KNOWN_STALE_TEMPLATE_VARIANTS.get(
        sha256(current.encode("utf-8")).hexdigest()
    )
    if known_variant is None:
        return False

    if "e2b" in identity or "e4b" in identity:
        variant = "compact"
    elif any(marker in identity for marker in _FULL_NAME_MARKERS):
        variant = "full"
    else:
        variant = known_variant
    owner.chat_template = _canonical_template(variant)
    logger.info(
        "Upgraded stale Gemma 4 chat template to bundled Google canonical "
        "2026-07-09 variant=%s (model=%s)",
        variant,
        model_name or "<unknown>",
    )
    return True


__all__ = ["upgrade_stale_gemma4_chat_template"]
