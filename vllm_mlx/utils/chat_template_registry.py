# SPDX-License-Identifier: Apache-2.0
"""Resolve declared chat-template contracts when a model is loaded.

Model profiles select a stable template ID.  This module owns the ID-to-asset
registry and the precedence rule; request rendering only consumes the resolved
``chat_template`` already installed on the tokenizer or processor.
"""

from __future__ import annotations

import logging
from functools import cache
from importlib.resources import files

logger = logging.getLogger(__name__)

ChatTemplateValue = str | dict[str, str]
ChatTemplateInput = ChatTemplateValue | list[dict[str, str]]

_TEMPLATE_ASSETS: dict[str, str] = {
    "gemma4_compact": "gemma4_compact.jinja",
    "gemma4_full": "gemma4_full.jinja",
}


@cache
def bundled_chat_template(template_id: str) -> str:
    """Return a bundled template by its validated registry ID."""

    try:
        asset = _TEMPLATE_ASSETS[template_id]
    except KeyError as exc:
        raise ValueError(f"unknown chat template ID: {template_id!r}") from exc
    return (
        files("vllm_mlx")
        .joinpath("templates")
        .joinpath(asset)
        .read_text(encoding="utf-8")
    )


def _template_owner(applicator):
    """Return the object whose ``chat_template`` is used for rendering."""

    if hasattr(applicator, "chat_template"):
        return applicator
    tokenizer = getattr(applicator, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "chat_template"):
        return tokenizer
    return None


def resolve_chat_template(
    applicator,
    template_id: str | None,
    *,
    explicit_template: ChatTemplateInput | None = None,
) -> bool:
    """Resolve one template onto a loaded tokenizer or processor.

    Precedence is explicit caller override, then the model profile's declared
    registry ID, then the checkpoint-provided template.  ``False`` means the
    checkpoint already had the selected value or no selection applied.
    """

    owner = _template_owner(applicator)
    if owner is None:
        return False
    if explicit_template is not None:
        # Tokenizer configs historically serialize named templates as a list
        # of {name, template} objects.  Transformers normalizes that config
        # shape to the runtime mapping consumed by apply_chat_template; do the
        # same here instead of restoring the raw, unhashable list post-load.
        selected: ChatTemplateValue = (
            {item["name"]: item["template"] for item in explicit_template}
            if isinstance(explicit_template, list)
            else explicit_template
        )
        source = "explicit override"
    elif template_id is not None:
        selected = bundled_chat_template(template_id)
        source = f"profile {template_id}"
    else:
        return False

    if getattr(owner, "chat_template", None) == selected:
        return False
    owner.chat_template = selected
    logger.info("Resolved chat template from %s", source)
    return True


def resolve_profile_chat_template(
    applicator,
    model_name: str,
    *,
    explicit_template: ChatTemplateInput | None = None,
) -> bool:
    """Resolve the template declared for an alias or exact repository path."""

    from ..model_aliases import resolve_profile

    profile = resolve_profile(model_name)
    template_id = profile.chat_template_id if profile is not None else None
    return resolve_chat_template(
        applicator,
        template_id,
        explicit_template=explicit_template,
    )


__all__ = [
    "bundled_chat_template",
    "resolve_chat_template",
    "resolve_profile_chat_template",
]
