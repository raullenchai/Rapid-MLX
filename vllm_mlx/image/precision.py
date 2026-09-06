# SPDX-License-Identifier: Apache-2.0
"""Explicit weight-source selection for the measured FLUX.2 Klein path."""

from __future__ import annotations

from pathlib import Path

FLUX2_KLEIN_Q4_ALIAS = "flux2-klein-4b"
FLUX2_KLEIN_BF16_ALIAS = "flux2-klein-4b-bf16"
FLUX2_KLEIN_Q4_REPO = "Runpod/FLUX.2-klein-4B-mflux-4bit"
FLUX2_KLEIN_BF16_REPO = "mflux-community/flux2-klein-4b-mflux-bf16"

IMAGE_WEIGHT_PRECISIONS = ("q4", "bf16")

_FLUX2_KLEIN_SOURCES = frozenset(
    {
        FLUX2_KLEIN_Q4_ALIAS.casefold(),
        FLUX2_KLEIN_BF16_ALIAS.casefold(),
        FLUX2_KLEIN_Q4_REPO.casefold(),
        FLUX2_KLEIN_BF16_REPO.casefold(),
        "black-forest-labs/flux.2-klein-4b",
    }
)


def resolve_image_weight_precision(model_name: str, precision: str) -> str:
    """Return the curated alias for an explicit image precision request.

    Only FLUX.2 Klein has an end-to-end q4/bf16 qualification today (#3058).
    Refusing other families prevents a seemingly generic flag from silently
    selecting an unmeasured checkpoint or changing a user-owned local model.
    """

    normalized_precision = (precision or "").casefold()
    if normalized_precision not in IMAGE_WEIGHT_PRECISIONS:
        raise ValueError(
            "image weight precision must be one of: "
            f"{', '.join(IMAGE_WEIGHT_PRECISIONS)}"
        )
    # Match ``resolve_model``'s repository-wide local-path precedence. A
    # same-named directory is user-owned checkpoint input, not permission to
    # discard it and silently select a remote curated checkpoint instead.
    if not model_name or Path(model_name).expanduser().is_dir():
        raise ValueError(
            "--image-weight-precision currently supports the curated "
            "FLUX.2 Klein model only, not local checkpoints"
        )
    if model_name.casefold() not in _FLUX2_KLEIN_SOURCES:
        raise ValueError(
            "--image-weight-precision currently supports FLUX.2 Klein only "
            "(use flux2-klein-4b); Z-Image and other diffusion families have "
            "not completed the q4/bf16 qualification"
        )
    return (
        FLUX2_KLEIN_BF16_ALIAS
        if normalized_precision == "bf16"
        else FLUX2_KLEIN_Q4_ALIAS
    )


def is_packaged_bf16_model(model_name: str) -> bool:
    """Whether *model_name* is an mflux-layout bf16 checkpoint."""

    return model_name.casefold() == FLUX2_KLEIN_BF16_REPO.casefold()
