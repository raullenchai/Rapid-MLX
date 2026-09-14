# SPDX-License-Identifier: Apache-2.0
"""Fail-closed eligibility for immutable, benchmark-qualified native-MTP pairs."""

from __future__ import annotations

from dataclasses import dataclass


class NativeMTPUnavailableError(ValueError):
    """Raised when an operator requests native MTP outside its qualified pair."""


@dataclass(frozen=True)
class NativeMTPPair:
    target_repo: str
    target_revision: str
    drafter_repo: str
    drafter_revision: str
    draft_tokens: int
    block_size: int
    drafter_model_type: str


QWEN36_35B_4BIT = NativeMTPPair(
    target_repo="mlx-community/Qwen3.6-35B-A3B-4bit",
    target_revision="38740b847e4cb78f352aba30aa41c76e08e6eb46",
    drafter_repo="mlx-community/Qwen3.6-35B-A3B-MTP-4bit",
    drafter_revision="0295b81421bf4d0fccca9a7c0fcfb1418dda3516",
    draft_tokens=2,
    block_size=3,
    drafter_model_type="qwen3_5_mtp",
)

GLM53_FLASH_4BIT = NativeMTPPair(
    target_repo="Vontra/GLM-5.3-Flash-MLX-4bit-MTP",
    target_revision="76add2a341a1cd90ad0e86bb69839ea9c35827c6",
    drafter_repo="rapid-mlx/GLM-5.3-Flash-MTP-4bit",
    drafter_revision="e9d62773d3e5272fb298830e8e06fadc4137ae2c",
    draft_tokens=1,
    block_size=2,
    drafter_model_type="glm5_next_mtp",
)

_QUALIFIED_PAIRS = {
    "qwen3.6-35b-4bit": QWEN36_35B_4BIT,
    "glm5.3-flash-4bit": GLM53_FLASH_4BIT,
}
_QUALIFIED_TARGETS = {pair.target_repo: pair for pair in _QUALIFIED_PAIRS.values()}


def resolve_native_mtp_pair(
    *,
    alias: str,
    target_repo: str,
    drafter_repo: str | None,
    draft_tokens: int,
) -> NativeMTPPair:
    """Return the immutable qualified pair or reject before loading weights."""

    pair = _QUALIFIED_PAIRS.get(alias)
    if pair is None and alias == target_repo:
        # ``serve org/repo`` is a first-class spelling everywhere else in the
        # registry. Permit it only when the user-typed value is itself the
        # exact qualified target; an arbitrary alias must not borrow another
        # model's eligibility.
        pair = _QUALIFIED_TARGETS.get(target_repo)
    if pair is None:
        raise NativeMTPUnavailableError(
            "native MTP is not qualified for this model alias"
        )
    if target_repo != pair.target_repo:
        raise NativeMTPUnavailableError(
            f"native MTP for {alias} requires target {pair.target_repo}"
        )
    if drafter_repo != pair.drafter_repo:
        raise NativeMTPUnavailableError(
            f"native MTP requires the {alias} alias's declared "
            f"sidecar ({pair.drafter_repo})"
        )
    if draft_tokens != pair.draft_tokens:
        raise NativeMTPUnavailableError(
            f"native MTP for {alias} requires "
            f"num_speculative_tokens={pair.draft_tokens}"
        )
    return pair


__all__ = [
    "NativeMTPPair",
    "NativeMTPUnavailableError",
    "GLM53_FLASH_4BIT",
    "QWEN36_35B_4BIT",
    "resolve_native_mtp_pair",
]
