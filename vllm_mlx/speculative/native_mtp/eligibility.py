# SPDX-License-Identifier: Apache-2.0
"""Fail-closed eligibility for the qualified Qwen3.6 native-MTP pair."""

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


QWEN36_35B_4BIT = NativeMTPPair(
    target_repo="mlx-community/Qwen3.6-35B-A3B-4bit",
    target_revision="38740b847e4cb78f352aba30aa41c76e08e6eb46",
    drafter_repo="mlx-community/Qwen3.6-35B-A3B-MTP-4bit",
    drafter_revision="0295b81421bf4d0fccca9a7c0fcfb1418dda3516",
    draft_tokens=2,
    block_size=3,
)


def resolve_native_mtp_pair(
    *,
    alias: str,
    target_repo: str,
    drafter_repo: str | None,
    draft_tokens: int,
) -> NativeMTPPair:
    """Return the immutable qualified pair or reject before loading weights."""

    pair = QWEN36_35B_4BIT
    if alias != "qwen3.6-35b-4bit" or target_repo != pair.target_repo:
        raise NativeMTPUnavailableError(
            "native MTP is currently qualified only for qwen3.6-35b-4bit"
        )
    if drafter_repo != pair.drafter_repo:
        raise NativeMTPUnavailableError(
            "native MTP requires the qwen3.6-35b-4bit alias's declared "
            f"sidecar ({pair.drafter_repo})"
        )
    if draft_tokens != pair.draft_tokens:
        raise NativeMTPUnavailableError(
            "native MTP for qwen3.6-35b-4bit requires "
            f"num_speculative_tokens={pair.draft_tokens}"
        )
    return pair


__all__ = [
    "NativeMTPPair",
    "NativeMTPUnavailableError",
    "QWEN36_35B_4BIT",
    "resolve_native_mtp_pair",
]
