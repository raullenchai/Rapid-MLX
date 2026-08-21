# SPDX-License-Identifier: Apache-2.0
"""Capability and recommendation tiers for speculative decoding.

Registry flags are recommendation evidence. They are not permission checks for
an operator-explicit target/drafter pair; runtime preflight owns compatibility.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from vllm_mlx.model_profile import ModelProfile


def looks_like_4bit(hf_path: str) -> bool:
    """Detect delimiter-bounded 4-bit quantization tags in repo names."""
    return bool(
        re.search(
            r"(?<![a-z0-9])(?:4[-_]?bit|mxfp4|nvfp4)(?![a-z0-9])",
            hf_path.lower(),
        )
    )


def _is_experimental_quantization(profile: ModelProfile) -> bool:
    return looks_like_4bit(profile.hf_path)


@dataclass(frozen=True)
class SpecCapability:
    method: str
    capable: bool | None
    recommendation: str
    explicit_opt_in: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def assess_method(profile: ModelProfile, method: str) -> SpecCapability:
    """Statically assess one alias; runtime-only facts remain unknown."""
    if profile.modality != "text":
        return SpecCapability(
            method, False, "incompatible", False, ("non-text serving lane",)
        )
    if method == "suffix":
        if profile.is_hybrid:
            return SpecCapability(
                method,
                False,
                "incompatible",
                False,
                ("target verifier does not support lossless rollback",),
            )
        tier = profile.suffix_decoding_tier
        verified = profile.supports_spec_decode and tier == "verified"
        return SpecCapability(
            method,
            True if verified else None,
            "verified" if verified else "experimental",
            not verified,
            warnings=()
            if verified
            else ("runtime validates lossless rollback support",),
        )
    if method == "dflash":
        if profile.is_moe:
            return SpecCapability(
                method, False, "incompatible", False, ("MoE verifier unsupported",)
            )
        verified = (
            profile.supports_dflash
            and bool(profile.dflash_draft_model)
            and not _is_experimental_quantization(profile)
        )
        return SpecCapability(
            method,
            True if verified else None,
            "verified" if verified else "experimental",
            not verified,
            warnings=("requires a structurally compatible drafter",),
        )
    if method == "ddtree":
        if profile.is_moe:
            return SpecCapability(
                method, False, "incompatible", False, ("MoE tree verifier unsupported",)
            )
        verified = (
            profile.supports_ddtree
            and bool(profile.ddtree_draft_model)
            and isinstance(profile.ddtree_speculative_tokens, int)
            and not isinstance(profile.ddtree_speculative_tokens, bool)
            and profile.ddtree_speculative_tokens > 0
            and isinstance(profile.ddtree_tree_budget, int)
            and not isinstance(profile.ddtree_tree_budget, bool)
            and profile.ddtree_tree_budget > 0
            and not _is_experimental_quantization(profile)
        )
        return SpecCapability(
            method,
            True if verified else None,
            "verified" if verified else "experimental",
            not verified,
            warnings=("requires drafter, speculative-token, and tree-budget metadata",),
        )
    if method == "mtp":
        verified = bool(profile.mtp_draft_model) and not _is_experimental_quantization(
            profile
        )
        return SpecCapability(
            method,
            None,
            "verified" if verified else "experimental",
            not verified,
            warnings=("runtime validates architecture and sidecar metadata",),
        )
    if method == "dspark":
        return SpecCapability(
            method,
            None,
            "experimental",
            True,
            warnings=("runtime validates embedded DSpark metadata",),
        )
    raise ValueError(f"unknown speculative method: {method}")


REGISTERED_METHODS = ("suffix", "mtp", "dflash", "ddtree", "dspark")
