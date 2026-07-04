# SPDX-License-Identifier: Apache-2.0
"""DDTree eligibility checks.

DDTree support is narrower than DFlash support because the verifier is
model-family specific. A model may have matching DFlash draft weights and
still be unsafe for DDTree until the target-side tree verifier has been
bench-validated.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm_mlx.model_aliases import AliasProfile
from vllm_mlx.speculative.dflash.eligibility import _looks_like_4bit


class DDTreeUnavailable(RuntimeError):  # noqa: N818
    """Raised when an alias fails a DDTree eligibility gate."""


@dataclass(frozen=True)
class EligibilityReport:
    alias: str | None
    supports_ddtree: bool
    is_moe: bool
    is_4bit: bool
    has_drafter: bool
    has_speculative_tokens: bool
    has_tree_budget: bool
    reasons: tuple[str, ...]


def report(profile: AliasProfile, alias: str | None = None) -> EligibilityReport:
    reasons: list[str] = []
    if not profile.supports_ddtree:
        reasons.append(
            "alias is not DDTree-enabled (set supports_ddtree=true only after "
            "benching this exact target/drafter pair)"
        )
    if profile.is_moe:
        reasons.append(
            "alias is MoE (is_moe=true) — DDTree verifier support is only "
            "validated for dense Qwen3.5/Qwen3-family targets in the MVP"
        )
    is_4bit = _looks_like_4bit(profile.hf_path)
    if is_4bit:
        reasons.append(
            f"main model hf_path={profile.hf_path!r} is 4-bit quantized; "
            "DDTree on 4-bit is not validated yet"
        )
    has_drafter = bool(profile.ddtree_draft_model)
    if profile.supports_ddtree and not has_drafter:
        reasons.append("supports_ddtree is set but ddtree_draft_model is empty")
    has_speculative_tokens = profile.ddtree_speculative_tokens is not None
    if profile.supports_ddtree and not has_speculative_tokens:
        reasons.append("supports_ddtree is set but ddtree_speculative_tokens is empty")
    has_tree_budget = profile.ddtree_tree_budget is not None
    if profile.supports_ddtree and not has_tree_budget:
        reasons.append("supports_ddtree is set but ddtree_tree_budget is empty")
    return EligibilityReport(
        alias=alias,
        supports_ddtree=profile.supports_ddtree,
        is_moe=profile.is_moe,
        is_4bit=is_4bit,
        has_drafter=has_drafter,
        has_speculative_tokens=has_speculative_tokens,
        has_tree_budget=has_tree_budget,
        reasons=tuple(reasons),
    )


def eligible_aliases() -> list[str]:
    try:
        from vllm_mlx.model_aliases import list_profiles

        return sorted(
            name
            for name, profile in list_profiles().items()
            if not report(profile).reasons
        )
    except Exception:  # noqa: BLE001
        return []


def check(profile: AliasProfile, alias: str | None = None) -> None:
    r = report(profile, alias=alias)
    if not r.reasons:
        return
    header = f"DDTree unavailable for {alias!r}" if alias else "DDTree unavailable"
    bullet = "\n  - ".join(r.reasons)
    eligible = eligible_aliases()
    if eligible:
        suffix = (
            f"Eligible aliases today: {', '.join(eligible)}. Run "
            "`rapid-mlx info <alias>` to inspect per-alias DDTree status."
        )
    else:
        suffix = (
            "No aliases currently pass every DDTree gate. Run "
            "`rapid-mlx info <alias>` to inspect per-alias DDTree status."
        )
    raise DDTreeUnavailable(f"{header}:\n  - {bullet}\n\n{suffix}")


def have_runtime() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("dtree_mlx.api") is not None
    except (ImportError, AttributeError):
        return False
