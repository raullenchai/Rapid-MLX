# SPDX-License-Identifier: Apache-2.0
"""DDTree eligibility checks.

DDTree support is narrower than DFlash support because the verifier is
model-family specific. A model may have matching DFlash draft weights and
still be unsafe for DDTree until the target-side tree verifier has been
bench-validated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from vllm_mlx.model_aliases import AliasProfile
from vllm_mlx.speculative.dflash.eligibility import _looks_like_4bit

logger = logging.getLogger(__name__)
_runtime_probe_error: str | None = None


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
    recommendation: str
    warnings: tuple[str, ...]
    reasons: tuple[str, ...]


def report(
    profile: AliasProfile,
    alias: str | None = None,
    *,
    explicit: bool = False,
    drafter_model: str | None = None,
    speculative_tokens: int | None = None,
    tree_budget: int | None = None,
) -> EligibilityReport:
    reasons: list[str] = []
    warnings: list[str] = []
    if not profile.supports_ddtree and not explicit:
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
        warnings.append(
            f"main model hf_path={profile.hf_path!r} is 4-bit quantized; "
            "DDTree on this quantization is not performance-validated"
        )
        if not explicit:
            reasons.append("4-bit DDTree requires explicit experimental opt-in")
    experimental_explicit = explicit and not profile.supports_ddtree
    has_drafter = (
        bool(drafter_model)
        if experimental_explicit
        else bool(drafter_model or profile.ddtree_draft_model)
    )
    if not has_drafter:
        reasons.append("DDTree requires an explicit drafter model")
    effective_tokens = (
        speculative_tokens
        if experimental_explicit or speculative_tokens is not None
        else profile.ddtree_speculative_tokens
    )
    has_speculative_tokens = (
        isinstance(effective_tokens, int)
        and not isinstance(effective_tokens, bool)
        and effective_tokens > 0
    )
    if not has_speculative_tokens:
        reasons.append("DDTree requires num_speculative_tokens")
    effective_budget = (
        tree_budget
        if experimental_explicit or tree_budget is not None
        else profile.ddtree_tree_budget
    )
    has_tree_budget = (
        isinstance(effective_budget, int)
        and not isinstance(effective_budget, bool)
        and effective_budget > 0
    )
    if not has_tree_budget:
        reasons.append("DDTree requires tree_budget")
    curated_pair = (
        not is_4bit
        and profile.supports_ddtree
        and has_drafter
        and has_speculative_tokens
        and has_tree_budget
        and (drafter_model is None or drafter_model == profile.ddtree_draft_model)
        and effective_tokens == profile.ddtree_speculative_tokens
        and effective_budget == profile.ddtree_tree_budget
    )
    if explicit and not curated_pair:
        warnings.append(
            "this target/drafter pair is experimental and has not been "
            "performance-validated by Rapid-MLX; it may be slower"
        )
    return EligibilityReport(
        alias=alias,
        supports_ddtree=profile.supports_ddtree,
        is_moe=profile.is_moe,
        is_4bit=is_4bit,
        has_drafter=has_drafter,
        has_speculative_tokens=has_speculative_tokens,
        has_tree_budget=has_tree_budget,
        recommendation=(
            "incompatible"
            if profile.is_moe
            else ("verified" if curated_pair else "experimental")
        ),
        warnings=tuple(warnings),
        reasons=tuple(reasons),
    )


def eligible_aliases() -> list[str]:
    from vllm_mlx.model_aliases import list_profiles

    return sorted(
        name for name, profile in list_profiles().items() if not report(profile).reasons
    )


def check(
    profile: AliasProfile,
    alias: str | None = None,
    *,
    explicit: bool = False,
    drafter_model: str | None = None,
    speculative_tokens: int | None = None,
    tree_budget: int | None = None,
) -> None:
    r = report(
        profile,
        alias=alias,
        explicit=explicit,
        drafter_model=drafter_model,
        speculative_tokens=speculative_tokens,
        tree_budget=tree_budget,
    )
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
    global _runtime_probe_error
    try:
        from dtree_mlx.api import DFlashGenerator  # noqa: F401

        _runtime_probe_error = None
        return True
    except Exception as exc:  # noqa: BLE001
        _runtime_probe_error = f"{type(exc).__name__}: {exc}"[:240]
        logger.debug("DDTree runtime probe failed: %s", _runtime_probe_error)
        return False


def runtime_probe_error() -> str | None:
    return _runtime_probe_error
