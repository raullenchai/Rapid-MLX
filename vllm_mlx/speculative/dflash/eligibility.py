# SPDX-License-Identifier: Apache-2.0
"""DFlash eligibility checks — gate the feature behind validated combos.

This is the single chokepoint between user intent
(``--speculative-config '{"method":"dflash"}'`` on the CLI) and the runtime hook.
Failures here surface as actionable error
messages at server-start, never as silent regressions at request time.

Gates derived from qualification bench data (see issue #264):
  - Alias must declare ``supports_dflash=True`` (explicit opt-in)
  - Alias must NOT be ``is_moe=True`` (MoE acceptance floors at ~1.5)
  - A curated alias pins both a drafter repository and its expected runtime
    algorithm. Quantized pairs remain experimental unless that exact registry
    entry has passed the benchmark gate.
  - Drafter HF path must be reachable (no auth-gated repo without token)
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm_mlx.model_aliases import AliasProfile
from vllm_mlx.spec_decode.capability import looks_like_4bit


class DFlashUnavailable(RuntimeError):  # noqa: N818 — domain-specific error name
    """Raised when an alias fails a DFlash eligibility gate.

    The message is end-user-facing: it explains *which* gate failed and
    *what* the user can do (switch alias, change quantization, etc.).
    """


@dataclass(frozen=True)
class EligibilityReport:
    """Structured eligibility result. Used by ``rapid-mlx info <alias>``
    to render a per-gate status table without re-checking each gate."""

    alias: str | None
    supports_dflash: bool
    is_moe: bool
    is_4bit: bool
    has_drafter: bool
    recommendation: str
    warnings: tuple[str, ...]
    reasons: tuple[str, ...]  # all failing-gate reasons (empty if eligible)


def _looks_like_4bit(hf_path: str) -> bool:
    """Heuristic: detect 4-bit quantization from the HF repo name.

    mlx-community publishes quants as ``-4bit``, ``-mxfp4``, ``-nvfp4``
    suffixes/segments. Mirrors the contract test's detection so a CLI
    error and a unit-test guard share one rule.
    """
    return looks_like_4bit(hf_path)


def report(
    profile: AliasProfile,
    alias: str | None = None,
    *,
    explicit: bool = False,
    drafter_model: str | None = None,
) -> EligibilityReport:
    """Compute the eligibility report without raising. Used by ``info``
    to render gate status — ``check`` is the raise-on-failure variant.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    if not profile.supports_dflash and not explicit:
        reasons.append(
            "alias is not DFlash-enabled (set supports_dflash=true in "
            "aliases.json after benching to validate ≥1.3× speedup)"
        )
    if profile.is_moe:
        reasons.append(
            "alias is MoE (is_moe=true) — DFlash acceptance floors at "
            "~1.5 tokens/round on expert-routing churn; regression "
            "measured on Qwen3.6-35B-A3B"
        )
    is_4bit = _looks_like_4bit(profile.hf_path)
    curated_pair = (
        profile.supports_dflash
        and bool(profile.dflash_algorithm)
        and bool(profile.dflash_target_revision)
        and bool(profile.dflash_draft_revision)
        and (not is_4bit or profile.dflash_algorithm == "dflash2")
        and bool(drafter_model or profile.dflash_draft_model)
        and (drafter_model is None or drafter_model == profile.dflash_draft_model)
    )
    if is_4bit and not curated_pair:
        warnings.append(
            f"main model hf_path={profile.hf_path!r} is 4-bit quantized; "
            "this pair has not been performance-validated and may be slower"
        )
        if not explicit:
            reasons.append("4-bit DFlash requires explicit experimental opt-in")
    if explicit and not profile.supports_dflash:
        has_drafter = bool(drafter_model)
    else:
        has_drafter = bool(drafter_model or profile.dflash_draft_model)
    if not has_drafter:
        # Should be caught at JSON-load time by _coerce, but defend
        # against direct AliasProfile construction in tests/code.
        reasons.append("DFlash requires an explicit drafter model")
    if explicit and not curated_pair:
        warnings.append(
            "this target/drafter pair is experimental and has not been "
            "performance-validated by Rapid-MLX; it may provide no speedup "
            "or may be slower than autoregressive decoding"
        )
    return EligibilityReport(
        alias=alias,
        supports_dflash=profile.supports_dflash,
        is_moe=profile.is_moe,
        is_4bit=is_4bit,
        has_drafter=has_drafter,
        recommendation=(
            "incompatible"
            if profile.is_moe
            else ("verified" if curated_pair else "experimental")
        ),
        warnings=tuple(warnings),
        reasons=tuple(reasons),
    )


def eligible_aliases() -> list[str]:
    """Return alias names whose AliasProfile currently passes every
    DFlash gate. Computed from the live ``aliases.json`` registry so
    error messages don't go stale as more aliases are validated.

    Kept tolerant: any import or registry error returns an empty list
    rather than raising, since this is only used to enrich error text.
    """
    try:
        from vllm_mlx.model_aliases import list_profiles

        return sorted(
            name
            for name, profile in list_profiles().items()
            if not report(profile).reasons
        )
    except Exception:  # noqa: BLE001 — diagnostic helper, never fatal
        return []


def is_registry_verified_pair(
    main_model_repo: str,
    main_model_revision: str | None,
    drafter_repo: str,
    drafter_revision: str | None,
    expected_algorithm: str | None,
) -> bool:
    """Return whether the exact runtime tuple is registry-qualified.

    The server uses this instead of trusting a caller-supplied bypass boolean.
    Every component must match one profile whose normal eligibility report is
    verified; arbitrary programmatic callers therefore retain the explicit
    experimental opt-in requirement.
    """
    if not all(
        (
            main_model_repo,
            main_model_revision,
            drafter_repo,
            drafter_revision,
            expected_algorithm,
        )
    ):
        return False
    try:
        from vllm_mlx.model_aliases import list_profiles

        for alias, profile in list_profiles().items():
            if (
                profile.hf_path == main_model_repo
                and profile.dflash_target_revision == main_model_revision
                and profile.dflash_draft_model == drafter_repo
                and profile.dflash_draft_revision == drafter_revision
                and profile.dflash_algorithm == expected_algorithm
            ):
                assessment = report(profile, alias=alias)
                if not assessment.reasons and assessment.recommendation == "verified":
                    return True
    except Exception:  # noqa: BLE001 — qualification lookup must fail closed
        return False
    return False


def check(
    profile: AliasProfile,
    alias: str | None = None,
    *,
    explicit: bool = False,
    drafter_model: str | None = None,
) -> None:
    """Raise ``DFlashUnavailable`` with an actionable message if any
    eligibility gate fails. Returns ``None`` on success."""
    r = report(profile, alias=alias, explicit=explicit, drafter_model=drafter_model)
    if not r.reasons:
        return
    header = f"DFlash unavailable for {alias!r}" if alias else "DFlash unavailable"
    bullet = "\n  - ".join(r.reasons)
    eligible = eligible_aliases()
    if eligible:
        suffix = (
            f"Eligible aliases today: {', '.join(eligible)}. Run "
            "`rapid-mlx info <alias>` to inspect per-alias DFlash status."
        )
    else:
        suffix = (
            "No aliases currently pass every DFlash gate. Run "
            "`rapid-mlx info <alias>` to inspect per-alias DFlash status."
        )
    raise DFlashUnavailable(f"{header}:\n  - {bullet}\n\n{suffix}")


def have_runtime() -> bool:
    """Return True iff mlx-vlm 0.5.0+ DFlash hooks are importable.

    Kept fast (no actual import on success path) so it's cheap to call
    in CLI startup and in ``rapid-mlx info`` rendering. Result is
    cached by ``importlib`` after first call.
    """
    try:
        # Probe the specific symbol DFlash needs — a partial install
        # (pre-0.5.0 mlx-vlm in our deps) would have `mlx_vlm` but no
        # `speculative.drafters.load_drafter`.
        from importlib.util import find_spec

        spec = find_spec("mlx_vlm.speculative.drafters")
        return spec is not None
    except (ImportError, AttributeError):
        return False
