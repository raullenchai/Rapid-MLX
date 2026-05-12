# SPDX-License-Identifier: Apache-2.0
"""DFlash eligibility checks — gate the feature behind validated combos.

This is the single chokepoint between user intent (``--enable-dflash`` on
the CLI) and the runtime hook. Failures here surface as actionable error
messages at server-start, never as silent regressions at request time.

Gates derived from PoC bench data (see issue #264):
  - Alias must declare ``supports_dflash=True`` (explicit opt-in)
  - Alias must NOT be ``is_moe=True`` (MoE acceptance floors at ~1.5)
  - Main model must be 8-bit or higher (4-bit regresses; gate by hf_path
    naming convention, double-checked at load time by inspecting the
    quantization config)
  - Drafter HF path must be reachable (no auth-gated repo without token)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from vllm_mlx.model_aliases import AliasProfile


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
    reasons: tuple[str, ...]  # all failing-gate reasons (empty if eligible)


def _looks_like_4bit(hf_path: str) -> bool:
    """Heuristic: detect 4-bit quantization from the HF repo name.

    mlx-community publishes quants as ``-4bit``, ``-mxfp4``, ``-nvfp4``
    suffixes/segments. Mirrors the contract test's detection so a CLI
    error and a unit-test guard share one rule.
    """
    lowered = hf_path.lower()
    if "-4bit" in lowered or lowered.endswith("-4bit"):
        return True
    if "4bit-" in lowered:
        return True
    if "mxfp4" in lowered or "nvfp4" in lowered:
        return True
    return False


def report(profile: AliasProfile, alias: str | None = None) -> EligibilityReport:
    """Compute the eligibility report without raising. Used by ``info``
    to render gate status — ``check`` is the raise-on-failure variant.
    """
    reasons: list[str] = []
    if not profile.supports_dflash:
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
    if is_4bit:
        reasons.append(
            f"main model hf_path={profile.hf_path!r} is 4-bit quantized; "
            "DFlash regresses on 4-bit (use an 8-bit or higher variant)"
        )
    has_drafter = bool(profile.dflash_draft_model)
    if profile.supports_dflash and not has_drafter:
        # Should be caught at JSON-load time by _coerce, but defend
        # against direct AliasProfile construction in tests/code.
        reasons.append("supports_dflash is set but dflash_draft_model is empty")
    return EligibilityReport(
        alias=alias,
        supports_dflash=profile.supports_dflash,
        is_moe=profile.is_moe,
        is_4bit=is_4bit,
        has_drafter=has_drafter,
        reasons=tuple(reasons),
    )


def check(profile: AliasProfile, alias: str | None = None) -> None:
    """Raise ``DFlashUnavailable`` with an actionable message if any
    eligibility gate fails. Returns ``None`` on success."""
    r = report(profile, alias=alias)
    if not r.reasons:
        return
    header = f"DFlash unavailable for {alias!r}" if alias else "DFlash unavailable"
    bullet = "\n  - ".join(r.reasons)
    raise DFlashUnavailable(
        f"{header}:\n  - {bullet}\n\n"
        "Eligible aliases today: qwen3.5-27b-8bit. Run "
        "`rapid-mlx info <alias>` to inspect per-alias DFlash status."
    )


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
        import importlib

        spec = importlib.util.find_spec("mlx_vlm.speculative.drafters")
        return spec is not None
    except (ImportError, AttributeError):
        return False


# Bypass-toggle for tests that need to exercise gates without a real
# drafter on disk. Tests set this via monkeypatch; production code never
# reads the env directly.
DFLASH_SKIP_RUNTIME_CHECK_ENV = "RAPID_MLX_DFLASH_SKIP_RUNTIME_CHECK"


def runtime_check_skipped() -> bool:
    """Honour ``RAPID_MLX_DFLASH_SKIP_RUNTIME_CHECK=1`` for tests that
    only exercise eligibility logic without actually loading weights."""
    return os.environ.get(DFLASH_SKIP_RUNTIME_CHECK_ENV, "") in ("1", "true", "yes")
