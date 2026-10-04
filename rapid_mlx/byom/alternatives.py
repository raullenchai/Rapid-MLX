# SPDX-License-Identifier: Apache-2.0
"""Suggest a runnable alternative when the BYOM preflight refuses a model.

Two sources, in order:

1. **An MLX build of the same model** (GGUF-only Hub repos). The refused repo
   must declare itself a QUANTIZATION of exactly one base repo
   (``base_model:quantized:<base>``; finetunes and merges are different
   models). One Hub search lists repos tagged ``mlx`` that are quantizations of
   that same base. A candidate is shown only when its metadata matches: same
   ``model_type`` as the base, parameter count within 5 %, an MLX (not AWQ /
   GPTQ / …) quantization, an architecture this install positively supports,
   and a comfortable fit in this Mac's memory. An adult-tagged candidate
   (``not-for-all-audiences`` / ``nsfw``) is shown only when the user's own
   source repo carries such a tag: we answer the user's request for the model
   they chose, but never introduce adult content they did not ask for. It is
   presented as an unreviewed third-party build with its publisher, size and
   license; nothing is ever substituted automatically, and no candidate is
   better than an unverified one.
2. **A catalog model of similar size** from the same recommendation policy the
   installer and Desktop use, restricted to tiers this Mac's memory supports.

All Hub work shares ONE short time budget, so a refusal is never held up for
long, and every failure simply yields fewer suggestions; this module never
raises into the CLI.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from rapid_mlx.byom import preflight as pf

_PARAM_TOLERANCE = 0.05
_SEARCH_LIMIT = 20
_MAX_CANDIDATES = 2
_PREFERRED_BITS = (4, 8, 6, 5, 3, 2)
_BUDGET_SECONDS = 8.0
# Hugging Face adult-content tags. A candidate carrying one is suggested only
# when the user's source repo carries one too (a build of the model the user
# chose); Rapid-MLX never curates or promotes such repos on its own.
_ADULT_TAGS = frozenset({"not-for-all-audiences", "nsfw"})


def is_adult_tagged(tags: Any) -> bool:
    return bool(_ADULT_TAGS & {str(tag).lower() for tag in tags})


@dataclass(frozen=True)
class Candidate:
    repo_id: str
    bits: int
    params: int
    license: str | None
    downloads: int

    @property
    def approx_bytes(self) -> int:
        # Affine MLX quantization keeps a scale and bias per group of 64.
        return int(self.params * (self.bits + 0.5) / 8)

    @property
    def publisher(self) -> str:
        return self.repo_id.split("/", 1)[0]


class _Budget:
    """One deadline shared by every Hub call made for a single suggestion."""

    def __init__(self, seconds: float) -> None:
        self._deadline = time.monotonic() + seconds

    def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        from rapid_mlx._download_gate import call_with_deadline

        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("suggestion budget spent")
        return call_with_deadline(fn, remaining, *args, **kwargs)


def _base_model_info(base: str, budget: _Budget) -> Any | None:
    try:
        from huggingface_hub import model_info

        return budget.call(model_info, base)
    except Exception:
        return None


def _expected_identity(
    inspection: pf.Inspection, budget: _Budget
) -> tuple[str, str, int] | None:
    """``(base repo, model_type, parameter count)`` a candidate must match."""
    if len(inspection.quantized_from) != 1 or inspection.params is None:
        # No declared base is unknown provenance; several has no single
        # identity an MLX build could be matched against.
        return None
    base = inspection.quantized_from[0]
    info = _base_model_info(base, budget)
    config = getattr(info, "config", None) if info is not None else None
    model_type = config.get("model_type") if isinstance(config, dict) else None
    if not isinstance(model_type, str) or not model_type:
        return None
    return base, model_type.lower(), inspection.params


def _list_mlx_builds(base: str, budget: _Budget) -> list[Any]:
    try:
        from huggingface_hub import HfApi

        def _search() -> list[Any]:
            return list(
                HfApi().list_models(
                    filter=["mlx", f"base_model:quantized:{base}"],
                    sort="downloads",
                    limit=_SEARCH_LIMIT,
                    expand=["safetensors", "cardData", "downloads", "config", "tags"],
                )
            )

        found: list[Any] = budget.call(_search)
        return found
    except Exception:
        return []


def _mlx_bits(config: dict[str, Any]) -> int | None:
    """Bits of an MLX quantization; ``None`` for anything else (AWQ, GPTQ…)."""
    for key in ("quantization", "quantization_config"):
        block = config.get(key)
        if isinstance(block, dict) and "quant_method" not in block:
            bits = block.get("bits")
            if isinstance(bits, int) and not isinstance(bits, bool):
                return bits
    return None


def _verified_candidate(
    info: Any,
    *,
    source: str,
    model_type: str,
    params: int,
    supported: frozenset[str] | None,
    ram_bytes: int | None,
    source_adult: bool = False,
) -> Candidate | None:
    repo_id = getattr(info, "id", None)
    config = getattr(info, "config", None)
    if (
        not isinstance(repo_id, str)
        or repo_id == source
        or not isinstance(config, dict)
    ):
        return None
    tags = pf.hub_tags(info)
    if "mlx" not in tags or (is_adult_tagged(tags) and not source_adult):
        return None
    candidate_type = config.get("model_type")
    if not isinstance(candidate_type, str) or candidate_type.lower() != model_type:
        return None
    bits = _mlx_bits(config)
    total = pf._hub_params(info)
    if bits is None or total is None:
        return None
    if abs(total - params) > params * _PARAM_TOLERANCE:
        return None
    if pf.architecture_supported(config, supported) is not True:
        return None
    candidate = Candidate(
        repo_id=repo_id,
        bits=bits,
        params=total,
        license=pf.card_license(getattr(info, "card_data", None)),
        downloads=int(getattr(info, "downloads", 0) or 0),
    )
    if not ram_bytes:
        return None
    served = pf.Inspection(
        ref=repo_id, is_local=False, files=(), weight_bytes=candidate.approx_bytes
    )
    verdict = pf.evaluate(
        served, refuse_oversize=True, supported=supported, ram_bytes=ram_bytes
    )
    # Only a comfortable fit: a suggestion must not trade one failure for
    # another (swap, OOM) on this Mac.
    return candidate if verdict.fit == "yes" else None


def _rank(candidate: Candidate) -> tuple[int, int]:
    preference = (
        _PREFERRED_BITS.index(candidate.bits)
        if candidate.bits in _PREFERRED_BITS
        else len(_PREFERRED_BITS)
    )
    return preference, -candidate.downloads


def find_mlx_builds(
    inspection: pf.Inspection,
    *,
    supported: frozenset[str] | None,
    ram_bytes: int | None,
    budget_seconds: float = _BUDGET_SECONDS,
) -> list[Candidate]:
    """Matching MLX builds of a refused GGUF repo, best first (may be empty)."""
    budget = _Budget(budget_seconds)
    identity = _expected_identity(inspection, budget)
    if identity is None:
        return []
    base, model_type, params = identity
    found = [
        candidate
        for candidate in (
            _verified_candidate(
                info,
                source=inspection.ref,
                source_adult=is_adult_tagged(inspection.tags),
                model_type=model_type,
                params=params,
                supported=supported,
                ram_bytes=ram_bytes,
            )
            for info in _list_mlx_builds(base, budget)
        )
        if candidate is not None
    ]
    return sorted(found, key=_rank)[:_MAX_CANDIDATES]


def similar_catalog_model(
    target_bytes: int | None, ram_bytes: int | None
) -> str | None:
    """A curated alias for this Mac, nearest in size to ``target_bytes``."""
    if not ram_bytes:
        return None
    try:
        from rapid_mlx.recommendations import load_recommendation_tiers

        tiers = load_recommendation_tiers(validate_catalog=False)
    except Exception:
        return None
    ram_gb = ram_bytes / float(1 << 30)
    fitting = [tier for tier in tiers if tier.floor_gb <= ram_gb]
    if not fitting:
        return None
    if target_bytes:
        target_gb = max(target_bytes / float(1 << 30), 0.1)
        picks = [pick for tier in fitting for pick in tier.picks]
        best = min(
            picks,
            key=lambda pick: (abs(math.log(pick.footprint_gb / target_gb)), pick.alias),
        )
        return best.alias
    # No size to match: this Mac's own "smart" recommendation.
    return fitting[-1].picks[0].alias


def render_candidates(candidates: list[Candidate], command: str) -> list[str]:
    lines = [
        "  MLX builds of the same base model (third-party, not reviewed by Rapid-MLX;",
        "  matched by base model, architecture and size):",
    ]
    for candidate in candidates:
        license_ = candidate.license or "license unknown"
        lines.append(
            f"    {candidate.repo_id} · {candidate.bits}-bit · "
            f"~{pf._gb(candidate.approx_bytes)} · {license_}"
        )
    lines.append(
        f"  To try one, check its model card first, then: rapid-mlx {command} "
        f"{candidates[0].repo_id}"
    )
    return lines


def render_catalog(alias: str, command: str, *, sized: bool) -> list[str]:
    header = (
        "A catalog model of similar size that fits your Mac:"
        if sized
        else "A catalog model that runs well on your Mac:"
    )
    return [f"  {header}", f"    rapid-mlx {command} {alias}"]


def _target_bytes(inspection: pf.Inspection, verdict: pf.Verdict) -> int | None:
    if verdict.failure == pf.INSUFFICIENT_MEMORY:
        # The refused model is too big by definition; aim for this Mac's
        # own recommendation instead of the nearest (still too big) size.
        return None
    if inspection.weight_bytes:
        return inspection.weight_bytes
    if inspection.params:
        return int(inspection.params * 0.55)  # ~4-bit MLX footprint
    return None


def suggest(
    inspection: pf.Inspection,
    verdict: pf.Verdict,
    *,
    command: str,
    supported: frozenset[str] | None,
    ram_bytes: int | None,
    targets: list[str] | None = None,
) -> tuple[list[str], bool]:
    """Extra failure lines naming something that will run, and whether they
    name an MLX build of the same model. Never raises.

    ``targets``, when given, receives the model references the lines tell the
    user to try (repo ids or a catalog alias), for local funnel telemetry.
    """
    try:
        if (
            verdict.failure == pf.UNSUPPORTED_FORMAT
            and verdict.format_label == "GGUF"
            and not inspection.is_local
        ):
            builds = find_mlx_builds(
                inspection, supported=supported, ram_bytes=ram_bytes
            )
            if builds:
                lines = render_candidates(builds, command)
                if targets is not None:
                    targets.extend(candidate.repo_id for candidate in builds)
                return lines, True
        target = _target_bytes(inspection, verdict)
        alias = similar_catalog_model(target, ram_bytes)
    except Exception:
        return [], False
    if alias is None:
        return [], False
    lines = render_catalog(alias, command, sized=target is not None)
    if targets is not None:
        targets.append(alias)
    return lines, False
