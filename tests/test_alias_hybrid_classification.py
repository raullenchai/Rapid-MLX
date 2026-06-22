# SPDX-License-Identifier: Apache-2.0
"""r6-A R6-C1 regression guard — dense Qwen3.5 / Qwen3.6 must NOT carry
``is_hybrid=True`` in the alias registry.

Background (Sasha R1/R2 dogfood reports):

  ``rapid-mlx serve qwen3.5-4b-4bit --port 9XYZ`` followed by ANY
  ``/v1/responses`` (or ``/v1/chat/completions``) request wedges on
  ``[metal::malloc] Resource limit (499000) exceeded`` at every
  generation step. ``--no-hybrid`` is the only workaround.

Root cause: ``aliases.json`` shipped ``is_hybrid=true`` for the dense
Qwen3.5 family (4B/9B/27B non-A3B) AND the runtime ArraysCache probe
in ``model_auto_config.enrich_model_config`` one-way-promoted
``is_hybrid`` to True for any model whose ``make_cache()`` returns
linear-attention layers (which dense ``model_type=qwen3_5`` weights
do). The hybrid scheduler's allocation pattern is incompatible with
the dense GatedDeltaNet cache layout at these sizes, surfacing as the
metal::malloc wedge.

Three-pronged fix, asserted here:

  1. ``aliases.json`` flips dense Qwen3.5 / Qwen3.6 to
     ``is_hybrid=false`` (and the MoE A3B/A10B siblings stay
     ``is_hybrid=true``).
  2. ``AliasProfile.is_hybrid_explicit=True`` pins the JSON-declared
     value so the runtime probe respects it.
  3. The auto-derivation regex in ``model_auto_config._MODEL_PATTERNS``
     requires an MoE marker (A3B / A10B / MoE) before stamping a
     Qwen3.5 / Qwen3.6 HF path as hybrid.
"""

from __future__ import annotations

import pytest

from vllm_mlx.model_aliases import list_profiles
from vllm_mlx.model_auto_config import detect_model_config

# Dense Qwen3.5 — these are the variants Sasha's repro wedged on.
DENSE_QWEN35_ALIASES = (
    "qwen3.5-4b-4bit",
    "qwen3.5-4b-8bit",
    "qwen3.5-9b-4bit",
    "qwen3.5-9b-8bit",
    "qwen3.5-27b-4bit",
    "qwen3.5-27b-8bit",
)
# Dense Qwen3.6 — same architecture story, same wedge risk.
DENSE_QWEN36_ALIASES = (
    "qwen3.6-27b-4bit",
    "qwen3.6-27b-8bit",
    "qwen3.6-27b-ud",
)
# MoE — A3B / A10B variants stay legitimately hybrid (the prefix-boundary
# snapshot + throttle path was originally written for these).
MOE_QWEN35_36_ALIASES = (
    "qwen3.5-35b-4bit",
    "qwen3.5-35b-8bit",
    "qwen3.5-122b-mxfp4",
    "qwen3.5-122b-8bit",
    "qwen3.6-35b-4bit",
    "qwen3.6-35b-6bit",
    "qwen3.6-35b-8bit",
    "qwen3.6-35b-ud",
    "qwen3.6-35b-dwq",
)


@pytest.mark.parametrize("alias", DENSE_QWEN35_ALIASES + DENSE_QWEN36_ALIASES)
def test_dense_qwen35_36_aliases_are_not_hybrid(alias: str) -> None:
    """Dense Qwen3.5 / Qwen3.6 aliases must NOT carry ``is_hybrid=True``.

    Wedge signature: ``metal::malloc Resource limit (499000)`` on every
    generation step. Caught by Sasha R1/R2 on ``qwen3.5-4b-4bit``;
    this guard extends the rule to every dense sibling so a future
    ``aliases.json`` edit can't silently re-introduce the wedge.
    """
    profiles = list_profiles()
    p = profiles[alias]
    assert not p.is_hybrid, (
        f"{alias}: is_hybrid=True — this is the r6-A R6-C1 wedge signature. "
        f"Dense Qwen3.5/3.6 variants must declare is_hybrid=false "
        f"(and is_hybrid_explicit=true so the runtime ArraysCache probe "
        f"respects the declaration)."
    )


@pytest.mark.parametrize("alias", DENSE_QWEN35_ALIASES + DENSE_QWEN36_ALIASES)
def test_dense_qwen35_36_aliases_pin_explicit_flag(alias: str) -> None:
    """``is_hybrid_explicit=True`` MUST be set on every dense Qwen3.5/3.6
    alias.

    Without the explicit pin, ``enrich_model_config``'s runtime probe
    re-flips ``is_hybrid`` to True at boot because the underlying
    weights (model_type=qwen3_5) carry linear-attention layers in
    ``make_cache()`` — that re-promotion is the silent regression
    path the JSON edit alone cannot close.
    """
    profiles = list_profiles()
    p = profiles[alias]
    assert p.is_hybrid_explicit, (
        f"{alias}: is_hybrid_explicit=False — the JSON-declared "
        f"is_hybrid=false would be silently re-promoted to True at boot "
        f"by enrich_model_config's ArraysCache probe. Set "
        f"is_hybrid_explicit=true to pin the declaration."
    )


@pytest.mark.parametrize("alias", MOE_QWEN35_36_ALIASES)
def test_moe_qwen35_36_aliases_remain_hybrid(alias: str) -> None:
    """MoE A3B / A10B variants stay ``is_hybrid=True`` — the
    prefix-boundary snapshot + throttle path was originally written for
    them and they don't share the dense wedge."""
    profiles = list_profiles()
    p = profiles[alias]
    assert p.is_hybrid, (
        f"{alias}: is_hybrid=False — MoE Qwen3.5/3.6 variants must keep "
        f"is_hybrid=true; the hybrid scheduler path is what they need."
    )
    assert p.is_moe, (
        f"{alias}: is_moe=False on an A3B/A10B alias — recheck the "
        f"hf_path; this alias may be mistakenly tagged as non-MoE."
    )


# =============================================================================
# Auto-derivation regex — bare Qwen3.5 / Qwen3.6 HF paths must fall
# through to the non-hybrid generic Qwen3 fallback unless an MoE marker
# is present.
# =============================================================================


@pytest.mark.parametrize(
    "hf_path",
    [
        "mlx-community/Qwen3.5-4B-MLX-4bit",
        "mlx-community/Qwen3.5-9B-4bit",
        "mlx-community/Qwen3.5-27B-4bit",
        "mlx-community/Qwen3.6-27B-4bit",
        # Synthetic path (no alias) — exercises the regex fallback rather
        # than the alias-profile lookup.
        "user/Qwen3.5-7B-Dense-Repack",
        "user/Qwen3.6-13B-Dense-Custom",
    ],
)
def test_bare_qwen35_36_path_resolves_to_non_hybrid(hf_path: str) -> None:
    """An HF path matching ``qwen3.5`` / ``qwen3.6`` without an MoE
    marker must NOT pick up ``is_hybrid=True`` from the auto-derivation
    regex. Pre-fix, the bare ``qwen3\\.5`` regex stamped every match as
    hybrid; this is the heuristic-side counterpart to the JSON fix above.
    """
    cfg = detect_model_config(hf_path)
    assert cfg is not None, (
        f"{hf_path}: detect_model_config returned None — the qwen3.5/3.6 "
        f"regex should still match the bare path (it just resolves to a "
        f"non-hybrid profile now)."
    )
    assert not cfg.is_hybrid, (
        f"{hf_path}: is_hybrid=True — the bare qwen3.5/3.6 regex stamped "
        f"a dense HF path as hybrid. Tighten the regex to require an "
        f"MoE marker (A3B / A10B / MoE)."
    )


@pytest.mark.parametrize(
    "hf_path",
    [
        "mlx-community/Qwen3.5-35B-A3B-4bit",
        "mlx-community/Qwen3.5-122B-A10B-8bit",
        "nightmedia/Qwen3.5-122B-A10B-Text-mxfp4-mlx",
        "mlx-community/Qwen3.6-35B-A3B-4bit",
        # Generic ``moe`` marker — covers future MoE variants that might
        # not use the A3B / A10B naming.
        "user/Qwen3.5-MoE-Custom",
    ],
)
def test_moe_marker_qwen35_36_path_resolves_to_hybrid(hf_path: str) -> None:
    """A path carrying an MoE marker (A3B / A10B / MoE) under the
    Qwen3.5 / Qwen3.6 family stays on the hybrid path — that's what the
    A3B / A10B variants need (their architecture pairs GatedDeltaNet
    with sparse experts and the scheduler path is benched for them).
    """
    cfg = detect_model_config(hf_path)
    assert cfg is not None, (
        f"{hf_path}: detect_model_config returned None — the MoE-marker "
        f"regex did not fire."
    )
    assert cfg.is_hybrid, (
        f"{hf_path}: is_hybrid=False — MoE-marker paths must resolve to "
        f"the hybrid branch of the qwen3.5/3.6 regex."
    )
    assert not cfg.supports_spec_decode, (
        f"{hf_path}: supports_spec_decode=True on a hybrid path — the "
        f"hybrid contract requires spec decode off."
    )


# =============================================================================
# Cross-check: every non-hybrid alias in the dense set must point at an
# hf_path that does NOT include an A3B / A10B / MoE marker. Catches
# accidental misclassification in the opposite direction.
# =============================================================================


@pytest.mark.parametrize("alias", DENSE_QWEN35_ALIASES + DENSE_QWEN36_ALIASES)
def test_dense_alias_hf_paths_have_no_moe_marker(alias: str) -> None:
    """Sanity check: an alias the JSON marks as ``is_hybrid=false`` must
    not point at an A3B / A10B / MoE-suffixed repo. If it did, either
    the JSON is wrong (the repo IS MoE) or the alias name is mis-named.
    """
    profiles = list_profiles()
    hf_path = profiles[alias].hf_path.lower()
    for marker in ("a3b", "a10b", "moe"):
        assert marker not in hf_path, (
            f"{alias}: hf_path={profiles[alias].hf_path!r} carries an "
            f"MoE marker {marker!r} but the alias is tagged as non-hybrid. "
            f"Either the JSON flag is wrong or the alias is misnamed."
        )
