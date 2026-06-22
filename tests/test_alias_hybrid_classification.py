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
from vllm_mlx.model_auto_config import (
    ModelConfig,
    detect_model_config,
    enrich_model_config,
)

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


# =============================================================================
# Runtime probe — the gate that closes the boot-time R6-C1 regression path
# =============================================================================


@pytest.fixture
def stub_arrays_cache(monkeypatch):
    """Inject a hermetic stub for ``mlx_lm.models.cache.ArraysCache``.

    The real ``ArraysCache`` is an ``mx.array``-backed structure whose
    import path (``mlx_lm.models.cache``) eagerly initializes the Metal
    device. On headless / no-Metal hosts (CI x86 runners, Linux build
    boxes) that import raises
    ``RuntimeError: [metal::load_device] No Metal device available``
    BEFORE the probe contract is exercised, which would make this
    regression guard environment-flaky.

    We swap a stub module into ``sys.modules['mlx_lm.models.cache']``
    so the ``from mlx_lm.models.cache import ArraysCache`` import inside
    ``enrich_model_config`` resolves to a plain class. ``isinstance``
    against this same class drives the probe's hybrid-promotion branch.
    """
    import sys
    import types

    # Build a minimal stub module that exposes a class named ``ArraysCache``.
    stub_cache_module = types.ModuleType("mlx_lm.models.cache")

    class ArraysCache:  # noqa: D401 — stub mirroring upstream class
        """Stub stand-in for the linear-attention cache marker class."""

        def __init__(self, *args, **kwargs):
            pass

    stub_cache_module.ArraysCache = ArraysCache

    # Ensure parent packages exist so the dotted import resolves cleanly
    # even if mlx_lm hasn't been imported at all in this test session.
    if "mlx_lm" not in sys.modules:
        monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    if "mlx_lm.models" not in sys.modules:
        monkeypatch.setitem(
            sys.modules, "mlx_lm.models", types.ModuleType("mlx_lm.models")
        )
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", stub_cache_module)

    return ArraysCache


def test_enrich_respects_is_hybrid_explicit_against_arrays_cache(
    stub_arrays_cache,
) -> None:
    """r6-A R6-C1 boot-path guard (codex r1 IMPORTANT).

    ``enrich_model_config``'s runtime probe sees ``ArraysCache`` layers
    on every dense Qwen3.5 / Qwen3.6 weight (``make_cache()`` returns
    the linear-attention cache type for ``model_type=qwen3_5``
    regardless of MoE-ness). Pre-fix, the probe one-way-flipped
    ``is_hybrid=True`` on any such model — silently undoing the JSON
    declaration that flagged the alias as non-hybrid. This guard pins
    the suppression contract: when the resolved ``ModelConfig`` carries
    ``is_hybrid_explicit=True``, the probe MUST leave ``is_hybrid``
    alone (only ``supports_spec_decode`` is forced off, which is a
    separate safety contract — spec decode is unsafe on
    linear-attention weights regardless of the routing decision).

    Hermeticity (codex r2 IMPORTANT): ``mlx_lm.models.cache`` is stubbed
    via the ``stub_arrays_cache`` fixture so this test does NOT require
    a Metal device — the contract is between ``enrich_model_config``
    and the ``ArraysCache`` marker class, not the underlying MLX arrays.
    """
    ArraysCache = stub_arrays_cache

    class HybridCacheModel:
        def make_cache(self):
            return [ArraysCache(size=1)]

    # Caller flagged the model as explicitly non-hybrid (the R6-C1 alias
    # contract). ``supports_spec_decode`` is also flipped on so we can
    # observe the probe forcing it off (the orthogonal safety gate).
    cfg_in = ModelConfig(
        is_hybrid=False,
        is_hybrid_explicit=True,
        supports_spec_decode=True,
    )
    cfg_out = enrich_model_config(cfg_in, HybridCacheModel())

    assert cfg_out.is_hybrid is False, (
        "enrich_model_config promoted is_hybrid=True even though "
        "is_hybrid_explicit=True was set — the R6-C1 boot-path "
        "regression has re-opened."
    )
    assert cfg_out.is_hybrid_explicit is True, (
        "is_hybrid_explicit must round-trip through enrich (replace())"
    )
    assert cfg_out.supports_spec_decode is False, (
        "supports_spec_decode must be forced off when ArraysCache is "
        "present, regardless of the routing decision (orthogonal "
        "safety contract — spec decode is unsafe on linear-attention "
        "weights)."
    )


def test_enrich_promotes_when_explicit_flag_unset(stub_arrays_cache) -> None:
    """Counterpart to ``test_enrich_respects_is_hybrid_explicit_*``: a
    caller that didn't set the explicit flag MUST still see the runtime
    probe promote ``is_hybrid=True``. The suppression is opt-in, not a
    blanket disable — legacy callers / brand-new HF paths without an
    alias profile rely on the safety-net promotion.

    Hermetic via the same ``stub_arrays_cache`` fixture (no Metal
    device required)."""
    ArraysCache = stub_arrays_cache

    class HybridCacheModel:
        def make_cache(self):
            return [ArraysCache(size=1)]

    cfg_in = ModelConfig(
        is_hybrid=False,
        is_hybrid_explicit=False,  # legacy / non-aliased path
        supports_spec_decode=True,
    )
    cfg_out = enrich_model_config(cfg_in, HybridCacheModel())

    assert cfg_out.is_hybrid is True, (
        "Probe should promote is_hybrid=True for ArraysCache models "
        "when the explicit flag is unset (legacy safety net)."
    )
    assert cfg_out.supports_spec_decode is False


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
