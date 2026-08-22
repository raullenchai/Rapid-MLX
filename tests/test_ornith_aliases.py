# SPDX-License-Identifier: Apache-2.0
"""Focused alias-routing contracts for the Ornith-1.5 family.

Ornith-1.5 is an NVIDIA-adjacent open LLM family (MIT) with official MLX
checkpoints. Both supported sizes (9B dense, 35B-A3B MoE) ship
``model_type=qwen3_5`` / ``qwen3_5_moe`` — the Qwen3.5 hybrid GatedDeltaNet
wrapper — with the *standard* 248320-vocab text head (unlike rapid-mlx's own
``qwen3.5-*`` short-gen 4M-vocab variants). They carry the Qwen3 XML
tool/thinking chat contract (Qwen2Tokenizer, ``<|im_end|>``), so the aliases
wire ``tool_call_parser="hermes"`` + ``reasoning_parser="qwen3"``, matching
the existing qwen3.5 family parser choice.

Dense-vs-MoE split (mirrors ``test_alias_hybrid_classification.py``):
  - 9B dense -> NOT hybrid, ``is_hybrid_explicit=True`` (the dense
    GatedDeltaNet cache layout wedges on metal::malloc under the hybrid
    scheduler path — same r6-A R6-C1 story as Qwen3.5-4B/9B).
  - 35B-A3B MoE -> hybrid (sparse experts + GatedDeltaNet, the path the
    prefix-boundary snapshot / throttle scheduler was benched for).
Both are bf16 unquantized and carry no MTP draft weights, so spec decode is
off family-wide and no ``mtp_draft_model`` is set.
"""

from __future__ import annotations

from vllm_mlx.model_aliases import list_profiles, resolve_profile
from vllm_mlx.model_auto_config import detect_model_config

ORNITH_9B = "ornith-1.5-9b-bf16"
ORNITH_35B = "ornith-1.5-35b-a3b-bf16"
ORNITH_9B_HF = "ornith-ai/Ornith-1.5-9B-MLX"
ORNITH_35B_HF = "ornith-ai/Ornith-1.5-35B-A3B-MLX"


# ---- SSOT alias profile -------------------------------------------------


def test_ornith_9b_alias_profile():
    """9B dense: non-hybrid with the explicit pin (dense GatedDeltaNet must
    stay off the hybrid scheduler), hermes/qwen3 parsers, no spec decode."""
    p = resolve_profile(ORNITH_9B)
    assert p is not None
    assert p.hf_path == ORNITH_9B_HF
    assert p.tool_call_parser == "hermes"
    assert p.reasoning_parser == "qwen3"
    assert p.is_hybrid is False
    assert p.is_hybrid_explicit is True
    assert p.is_moe is False
    assert p.supports_spec_decode is False
    assert p.modality == "text"


def test_ornith_35b_alias_profile():
    """35B-A3B MoE: hybrid + MoE (the hybrid scheduler path is what the
    sparse-expert variant needs), same parsers, no spec decode."""
    p = resolve_profile(ORNITH_35B)
    assert p is not None
    assert p.hf_path == ORNITH_35B_HF
    assert p.tool_call_parser == "hermes"
    assert p.reasoning_parser == "qwen3"
    assert p.is_hybrid is True
    assert p.is_moe is True
    assert p.supports_spec_decode is False
    assert p.modality == "text"


def test_ornith_aliases_are_registered():
    """Both aliases must be in the registry (list_aliases lower-bound and
    orphan checks in ``test_model_profiles_ssot.py`` also cover this, but a
    non-existent alias resolving to ``None`` here is the sharpest signal)."""
    profiles = list_profiles()
    assert ORNITH_9B in profiles
    assert ORNITH_35B in profiles


# ---- detect_model_config routing ---------------------------------------


def test_detect_ornith_alias_routes_9b_non_hybrid():
    """Alias-first lookup: the 9B alias must resolve to the non-hybrid
    profile (not get stamped hybrid by any heuristic)."""
    cfg = detect_model_config(ORNITH_9B)
    assert cfg is not None
    assert cfg.tool_call_parser == "hermes"
    assert cfg.is_hybrid is False
    assert cfg.is_hybrid_explicit is True
    assert cfg.supports_spec_decode is False


def test_detect_ornith_alias_routes_35b_hybrid():
    cfg = detect_model_config(ORNITH_35B)
    assert cfg is not None
    assert cfg.tool_call_parser == "hermes"
    assert cfg.is_hybrid is True
    assert cfg.is_moe is True
    assert cfg.supports_spec_decode is False


def test_detect_ornith_unregistered_repack_falls_back_to_regex():
    """Unregistered repacks must route through ``_MODEL_PATTERNS`` rather
    than resolving one of the official HF paths through the alias registry."""
    cfg9 = detect_model_config("community/Ornith-1.5-9B-MLX-repack")
    assert cfg9 is not None
    assert cfg9.is_hybrid is False
    assert cfg9.supports_spec_decode is False

    cfg35 = detect_model_config("community/Ornith-1.5-35B-A3B-MLX-repack")
    assert cfg35 is not None
    assert cfg35.is_hybrid is True
    assert cfg35.is_moe is True
    assert cfg35.supports_spec_decode is False


def test_detect_ornith_local_snapshot_dir():
    """The final local-directory segment can identify an Ornith repack."""
    cfg = detect_model_config("/tmp/ornith-1.5-35b-a3b-checkout")
    assert cfg is not None
    assert cfg.is_hybrid is True
    assert cfg.is_moe is True


def test_ornith_parent_directory_does_not_hijack_unrelated_checkpoint():
    """An Ornith-named parent must not stamp its routing onto a child model."""
    cfg = detect_model_config("/models/ornith-1.5-35b-a3b/Qwen3-4B")
    assert cfg is not None
    assert cfg.is_hybrid is False
    assert cfg.is_moe is False
    assert cfg.supports_spec_decode is True


def test_ornith_prefix_collision_does_not_match_family_fallback():
    """Words beginning with ``ornith`` are not Ornith-1.5 checkpoints."""
    assert detect_model_config("community/ornithology-9b") is None
    assert detect_model_config("community/ornithopter-35b-a3b") is None


def test_ornith_size_marker_substrings_do_not_match_family_fallback():
    """Size and architecture markers must be complete name tokens."""
    assert detect_model_config("community/Ornith-1.5-135B") is None
    assert detect_model_config("community/Ornith-1.5-19B") is None
    assert detect_model_config("community/Ornith-1.5-condensed") is None


# ---- cross-check: hf_path markers match the JSON flags -----------------


def test_ornith_hf_path_markers_match_classification():
    """Sanity: the alias the JSON marks hybrid/MoE must point at an A3B repo,
    and the non-hybrid one at a repo without an MoE marker."""
    profiles = list_profiles()
    assert "a3b" in profiles[ORNITH_35B].hf_path.lower()
    assert not any(
        m in profiles[ORNITH_9B].hf_path.lower() for m in ("a3b", "a10b", "moe")
    )
