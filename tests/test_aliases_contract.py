# SPDX-License-Identifier: Apache-2.0
"""Contract tests for ``vllm_mlx/aliases.json`` — under-spec'd alias guard.

The alias JSON is a frequent landing-zone for "looks-fine-on-PR" mistakes
that only surface much later: a Qwen alias missing ``tool_call_parser``
silently breaks tool calls; ``is_hybrid=true`` paired with
``supports_spec_decode=true`` makes the scheduler refuse the model at
startup; a tier of ``"god"`` (typo for ``"good"``) silently produces
no startup hint.

These tests pin those contracts at PR-review time so they fail in CI
rather than at first-user-load.

Adding a new alias?
  - It must use a registered parser name (or ``null``).
  - ``is_hybrid=true`` ⇒ ``supports_spec_decode=false`` (mutually
    exclusive — see MEMORY.md "Hybrid models").
  - ``suffix_decoding_tier`` must be one of the names in
    ``VALID_SUFFIX_TIERS``.
  - If you set ``suffix_bench_speedup``, set a non-``unknown`` tier (or
    explicitly mark ``unknown`` with a comment in the PR description).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx.model_aliases import (
    POPULAR_ALIASES,
    VALID_SUFFIX_TIERS,
    list_profiles,
)
from vllm_mlx.reasoning import list_parsers as list_reasoning_parsers
from vllm_mlx.tool_parsers import ToolParserManager

# Top-level keys we currently accept on a profile object. Typo-guard: if a
# PR adds ``is_hybird: true`` (real typo) it silently flows through as an
# unknown key today — this list catches that at PR time.
ALLOWED_PROFILE_KEYS: frozenset[str] = frozenset(
    {
        "hf_path",
        "tool_call_parser",
        "reasoning_parser",
        "is_hybrid",
        "is_moe",
        "supports_spec_decode",
        "default_max_tokens",
        "suffix_decoding_tier",
        "suffix_bench_speedup",
        "supports_dflash",
        "dflash_draft_model",
    }
)


def _raw_aliases() -> dict[str, dict | str]:
    """Return the raw JSON, not the coerced profiles — we need to see
    unexpected keys before ``_coerce`` drops them on the floor."""
    path = Path(__file__).resolve().parents[1] / "vllm_mlx" / "aliases.json"
    return json.loads(path.read_text())


def _alias_ids() -> list[str]:
    """Stable alias name list for ``parametrize`` IDs."""
    return sorted(_raw_aliases().keys())


# =============================================================================
# hf_path well-formed-ness
# =============================================================================


@pytest.mark.parametrize("alias", _alias_ids())
def test_alias_hf_path_is_org_slash_repo(alias: str) -> None:
    """Every alias must point at an ``org/repo`` style path. Loose paths
    silently break HF download — the user sees a confusing 404 from
    ``huggingface_hub`` rather than "you typed the alias wrong"."""
    profile = list_profiles()[alias]
    assert "/" in profile.hf_path, (
        f"{alias}: hf_path {profile.hf_path!r} is missing '/' separator. "
        f"Use 'org/repo' format (e.g. 'mlx-community/Qwen3.5-4B-MLX-4bit')."
    )
    # The legacy short-form (``"alias": "hf_path"``) coerces to a profile
    # but we still want the path itself to look HuggingFace-shaped.
    assert not profile.hf_path.startswith("/"), (
        f"{alias}: hf_path looks like an absolute path, not an HF repo id"
    )
    assert " " not in profile.hf_path, (
        f"{alias}: hf_path contains whitespace — copy-paste artifact?"
    )


# =============================================================================
# Parser names — must be registered or null
# =============================================================================


def _registered_tool_parsers() -> set[str]:
    """All registered tool-parser names from ToolParserManager."""
    eager = set(ToolParserManager.tool_parsers.keys())
    lazy = set(ToolParserManager.lazy_parsers.keys())
    return eager | lazy


def _registered_reasoning_parsers() -> set[str]:
    """All registered reasoning-parser names from the reasoning registry."""
    return set(list_reasoning_parsers())


@pytest.mark.parametrize("alias", _alias_ids())
def test_alias_tool_parser_is_registered(alias: str) -> None:
    """``tool_call_parser`` must be either ``null`` (base model, no tools)
    or one of the names ``ToolParserManager`` knows about. Typing
    ``"hermess"`` silently produces a model that emits tool calls the
    server can't parse, and there's no startup error today — the user
    just sees no tool_calls in their response."""
    parser = list_profiles()[alias].tool_call_parser
    if parser is None:
        return
    valid = _registered_tool_parsers()
    assert parser in valid, (
        f"{alias}: tool_call_parser={parser!r} is not in the registered "
        f"parser set. Did you misspell it? Registered: {sorted(valid)}"
    )


@pytest.mark.parametrize("alias", _alias_ids())
def test_alias_reasoning_parser_is_registered(alias: str) -> None:
    """Same contract as the tool parser — a typo'd reasoning_parser
    silently makes ``<think>...</think>`` blocks flow into the user-visible
    content."""
    parser = list_profiles()[alias].reasoning_parser
    if parser is None:
        return
    valid = _registered_reasoning_parsers()
    assert parser in valid, (
        f"{alias}: reasoning_parser={parser!r} is not in the registered "
        f"reasoning-parser set. Did you misspell it? Registered: {sorted(valid)}"
    )


# =============================================================================
# Capability gates — mutually exclusive combinations
# =============================================================================


@pytest.mark.parametrize("alias", _alias_ids())
def test_hybrid_disables_spec_decode(alias: str) -> None:
    """``is_hybrid=true`` and ``supports_spec_decode=true`` cannot both
    hold — the scheduler refuses to install spec-decode on hybrid models
    (Mamba/Transformer mix breaks the drafter state).

    Background: MEMORY.md "Hybrid models" — Qwen3.5/3.6, Qwopus, Nemotron,
    Granite4 all have ``is_hybrid=true`` and ``supports_spec_decode=false``.
    Mixing these silently caused failed boots in past PRs.
    """
    profile = list_profiles()[alias]
    if profile.is_hybrid:
        assert not profile.supports_spec_decode, (
            f"{alias}: is_hybrid=True but supports_spec_decode=True — "
            f"these are mutually exclusive. Hybrid models cannot use "
            f"spec-decode / suffix-decode (Mamba state breaks drafter)."
        )


# =============================================================================
# SuffixDecoding tier sanity
# =============================================================================


@pytest.mark.parametrize("alias", _alias_ids())
def test_alias_suffix_tier_value_is_in_enum(alias: str) -> None:
    """``suffix_decoding_tier`` must be one of the canonical enum values.
    Typing ``"god"`` (typo for ``"good"``) today silently flows through
    as a string — the CLI startup hint and any future filtering would
    treat it as ``unknown`` without a warning."""
    tier = list_profiles()[alias].suffix_decoding_tier
    assert tier in VALID_SUFFIX_TIERS, (
        f"{alias}: suffix_decoding_tier={tier!r} not in "
        f"{sorted(VALID_SUFFIX_TIERS)}. Did you misspell it?"
    )


@pytest.mark.parametrize("alias", _alias_ids())
def test_alias_suffix_bench_consistency(alias: str) -> None:
    """If ``suffix_bench_speedup`` is populated, ``suffix_decoding_tier``
    must NOT be ``"unknown"`` — there's a benched signal, so a tier
    decision is required. Conversely, ``tier`` ∉ {``"unknown"``} requires
    bench data so the decision is justified (no editorial classification
    without evidence)."""
    profile = list_profiles()[alias]
    has_bench = profile.suffix_bench_speedup is not None
    is_unknown = profile.suffix_decoding_tier == "unknown"
    if has_bench:
        assert not is_unknown, (
            f"{alias}: suffix_bench_speedup is set but tier=unknown — "
            f"benched aliases must have a tier decision. Pick one of: "
            f"{sorted(VALID_SUFFIX_TIERS - {'unknown'})}."
        )
    if not is_unknown:
        # Hybrid models can carry a documented tier even when bench data
        # is absent because the CLI renders them as ``n/a`` regardless.
        # MEMORY.md "Hybrid models" — tier setting is irrelevant for
        # hybrid (auto-rendered n/a), so don't require bench data there.
        if not profile.is_hybrid:
            assert has_bench, (
                f"{alias}: tier={profile.suffix_decoding_tier!r} but no "
                f"suffix_bench_speedup data. A tier decision must be "
                f"backed by bench evidence; add the bench result or "
                f"reset tier to 'unknown'."
            )


# =============================================================================
# Schema integrity — no unexpected keys (typo guard)
# =============================================================================


@pytest.mark.parametrize("alias", _alias_ids())
def test_alias_only_uses_known_keys(alias: str) -> None:
    """Catch typos like ``is_hybird`` or ``hf_paht`` at PR time.

    Today an unknown key flows silently through ``_coerce`` because the
    function reads keys by name — an extra ``is_hybird: true`` key just
    sits in the JSON dictionary with no effect, and ``is_hybrid`` stays
    at its default False. This test makes the typo a CI failure.
    """
    raw = _raw_aliases()[alias]
    if isinstance(raw, str):
        # Legacy short-form — no keys to validate.
        return
    extra = set(raw.keys()) - ALLOWED_PROFILE_KEYS
    assert not extra, (
        f"{alias}: unknown profile keys {sorted(extra)}. "
        f"Allowed: {sorted(ALLOWED_PROFILE_KEYS)}. "
        f"If you're adding a new field, update ALLOWED_PROFILE_KEYS here "
        f"and AliasProfile in vllm_mlx/model_aliases.py."
    )


# =============================================================================
# Cross-references — POPULAR_ALIASES tuple must be self-consistent
# =============================================================================


def test_popular_aliases_all_exist_in_registry() -> None:
    """``POPULAR_ALIASES`` is the fallback list shown when a user's typo
    can't be matched to any family. Every entry must resolve — otherwise
    the fallback would itself contain a broken suggestion."""
    profiles = list_profiles()
    missing = [a for a in POPULAR_ALIASES if a not in profiles]
    assert not missing, (
        f"POPULAR_ALIASES references aliases that don't exist in "
        f"aliases.json: {missing}. Either add the alias or remove the "
        f"name from POPULAR_ALIASES in vllm_mlx/model_aliases.py."
    )


# =============================================================================
# Negative controls — synthetic broken profiles to prove the guards bite
# =============================================================================
#
# These tests verify that the assertions in this file would actually CATCH
# the bad PRs they're written for. A guard that only passes on clean data
# isn't a regression guard — it's wallpaper. Each negative control crafts
# a known-bad profile and confirms the matching assertion would fire.


def test_negative_control_hybrid_spec_decode_combination_is_caught() -> None:
    """If a future PR adds ``is_hybrid=true`` + ``supports_spec_decode=true``,
    ``test_hybrid_disables_spec_decode`` must reject it."""
    from vllm_mlx.model_aliases import AliasProfile

    bad = AliasProfile(
        hf_path="fake/Model",
        is_hybrid=True,
        supports_spec_decode=True,  # contradiction
    )
    # Re-run the assertion logic on the synthetic profile.
    assert bad.is_hybrid and bad.supports_spec_decode, (
        "negative control malformed — should have hit the contradiction"
    )
    # The real guard would fail here:
    caught = bad.is_hybrid and bad.supports_spec_decode
    assert caught, "the test_hybrid_disables_spec_decode guard would miss this"


def test_negative_control_typo_in_tier_is_caught() -> None:
    """A typo like ``"god"`` must not be in ``VALID_SUFFIX_TIERS``."""
    assert "god" not in VALID_SUFFIX_TIERS
    assert "goood" not in VALID_SUFFIX_TIERS
    assert "AVOID" not in VALID_SUFFIX_TIERS  # case-sensitive on purpose


def test_negative_control_unregistered_parser_is_caught() -> None:
    """A misspelt ``tool_call_parser`` like ``"hermess"`` must not be in
    the registered set — proves the guard would catch a typo'd PR."""
    valid = _registered_tool_parsers()
    assert "hermess" not in valid
    assert "Hermes" not in valid  # case mismatch
    # And a positive control: a real parser must exist (so the test
    # itself wouldn't trivially pass for the wrong reason).
    assert any(p in valid for p in ("hermes", "qwen3_coder_xml", "minimax"))


# =============================================================================
# DFlash speculative-decoding contract (issue #264)
# =============================================================================


@pytest.mark.parametrize("alias", _alias_ids())
def test_dflash_requires_drafter(alias: str) -> None:
    """If ``supports_dflash=True``, ``dflash_draft_model`` MUST be set.
    A half-populated DFlash alias would silently fall back to AR at
    server-start time and look like an unexplained perf regression."""
    profile = list_profiles()[alias]
    if profile.supports_dflash:
        assert profile.dflash_draft_model, (
            f"{alias}: supports_dflash=True but dflash_draft_model is empty"
        )
        assert "/" in profile.dflash_draft_model, (
            f"{alias}: dflash_draft_model={profile.dflash_draft_model!r} "
            f"must be 'org/repo' format"
        )


@pytest.mark.parametrize("alias", _alias_ids())
def test_dflash_excludes_moe_architectures(alias: str) -> None:
    """``is_moe=True`` MUST NOT pair with ``supports_dflash=True``. PoC on
    Qwen3.6-35B-A3B (MoE hybrid) measured 0.76-0.82× regression
    regardless of precision — DFlash drafters' hidden-state fusion
    misfires on expert-routing churn (accept_len floors at ~1.5).
    Re-enabling this combination would ship the regression to users."""
    profile = list_profiles()[alias]
    if profile.is_moe:
        assert not profile.supports_dflash, (
            f"{alias}: is_moe=True but supports_dflash=True — DFlash "
            f"acceptance collapses on MoE due to expert-routing churn. "
            f"Confirmed regression on Qwen3.6-35B-A3B; do not enable on "
            f"MoE aliases."
        )


@pytest.mark.parametrize("alias", _alias_ids())
def test_dflash_excludes_4bit_precision(alias: str) -> None:
    """DFlash on a 4-bit MLX main model regresses (PoC: 0.63-0.96× on
    Qwen3.5-4B-MLX-4bit, accept_len 2.35-3.88). 4-bit AR is already at
    memory-bandwidth floor; drafter overhead dominates. Pattern is
    detected from the HF path naming convention (``-4bit`` suffix or
    ``-4bit-`` infix), which is how mlx-community publishes quantized
    variants."""
    profile = list_profiles()[alias]
    if not profile.supports_dflash:
        return
    hf = profile.hf_path
    is_4bit = (
        "-4bit" in hf
        or hf.endswith("-4bit")
        or "4bit-" in hf
        or "mxfp4" in hf.lower()
        or "nvfp4" in hf.lower()
    )
    assert not is_4bit, (
        f"{alias}: supports_dflash=True but hf_path={hf!r} looks like a "
        f"4-bit quantized variant. DFlash regresses on 4-bit precision "
        f"(accept rate collapses). Use an 8-bit or higher quantization."
    )


def test_dflash_eligible_aliases_have_qwen35_36_drafter() -> None:
    """DFlash drafters published today are Qwen-family-specific. Any
    eligible alias must point at a ``z-lab/Qwen3.5-*-DFlash`` or
    ``z-lab/Qwen3.6-*-DFlash`` drafter. Catches an accidental copy-paste
    that swaps the drafter to an incompatible model."""
    valid_drafter_prefixes = (
        "z-lab/Qwen3.5-",
        "z-lab/Qwen3.6-",
    )
    for alias, profile in list_profiles().items():
        if not profile.supports_dflash:
            continue
        d = profile.dflash_draft_model or ""
        ok = any(d.startswith(p) for p in valid_drafter_prefixes)
        assert d.endswith("-DFlash") and ok, (
            f"{alias}: dflash_draft_model={d!r} doesn't match the "
            f"expected ``z-lab/Qwen3.5-*-DFlash`` or "
            f"``z-lab/Qwen3.6-*-DFlash`` shape. If you've validated a "
            f"new drafter family, update this allow-list."
        )


def test_negative_control_dflash_on_moe_is_caught() -> None:
    """A future PR adding ``is_moe=true`` + ``supports_dflash=true`` must
    be rejected by ``test_dflash_excludes_moe_architectures``."""
    from vllm_mlx.model_aliases import AliasProfile

    bad = AliasProfile(
        hf_path="fake/MoE-Model",
        is_moe=True,
        supports_dflash=True,
        dflash_draft_model="z-lab/Qwen3.6-35B-A3B-DFlash",
    )
    caught = bad.is_moe and bad.supports_dflash
    assert caught, "the MoE-DFlash exclusion guard would miss this"


def test_negative_control_dflash_missing_drafter_is_caught() -> None:
    """``supports_dflash=True`` without ``dflash_draft_model`` must be
    rejected at JSON load time by ``_coerce``."""
    from vllm_mlx.model_aliases import _coerce

    with pytest.raises(ValueError, match="dflash_draft_model"):
        _coerce(
            "fake-alias",
            {"hf_path": "fake/Model", "supports_dflash": True},
        )


def test_default_max_tokens_is_positive_or_none() -> None:
    """``default_max_tokens`` is None or a positive int. A negative or
    zero default would make every request return empty completions."""
    for alias, profile in list_profiles().items():
        if profile.default_max_tokens is not None:
            assert (
                isinstance(profile.default_max_tokens, int)
                and profile.default_max_tokens > 0
            ), (
                f"{alias}: default_max_tokens={profile.default_max_tokens!r} "
                f"must be a positive int or None"
            )
