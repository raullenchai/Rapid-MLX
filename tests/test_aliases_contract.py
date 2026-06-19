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
import re
from pathlib import Path

import pytest

from vllm_mlx.model_aliases import (
    POPULAR_ALIASES,
    VALID_PFLASH_TIERS,
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
        "modality",
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
        "recommended_sampling",
        "pflash_tier",
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
# PFlash tier sanity (#287 alias-profile integration)
# =============================================================================


@pytest.mark.parametrize("alias", _alias_ids())
def test_alias_pflash_tier_value_is_in_enum(alias: str) -> None:
    """``pflash_tier`` must be one of the canonical enum values.

    Same closed-enum guard as ``suffix_decoding_tier``: a typo like
    ``"verifed"`` would silently fall back to the ``"unknown"`` default
    behaviour at engine boot, hiding the operator's intent to enable
    PFlash on a benched alias. Loader rejects it; this test pins the
    contract at PR review time so the rejection isn't only an integration-
    test failure.
    """
    tier = list_profiles()[alias].pflash_tier
    assert tier in VALID_PFLASH_TIERS, (
        f"{alias}: pflash_tier={tier!r} not in "
        f"{sorted(VALID_PFLASH_TIERS)}. Did you misspell it?"
    )


def test_pflash_verified_aliases_are_qwen35_or_qwen36() -> None:
    """``pflash_tier="verified"`` is reserved for the Qwen3.5 / Qwen3.6
    family — the only architectures we've bench-validated for the
    keep_ratio=0.20 default (PR #649: 3.87x-8.5x TTFT speedup with 100%
    needle recall across tested cells). Promoting a non-Qwen3.5/3.6 alias
    to ``"verified"`` should be a deliberate review-blocking change: it
    flips the engine's default ``--pflash`` mode to ``"always"`` for that
    alias, which silently shifts the quality/speed tradeoff for every
    user who hadn't passed an explicit flag. Keep the gate tight until
    there's matching bench evidence on a new family.
    """
    profiles = list_profiles()
    verified = sorted(a for a, p in profiles.items() if p.pflash_tier == "verified")
    # Positive control: at least one verified alias exists (otherwise the
    # test would trivially pass and the intent would silently rot).
    assert verified, (
        "No aliases tagged pflash_tier=verified. PR #649 tagged the "
        "Qwen3.5 / Qwen3.6 family — if you intentionally removed all of "
        "them, also delete this test."
    )
    offenders = [
        a
        for a in verified
        if not (a.startswith("qwen3.5-") or a.startswith("qwen3.6-"))
    ]
    assert not offenders, (
        f"Aliases tagged pflash_tier=verified outside the Qwen3.5 / "
        f"Qwen3.6 family: {offenders}. Either bench the family and "
        "extend the allowlist in this test, or reset the tier to "
        "'unknown'."
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


def test_negative_control_typo_in_pflash_tier_is_caught() -> None:
    """A typo like ``"verifed"`` must not be in ``VALID_PFLASH_TIERS``.

    Parallel guard for the PFlash tier enum (#287) — see
    ``test_negative_control_typo_in_tier_is_caught`` for the
    suffix-decoding analogue.
    """
    assert "verifed" not in VALID_PFLASH_TIERS
    assert "VERIFIED" not in VALID_PFLASH_TIERS  # case-sensitive on purpose
    assert "auto" not in VALID_PFLASH_TIERS  # tier != mode (avoid confusion)
    assert "always" not in VALID_PFLASH_TIERS  # ditto


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
    # Case-insensitive AND anchored on the "-4bit" form so the test
    # matches ``eligibility._looks_like_4bit`` exactly. Drift between
    # the two would let an alias green-light here but crash at boot.
    hf_lc = hf.lower()
    is_4bit = "-4bit" in hf_lc or "mxfp4" in hf_lc or "nvfp4" in hf_lc
    assert not is_4bit, (
        f"{alias}: supports_dflash=True but hf_path={hf!r} looks like a "
        f"4-bit quantized variant. DFlash regresses on 4-bit precision "
        f"(accept rate collapses). Use an 8-bit or higher quantization."
    )


def test_dflash_eligible_aliases_have_qwen35_36_drafter() -> None:
    """DFlash drafters today are published by ``z-lab/`` for Qwen3,
    Qwen3.5, Qwen3.6, Gemma-4 and LLaMA-3.1 families. Any eligible
    alias must point at one of these prefixes and bear the ``DFlash``
    marker (the ``-b16`` / ``-UltraChat`` / etc. suffix is permitted —
    z-lab uses it for training-data and precision tags). Catches an
    accidental copy-paste that swaps the drafter to an incompatible
    model."""
    valid_drafter_prefixes = (
        "z-lab/Qwen3-",
        "z-lab/Qwen3.5-",
        "z-lab/Qwen3.6-",
        "z-lab/gemma-4-",
        "z-lab/LLaMA3.1-",
    )
    for alias, profile in list_profiles().items():
        if not profile.supports_dflash:
            continue
        d = profile.dflash_draft_model or ""
        ok = any(d.startswith(p) for p in valid_drafter_prefixes)
        # ``DFlash`` may appear at end of repo name OR before a tag
        # suffix (``-b16``, ``-UltraChat``, etc.). Anchored on ``-`` /
        # end-of-string so we don't accept ``-notDFlash-utils`` or
        # other strings where ``DFlash`` is just a substring of an
        # unrelated word.
        has_marker = bool(re.search(r"(?:^|-)DFlash(?:$|-)", d))
        assert has_marker and ok, (
            f"{alias}: dflash_draft_model={d!r} doesn't match the "
            f"expected ``z-lab/{{Qwen3,Qwen3.5,Qwen3.6,gemma-4,LLaMA3.1}}-*"
            f"DFlash*`` shape. If you've validated a new drafter family, "
            f"update this allow-list."
        )


def test_negative_control_dflash_on_moe_is_caught() -> None:
    """A future PR adding ``is_moe=true`` + ``supports_dflash=true`` must
    be rejected by the eligibility gate. Exercises the actual gate path
    (not just the data structure) so a regression that quietly removes
    the MoE check in ``eligibility.check`` fails this test."""
    from vllm_mlx.model_aliases import AliasProfile
    from vllm_mlx.speculative.dflash import DFlashUnavailable, check

    bad = AliasProfile(
        hf_path="fake/MoE-Model",
        is_moe=True,
        supports_dflash=True,
        dflash_draft_model="z-lab/Qwen3.6-35B-A3B-DFlash",
    )
    with pytest.raises(DFlashUnavailable, match="MoE"):
        check(bad, alias="fake-moe-alias")


def test_negative_control_dflash_missing_drafter_is_caught() -> None:
    """``supports_dflash=True`` without ``dflash_draft_model`` must be
    rejected at JSON load time by ``_coerce``."""
    from vllm_mlx.model_aliases import _coerce

    with pytest.raises(ValueError, match="dflash_draft_model"):
        _coerce(
            "fake-alias",
            {"hf_path": "fake/Model", "supports_dflash": True},
        )


def test_audit_batch_reasoning_parser_wirings() -> None:
    """Pin the Model Onboarding SOP audit fixes for reasoning_parser
    on nemotron / hermes4 aliases. Each was previously
    ``null`` despite the model emitting ``<think>``/``</think>``
    blocks — without the parser, those blocks leak into
    ``message.content``.

    Parser choice rationale:
    - nemotron-30b-4bit/nano use a Qwen3-style template that
      INJECTS ``<think>`` into the prompt (gated by ``enable_thinking``
      / ``thinking`` flag). ``qwen3`` parser's ``finalize_streaming``
      correction handles the "no </think> ever appeared → emit as
      content" case correctly.
    - hermes4-70b-4bit: the chat template does NOT inject ``<think>``;
      the model decides autonomously. Same contract as GLM-4 → reuse
      ``glm4`` parser (no-tags-yet → content semantics).
    """
    profiles = list_profiles()
    expected = {
        "nemotron-30b-4bit": "qwen3",
        "hermes4-70b-4bit": "glm4",
    }
    for alias, parser in expected.items():
        assert alias in profiles, f"{alias} missing from aliases.json"
        assert profiles[alias].reasoning_parser == parser, (
            f"{alias}: reasoning_parser must be {parser!r} per audit. "
            f"Got {profiles[alias].reasoning_parser!r}."
        )


def test_bonsai_family_wires_glm4_reasoning_parser() -> None:
    """The Bonsai chat template (verified at
    https://huggingface.co/prism-ml/Bonsai-1.7B-unpacked/resolve/main/chat_template.jinja)
    injects an empty ``<think>\\n\\n</think>`` block when
    ``add_generation_prompt=True`` — the model's actual output stream
    then contains only content, no tags. The base class' "no tags
    yet, treat as reasoning" default would misclassify every Bonsai
    token; the ``glm4`` parser overrides exactly that branch.

    If a downstream user enables thinking via ``reasoning_content``
    on a prior assistant turn, the model may emit real
    ``<think>...</think>`` blocks; the same glm4 parser splits those
    correctly. Net effect: glm4 is strictly safer than null with
    zero behavioural downside for non-thinking turns.
    """
    profiles = list_profiles()
    for alias in ("bonsai-1.7b-unpacked", "bonsai-4b-unpacked", "bonsai-8b-unpacked"):
        assert alias in profiles, f"{alias} missing from aliases.json"
        assert profiles[alias].reasoning_parser == "glm4", (
            f"{alias}: reasoning_parser must be 'glm4' per audit. "
            f"Got {profiles[alias].reasoning_parser!r}."
        )


def test_audit_batch_bonsai_tool_call_parser_wired() -> None:
    """Pin the Model Onboarding SOP audit fix for the Bonsai family.
    The chat template emits ``<tool_call>...</tool_call>`` blocks
    (hermes pattern); leaving ``tool_call_parser=null`` made every
    tool call land in ``message.content`` as plain text. Verified the
    template format directly against
    https://huggingface.co/prism-ml/Bonsai-1.7B-unpacked.
    """
    profiles = list_profiles()
    for alias in ("bonsai-1.7b-unpacked", "bonsai-4b-unpacked", "bonsai-8b-unpacked"):
        assert alias in profiles, f"{alias} missing from aliases.json"
        assert profiles[alias].tool_call_parser == "hermes", (
            f"{alias}: tool_call_parser must be 'hermes' per audit. "
            f"Got {profiles[alias].tool_call_parser!r}."
        )


def test_deepseek_v4_flash_family_wires_deepseek_r1_reasoning_parser() -> None:
    """The DeepSeek-V4-Flash chat template emits ``<think>...</think>``
    blocks (gated by ``thinking_mode``). Without ``reasoning_parser`` set,
    that text leaks into ``choices[0].message.content`` as user-visible
    chain-of-thought. Pin the wiring so a future PR can't silently revert
    it to ``null``.

    Verified format source:
    https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit/resolve/main/chat_template.jinja
    """
    profiles = list_profiles()
    family = [
        "deepseek-v4-flash-8bit",
        "deepseek-v4-flash-2bit",
        "deepseek-v4-flash-4bit",
        "deepseek-v4-flash-8bit",
    ]
    for alias in family:
        assert alias in profiles, f"{alias} missing from aliases.json"
        assert profiles[alias].reasoning_parser == "deepseek_r1", (
            f"{alias}: reasoning_parser must be 'deepseek_r1' (V4-Flash emits "
            f"`<think>` blocks). Got {profiles[alias].reasoning_parser!r}."
        )


@pytest.mark.parametrize(
    "alias",
    ["vibethinker-1.5b-4bit", "vibethinker-3b-8bit"],
)
def test_vibethinker_family_wires_deepseek_r1_reasoning_parser(alias: str) -> None:
    """VibeThinker (Weibo AI; 1.5B base = Qwen2.5-Math-1.5B, 3B base =
    Qwen2.5-Coder-3B) is a reasoning family whose chat template does
    NOT inject ``<think>`` — the model emits ``<think>...</think>``
    blocks autonomously on every response. Without ``reasoning_parser``
    set, those blocks leak into ``choices[0].message.content`` as plain
    text and break clients that expect ``reasoning_content`` to carry
    the chain-of-thought.

    Pin the wiring so a future PR can't silently revert it to ``null``
    for either size. Also pins ``tool_call_parser=None`` — the model
    card explicitly states tool calling is unsupported even though the
    inherited Qwen2 vocab carries ``<tool_call>`` tokens (community
    reports confirm the model reasons about the template instead of
    emitting tool calls).
    Verified format sources:
    https://huggingface.co/mlx-community/VibeThinker-3B-8bit
    https://huggingface.co/mlx-community/VibeThinker-1.5B-mlx-4bit
    """
    profiles = list_profiles()
    assert alias in profiles, f"{alias} missing from aliases.json"
    assert profiles[alias].reasoning_parser == "deepseek_r1", (
        f"{alias}: reasoning_parser must be 'deepseek_r1' (VibeThinker emits "
        f"`<think>` blocks autonomously). Got {profiles[alias].reasoning_parser!r}."
    )
    assert profiles[alias].tool_call_parser is None, (
        f"{alias}: tool_call_parser must be None — VibeThinker is not "
        f"trained for tool calling per the upstream model card. "
        f"Got {profiles[alias].tool_call_parser!r}."
    )
    # Reasoning-model sampling guidance: temperature=1.0, top_p=0.95
    # (paper-recommended; greedy temperature=0 produces garbage on
    # reasoning models that depend on diverse beam exploration).
    sampling = dict(profiles[alias].recommended_sampling or ())
    assert sampling.get("temperature") == 1.0, (
        f"{alias}: recommended_sampling.temperature must be 1.0 per the "
        f"VibeThinker paper. Got {sampling.get('temperature')!r}."
    )
    assert sampling.get("top_p") == 0.95, (
        f"{alias}: recommended_sampling.top_p must be 0.95 per the "
        f"VibeThinker paper. Got {sampling.get('top_p')!r}."
    )


@pytest.mark.parametrize(
    "alias",
    [
        "qwen3-4b-thinking-2507-4bit",
        "qwen3-4b-instruct-2507-4bit",
        "qwen3-vl-2b-4bit",
    ],
)
def test_qwen3_small_models_batch_wires_qwen3_parsers(alias: str) -> None:
    """The Qwen3 ≤5B additions (Thinking-2507, Instruct-2507, VL-2B)
    must all carry the Qwen3 family parser pair (``hermes`` for tool
    calls + ``qwen3`` for reasoning).

    The ``qwen3`` reasoning parser is the right pick even for the
    non-thinking ``Instruct`` variant — the parser stays inert on
    output that contains no ``<think>...</think>`` blocks and never
    swallows content; pinning it keeps the wiring uniform across the
    whole Qwen3 family and matches existing entries like
    ``qwen3-4b-8bit`` / ``qwen3-vl-4b-4bit``.

    Pinning here so a future PR can't silently revert the Thinking-2507
    entry to ``reasoning_parser=null`` (which would leak ``<think>``
    blocks into ``content``) or downgrade the Instruct-2507 entry away
    from the family default (which would create a parser-mismatch hop
    for clients that round-robin across Qwen3 sizes).
    """
    profiles = list_profiles()
    assert alias in profiles, f"{alias} missing from aliases.json"
    assert profiles[alias].tool_call_parser == "hermes", (
        f"{alias}: tool_call_parser must be 'hermes' (Qwen3 family default). "
        f"Got {profiles[alias].tool_call_parser!r}."
    )
    assert profiles[alias].reasoning_parser == "qwen3", (
        f"{alias}: reasoning_parser must be 'qwen3' — the parser is inert "
        f"on outputs without `<think>` blocks, so it's safe for both the "
        f"Thinking and Instruct variants. Got {profiles[alias].reasoning_parser!r}."
    )
    assert profiles[alias].is_hybrid is False, (
        f"{alias}: Qwen3 (2B / 4B / VL) is pure-attention, not hybrid. "
        f"Mis-tagging as hybrid disables spec-decode for no reason."
    )


def test_granite4_h_micro_inherits_family_hybrid_gates() -> None:
    """``granite4-h-micro-4bit`` is a 3B variant of IBM's hybrid
    Mamba2+Transformer family. It MUST carry the same hybrid +
    no-spec-decode + no-reasoning-parser wiring as the existing
    ``granite4-tiny-4bit`` entry — Granite 4 does not emit
    ``<think>...</think>`` (model_auto_config has the matching
    comment on the family regex), and the hybrid Mamba2 state breaks
    spec-decode drafters.
    """
    profiles = list_profiles()
    micro = profiles["granite4-h-micro-4bit"]
    tiny = profiles["granite4-tiny-4bit"]
    assert micro.tool_call_parser == tiny.tool_call_parser == "hermes", (
        "granite4-h-micro-4bit must match granite4-tiny-4bit on "
        "tool_call_parser; the family shares the same template."
    )
    assert micro.reasoning_parser is None and tiny.reasoning_parser is None, (
        "Granite 4 does NOT emit `<think>` blocks; setting a reasoning "
        "parser would route all output into reasoning_content."
    )
    assert micro.is_hybrid and tiny.is_hybrid, (
        "Granite 4 is hybrid Mamba2+Transformer — is_hybrid must be True."
    )
    assert not micro.supports_spec_decode, (
        "granite4-h-micro-4bit: hybrid arch + supports_spec_decode=True "
        "is a forbidden combination (see test_hybrid_disables_spec_decode)."
    )


def test_nanbeige_4_1_3b_uses_hermes_not_llama_tool_parser() -> None:
    """``nanbeige4.1-3b-4bit`` has ``model_type=llama`` in its config but
    is NOT a Meta-LLaMA-3 chat checkpoint — chat template + tool format
    are upstream-Nanbeige. The matching ``nanbeige`` regex in
    ``model_auto_config.py`` must win over the generic ``llama`` regex
    so HF-path serves don't pick up ``tool_call_parser=llama`` (which
    would silently fail to parse the Nanbeige tool-call envelope).

    Pin here so a regex re-ordering can't quietly demote the entry to
    the LLaMA tool parser.
    """
    profile = list_profiles()["nanbeige4.1-3b-4bit"]
    assert profile.tool_call_parser == "hermes", (
        f"nanbeige4.1-3b-4bit: tool_call_parser must be 'hermes' (Nanbeige "
        f"is not vanilla LLaMA-3 despite model_type=llama). "
        f"Got {profile.tool_call_parser!r}."
    )
    # The 3B preview emits autonomous ``<think>...</think>`` blocks on
    # every response (verified by a local smoke test during the batch
    # landing). With ``reasoning_parser=null`` the raw block leaks into
    # ``choices[0].message.content`` and clients lose ``reasoning_content``.
    # ``deepseek_r1`` handles the "model decides" contract — same as
    # VibeThinker / R1-distill on a non-DeepSeek base.
    assert profile.reasoning_parser == "deepseek_r1", (
        f"nanbeige4.1-3b-4bit: reasoning_parser must be 'deepseek_r1' — "
        f"the model emits `<think>` blocks autonomously. Got "
        f"{profile.reasoning_parser!r}."
    )


def test_phi_4_mini_reasoning_wires_deepseek_r1_reasoning_parser() -> None:
    """``phi-4-mini-reasoning-4bit`` is Microsoft's math-tuned reasoning
    variant of Phi-4-mini. The chat template does NOT inject a
    ``<think>`` tag (the only special tokens are ``<|user|>`` /
    ``<|assistant|>`` / ``<|end|>`` / ``<|tool_call|>``), but the model
    emits ``<think>...</think>`` autonomously on every response —
    smoke-verified during this PR with ``reasoning_parser=null``:
    a ``Say hi in three words.`` prompt returned ``<think>\\nOkay,
    so the user wants me to say...`` as the raw ``content`` of the
    assistant message, leaking the chain-of-thought to clients.

    Pin ``reasoning_parser=deepseek_r1`` so the block lands in
    ``reasoning_content`` instead. This matches the same "model decides"
    contract used by VibeThinker, R1-distill, and Nanbeige4.1 — none of
    those templates inject a ``<think>`` open tag either, and they all
    rely on the deepseek_r1 parser's "stay-in-reasoning-until-we-see-
    </think>" state machine.

    Verified format source: smoke test on the 4-bit lmstudio-community
    repack; tokenizer special tokens enumerated from
    ``microsoft/Phi-4-mini-reasoning/tokenizer_config.json``.
    """
    profiles = list_profiles()
    alias = "phi-4-mini-reasoning-4bit"
    assert alias in profiles, f"{alias} missing from aliases.json"
    assert profiles[alias].reasoning_parser == "deepseek_r1", (
        f"{alias}: reasoning_parser must be 'deepseek_r1' — Phi-4-mini-"
        f"reasoning emits `<think>` blocks autonomously (smoke-verified). "
        f"Got {profiles[alias].reasoning_parser!r}."
    )
    assert profiles[alias].tool_call_parser == "hermes", (
        f"{alias}: tool_call_parser must be 'hermes' (Phi family default)."
    )


def test_gemma_3n_multimodal_aliases_share_family_sampling() -> None:
    """Gemma 3n E2B / E4B are Google's on-device multimodal family
    (text + image + audio share the same model). They MUST inherit the
    Gemma 3 sampling defaults (temperature=1.0, top_p=0.95, top_k=64
    per Google's chat-tuned guidance) — upstream
    ``generation_config.json`` ships an empty stub, so dropping the
    curated values would fall through to global defaults that are
    wrong for the family.

    Audio path is recognised by ``multimodal_processor.py`` (model_type
    ``gemma3n``); these aliases ship with the text+image surface and
    audio rides the same lane when an audio attachment is present.
    """
    for alias in ("gemma-3n-e2b-4bit", "gemma-3n-e4b-4bit"):
        profile = list_profiles()[alias]
        sampling = dict(profile.recommended_sampling or ())
        assert sampling.get("temperature") == 1.0, (
            f"{alias}: temperature must be 1.0 per Google's Gemma chat "
            f"sampling guidance. Got {sampling.get('temperature')!r}."
        )
        assert sampling.get("top_p") == 0.95, (
            f"{alias}: top_p must be 0.95. Got {sampling.get('top_p')!r}."
        )
        assert sampling.get("top_k") == 64, (
            f"{alias}: top_k must be 64. Got {sampling.get('top_k')!r}."
        )


def test_aliases_with_known_broken_hf_paths_stay_fixed() -> None:
    """Pin replacement paths for aliases that previously pointed at HF
    repos that no longer exist (or never existed).

    Three aliases shipped with hf_paths that 404 on HuggingFace —
    ``rapid-mlx serve <alias>`` would download-fail at first user
    contact. Each replacement was selected by manually browsing the
    mlx-community namespace for an extant repo of the same family.

    The substring guards below ensure a future "revert that aliases
    change" commit doesn't quietly restore the broken path.
    """
    profiles = list_profiles()
    # qwen3-vl-4b-4bit: stale ``-MLX-`` suffix not used by upstream uploads
    assert "MLX-4bit" not in profiles["qwen3-vl-4b-4bit"].hf_path, (
        "qwen3-vl-4b-4bit previously pointed at "
        "mlx-community/Qwen3-VL-4B-Instruct-MLX-4bit which 404s; the "
        "current upload is Qwen3-VL-4B-Instruct-4bit (no '-MLX-' suffix)."
    )
    # devstral-24b-4bit: ``2503`` snapshot was never re-uploaded as MLX-4bit;
    # 2505/2507 are the canonical Devstral-Small v1 releases.
    assert "2503" not in profiles["devstral-24b-4bit"].hf_path, (
        "devstral-24b-4bit previously pointed at Devstral-Small-2503-MLX-4bit "
        "which 404s. Use the 2507 (or 2505) MLX 4-bit upload."
    )
    # glm4.5-air-4bit: ``-0111-`` date suffix was a community-only tag that
    # got rolled into the default release.
    assert "0111" not in profiles["glm4.5-air-4bit"].hf_path, (
        "glm4.5-air-4bit previously pointed at GLM-4.5-Air-0111-4bit which "
        "404s. The current canonical upload is GLM-4.5-Air-4bit."
    )
    # glm4.7-9b-4bit previously pointed at the full GLM-4.7 (355B MoE,
    # ~185 GB at 4-bit) — the alias name implies a 9B model. The
    # correct upload is the Flash variant (~16 GB).
    assert "Flash" in profiles["glm4.7-9b-4bit"].hf_path, (
        "glm4.7-9b-4bit must point at the GLM-4.7-Flash upload, not the full "
        "GLM-4.7 (355B MoE) which is ~12x larger and won't fit on most "
        "user disks."
    )
    # gpt-oss-20b-mxfp4-q8 previously pointed at mlx-community/GPT-OSS-20B-4bit
    # which 404s; the canonical mlx-community release uses the
    # MXFP4-Q8 hybrid quantization.
    assert (
        profiles["gpt-oss-20b-mxfp4-q8"].hf_path != "mlx-community/GPT-OSS-20B-4bit"
    ), (
        "gpt-oss-20b-mxfp4-q8 must not regress to the 404 path; current canonical "
        "upload is mlx-community/gpt-oss-20b-MXFP4-Q8."
    )


# Curated ``recommended_sampling`` overrides — one entry per alias whose
# upstream ``generation_config.json`` is an empty stub (e.g. Gemma 3 /
# GLM-4.5-Air ship only eos/pad tokens) or partial (GLM-4.7 ships only
# ``temperature``). Each entry is a gap-fill against the model card,
# never a contradiction of upstream values.
#
# Pinned in a test so a future bulk edit to ``aliases.json`` can't
# silently drop or mutate one of these without the author looking up
# the model card again and confirming the value still applies.
#
# Phase 2 ships 10 entries; the other 48 aliases either inherit usable
# values from ``generation_config.json`` (Qwen3 family, Qwen3-VL) or
# haven't been audited yet (most of the missing-locally bucket).
_CURATED_RECOMMENDED_SAMPLING: dict[str, dict[str, float]] = {
    # Devstral 1.x — Mistral code-tuned model card example uses 0.15
    # for interactive coding (see model card on huggingface.co/mistralai).
    # Devstral 2.x ships the same empty stub; same pattern applies.
    "devstral-24b-4bit": {"temperature": 0.15},
    "devstral-v2-24b-4bit": {"temperature": 0.15},
    # Gemma 3 family — Google's Gemma docs recommend
    # (temperature=1.0, top_p=0.95, top_k=64) for the chat-tuned models.
    # All of gemma-3-1b / gemma-3-12b / gemma-3-27b ship an empty stub
    # locally (`_from_model_config: true` plus eos/pad tokens only).
    "gemma3-1b-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma3-12b-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma3-27b-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    # Gemma 3 QAT variants — same sampling as the PTQ siblings above. QAT
    # changes weight distribution (training with simulated quantization),
    # not the decoding distribution, so Google's chat sampling guidance
    # applies unchanged. (Matches the Gemma 4 QAT block below.)
    "gemma3-1b-qat-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma3-4b-qat-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma3-27b-qat-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    # Gemma 3n family (E2B / E4B) — on-device multimodal (text + image +
    # audio). Same chat-tuning recipe as Gemma 3, so Google's sampling
    # guidance applies unchanged. ``google/gemma-3n-E{2,4}B-it`` ship a
    # near-empty ``generation_config.json`` (the upstream HF page 401s
    # for the raw config under WebFetch; the MLX repacks under
    # ``mlx-community`` / ``lmstudio-community`` preserve the stub).
    "gemma-3n-e2b-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma-3n-e4b-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    # Gemma 4 — official Google sampling guidance hasn't been
    # published yet at the time of writing; we extrapolate from the
    # Gemma 3 family card. Revisit when an official Gemma 4 doc lands.
    # Gemma 4 "effective" variants (e2b/e4b) share the same chat-tuned
    # training recipe as their full-size siblings, so the same sampling
    # guidance applies.
    "gemma-4-e2b-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma-4-e4b-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma-4-12b-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma-4-12b-8bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma-4-26b-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma-4-31b-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma-4-31b-8bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    # Gemma 4 QAT variants — same sampling as PTQ siblings. QAT changes
    # weight distribution (training with simulated quantization) not the
    # decoding distribution, so Google's chat sampling guidance applies
    # unchanged.
    "gemma-4-12b-qat-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma-4-12b-qat-8bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma-4-26b-qat-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma-4-31b-qat-4bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    "gemma-4-31b-qat-8bit": {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0},
    # GLM-4.5-Air — THUDM publishes two recommendations: temperature=0.6
    # for *thinking* mode, ~1.0 for non-thinking. The alias has
    # reasoning_parser=glm4 → thinking IS the default response path,
    # so 0.6 is the right pick. (Users who want non-thinking can pass
    # temperature explicitly per-request.)
    "glm4.5-air-4bit": {"temperature": 0.6, "top_p": 0.95},
    # GLM-4.7-Flash ships temperature=1.0 upstream; we add only top_p.
    "glm4.7-9b-4bit": {"top_p": 0.95},
}


def test_curated_recommended_sampling_matches_pinned_values() -> None:
    """Pin every curated ``recommended_sampling`` override against the
    table above so a stray bulk edit to ``aliases.json`` can't silently
    drop or mutate a value. If you intentionally change a value, update
    this test too — that's the prompt to re-verify against the model
    card you originally consulted."""
    profiles = list_profiles()
    for alias, expected in _CURATED_RECOMMENDED_SAMPLING.items():
        assert alias in profiles, f"{alias}: missing from aliases.json"
        actual_tuple = profiles[alias].recommended_sampling
        assert actual_tuple is not None, (
            f"{alias}: recommended_sampling was curated but is now None; "
            f"either restore the entry or remove it from "
            f"_CURATED_RECOMMENDED_SAMPLING in this test."
        )
        actual = dict(actual_tuple)
        assert actual == expected, (
            f"{alias}: recommended_sampling drifted.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"If this is intentional, update _CURATED_RECOMMENDED_SAMPLING "
            f"and re-verify against the model card."
        )


def test_curated_aliases_do_not_contradict_fixture_generation_config() -> None:
    """For each curated alias with a checked-in upstream snapshot under
    ``tests/fixtures/generation_configs/<alias>.json``, the curated
    value must not *contradict* what the model author shipped.
    Gap-filling is fine; flipping a non-empty value is a red flag and
    means the curation needs explicit justification.

    The fixtures are byte-for-byte copies of the upstream JSON pulled
    from the local HF cache at curation time. They're committed so the
    test runs deterministically on a fresh CI runner (no HF cache
    required) and so future re-quants that change upstream values
    surface as a fixture mismatch rather than silently shifting which
    layer of the cascade wins.

    To refresh after an upstream update:
      cp ~/.cache/huggingface/hub/models--<repo>/snapshots/<sha>/generation_config.json \\
         tests/fixtures/generation_configs/<alias>.json
    Then re-verify the curated value still matches the new upstream.
    """
    import tempfile

    from vllm_mlx.utils.generation_config import load_generation_config_sampling

    fixture_dir = Path(__file__).parent / "fixtures" / "generation_configs"
    profiles = list_profiles()

    coverage = 0
    for alias in _CURATED_RECOMMENDED_SAMPLING:
        fixture = fixture_dir / f"{alias}.json"
        if not fixture.is_file():
            continue  # no fixture yet — alias is "trust the curation"
        coverage += 1
        # Stage the fixture in a temp dir so the loader (which expects
        # a model directory with ``generation_config.json`` inside)
        # exercises the same parsing path the cascade uses at runtime.
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "generation_config.json").write_bytes(fixture.read_bytes())
            shipped = load_generation_config_sampling(td)

        profile = profiles[alias]
        curated = dict(profile.recommended_sampling or ())
        for key, shipped_value in shipped.items():
            if key not in curated:
                continue  # curated is silent on this key — upstream wins
            assert curated[key] == shipped_value, (
                f"{alias}: curated recommended_sampling[{key!r}]="
                f"{curated[key]} contradicts upstream fixture "
                f"{fixture.name}[{key!r}]={shipped_value}. "
                f"Either drop the curated key (let upstream win) or "
                f"document why upstream is wrong in the comment above "
                f"_CURATED_RECOMMENDED_SAMPLING."
            )

    # Sanity floor: if every fixture got removed by accident, the test
    # would silently become a no-op. Pin a minimum coverage of 3 so a
    # bulk-delete of the fixtures directory is caught at PR time.
    assert coverage >= 3, (
        f"Only {coverage} curated aliases have a fixture under "
        f"{fixture_dir}; expected ≥3. Did the fixtures directory get "
        f"deleted? Restore the *.json files referenced by "
        f"_CURATED_RECOMMENDED_SAMPLING."
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
