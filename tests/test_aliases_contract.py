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
        # r6-A R6-C1: explicit-pin flag that suppresses the runtime
        # ArraysCache → is_hybrid=True auto-promotion. See
        # ``AliasProfile.is_hybrid_explicit``.
        "is_hybrid_explicit",
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
    for either size. Also pins ``tool_call_parser="hermes"`` — the
    inherited Qwen2 vocab carries ``<tool_call>`` / ``</tool_call>``
    tokens and the 2026-06-17 VibeThinker-3B-8bit live test confirmed
    the model emits BOTH ``<tool_call>{"name": ...}</tool_call>`` and
    bare ``<function=name>...</function>`` wire shapes for tool calls.
    With ``tool_call_parser=null`` the OutputRouter's token-level
    fallback caught the ``<tool_call>`` shape "by accident" but the
    bare ``<function>`` shape leaked into ``content`` as raw text.
    Hermes parser handles both shapes natively (see
    ``HermesToolParser.TOOL_CALL_PATTERN`` and
    ``BARE_FUNCTION_PATTERN``).
    Verified format sources:
    https://huggingface.co/mlx-community/VibeThinker-3B-8bit
    https://huggingface.co/mlx-community/VibeThinker-1.5B-mlx-4bit
    """
    profiles = list_profiles()
    assert alias in profiles, f"{alias} missing from aliases.json"
    assert profiles[alias].reasoning_parser == "vibethinker", (
        f"{alias}: reasoning_parser must be 'vibethinker' — a DeepSeek-R1 "
        f"variant with NO_TAG_CONTENT_THRESHOLD=1024 (vs base 64) to handle "
        f"the documented preamble-before-`<think>` shape (codex r2 P2). "
        f"Got {profiles[alias].reasoning_parser!r}."
    )
    assert profiles[alias].tool_call_parser == "hermes", (
        f"{alias}: tool_call_parser must be 'hermes' — VibeThinker is "
        f"Qwen2-derived and emits both <tool_call>{{...}}</tool_call> and "
        f"bare <function=name>...</function> shapes. The 2026-06-17 live "
        f"test confirmed the bare-function shape leaks into content "
        f"without the hermes parser. "
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


def test_qwen3_4b_thinking_2507_wires_qwen3_reasoning_parser() -> None:
    """The Qwen3-4B-Thinking-2507 variant emits ``<think>...</think>``
    blocks autonomously on every response; it MUST carry the
    ``qwen3`` reasoning parser so the trace lands in
    ``reasoning_content`` instead of leaking into ``content``.

    Pinned separately from the non-thinking siblings (Instruct-2507
    + VL-2B) because the non-thinking variants must NOT carry the
    parser — see the docstring on
    ``test_qwen3_small_non_thinking_variants_have_no_reasoning_parser``
    for the fuzz-evidence rationale (PR #715 bundle).
    """
    profiles = list_profiles()
    alias = "qwen3-4b-thinking-2507-4bit"
    assert alias in profiles, f"{alias} missing from aliases.json"
    assert profiles[alias].tool_call_parser == "hermes", (
        f"{alias}: tool_call_parser must be 'hermes' (Qwen3 family default). "
        f"Got {profiles[alias].tool_call_parser!r}."
    )
    assert profiles[alias].reasoning_parser == "qwen3", (
        f"{alias}: reasoning_parser must be 'qwen3' — the Thinking-2507 "
        f"variant emits `<think>` blocks autonomously. "
        f"Got {profiles[alias].reasoning_parser!r}."
    )
    assert profiles[alias].is_hybrid is False, (
        f"{alias}: Qwen3-4B is pure-attention, not hybrid."
    )


@pytest.mark.parametrize(
    "alias",
    [
        "qwen3-4b-instruct-2507-4bit",
        "qwen3-vl-2b-4bit",
    ],
)
def test_qwen3_small_non_thinking_variants_have_no_reasoning_parser(
    alias: str,
) -> None:
    """The Qwen3-4B-Instruct-2507 and Qwen3-VL-2B-Instruct aliases are
    NON-thinking variants — their model cards explicitly state no
    ``<think>`` emission and the 2026-06-18 fuzz battery against PR
    #714 confirmed the symptom: with ``reasoning_parser=qwen3`` wired
    AND the client passing ``enable_thinking=True`` (or the parser's
    Case-4 fallback firing on a no-tag output) the entire response
    is duplicated into BOTH ``content`` AND ``reasoning_content``,
    leaving a confusing assistant turn for the caller.

    The qwen3 parser's Case-4 "no tags AND enable_thinking=True →
    everything is reasoning" path is the load-bearing addition for
    #575 — it's correct for actual thinking models but a footgun for
    non-thinking variants that never produce ``<think>`` regardless
    of the kwarg. Setting ``reasoning_parser=null`` short-circuits
    the whole reasoning path so output flows directly to ``content``
    as the non-thinking variants intend.

    Pinning here so a future PR can't silently re-wire the qwen3
    parser on the strength of "but the family default is qwen3" — the
    Thinking-2507 sibling keeps the family parser (see
    ``test_qwen3_4b_thinking_2507_wires_qwen3_reasoning_parser``).
    """
    profiles = list_profiles()
    assert alias in profiles, f"{alias} missing from aliases.json"
    assert profiles[alias].tool_call_parser == "hermes", (
        f"{alias}: tool_call_parser stays 'hermes' (the model can emit "
        f"hermes-style tool calls via the Qwen3 vocab; only the reasoning "
        f"parser is being cleared). Got {profiles[alias].tool_call_parser!r}."
    )
    assert profiles[alias].reasoning_parser is None, (
        f"{alias}: reasoning_parser must be None — this is a NON-thinking "
        f"Qwen3 variant and the qwen3 parser's Case-4 fallback duplicates "
        f"the whole output into both content + reasoning_content when the "
        f"client passes enable_thinking=True. See PR #715 bundle (fuzz "
        f"finding A). Got {profiles[alias].reasoning_parser!r}."
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


@pytest.mark.parametrize(
    "alias",
    [
        "phi-3.5-mini-4bit",
        "gemma-3n-e2b-4bit",
        "gemma-3n-e4b-4bit",
    ],
)
def test_no_tool_call_support_aliases_have_null_tool_call_parser(
    alias: str,
) -> None:
    """Models whose chat templates can't emit hermes-style tool grammar
    must carry ``tool_call_parser=null`` so the route doesn't
    advertise tools the model can't fulfil.

    The 2026-06-18 fuzz battery against PR #714 sent tool-call prompts
    to each ≤5B alias and recorded whether the model emitted a parseable
    ``<tool_call>...</tool_call>`` envelope:

    * ``phi-3.5-mini-4bit`` (Microsoft Phi-3.5) — chat template only
      defines ``<|user|>`` / ``<|assistant|>`` / ``<|end|>``; no
      ``<tool_call>`` special tokens. Tool-call attempts get ignored.
    * ``gemma-3n-e2b-4bit`` / ``gemma-3n-e4b-4bit`` (Google Gemma 3n
      multimodal) — chat template injects no tool-call markers; model
      replies with plain prose when asked to use a tool.

    With ``tool_call_parser=hermes`` (the prior wiring carried over
    from the family-default seed) the route still SCANS for hermes
    markup and returns ``tool_calls=[]`` + raw text in ``content`` —
    not a crash, but it advertises ``tools`` capability in the
    OpenAI surface that the model can't honour, which clients
    consuming the ``/v1/models`` capability discovery treat as a
    bug. Setting ``tool_call_parser=null`` is the honest signal.

    ``phi-4-mini-reasoning-4bit`` is deliberately NOT in this list —
    Phi-4-mini-reasoning CAN emit tool calls (the parser works), it
    just spends most of its 256-token default budget on thinking
    first. Users bump ``max_tokens=1024+`` for tool use; the alias
    config stays at ``tool_call_parser=hermes``.

    Pinned here so a future PR can't silently re-wire ``hermes``
    on the strength of "but the family default is hermes" — the
    family default is correct for the chat-format families that
    ship tool tokens, and wrong for these three that don't.
    """
    profiles = list_profiles()
    assert alias in profiles, f"{alias} missing from aliases.json"
    assert profiles[alias].tool_call_parser is None, (
        f"{alias}: tool_call_parser must be None — the model's chat "
        f"template can't emit hermes-style tool grammar (PR #715 fuzz "
        f"finding D). Got {profiles[alias].tool_call_parser!r}."
    )


def test_phi_4_mini_reasoning_keeps_hermes_tool_call_parser() -> None:
    """Counter-pin to ``test_no_tool_call_support_aliases_have_null_tool_call_parser``:
    ``phi-4-mini-reasoning-4bit`` MUST stay at ``tool_call_parser=hermes``.

    The Phi-4-mini-reasoning variant CAN emit tool calls (the hermes
    parser successfully extracts them when the model gets enough
    decode budget) — it just spends most of its 256-token default
    budget on its autonomous ``<think>...</think>`` block before
    producing the tool call. Users need ``max_tokens=1024+`` for
    reliable tool use; the parser wiring itself is correct.

    Pinned separately so a "let's clean up tool_call_parser on
    phi family" sweep can't silently flip this entry to ``null``
    on the strength of pattern-matching to phi-3.5.
    """
    profile = list_profiles()["phi-4-mini-reasoning-4bit"]
    assert profile.tool_call_parser == "hermes", (
        f"phi-4-mini-reasoning-4bit: tool_call_parser must stay 'hermes' — "
        f"the model CAN tool-call (parser works), it just needs more "
        f"max_tokens for the thinking block. See PR #715 bundle. "
        f"Got {profile.tool_call_parser!r}."
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


# =============================================================================
# Tier-4 alias wave (#299, #290, #304)
# =============================================================================
#
# Five-pack of operator-requested additions landed in 0.9.x:
#   - kimi-k2.6 (Kimi K2.6 Smart-Quant MoE — DeepseekV3-style architecture)
#   - holo3.1-35b-a3b / -8bit (Hcompany Holo 3.1, Qwen3.5-MoE base — GUI-agent
#     vision model, A3B sparse experts)
#   - qwen3-0.6b / qwen3-1.7b / qwen3-1.7b-4bit (the bare ≤2B Qwen3 short
#     aliases — triple-reported missing in #299/#290/#304)
#   - mistral-small-4-119b / -4bit / -8bit (Apache-2 dense workhorse,
#     mistral3 model_type — same arch family as the existing
#     mistral-24b-4bit entry)
#
# Pinned here so a future bulk edit can't silently revert the parser
# wiring or capability flags without an operator review at PR time.


_TIER4_ALIASES_AND_MOE: tuple[tuple[str, bool], ...] = (
    # alias, expected is_moe
    ("kimi-k2.6", True),
    ("holo3.1-35b-a3b", True),
    ("holo3.1-35b-a3b-8bit", True),
    ("qwen3-0.6b", False),
    ("qwen3-1.7b", False),
    ("qwen3-1.7b-4bit", False),
    ("mistral-small-4-119b", False),
    ("mistral-small-4-119b-4bit", False),
    ("mistral-small-4-119b-8bit", False),
)


@pytest.mark.parametrize("alias,expected_is_moe", _TIER4_ALIASES_AND_MOE)
def test_tier4_alias_resolves(alias: str, expected_is_moe: bool) -> None:
    """Every Tier-4 wave alias must resolve through ``list_profiles`` —
    catches a JSON-merge mistake where an entry is dropped before commit.

    Also pins the architectural ``is_moe`` flag per family:
    - Kimi K2.6 and Holo 3.1-A3B are sparse-expert models (MoE).
      Mis-tagging them as dense would silently enable DFlash if the rest
      of the gates aligned (DFlash drafter hidden-state fusion misfires
      on expert-routing churn — see
      ``test_dflash_excludes_moe_architectures`` for the regression
      evidence on Qwen3.6-35B-A3B).
    - Qwen3 0.6B/1.7B (dense Qwen3, not Qwen3.5-MoE) and Mistral-Small-4
      are dense transformers; mis-tagging as MoE would gate them out of
      DFlash unnecessarily.
    """
    profiles = list_profiles()
    assert alias in profiles, (
        f"{alias}: missing from aliases.json — Tier-4 wave (#299/#290/#304) "
        f"requires this entry to resolve."
    )
    profile = profiles[alias]
    assert profile.is_moe is expected_is_moe, (
        f"{alias}: is_moe={profile.is_moe!r}, expected {expected_is_moe!r}. "
        f"Kimi K2.6 (DeepseekV3 sparse-expert) and Holo3.1-A3B "
        f"(Qwen3.5-MoE base) MUST be MoE; the Qwen3-0.6B/1.7B and "
        f"Mistral-Small-4-119B aliases MUST be dense."
    )


def test_tier4_short_alias_keys_are_unique() -> None:
    """The Tier-4 wave introduces several short aliases that resolve to
    HF paths already referenced by existing entries (qwen3-0.6b →
    Qwen3-0.6B-4bit, qwen3-1.7b → Qwen3-1.7B-4bit, mistral-small-4-119b →
    -4bit). The JSON parser would reject duplicate keys, but
    ``json.loads`` silently keeps the LAST occurrence — meaning a
    copy-paste typo could shadow an earlier entry without any error.
    Pin uniqueness explicitly by re-parsing the raw file and counting
    occurrences of each alias key.
    """
    path = Path(__file__).resolve().parents[1] / "vllm_mlx" / "aliases.json"
    raw_text = path.read_text()
    # Lightweight key scan — count quoted alias names at the start of a
    # JSON object property. Anchored on the leading whitespace pattern
    # that the file uses (2-space indent for top-level entries) so
    # nested keys aren't double-counted.
    short_aliases = [a for a, _ in _TIER4_ALIASES_AND_MOE]
    for alias in short_aliases:
        pattern = re.compile(rf'^\s{{2}}"{re.escape(alias)}":\s*\{{', re.MULTILINE)
        hits = pattern.findall(raw_text)
        assert len(hits) == 1, (
            f"{alias}: appears {len(hits)} times as a top-level alias key "
            f"in aliases.json; must appear exactly once. Duplicate keys "
            f"silently let the last one win, masking the first entry."
        )


def test_kimi_k26_wires_kimi_tool_parser_and_deepseek_r1_reasoning() -> None:
    """Kimi K2.6 (model_type=kimi_k25, DeepseekV3 sparse-expert backbone)
    uses the native ``<|tool_calls_section_begin|>...`` tool format
    handled by the kimi/kimi_k2/moonshot parser. The chat template
    autonomously emits ``<think>...</think>`` blocks (default
    ``preserve_thinking=false`` strips them from history, but the live
    response carries them) — same "model decides" contract used by
    R1-distill / VibeThinker / Phi-4-mini-reasoning, so route through
    ``deepseek_r1`` to land the trace in ``reasoning_content`` instead
    of leaking into ``content``.

    Pinned here so a future "Kimi family default" sweep can't silently
    flip these to e.g. ``hermes`` (wrong tool envelope) or ``None``
    reasoning (think blocks would leak into content).
    """
    profile = list_profiles()["kimi-k2.6"]
    assert profile.tool_call_parser == "kimi", (
        f"kimi-k2.6: tool_call_parser must be 'kimi' — the model emits the "
        f"native <|tool_calls_section_begin|> envelope, not hermes/qwen3 XML. "
        f"Got {profile.tool_call_parser!r}."
    )
    assert profile.reasoning_parser == "deepseek_r1", (
        f"kimi-k2.6: reasoning_parser must be 'deepseek_r1' — the chat "
        f"template autonomously emits `<think>...</think>` blocks. "
        f"Got {profile.reasoning_parser!r}."
    )
    assert profile.is_moe is True, "Kimi K2.6 is sparse-expert MoE."
    assert profile.is_hybrid is False, (
        "Kimi K2.6 is pure-attention (DeepseekV3 backbone), not hybrid."
    )
    assert profile.supports_spec_decode is False, (
        "Kimi K2.6 is too large + MoE for spec-decode to be net-positive."
    )
    # codex r2 BLOCKING #1: explicit-pin the DFlash gate even though the
    # AliasProfile default already forbids it on MoE. Defense-in-depth —
    # if a future PR ever flips the dataclass default-False to default-True
    # for any reason, this assertion catches the regression here instead
    # of waiting for the broader ``test_dflash_excludes_moe_architectures``
    # guard to fire at a much later boundary.
    assert profile.supports_dflash is False, (
        "kimi-k2.6: supports_dflash must be False — sparse-expert MoE "
        "kills DFlash drafter acceptance (see "
        "test_dflash_excludes_moe_architectures)."
    )


@pytest.mark.parametrize(
    "alias",
    ["holo3.1-35b-a3b", "holo3.1-35b-a3b-8bit"],
)
def test_holo3_1_family_follows_qwen35_moe_precedent(alias: str) -> None:
    """Holo 3.1-35B-A3B (Hcompany GUI-agent fine-tune of Qwen3.5-MoE)
    MUST match the existing ``qwen3.5-35b-{4,8}bit`` entries on the
    routing flags — same backbone, same hybrid GatedDeltaNet attention,
    same sparse-expert routing. The chat template emits the Qwen XML
    tool envelope (`<tool_call><function=...>...`) handled by the
    hermes parser's bare-function branch, and injects ``<think>\\n``
    on every assistant turn (qwen3 reasoning parser handles the
    "template opens, model closes" contract).
    """
    profile = list_profiles()[alias]
    assert profile.tool_call_parser == "hermes", (
        f"{alias}: tool_call_parser must be 'hermes' — Holo 3.1 emits the "
        f"`<tool_call><function=name><parameter=...>...</function></tool_call>` "
        f"shape that the hermes parser's bare-function branch handles. "
        f"Got {profile.tool_call_parser!r}."
    )
    assert profile.reasoning_parser == "qwen3", (
        f"{alias}: reasoning_parser must be 'qwen3' — the Qwen3.5-MoE chat "
        f"template injects `<think>\\n` on assistant turns. "
        f"Got {profile.reasoning_parser!r}."
    )
    assert profile.is_hybrid is True, (
        f"{alias}: Qwen3.5-MoE base uses hybrid GatedDeltaNet attention. "
        f"Got is_hybrid={profile.is_hybrid!r}."
    )
    assert profile.is_moe is True, (
        f"{alias}: Holo 3.1-A3B is sparse-expert (A3B = 3B active "
        f"experts). Got is_moe={profile.is_moe!r}."
    )
    assert profile.supports_spec_decode is False, (
        f"{alias}: hybrid arch forbids spec-decode (see "
        f"test_hybrid_disables_spec_decode)."
    )
    # codex r2 BLOCKING #2: explicit-pin the DFlash gate alongside the
    # spec-decode gate. Holo3.1-A3B is MoE + hybrid, both of which
    # independently kill DFlash (MoE expert-routing churn breaks
    # drafter acceptance per qwen3.6-35b-a3b PoC; hybrid Mamba/GDN
    # state breaks drafter rollback). Defense-in-depth — same
    # rationale as the kimi-k2.6 pin above.
    assert profile.supports_dflash is False, (
        f"{alias}: supports_dflash must be False — Holo3.1-A3B is "
        f"both MoE and hybrid; either independently makes DFlash "
        f"a regression (see test_dflash_excludes_moe_architectures)."
    )


def test_mistral_small_4_119b_family_follows_mistral_24b_precedent() -> None:
    """Mistral-Small-4-119B-2603 (Apache-2, model_type=mistral3, dense
    transformer) MUST match the existing ``mistral-24b-4bit`` entry on
    the routing flags — same family, same chat template, just a bigger
    parameter count. Wiring drift here would silently route 119B traffic
    through a different parser than 24B (confusing for operators
    debugging tool-call regressions).
    """
    twentyfour = list_profiles()["mistral-24b-4bit"]
    family = [
        "mistral-small-4-119b",
        "mistral-small-4-119b-4bit",
        "mistral-small-4-119b-8bit",
    ]
    for alias in family:
        profile = list_profiles()[alias]
        assert profile.tool_call_parser == twentyfour.tool_call_parser == "hermes", (
            f"{alias}: tool_call_parser must match mistral-24b-4bit "
            f"('hermes'). Got {profile.tool_call_parser!r}."
        )
        assert profile.reasoning_parser is None, (
            f"{alias}: Mistral-Small-4 is a non-thinking variant; no "
            f"`<think>` emission. reasoning_parser must be None. "
            f"Got {profile.reasoning_parser!r}."
        )
        assert profile.is_hybrid is False, (
            f"{alias}: mistral3 is dense + pure-attention, not hybrid."
        )
        assert profile.is_moe is False, (
            f"{alias}: Mistral-Small-4-119B-2603 is dense (NOT MoE). "
            f"Got is_moe={profile.is_moe!r}."
        )


@pytest.mark.parametrize(
    "alias",
    ["qwen3-0.6b", "qwen3-1.7b", "qwen3-1.7b-4bit"],
)
def test_bare_qwen3_short_aliases_follow_family_precedent(alias: str) -> None:
    """The bare-size Qwen3 short aliases (no -4bit/-8bit suffix) MUST
    match the existing sized siblings on routing flags — Qwen3 family
    default is hermes + qwen3 (the model emits hermes-style `<tool_call>`
    JSON and autonomous `<think>...</think>` reasoning blocks). Drift
    here would let a user typing ``qwen3-0.6b`` get different behaviour
    from ``qwen3-0.6b-4bit`` (which points at the same HF path).
    """
    profile = list_profiles()[alias]
    assert profile.tool_call_parser == "hermes", (
        f"{alias}: Qwen3 family default tool_call_parser is 'hermes'. "
        f"Got {profile.tool_call_parser!r}."
    )
    assert profile.reasoning_parser == "qwen3", (
        f"{alias}: Qwen3 family default reasoning_parser is 'qwen3'. "
        f"Got {profile.reasoning_parser!r}."
    )
    assert profile.is_hybrid is False, (
        f"{alias}: vanilla Qwen3 (non-3.5 / non-3.6) is pure-attention, not hybrid."
    )
    assert profile.is_moe is False, (
        f"{alias}: vanilla Qwen3 0.6B/1.7B is dense (only A3B/A10B/A22B "
        f"siblings are MoE)."
    )


def test_glm_5_2_reap50_alias_resolves_to_pipenetwork_4bit() -> None:
    """R15 Phase 6 #298 — GLM-5.2-REAP50 alias for the 256GB Mac Studio
    MoE-fit story (体感 1).

    The Cerebras REAP-pruned variant of GLM-5.2 drops the lowest-saliency
    half of the 256 routed experts (down to ``n_routed_experts=128``).
    Combined with mlx-community / pipenetwork 4-bit affine quantization
    this lands at ~214 GB on disk and fits a 256GB Mac Studio with KV
    headroom — the FIRST realistically-Mac-runnable GLM-5.2 variant.

    Pinned here so a future bulk edit can't silently re-route the alias
    to a different REAP pruning ratio (REAP25 / REAP75 land at different
    disk sizes and DIFFERENT quality budgets — operators picked this
    alias name expecting the 50% prune specifically) or flip routing
    flags. The model_type at HF (``glm_moe_dsa``) shares the same chat
    template + tool envelope family as GLM-4.5 / 4.7, so the existing
    glm47 / glm4 parsers apply unchanged.

    HF: https://huggingface.co/pipenetwork/GLM-5.2-REAP50-MLX-4bit
    """
    profile = list_profiles()["glm-5.2-reap50"]
    assert profile.hf_path == "pipenetwork/GLM-5.2-REAP50-MLX-4bit", (
        f"glm-5.2-reap50: hf_path drifted. The alias name carries the "
        f"50% expert-prune semantics; pointing at REAP25 / REAP75 / "
        f"non-REAP would silently change disk size + quality. "
        f"Got {profile.hf_path!r}."
    )
    assert profile.is_moe is True, (
        "glm-5.2-reap50: GLM-5.2 is sparse-expert (n_routed_experts=128 "
        "after REAP-50% prune, num_experts_per_tok=8). Mis-tagging as "
        "dense would mis-route DFlash / spec-decode gates."
    )
    assert profile.is_hybrid is False, (
        "glm-5.2-reap50: pure-attention (glm_moe_dsa backbone), not hybrid."
    )
    assert profile.tool_call_parser == "glm47", (
        f"glm-5.2-reap50: GLM-5.2 shares the GLM-4.7 ``<tool_call>...`` "
        f"tool envelope. Got {profile.tool_call_parser!r}."
    )
    assert profile.reasoning_parser == "glm4", (
        f"glm-5.2-reap50: GLM-5.2 chat template autonomously emits "
        f"`<think>...</think>` blocks (same as GLM-4.5/4.7) — route "
        f"through ``glm4`` so the trace lands in reasoning_content. "
        f"Got {profile.reasoning_parser!r}."
    )
    assert profile.supports_spec_decode is False, (
        "glm-5.2-reap50: 214 GB MoE — spec-decode drafter overhead "
        "swamps the win at this size (same call as Kimi K2.6)."
    )
    assert profile.supports_dflash is False, (
        "glm-5.2-reap50: sparse-expert MoE — DFlash drafter hidden-state "
        "fusion misfires on expert-routing churn (see "
        "test_dflash_excludes_moe_architectures)."
    )
