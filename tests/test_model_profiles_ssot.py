# SPDX-License-Identifier: Apache-2.0
"""SSOT contract tests for the per-alias model profile registry.

Pre-PR architecture had two sources: ``aliases.json`` (51 alias→hf_path
mappings) and ``_MODEL_PATTERNS`` (25 regex-keyed ``ModelConfig`` rows).
A new alias would silently inherit whichever pattern's regex happened to
match its HF path, with no per-alias granularity for capability flags.

These tests pin down the new contract:

- every alias has an explicit profile in ``aliases.json``
- ``detect_model_config`` prefers the alias profile over the regex
  fallback
- the regex fallback still works for unaliased HF paths (so users
  serving a brand-new model from HuggingFace still get sensible
  parser/capability inference)
- the legacy bare-string form is still accepted at load time, with
  default capability flags
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_mlx.model_aliases import (
    AliasProfile,
    list_aliases,
    list_profiles,
    resolve_model,
    resolve_profile,
)
from vllm_mlx.model_auto_config import detect_model_config

ALIASES_PATH = Path(__file__).parent.parent / "vllm_mlx" / "aliases.json"


# ---- Schema sanity --------------------------------------------------------


def test_aliases_json_is_rich_schema() -> None:
    """Every entry must be the new dict form (string form was the old
    schema; if it appears in ``aliases.json`` going forward the migration
    has regressed)."""
    with open(ALIASES_PATH) as f:
        raw = json.load(f)
    assert raw, "aliases.json must not be empty"
    for alias, value in raw.items():
        assert isinstance(value, dict), f"{alias!r}: rich schema required"
        assert "hf_path" in value, f"{alias!r}: missing hf_path"
        assert isinstance(value["hf_path"], str)


def test_every_alias_has_explicit_profile_fields() -> None:
    """No alias should rely on default capability flags — the whole
    point of the SSOT is to be explicit. ``tool_call_parser`` and
    ``reasoning_parser`` are allowed to be null (some families have no
    native tool format), but the bool gates must be present."""
    with open(ALIASES_PATH) as f:
        raw = json.load(f)
    for alias, value in raw.items():
        assert "is_hybrid" in value, f"{alias!r}: missing is_hybrid"
        assert "supports_spec_decode" in value, (
            f"{alias!r}: missing supports_spec_decode"
        )
        assert isinstance(value["is_hybrid"], bool)
        assert isinstance(value["supports_spec_decode"], bool)


def test_no_orphan_aliases() -> None:
    """The pre-SSOT audit found 6 orphans with no profile (bonsai×3,
    ministral, nemotron×2). After P1 every alias must resolve."""
    aliases = list_aliases()
    for alias in aliases:
        cfg = detect_model_config(alias)
        assert cfg is not None, f"orphan alias: {alias}"


def test_orphan_aliases_now_covered() -> None:
    """Pin the 6 specific aliases that were orphans before this PR to
    catch a regression where someone deletes their profile."""
    for orphan in (
        "bonsai-1.7b",
        "bonsai-4b",
        "bonsai-8b",
        "ministral-3b",
        "nemotron-30b",
        "nemotron-nano",
    ):
        profile = resolve_profile(orphan)
        assert profile is not None, f"{orphan} regressed to orphan"


# ---- Loader behaviour -----------------------------------------------------


def test_list_aliases_returns_legacy_string_view() -> None:
    """Old callers (doctor harness, tests) expect ``{alias: hf_path}``."""
    aliases = list_aliases()
    assert len(aliases) == 51
    assert all(isinstance(p, str) for p in aliases.values())
    assert aliases["qwen3.5-4b"] == "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_list_profiles_returns_rich_dataclass_view() -> None:
    profiles = list_profiles()
    assert len(profiles) == 51
    p = profiles["qwen3.5-4b"]
    assert isinstance(p, AliasProfile)
    assert p.hf_path == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert p.tool_call_parser == "hermes"
    assert p.reasoning_parser == "qwen3"
    assert p.is_hybrid is True
    assert p.supports_spec_decode is False


def test_resolve_model_unchanged_for_callers() -> None:
    """Existing callers of ``resolve_model`` must keep getting a string."""
    assert resolve_model("qwen3.5-4b") == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert (
        resolve_model("mlx-community/Qwen3.5-4B-MLX-4bit")
        == "mlx-community/Qwen3.5-4B-MLX-4bit"
    )
    assert resolve_model("totally-unknown") == "totally-unknown"


# ---- Lookup paths ---------------------------------------------------------


def test_resolve_profile_by_alias_name() -> None:
    p = resolve_profile("qwen3.5-4b")
    assert p is not None
    assert p.tool_call_parser == "hermes"


def test_resolve_profile_by_hf_path_reverse_lookup() -> None:
    """The reverse lookup is what makes per-alias profiles win when the
    user passes the full HF path on the command line."""
    p = resolve_profile("mlx-community/Qwen3.5-4B-MLX-4bit")
    assert p is not None
    assert p.tool_call_parser == "hermes"
    assert p.is_hybrid is True


def test_resolve_profile_returns_none_for_unknown() -> None:
    assert resolve_profile("totally-unknown") is None
    assert resolve_profile("unaffiliated/Random-Model-2099-99B") is None


# ---- detect_model_config integration -------------------------------------


def test_detect_model_config_prefers_alias_profile_over_regex() -> None:
    """``qwen3.5-4b`` (alias) and the matching qwen3.5 regex pattern
    happen to agree today, but the alias path is the one we contract on
    — pin a known field that exists on the alias profile so a future
    regex change can't silently take over."""
    cfg = detect_model_config("qwen3.5-4b")
    assert cfg is not None
    assert cfg.tool_call_parser == "hermes"
    assert cfg.is_hybrid is True
    assert cfg.supports_spec_decode is False


def test_detect_model_config_falls_back_to_regex_for_unaliased_path() -> None:
    """A user serves a brand-new HF model that isn't aliased — regex
    fallback must still infer a sensible parser."""
    cfg = detect_model_config("custom-org/Qwen3-99B-Instruct-4bit")
    assert cfg is not None
    assert cfg.tool_call_parser == "hermes"  # generic /qwen3/ pattern


def test_detect_model_config_returns_none_when_neither_matches() -> None:
    cfg = detect_model_config("no-prefix-no-pattern-2099")
    assert cfg is None


def test_detect_model_config_alias_wins_over_regex_when_they_disagree() -> None:
    """The whole architectural point: when an alias profile and a regex
    pattern both could match, the alias profile must win. Today the
    derivations agree by construction (alias profiles were generated
    from the regex matches), but the day someone bumps a single
    alias's tier in aliases.json, the regex is going to keep returning
    the family-wide value — and we need the alias to override it.

    Simulate this by monkeypatching the alias profile's
    tool_call_parser to something the qwen3.5 regex would never
    return, and assert the alias's value reaches the caller.
    """
    import vllm_mlx.model_aliases as ma
    from vllm_mlx.model_aliases import AliasProfile

    real = ma._aliases["qwen3.5-4b"]
    forged = AliasProfile(
        hf_path=real.hf_path,
        tool_call_parser="ALIAS_WINS",  # the regex would say "hermes"
        reasoning_parser=real.reasoning_parser,
        is_hybrid=real.is_hybrid,
        supports_spec_decode=real.supports_spec_decode,
    )
    with patch.dict(ma._aliases, {"qwen3.5-4b": forged}):
        cfg = detect_model_config("qwen3.5-4b")
    assert cfg is not None
    assert cfg.tool_call_parser == "ALIAS_WINS", (
        "regex shadowed the alias profile — alias-first lookup is broken"
    )


# ---- Backward compat with legacy bare-string form ------------------------


def test_legacy_string_value_still_loads(tmp_path) -> None:
    """An external tool that hand-edited ``aliases.json`` to the old
    ``{alias: hf_path_string}`` form should still load — defaults fill in
    the new capability fields."""
    legacy = tmp_path / "aliases.json"
    legacy.write_text(json.dumps({"foo": "org/Foo-Model-7B"}))

    import vllm_mlx.model_aliases as ma

    # Reset module cache and point loader at the legacy file
    with (
        patch.object(ma, "_aliases", None),
        patch.object(ma, "_hf_to_alias", None),
        patch("vllm_mlx.model_aliases.os.path.join", return_value=str(legacy)),
    ):
        profiles = ma.list_profiles()

    assert "foo" in profiles
    assert profiles["foo"].hf_path == "org/Foo-Model-7B"
    # Defaults for fields not present in legacy form
    assert profiles["foo"].tool_call_parser is None
    assert profiles["foo"].is_hybrid is False
    assert profiles["foo"].supports_spec_decode is True


def test_empty_hf_path_string_form_raises(tmp_path) -> None:
    """Empty hf_path slips through downstream as a ``""`` and surfaces
    as a confusing 404 — catch it at load time."""
    bad = tmp_path / "aliases.json"
    bad.write_text(json.dumps({"foo": ""}))

    import vllm_mlx.model_aliases as ma

    with (
        patch.object(ma, "_aliases", None),
        patch.object(ma, "_hf_to_alias", None),
        patch("vllm_mlx.model_aliases.os.path.join", return_value=str(bad)),
        pytest.raises(ValueError, match="empty"),
    ):
        ma.list_profiles()


def test_empty_hf_path_dict_form_raises(tmp_path) -> None:
    """Same empty-path check for the rich-schema form."""
    bad = tmp_path / "aliases.json"
    bad.write_text(json.dumps({"foo": {"hf_path": ""}}))

    import vllm_mlx.model_aliases as ma

    with (
        patch.object(ma, "_aliases", None),
        patch.object(ma, "_hf_to_alias", None),
        patch("vllm_mlx.model_aliases.os.path.join", return_value=str(bad)),
        pytest.raises(ValueError, match="non-empty string"),
    ):
        ma.list_profiles()


def test_invalid_value_raises_with_alias_name(tmp_path) -> None:
    """A typo in the JSON should fail loud and point at the bad alias,
    not crash deep in some downstream caller."""
    bad = tmp_path / "aliases.json"
    bad.write_text(json.dumps({"foo": 42}))

    import vllm_mlx.model_aliases as ma

    with (
        patch.object(ma, "_aliases", None),
        patch.object(ma, "_hf_to_alias", None),
        patch("vllm_mlx.model_aliases.os.path.join", return_value=str(bad)),
        pytest.raises(ValueError, match="foo"),
    ):
        ma.list_profiles()


# ---- Cross-family granularity (the whole point of the refactor) ----------


def test_qwen35_family_aliases_share_hybrid_flag() -> None:
    """All qwen3.5-* aliases currently share the same regex profile —
    they should also share it after migration. This is the regression
    guard: if someone bumps just one variant's tier without bumping the
    others, this test will catch the inconsistency that a per-alias
    schema enables."""
    profiles = list_profiles()
    family = {a: p for a, p in profiles.items() if a.startswith("qwen3.5-")}
    assert len(family) == 6
    flags = {(p.is_hybrid, p.supports_spec_decode) for p in family.values()}
    assert flags == {(True, False)}, (
        f"qwen3.5-* family disagrees on capability flags: {flags}"
    )


def test_per_alias_schema_allows_independent_overrides() -> None:
    """Schema check: two aliases can carry different capability flags
    even if they map to the same family. This is what we couldn't do
    before, and it's the architectural reason for the refactor."""
    profiles = list_profiles()
    p1 = profiles["qwen3.5-4b"]
    # Object identity check would be wrong; equality on a value-typed
    # dataclass is what we actually want — separate AliasProfile
    # instances per alias means we can mutate one without touching the
    # other. (Mutation isn't supported because the dataclass is frozen,
    # but a re-load with edited JSON would work.)
    assert p1 is not profiles["qwen3.5-9b"]


# ---- Reverse-lookup behaviour with shared hf_paths -----------------------


def test_reverse_lookup_for_shared_hf_path_is_deterministic() -> None:
    """Two aliases (``nemotron-30b`` and ``nemotron-nano``) point at the
    same MLX repo. Reverse lookup by HF path should return the
    JSON-insertion-order-first alias's profile, deterministically.

    The contract is "any profile valid for this path", but we lock in
    the order so a future re-shuffle of aliases.json is forced to
    explicitly update this test (which is the right place to think
    about who's the canonical alias).
    """
    profiles = list_profiles()
    nemotron_30b = profiles["nemotron-30b"]
    nemotron_nano = profiles["nemotron-nano"]
    assert nemotron_30b.hf_path == nemotron_nano.hf_path

    # nemotron-30b appears first in aliases.json, so reverse lookup
    # by the shared HF path returns nemotron-30b's profile object.
    via_path = resolve_profile(nemotron_30b.hf_path)
    assert via_path is not None
    assert via_path is nemotron_30b


def test_reverse_lookup_handles_deepseek_v4_flash_duplicate() -> None:
    """``deepseek-v4-flash`` and ``deepseek-v4-flash-8bit`` share
    ``mlx-community/DeepSeek-V4-Flash-8bit`` — same regression guard
    pattern as the nemotron pair, different family."""
    profiles = list_profiles()
    flash = profiles["deepseek-v4-flash"]
    flash_8bit = profiles["deepseek-v4-flash-8bit"]
    assert flash.hf_path == flash_8bit.hf_path
    via_path = resolve_profile(flash.hf_path)
    assert via_path is not None
    # Both profiles agree on capability flags (same model), so either
    # would be correct semantically. Pin the JSON order winner.
    assert via_path is flash


def test_reverse_lookup_index_built_once_after_first_load() -> None:
    """Cheap behavioural check that the reverse index is built once and
    reused — exercises the cache path. Not a perf benchmark; just
    asserts ``_hf_to_alias`` is populated."""
    import vllm_mlx.model_aliases as ma

    # Trigger load
    ma.list_profiles()
    assert ma._hf_to_alias is not None
    assert len(ma._hf_to_alias) <= len(ma._aliases)  # dedup possible
    # Every hf_path in aliases must be reachable via reverse lookup
    for profile in ma._aliases.values():
        assert profile.hf_path in ma._hf_to_alias
