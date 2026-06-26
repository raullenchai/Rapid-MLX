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
        "bonsai-1.7b-unpacked",
        "bonsai-4b-unpacked",
        "bonsai-8b-unpacked",
        "ministral-3b-4bit",
        "nemotron-30b-4bit",
    ):
        profile = resolve_profile(orphan)
        assert profile is not None, f"{orphan} regressed to orphan"


# ---- Loader behaviour -----------------------------------------------------


def test_list_aliases_returns_legacy_string_view() -> None:
    """Old callers (doctor harness, tests) expect ``{alias: hf_path}``.

    Lower-bound (``>=``) on count instead of exact equality: an exact
    count is a merge-conflict magnet — every alias add/remove demands
    a manual bump here, and the count itself isn't a contract. The
    lower bound still catches accidental bulk-deletion of aliases.
    Per-entry contracts are pinned by the focused assertions below."""
    aliases = list_aliases()
    assert len(aliases) >= 65
    assert all(isinstance(p, str) for p in aliases.values())
    assert aliases["qwen3.5-4b-4bit"] == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert aliases["qwen3-0.6b-8bit"] == "mlx-community/Qwen3-0.6B-8bit"


def test_list_profiles_returns_rich_dataclass_view() -> None:
    profiles = list_profiles()
    assert len(profiles) >= 65
    p = profiles["qwen3.5-4b-4bit"]
    assert isinstance(p, AliasProfile)
    assert p.hf_path == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert p.tool_call_parser == "hermes"
    assert p.reasoning_parser == "qwen3"
    # r6-A R6-C1: dense Qwen3.5-4B is no longer hybrid (the metal::malloc
    # wedge surface). The MoE A3B siblings retain ``is_hybrid=True`` —
    # see ``test_qwen35_dense_aliases_not_hybrid`` for the symmetrical
    # guard.
    assert p.is_hybrid is False
    assert p.is_hybrid_explicit is True
    assert p.supports_spec_decode is False

    # qwen3-0.6b-8bit — canonical smoke-test model, registered as a
    # first-class alias so onboarding scripts don't need to bypass
    # the registry with a raw HF path. Standard Qwen3 dense profile.
    p06 = profiles["qwen3-0.6b-8bit"]
    assert isinstance(p06, AliasProfile)
    assert p06.hf_path == "mlx-community/Qwen3-0.6B-8bit"
    assert p06.tool_call_parser == "hermes"
    assert p06.reasoning_parser == "qwen3"
    assert p06.is_hybrid is False
    assert p06.is_moe is False
    assert p06.supports_spec_decode is True


def test_resolve_model_unchanged_for_callers() -> None:
    """Existing callers of ``resolve_model`` must keep getting a string."""
    assert resolve_model("qwen3.5-4b-4bit") == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert (
        resolve_model("mlx-community/Qwen3.5-4B-MLX-4bit")
        == "mlx-community/Qwen3.5-4B-MLX-4bit"
    )
    assert resolve_model("totally-unknown") == "totally-unknown"


# ---- Lookup paths ---------------------------------------------------------


def test_resolve_profile_by_alias_name() -> None:
    p = resolve_profile("qwen3.5-4b-4bit")
    assert p is not None
    assert p.tool_call_parser == "hermes"


def test_resolve_profile_by_hf_path_reverse_lookup() -> None:
    """The reverse lookup is what makes per-alias profiles win when the
    user passes the full HF path on the command line."""
    p = resolve_profile("mlx-community/Qwen3.5-4B-MLX-4bit")
    assert p is not None
    assert p.tool_call_parser == "hermes"
    # r6-A R6-C1: dense Qwen3.5-4B is no longer hybrid (see
    # ``test_list_profiles_returns_rich_dataclass_view`` for the
    # rationale block).
    assert p.is_hybrid is False
    assert p.is_hybrid_explicit is True


def test_resolve_profile_returns_none_for_unknown() -> None:
    assert resolve_profile("totally-unknown") is None
    assert resolve_profile("unaffiliated/Random-Model-2099-99B") is None


# ---- detect_model_config integration -------------------------------------


def test_detect_model_config_prefers_alias_profile_over_regex() -> None:
    """``qwen3.5-4b-4bit`` (alias) and the matching qwen3.5 regex pattern
    happen to agree today, but the alias path is the one we contract on
    — pin a known field that exists on the alias profile so a future
    regex change can't silently take over."""
    cfg = detect_model_config("qwen3.5-4b-4bit")
    assert cfg is not None
    assert cfg.tool_call_parser == "hermes"
    # r6-A R6-C1: dense Qwen3.5-4B is now non-hybrid in the alias
    # profile AND the regex fallback. ``is_hybrid_explicit=True`` is
    # threaded through to gate the runtime ArraysCache probe.
    assert cfg.is_hybrid is False
    assert cfg.is_hybrid_explicit is True
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

    real = ma._aliases["qwen3.5-4b-4bit"]
    forged = AliasProfile(
        hf_path=real.hf_path,
        tool_call_parser="ALIAS_WINS",  # the regex would say "hermes"
        reasoning_parser=real.reasoning_parser,
        is_hybrid=real.is_hybrid,
        supports_spec_decode=real.supports_spec_decode,
    )
    with patch.dict(ma._aliases, {"qwen3.5-4b-4bit": forged}):
        cfg = detect_model_config("qwen3.5-4b-4bit")
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


def test_qwen35_family_split_dense_vs_moe_hybrid_flag() -> None:
    """r6-A R6-C1: post-fix, the qwen3.5-* family splits on the hybrid
    flag — dense variants (4B/9B/27B non-A3B) are ``is_hybrid=False``
    (the metal::malloc wedge surface), MoE A3B/A10B variants stay
    ``is_hybrid=True``. The split is intentional and load-bearing: it
    keeps the dense path off the hybrid scheduler while preserving the
    MoE A3B/A10B routing the prefix-boundary snapshot was originally
    written for. ``supports_spec_decode=False`` continues to hold across
    the entire family because the underlying architecture (GatedDeltaNet
    layers in both dense and MoE) still rules out spec decode regardless
    of the routing choice.
    """
    profiles = list_profiles()
    family = {a: p for a, p in profiles.items() if a.startswith("qwen3.5-")}
    assert len(family) == 15
    # supports_spec_decode is uniformly False across the entire
    # qwen3.5-* family — that contract is unchanged.
    assert {p.supports_spec_decode for p in family.values()} == {False}, (
        f"qwen3.5-* family disagrees on supports_spec_decode: "
        f"{[(a, p.supports_spec_decode) for a, p in family.items()]}"
    )
    # The hybrid flag now splits on MoE markers. Pin the partition so
    # accidentally re-flipping a dense alias to is_hybrid=True trips
    # CI rather than waiting for another dogfood report.
    dense = {a for a, p in family.items() if not p.is_hybrid and not p.is_moe}
    moe_hybrid = {a for a, p in family.items() if p.is_hybrid and p.is_moe}
    assert dense | moe_hybrid == set(family), (
        f"qwen3.5-* family includes an alias that is neither dense-"
        f"non-hybrid nor moe-hybrid: "
        f"{set(family) - dense - moe_hybrid}"
    )
    # Every dense alias must pin is_hybrid_explicit=True so the runtime
    # probe respects the JSON declaration.
    missing_explicit = [a for a in dense if not family[a].is_hybrid_explicit]
    assert not missing_explicit, (
        f"qwen3.5-* dense aliases missing is_hybrid_explicit=True: {missing_explicit}"
    )


def test_per_alias_schema_allows_independent_overrides() -> None:
    """Schema check: two aliases can carry different capability flags
    even if they map to the same family. This is what we couldn't do
    before, and it's the architectural reason for the refactor."""
    profiles = list_profiles()
    p1 = profiles["qwen3.5-4b-4bit"]
    # Object identity check would be wrong; equality on a value-typed
    # dataclass is what we actually want — separate AliasProfile
    # instances per alias means we can mutate one without touching the
    # other. (Mutation isn't supported because the dataclass is frozen,
    # but a re-load with edited JSON would work.)
    assert p1 is not profiles["qwen3.5-9b-4bit"]


# ---- Reverse-lookup behaviour with shared hf_paths -----------------------
#
# The original two tests in this section pinned the duplicate-hf_path
# tie-break for ``(nemotron-30b, nemotron-nano)`` and
# ``(deepseek-v4-flash, deepseek-v4-flash-8bit)``. After the explicit-quant
# alias rename, those codename aliases are gone (see the PR description for
# ``feat/explicit-alias-naming``) and aliases.json no longer has any pair
# pointing at the same hf_path, so the tie-break is unreachable from the
# current registry. The reverse-lookup *mechanism* is still exercised by
# ``test_reverse_lookup_index_built_once_after_first_load`` below.


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
