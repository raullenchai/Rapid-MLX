"""#3115: ``qwen3.5-4b-4bit`` ships its MTP preset default-off.

Measured single-stream decode with the k=2 MTP drafter was 25-37% slower on
M2 Pro (32 GB) and M3 Ultra, and concurrency showed no gain, so the product
default is off while the continuous-MTP qualification tier stays ``verified``
(explicit opt-in still runs the qualified route).  The new catalog field
``mtp_default_enabled`` carries that default independently of the tier.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vllm_mlx import cli, model_aliases


def _args(model: str, payload: str | None = None, **overrides) -> SimpleNamespace:
    base = dict(
        model=model,
        speculative_config=payload,
        enable_ddtree=False,
        enable_dflash=False,
        enable_mtp=False,
        no_spec_decode=False,
        spec_decode="none",
        dflash_drafter_path="",
        mtp_num_draft_tokens=1,
        mtp_optimistic=False,
        mtp_sidecar=None,
        mtp_max_k=None,
        mtp_disable_auto_k=False,
        force_spec_decode=False,
        suffix_max_tree_depth=None,
        suffix_max_spec_factor=None,
        suffix_max_spec_offset=None,
        suffix_max_cached_requests=None,
        suffix_max_suffix_len=None,
        suffix_min_confidence=None,
        suffix_min_draft_len=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --- catalog -----------------------------------------------------------------


def test_catalog_ships_4b_default_off_but_qualified() -> None:
    profile = model_aliases.resolve_profile("qwen3.5-4b-4bit")
    assert profile is not None
    assert profile.mtp_continuous_batching_tier == "verified"
    assert profile.mtp_default_enabled is False
    assert profile.mtp_draft_model  # preset still declared for opt-in


@pytest.mark.parametrize(
    "alias", ["qwen3.5-9b-4bit", "qwen3.6-27b-4bit", "qwen3.8-27b-4bit"]
)
def test_other_verified_artifacts_keep_default_on(alias: str) -> None:
    profile = model_aliases.resolve_profile(alias)
    assert profile is not None
    assert profile.mtp_continuous_batching_tier == "verified"
    assert profile.mtp_default_enabled is True


def test_field_defaults_true_and_rejects_non_bool() -> None:
    from vllm_mlx.model_aliases import _coerce as _alias_profile_from_entry

    base = {
        "hf_path": "org/model",
        "supports_native_mtp": True,
        "mtp_speculative_tokens": 2,
    }
    assert _alias_profile_from_entry("x", dict(base)).mtp_default_enabled is True
    assert (
        _alias_profile_from_entry(
            "x", dict(base, mtp_default_enabled=False)
        ).mtp_default_enabled
        is False
    )
    with pytest.raises(ValueError, match="mtp_default_enabled"):
        _alias_profile_from_entry("x", dict(base, mtp_default_enabled="no"))


# --- CLI auto-select ---------------------------------------------------------


def test_serve_4b_without_flags_stays_on_plain_decode() -> None:
    args = _args("qwen3.5-4b-4bit")
    cli._normalize_speculative_config_or_exit(args)
    assert args._speculative_config is None
    assert args.spec_decode == "none"


def test_serve_9b_without_flags_still_auto_selects_mtp() -> None:
    args = _args("qwen3.5-9b-4bit")
    cli._normalize_speculative_config_or_exit(args)
    assert args._speculative_config is not None
    assert args._speculative_config.method == "mtp"
    assert args.mtp_continuous_batching is True


def test_explicit_opt_in_runs_qualified_continuous_route() -> None:
    args = _args("qwen3.5-4b-4bit", json.dumps({"method": "mtp"}))
    cli._normalize_speculative_config_or_exit(args)
    assert args._speculative_config.method == "mtp"
    assert args.spec_decode == "mtp"
    assert args.mtp_continuous_batching_tier == "verified"
    assert args.mtp_continuous_batching is True


def test_default_helper_fails_closed(monkeypatch) -> None:
    assert cli._alias_mtp_default_enabled(None) is False
    assert cli._alias_mtp_default_enabled("someone/unknown") is False

    def _raise(_model: str):
        raise RuntimeError("broken alias registry")

    monkeypatch.setattr(model_aliases, "resolve_profile", _raise)
    assert cli._alias_mtp_default_enabled("qwen3.5-9b-4bit") is False


# --- Desktop contract --------------------------------------------------------


def test_models_json_carries_default_flag() -> None:
    payload = cli._available_models_json_payload()
    rows = {
        row["alias"]: row
        for section in payload.values()
        if isinstance(section, list)
        for row in section
    }
    assert rows["qwen3.5-4b-4bit"]["mtp_default_enabled"] is False
    assert rows["qwen3.5-4b-4bit"]["mtp_continuous_batching_tier"] == "verified"
    assert rows["qwen3.5-9b-4bit"]["mtp_default_enabled"] is True
