from __future__ import annotations

import json
from pathlib import Path

import pytest

from rapid_mlx import recommendations as rec
from rapid_mlx.model_aliases import _coerce, list_profiles

ROOT = Path(__file__).resolve().parents[1]
ROWS = json.loads((ROOT / "tests/fixtures/moe_recommendations_golden.json").read_text())


@pytest.mark.parametrize("row", ROWS)
def test_recommendation_shared_chip_vectors(row):
    chip, ram = row["chip"], row["ram_gib"]
    assert rec.chip_class(chip) == row["chip_class"]
    payload = rec.recommendation_payload(ram, chip=chip)
    assert payload["picks"][0]["alias"] == row["primary"]
    assert rec.fitting_pick(ram, 1000, chip=chip) == row["primary"]
    assert (
        rec.select_starter_model(ram, set(list_profiles()), chip=chip) == row["primary"]
    )
    assert rec.select_starter_model(ram, set(), chip=chip) == "lfm2.5-2.6b-4bit"


def test_recommendation_parameter_metadata_survives_catalog_and_payload():
    from rapid_mlx.catalog.legacy import build_legacy_catalog_snapshot

    profiles = list_profiles()
    payload = rec.recommendation_payload(48, chip="Apple M4 Pro")
    moe, dense = payload["picks"]
    assert (moe["total_params_b"], moe["active_params_b"]) == (35, 3)
    assert dense["total_params_b"] is None
    # Memory gates and role meaning remain unchanged when order changes.
    assert [p["role"] for p in payload["picks"]] == ["fast", "smart"]
    assert profiles[moe["alias"]].min_memory_gb == 48
    snapshot = build_legacy_catalog_snapshot()
    rows = {r["alias"]: r for r in snapshot["aliases"]}
    for alias, profile in profiles.items():
        if profile.active_params_b is not None:
            assert rows[alias]["presentation"] == {
                "total_params_million": round(profile.total_params_b * 1000),
                "active_params_million": round(profile.active_params_b * 1000),
            }
    assert profiles["qwen3.6-35b-6bit"].active_params_b == 3


@pytest.mark.parametrize(
    "total,active",
    [(35, None), (None, 3), (True, 3), (35, 0), (35, 36), (float("inf"), 3), (35, "3")],
)
def test_alias_rejects_invalid_parameter_metadata(total, active):
    with pytest.raises(ValueError, match="invalid total_params_b"):
        _coerce(
            "test",
            {
                "hf_path": "test/model",
                "is_moe": True,
                "total_params_b": total,
                "active_params_b": active,
            },
        )


def test_alias_rejects_dense_parameter_label():
    with pytest.raises(ValueError, match="invalid total_params_b"):
        _coerce(
            "test",
            {"hf_path": "test/model", "total_params_b": 35, "active_params_b": 3},
        )


@pytest.mark.parametrize(
    "row",
    json.loads((ROOT / "tests/fixtures/moe_parameter_labels_golden.json").read_text()),
)
def test_recommendation_shared_parameter_labels(row):
    assert rec.parameter_label(row["total"], row["active"]) == row["label"]


def test_recommendation_cli_text_labels(monkeypatch, capsys):
    from argparse import Namespace

    from rapid_mlx import cli, decode_expectations

    monkeypatch.setattr(decode_expectations, "detected_chip", lambda: "M4 Pro")
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 100.0)
    cli.recipe_command(Namespace(max_ram=48, json=False))
    text = capsys.readouterr().out
    assert "1. Fast — qwen3.6-35b-4bit" in text
    assert "35B total · 3B active" in text


@pytest.mark.parametrize(
    "classes", [None, [], ["max"], ["base", "base"], "base", [True]]
)
def test_recommendation_rejects_bad_chip_qualifiers(monkeypatch, classes):
    from copy import deepcopy

    policy = deepcopy(rec._read_recommendation_policy())
    policy["tiers"][-1]["prefer_fast_chip_classes"] = classes
    monkeypatch.setattr(rec, "_read_recommendation_policy", lambda: policy)
    rec.load_recommendation_tiers.cache_clear()
    try:
        with pytest.raises(ValueError, match="invalid prefer_fast_chip_classes"):
            rec.load_recommendation_tiers(validate_catalog=False)
    finally:
        rec.load_recommendation_tiers.cache_clear()


def test_recommendation_old_policy_retains_dense_primary(monkeypatch):
    from copy import deepcopy

    policy = deepcopy(rec._read_recommendation_policy())
    for tier in policy["tiers"]:
        tier.pop("prefer_fast_chip_classes", None)
    monkeypatch.setattr(rec, "_read_recommendation_policy", lambda: policy)
    rec.load_recommendation_tiers.cache_clear()
    try:
        assert (
            rec.recommendation_tier(48, chip="M4 Pro", validate_catalog=False)
            .picks[0]
            .alias
            == "qwen3.8-27b-4bit"
        )
    finally:
        rec.load_recommendation_tiers.cache_clear()
