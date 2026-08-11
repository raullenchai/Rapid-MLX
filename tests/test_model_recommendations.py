from __future__ import annotations

import json
from argparse import Namespace

from vllm_mlx import cli
from vllm_mlx.model_aliases import list_aliases
from vllm_mlx.recommendations import load_recommendation_tiers, recommendation_tier


def test_every_tier_has_exactly_smart_and_fast() -> None:
    tiers = load_recommendation_tiers()
    assert [tier.floor_gb for tier in tiers] == [8, 16, 18, 24, 32, 48, 64, 96]
    aliases = set(list_aliases())
    for tier in tiers:
        assert [pick.role for pick in tier.picks] == ["smart", "fast"]
        assert all(pick.alias in aliases for pick in tier.picks)
        assert all(pick.footprint_gb < tier.floor_gb * 0.75 for pick in tier.picks)


def test_tier_rounds_down_and_clamps() -> None:
    assert recommendation_tier(4).floor_gb == 8
    assert recommendation_tier(20).floor_gb == 18
    assert recommendation_tier(256).floor_gb == 96


def test_recipe_json_is_stable_and_has_two_picks(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    cli.recipe_command(Namespace(max_ram=32, json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["tier_floor_gb"] == 32
    assert [pick["role"] for pick in payload["picks"]] == ["smart", "fast"]
    assert [pick["alias"] for pick in payload["picks"]] == [
        "gemma-4-26b-4bit",
        "qwen3.5-4b-4bit",
    ]
    assert payload["picks"][0]["launch_flags"] == [
        "--no-mllm",
        "--kv-cache-dtype",
        "bf16",
        "--cache-memory-mb",
        "512",
    ]


def test_recipe_text_prints_ready_to_run_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    cli.recipe_command(Namespace(max_ram=18, json=False))
    output = capsys.readouterr().out
    assert "Smart — qwen3.5-9b-4bit" in output
    assert "Fast — qwen3.5-4b-4bit" in output
    assert "rapid-mlx serve qwen3.5-9b-4bit" in output
