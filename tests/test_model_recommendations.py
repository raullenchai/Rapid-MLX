from __future__ import annotations

import json
import shutil
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

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
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 100.0)
    cli.recipe_command(Namespace(max_ram=32, json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["tier_floor_gb"] == 32
    assert [pick["role"] for pick in payload["picks"]] == ["smart", "fast"]
    assert [pick["alias"] for pick in payload["picks"]] == [
        "qwen3.8-27b-4bit",
        "qwen3.5-4b-4bit",
    ]
    # qwen3.8-27b-4bit needs no tier flags — MTP is baked into the alias
    # and the measured 8K peak (20.0 GB) fits the 32 GB tier bare. The
    # flag-carrying pick moved out of the recommendation table with
    # gemma-4-26b; flag propagation is still covered by
    # test_ram_tier_recommendations_agree.py against the SSOT.
    assert payload["picks"][0]["launch_flags"] == []
    assert payload["free_disk_gb"] == 100.0
    assert payload["picks"][0]["disk_fit"] is True
    assert payload["picks"][0]["download_size_gb"] == 15.2
    assert payload["picks"][0]["required_disk_gb"] == 16.72


def test_recipe_text_prints_ready_to_run_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 100.0)
    cli.recipe_command(Namespace(max_ram=18, json=False))
    output = capsys.readouterr().out
    assert "Smart — qwen3.5-9b-4bit" in output
    assert "Fast — qwen3.5-4b-4bit" in output
    assert "rapid-mlx serve qwen3.5-9b-4bit" in output


def test_recipe_suppresses_command_for_pick_that_will_not_fit(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 16.0)

    cli.recipe_command(Namespace(max_ram=32, json=False))

    output = capsys.readouterr().out
    assert "Smart — qwen3.8-27b-4bit" in output
    assert "won't fit: needs ~16.72 GB" in output
    assert "16.00 GB free" in output
    assert "rapid-mlx serve qwen3.8-27b-4bit" not in output
    assert "rapid-mlx serve qwen3.5-4b-4bit" in output


def test_recipe_uses_download_size_not_peak_ram_for_disk_fit(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 18.0)

    cli.recipe_command(Namespace(max_ram=32, json=False))

    output = capsys.readouterr().out
    # The displayed 20.0 GB is measured peak RAM. The manifest records a
    # 15.2 GiB download (~16.7 GiB with headroom), so 18 GiB really does fit.
    assert "Smart — qwen3.8-27b-4bit" in output
    assert "won't fit" not in output
    assert "rapid-mlx serve qwen3.8-27b-4bit" in output


def test_recipe_labels_footprint_as_ram_to_disambiguate_from_disk(
    monkeypatch, capsys
) -> None:
    # The footprint is measured 8K peak RAM, a different axis from the on-disk
    # download size in the won't-fit line. Labeling it "GB RAM" stops the two
    # numbers ("20.0 GB" vs "needs ~16.72 GB") from reading as a contradiction.
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 16.0)

    cli.recipe_command(Namespace(max_ram=32, json=False))

    output = capsys.readouterr().out
    assert "20.0 GB RAM" in output
    # The unlabeled footprint ("20.0 GB ·") must not survive — the RAM label is
    # exactly what tells it apart from the disk requirement printed below.
    assert "20.0 GB ·" not in output


def test_recipe_rounding_does_not_reject_a_download_that_really_fits(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 16.75)

    cli.recipe_command(Namespace(max_ram=32, json=False))

    output = capsys.readouterr().out
    assert "won't fit" not in output
    assert "rapid-mlx serve qwen3.8-27b-4bit" in output


def test_recipe_failing_rounding_boundary_displays_distinct_values(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 16.719)

    cli.recipe_command(Namespace(max_ram=32, json=False))

    output = capsys.readouterr().out
    assert "needs ~16.72 GB including download headroom; 16.71 GB free" in output
    assert "rapid-mlx serve qwen3.8-27b-4bit" not in output


def test_recipe_complete_cache_does_not_require_free_download_space(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        cli,
        "_scan_hf_cache_models",
        lambda: [("rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX", 20 << 30, 0.0)],
    )
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda _repo: True)
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 1.0)

    cli.recipe_command(Namespace(max_ram=48, json=True))

    payload = json.loads(capsys.readouterr().out)
    smart = payload["picks"][0]
    assert smart["cached"] is True
    assert smart["download_size_gb"] == 0.0
    assert smart["required_disk_gb"] == 0.0
    assert smart["disk_fit"] is True


def test_recipe_unknown_disk_space_does_not_hide_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: None)

    cli.recipe_command(Namespace(max_ram=48, json=False))

    output = capsys.readouterr().out
    assert "won't fit" not in output
    assert "rapid-mlx serve qwen3.8-27b-4bit" in output


def test_recipe_unknown_download_size_does_not_claim_fit(monkeypatch, capsys) -> None:
    from vllm_mlx import model_sizes

    monkeypatch.setattr(cli, "_scan_hf_cache_models", lambda: [])
    monkeypatch.setattr(cli, "_recipe_free_disk_gb", lambda: 1.0)
    monkeypatch.setattr(model_sizes, "size_bytes", lambda _repo: None)

    cli.recipe_command(Namespace(max_ram=32, json=True))

    payload = json.loads(capsys.readouterr().out)
    assert all(pick["disk_fit"] is None for pick in payload["picks"])
    assert all(pick["required_disk_gb"] is None for pick in payload["picks"])


def test_recipe_disk_probe_uses_hf_cache_filesystem_and_existing_ancestor(
    monkeypatch, tmp_path
) -> None:
    from huggingface_hub import constants

    cache_volume = tmp_path / "external-volume"
    cache_volume.mkdir()
    monkeypatch.setattr(
        constants, "HF_HUB_CACHE", str(cache_volume / "not-created" / "hub")
    )
    seen = []

    def fake_disk_usage(path):
        seen.append(path)
        return SimpleNamespace(free=17 * (1 << 30))

    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)

    assert cli._recipe_free_disk_gb() == 17.0
    assert seen == [cache_volume]


def test_recipe_disk_probe_failure_is_unknown(monkeypatch, tmp_path) -> None:
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))

    def fail(_path):
        raise OSError("volume disappeared")

    monkeypatch.setattr(shutil, "disk_usage", fail)
    assert cli._recipe_free_disk_gb() is None


def test_recipe_inaccessible_cache_path_is_unknown(monkeypatch) -> None:
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", "/inaccessible/hf/hub")

    def fail_exists(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "exists", fail_exists)
    assert cli._recipe_free_disk_gb() is None
