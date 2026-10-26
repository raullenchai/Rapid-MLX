# SPDX-License-Identifier: Apache-2.0
"""Offline coverage and selection contract for packaged model digests."""

import json
from pathlib import Path

import pytest

from scripts.build_model_digests import OUTPUT, check, selections


def test_packaged_digests_cover_every_mirrored_pin_offline():
    check(json.loads(OUTPUT.read_text()), selections())


def test_missing_or_changed_alias_pin_fails_offline():
    expected = selections()
    payload = json.loads(OUTPUT.read_text())
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="digest schema"):
        check(payload, expected)
    payload["schema_version"] = 1
    alias = "qwen3-0.6b-4bit"
    payload["aliases"][alias]["revision"] = "f" * 40
    with pytest.raises(ValueError, match="revision or selection drift"):
        check(payload, expected)
    del payload["aliases"][alias]
    with pytest.raises(ValueError, match="alias drift"):
        check(payload, expected)


def test_missing_selected_file_fails_offline():
    payload = json.loads(OUTPUT.read_text())
    files = payload["aliases"]["qwen3-0.6b-4bit"]["files"]
    files.pop()
    with pytest.raises(ValueError, match="incomplete digest inventory"):
        check(payload, selections())


def test_image_runtime_pin_must_match_alias_pin(monkeypatch):
    from rapid_mlx._download_gate import IMAGE_MODEL_REVISIONS, SDXL_REPO

    monkeypatch.setitem(IMAGE_MODEL_REVISIONS, SDXL_REPO, "f" * 40)
    with pytest.raises(ValueError, match="image runtime pin differs"):
        selections()


def test_subfolder_inventory_contains_only_selected_files():
    payload = json.loads(Path(OUTPUT).read_text())
    entry = payload["aliases"]["lfm2.5-2.6b-4bit"]
    assert entry["allow_patterns"] == ["4bit/*"]
    assert entry["files"]
    assert all(file["path"].startswith("4bit/") for file in entry["files"])


def test_packaged_pin_size_preflight_never_queries_hub(monkeypatch, tmp_path):
    from rapid_mlx import _download_gate, cli
    from rapid_mlx.model_aliases import resolve_revision

    repo = "mlx-community/Qwen3-0.6B-4bit"
    pin = resolve_revision(repo)
    assert pin is not None
    monkeypatch.setattr(
        _download_gate,
        "_model_info_with_timeout",
        lambda *args: pytest.fail("Hub size metadata queried"),
    )
    assert _download_gate.estimate_repo_size_bytes(repo) > 0
    monkeypatch.setattr(
        "huggingface_hub.model_info",
        lambda *args, **kwargs: pytest.fail("Hub disk metadata queried"),
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    cli._check_disk_space(repo, revision_override=pin)


def test_image_file_selection_disk_preflight_uses_package(monkeypatch, tmp_path):
    from rapid_mlx import cli
    from rapid_mlx._download_gate import (
        IMAGE_MODEL_DATA_FILES,
        IMAGE_MODEL_REVISIONS,
        SDXL_REPO,
    )

    monkeypatch.setattr(
        "huggingface_hub.model_info",
        lambda *args, **kwargs: pytest.fail("Hub image metadata queried"),
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    cli._check_disk_space(
        SDXL_REPO,
        force=True,
        revision_override=IMAGE_MODEL_REVISIONS[SDXL_REPO],
        allow_patterns=list(IMAGE_MODEL_DATA_FILES[SDXL_REPO]),
    )


def test_explicit_packaged_subfolder_selector_skips_hub(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_revision

    repo = "LiquidAI/LFM2.5-2.6B-MLX"
    monkeypatch.setattr(
        "huggingface_hub.HfApi",
        lambda: pytest.fail("Hub variant listing queried"),
    )
    assert cli._resolve_variant_allow_patterns(
        repo, "4", None, resolve_revision(repo)
    ) == ["4bit/*"]
