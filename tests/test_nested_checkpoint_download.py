# SPDX-License-Identifier: Apache-2.0
"""A root checkpoint shipped with per-quantization sibling folders.

The loader reads only the root, so download, disk pricing and the
``--check`` estimate must skip ``4-bit/``-style folders (but keep the
``mtp/`` head sidecar a root checkpoint may load).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from rapid_mlx.byom.preflight import (
    nested_checkpoint_dirs,
    nested_checkpoint_ignore_patterns,
)
from rapid_mlx.cli import _missing_download_bytes, _nested_checkpoint_ignore
from tests.pull_integrity_helpers import (  # noqa: F401
    hub_integrity_metadata,
    snapshot_tree,
)

GB = 1024**3

ROOT_PLUS_VARIANTS = {
    "config.json": 1_000,
    "tokenizer.json": 1_000,
    "model-00001-of-00002.safetensors": 8 * GB,
    "model-00002-of-00002.safetensors": 7 * GB,
    "4-bit/config.json": 1_000,
    "4-bit/model-00001-of-00002.safetensors": 15 * GB,
    "8-bit/config.json": 1_000,
    "8-bit/model.safetensors": 27 * GB,
    "mtp/config.json": 1_000,
    "mtp/model.safetensors": 1 * GB,
}


def test_detects_sibling_quantization_folders_but_keeps_mtp():
    assert nested_checkpoint_dirs(ROOT_PLUS_VARIANTS) == ("4-bit", "8-bit")
    assert nested_checkpoint_ignore_patterns(ROOT_PLUS_VARIANTS) == [
        "4-bit/*",
        "8-bit/*",
    ]


def test_no_root_checkpoint_means_no_narrowing():
    # Subfolder-only repos (and component layouts) need their folders.
    files = ["README.md", "4bit/config.json", "4bit/model.safetensors"]
    assert nested_checkpoint_dirs(files) == ()
    assert nested_checkpoint_ignore_patterns(files) is None


def test_folders_without_a_checkpoint_are_kept():
    files = ["config.json", "model.safetensors", "voices/af.safetensors"]
    assert nested_checkpoint_dirs(files) == ()


def test_glob_metacharacters_in_folder_names_are_escaped():
    files = [
        "config.json",
        "model.safetensors",
        "q[4]/config.json",
        "q[4]/m.safetensors",
    ]
    assert nested_checkpoint_ignore_patterns(files) == ["q[[]4]/*"]


def test_download_estimate_prices_only_the_root_checkpoint():
    info = SimpleNamespace(
        sha="sha",
        siblings=[
            SimpleNamespace(rfilename=name, size=size)
            for name, size in ROOT_PLUS_VARIANTS.items()
        ],
    )
    with (
        patch("huggingface_hub.try_to_load_from_cache", return_value=None),
        patch("huggingface_hub.model_info", return_value=info),
    ):
        missing = _missing_download_bytes("someone/Root-Plus-Variants-MLX")
    assert missing == 16 * GB + 3_000


def test_pull_ignore_patterns_are_best_effort():
    with patch(
        "huggingface_hub.list_repo_files", return_value=list(ROOT_PLUS_VARIANTS)
    ):
        assert _nested_checkpoint_ignore("someone/x", None) == ["4-bit/*", "8-bit/*"]
    with patch("huggingface_hub.list_repo_files", side_effect=OSError("offline")):
        assert _nested_checkpoint_ignore("someone/x", None) is None


def test_estimate_keeps_catalog_subfolder_selection():
    info = SimpleNamespace(
        sha="sha",
        siblings=[
            SimpleNamespace(rfilename=name, size=size)
            for name, size in ROOT_PLUS_VARIANTS.items()
        ],
    )
    with (
        patch("huggingface_hub.try_to_load_from_cache", return_value=None),
        patch("huggingface_hub.model_info", return_value=info),
        patch("rapid_mlx.model_aliases.checkpoint_prefix", return_value="4-bit/"),
        patch("rapid_mlx.model_aliases.subfolder_allow_patterns", return_value=None),
    ):
        missing = _missing_download_bytes("someone/Root-Plus-Variants-MLX")
    assert missing == 15 * GB + 1_000


def test_pull_skips_sibling_checkpoint_folders(
    hub_online_env, ample_pull_disk_space, monkeypatch, tmp_path, capsys
):
    import argparse

    from rapid_mlx import cli

    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *_a, **_kw: False)
    monkeypatch.setattr(cli, "_hf_cache_root", lambda _repo: tmp_path)
    monkeypatch.setattr(
        cli, "_nested_checkpoint_ignore", lambda *_a: ["4-bit/*", "8-bit/*"]
    )
    calls = []

    def download(repo, **kwargs):
        calls.append(kwargs)
        return str(snapshot_tree(tmp_path))

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    cli._pull_repository(
        argparse.Namespace(model="owner/repo", bits=None, format=None),
        revision_override="a" * 40,
    )
    assert calls == [{"revision": "a" * 40, "ignore_patterns": ["4-bit/*", "8-bit/*"]}]
    assert "fetching only the root checkpoint" in capsys.readouterr().out
