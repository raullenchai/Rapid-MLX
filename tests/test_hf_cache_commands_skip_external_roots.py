# SPDX-License-Identifier: Apache-2.0
"""``pull`` and ``rm`` operate on the Hub repo id, never an external folder.

A complete model inside a default external model folder is served in place
by ``serve`` (zero download). The commands that download into or delete from
the Hugging Face cache must keep resolving the alias to its Hub repo id:
handing the downloader a filesystem path crashed ``pull`` with a repo-id
validation traceback, and ``rm`` reported the folder as missing from the
cache.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rapid_mlx import cli, model_aliases
from tests.pull_integrity_helpers import (  # noqa: F401
    hub_integrity_metadata,
    snapshot_tree,
)

ALIAS = "deepseek-r1-8b-4bit"
REPO = "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"


@pytest.fixture
def external_model(monkeypatch, tmp_path) -> Path:
    """A fake HOME whose default external folder holds a complete model."""
    home = tmp_path / "home"
    model = home / ".lmstudio" / "models" / REPO
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type": "qwen3"}')
    (model / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("RAPID_MLX_EXTRA_MODEL_ROOTS", raising=False)
    monkeypatch.setattr(
        model_aliases,
        "DEFAULT_EXTERNAL_MODEL_ROOTS",
        ("~/.lmstudio/models", "~/.cache/lm-studio/models"),
    )
    monkeypatch.setattr(
        model_aliases, "_managed_hub_model_is_runnable", lambda _name: False
    )
    assert model_aliases.resolve_model(ALIAS) == os.path.realpath(model)
    return model


@pytest.mark.parametrize("name", [ALIAS, REPO])
def test_resolver_without_external_roots_returns_hub_id(external_model, name):
    assert model_aliases.resolve_model(name) == os.path.realpath(external_model)
    assert model_aliases.resolve_hub_model(name) == REPO


def test_serve_still_uses_external_folder(external_model, monkeypatch):
    reached = []
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", ALIAS])
    monkeypatch.setattr(cli, "serve_command", lambda args: reached.append(args.model))

    cli.main()

    assert reached == [os.path.realpath(external_model)]


def test_pull_downloads_hub_repo_not_external_folder(
    hub_online_env, external_model, monkeypatch, tmp_path, capsys
):
    revision = model_aliases.resolve_revision(REPO)
    repo_root = tmp_path / "hub" / f"models--{REPO.replace('/', '--')}"
    (repo_root / "refs").mkdir(parents=True)
    (repo_root / "refs" / "main").write_text(revision)
    (repo_root / "blobs").mkdir()
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
    downloads = []

    def _download(*args, **kwargs):
        downloads.append(kwargs.get("repo_id", args[0] if args else None))
        (repo_root / "blobs" / "cafe").write_bytes(b"\x00" * 2048)
        return str(snapshot_tree(repo_root, revision))

    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "pull", ALIAS])
    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", side_effect=_download),
    ):
        cli.main()

    assert downloads == [REPO]
    out = capsys.readouterr().out
    assert f"Alias: {ALIAS} → {REPO}" in out
    assert str(external_model) not in out


def test_rm_targets_hf_cache_not_external_folder(external_model, monkeypatch, capsys):
    deleted = []
    executed = []
    repo = SimpleNamespace(
        repo_id=REPO,
        repo_type="model",
        size_on_disk=2048,
        revisions=[SimpleNamespace(commit_hash="rev1")],
    )
    cache = SimpleNamespace(
        repos=[repo],
        delete_revisions=lambda *revs: (
            deleted.extend(revs)
            or SimpleNamespace(execute=lambda: executed.append(True))
        ),
    )
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "rm", ALIAS, "-y"])

    cli.main()

    assert deleted == ["rev1"]
    assert executed == [True]
    out = capsys.readouterr().out
    assert "not in the HuggingFace cache" not in out
    assert "Freed" in out
    assert (external_model / "model.safetensors").exists()
