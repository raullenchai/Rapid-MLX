# SPDX-License-Identifier: Apache-2.0
"""Tests for #2340 — serve/load resolution of a ``--bits/--format`` pulled variant.

``rapid-mlx pull --bits 4 <multi-variant-repo>`` fetches only ``4bit/``. A later
``serve <repo>`` must reach that same subfolder, not the weightless repo root.
The pulled variant is persisted as a small marker in the HF cache at pull time
(``_download_gate.persist_pulled_variant``) and recovered at load time
(``_resolve_subfolder_checkpoint`` ``pulled_variant`` fallback).

These tests pin that:

1. ``persist_pulled_variant`` / ``pulled_variant`` round-trip: a narrowed pull
   records the folder, while a later ordinary pull clears that choice.
2. the core #2340 scenario: a repo id pulled with ``--bits`` resolves, at load
   time, to that exact snapshot subfolder, even when a catalog reverse lookup
   names a different default subfolder.
3. without a recorded variant (no ``--bits/--format`` pull), the same raw repo
   id resolves to the repo root as before — the marker never invents one.
4. a genuinely absent variant still raises the actionable message (the pulled
   marker points at a folder that is neither present nor complete).

Both an unregistered repo and the real cataloged multi-variant layout are
covered. ``snapshot_download`` and the marker path are mocked; no weights and
no network are touched.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub import RepoFile, RepoFolder

from vllm_mlx import _download_gate
from vllm_mlx.utils.tokenizer import _resolve_subfolder_checkpoint

# A raw multi-variant repo id with NO catalog alias — ``resolve_subfolder``
# and ``resolve_model`` both leave it untouched, so only the marker fallback
# can select the subfolder.
RAW_REPO = "acme/MultiVariant-MLX"
CATALOG_REPO = "LiquidAI/LFM2.5-2.6B-MLX"
CATALOG_ALIAS = "lfm2.5-2.6b-4bit"


def _multi_variant_tree():
    """A multi-variant repo layout: several quant folders + root README only."""
    return [
        RepoFolder(path="4bit", oid="a"),
        RepoFolder(path="8bit", oid="b"),
        RepoFile(path="README.md", size=100, oid="c"),
    ]


@pytest.fixture
def marker_path(tmp_path, monkeypatch):
    """Point the variant marker at a fresh tmp file and return its path."""
    target = tmp_path / "models--acme--MultiVariant-MLX" / ".rapid-mlx" / "variant"
    monkeypatch.setattr(
        _download_gate, "_variant_marker_path", lambda repo_id: str(target)
    )
    return target


def test_persist_and_read_round_trip(marker_path):
    assert _download_gate.pulled_variant(RAW_REPO) is None
    assert _download_gate.persist_pulled_variant(RAW_REPO, "4bit") is True
    assert _download_gate.pulled_variant(RAW_REPO) == "4bit"
    assert marker_path.read_text() == "4bit"


def test_persist_overwrites_a_previous_variant(marker_path):
    _download_gate.persist_pulled_variant(RAW_REPO, "8bit")
    _download_gate.persist_pulled_variant(RAW_REPO, "4bit")
    assert _download_gate.pulled_variant(RAW_REPO) == "4bit"


def test_persist_empty_is_a_noop(marker_path):
    assert _download_gate.persist_pulled_variant(RAW_REPO, "") is False
    assert _download_gate.pulled_variant(RAW_REPO) is None
    assert not marker_path.exists()


def test_clear_removes_a_previous_variant(marker_path):
    _download_gate.persist_pulled_variant(RAW_REPO, "8bit")

    _download_gate.clear_pulled_variant(RAW_REPO)

    assert _download_gate.pulled_variant(RAW_REPO) is None
    assert not marker_path.exists()


def test_clear_missing_marker_is_a_noop(marker_path):
    _download_gate.clear_pulled_variant(RAW_REPO)
    assert not marker_path.exists()


@pytest.mark.parametrize(
    "invalid", ["../escape", "/absolute", "C:/drive", "4bit\\escape", "4bit/"]
)
def test_invalid_marker_values_fail_closed(marker_path, invalid):
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(invalid)
    assert _download_gate.pulled_variant(RAW_REPO) is None


def test_glob_metacharacters_round_trip_as_literal_variant(marker_path):
    _download_gate.persist_pulled_variant(RAW_REPO, "quant[4]*?")
    assert _download_gate.pulled_variant(RAW_REPO) == "quant[4]*?"


def test_missing_marker_returns_none_without_error(marker_path):
    # No task-scoped file means "never narrowed" — never an exception.
    assert _download_gate.pulled_variant(RAW_REPO) is None


def test_serve_resolves_a_pulled_bits_variant(monkeypatch, tmp_path, marker_path):
    """The core #2340 scenario: ``pull --bits 4 RAW_REPO`` then serve RAW_REPO.

    The repo has no catalog alias, so the ONLY thing that can select the
    subfolder is the persisted marker. The served load must land on the
    ``4bit/`` snapshot subfolder (with the ``4bit/*`` narrow allow-pattern),
    not on the weightless repo root.
    """
    snapshot = tmp_path / "snapshots" / "deadbeef"
    (snapshot / "4bit").mkdir(parents=True)
    (snapshot / "4bit" / "config.json").write_text("{}")
    (snapshot / "4bit" / "model.safetensors").write_bytes(b"\x00" * 16)

    seen: dict[str, object] = {}

    def fake_snapshot_download(repo_id, **kwargs):
        seen["repo_id"] = repo_id
        seen["allow_patterns"] = kwargs.get("allow_patterns")
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    # Record what the pull would have persisted (``_variant_name == "4bit"``).
    _download_gate.persist_pulled_variant(RAW_REPO, "4bit")

    resolved = _resolve_subfolder_checkpoint(RAW_REPO)

    assert resolved == os.path.join(str(snapshot), "4bit")
    assert seen["repo_id"] == RAW_REPO
    assert seen["allow_patterns"] == ["4bit/*"], (
        "the loader must fetch only the pulled variant, not all quantizations"
    )
    # The handed-over directory must be where mlx-lm's loader glob will hit.
    assert Path(resolved, "config.json").exists()
    assert list(Path(resolved).glob("model*.safetensors"))


def test_pulled_variant_overrides_catalog_default(monkeypatch, tmp_path, marker_path):
    """An explicit 8-bit pull wins over this repo's catalog 4-bit default."""
    from vllm_mlx.model_aliases import resolve_subfolder

    assert resolve_subfolder(CATALOG_REPO) == "4bit", "precondition: catalog default"

    snapshot = tmp_path / "snapshots" / "cafebabe"
    (snapshot / "8bit").mkdir(parents=True)
    (snapshot / "8bit" / "config.json").write_text("{}")
    (snapshot / "8bit" / "model.safetensors").write_bytes(b"\x00" * 16)

    seen: dict[str, object] = {}

    def fake_snapshot_download(repo_id, **kwargs):
        seen["repo_id"] = repo_id
        seen["allow_patterns"] = kwargs.get("allow_patterns")
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    _download_gate.persist_pulled_variant(CATALOG_REPO, "8bit")

    resolved = _resolve_subfolder_checkpoint(CATALOG_REPO)

    assert resolved == os.path.join(str(snapshot), "8bit")
    assert seen == {"repo_id": CATALOG_REPO, "allow_patterns": ["8bit/*"]}


def test_explicit_alias_overrides_repo_variant_marker(
    monkeypatch, tmp_path, marker_path
):
    """An explicit 4-bit alias keeps its meaning despite an 8-bit repo marker."""
    from vllm_mlx.model_aliases import resolve_model, resolve_subfolder

    assert resolve_model(CATALOG_ALIAS) == CATALOG_REPO
    assert resolve_subfolder(CATALOG_ALIAS) == "4bit"

    snapshot = tmp_path / "snapshots" / "feedface"
    (snapshot / "4bit").mkdir(parents=True)
    (snapshot / "4bit" / "config.json").write_text("{}")
    (snapshot / "4bit" / "model.safetensors").write_bytes(b"\x00" * 16)
    seen: dict[str, object] = {}

    def fake_snapshot_download(repo_id, **kwargs):
        seen["repo_id"] = repo_id
        seen["allow_patterns"] = kwargs.get("allow_patterns")
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    _download_gate.persist_pulled_variant(CATALOG_REPO, "8bit")

    resolved = _resolve_subfolder_checkpoint(CATALOG_ALIAS)

    assert resolved == os.path.join(str(snapshot), "4bit")
    assert seen == {"repo_id": CATALOG_REPO, "allow_patterns": ["4bit/*"]}


def test_serving_checkpoint_resolves_explicit_alias_before_lane_selection(
    monkeypatch, tmp_path, marker_path
):
    """The server checkpoint choke point retains CLI alias precedence."""
    from types import SimpleNamespace

    import huggingface_hub

    from vllm_mlx import server

    snapshot = tmp_path / "snapshots" / "01234567"
    checkpoint = snapshot / "4bit"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"\x00" * 16)
    seen: dict[str, object] = {}

    def fake_snapshot_download(repo_id, **kwargs):
        seen["repo_id"] = repo_id
        seen["allow_patterns"] = kwargs.get("allow_patterns")
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        server, "_prefetch_routing_metadata", lambda _name: str(checkpoint)
    )
    monkeypatch.setattr(server, "_ensure_routing_config", lambda _path: None)
    monkeypatch.setattr(
        server,
        "resolve_serving_lane_decision",
        lambda *args, **kwargs: SimpleNamespace(
            auto_text_fallback=False,
            reason="text_checkpoint",
        ),
    )
    _download_gate.persist_pulled_variant(CATALOG_REPO, "8bit")

    resolved = server._resolve_serving_checkpoint(CATALOG_ALIAS)

    assert resolved.model_path == CATALOG_REPO
    assert resolved.load_path == str(checkpoint)
    assert seen == {"repo_id": CATALOG_REPO, "allow_patterns": ["4bit/*"]}


def test_serve_uses_format_marker_folder(monkeypatch, tmp_path, marker_path):
    """``--format <folder>`` records the literal folder; serve joins it."""
    snapshot = tmp_path / "snap" / "abc"
    (snapshot / "mxfp4").mkdir(parents=True)
    (snapshot / "mxfp4" / "config.json").write_text("{}")
    (snapshot / "mxfp4" / "model.safetensors").write_bytes(b"\x00" * 16)

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, **kw: str(snapshot),
    )
    _download_gate.persist_pulled_variant(RAW_REPO, "mxfp4")

    assert _resolve_subfolder_checkpoint(RAW_REPO) == os.path.join(
        str(snapshot), "mxfp4"
    )


def test_serve_escapes_format_marker_glob_metacharacters(
    monkeypatch, tmp_path, marker_path
):
    """A literal format survives the real pull-to-serve marker transition."""
    import argparse

    from vllm_mlx import cli

    variant = "quant[4]*?"
    snapshot = tmp_path / "snap" / "def"
    (snapshot / variant).mkdir(parents=True)
    (snapshot / variant / "config.json").write_text("{}")
    (snapshot / variant / "model.safetensors").write_bytes(b"\x00" * 16)
    seen_patterns: list[object] = []

    def fake_snapshot_download(repo_id, **kwargs):
        seen_patterns.append(kwargs.get("allow_patterns"))
        return str(snapshot)

    args = argparse.Namespace(
        model=RAW_REPO,
        bits=None,
        format=variant,
        _original_alias=RAW_REPO,
    )
    with (
        patch(
            "huggingface_hub.HfApi.list_repo_tree",
            return_value=[RepoFolder(path=variant, oid="variant")],
        ),
        patch("huggingface_hub.snapshot_download", fake_snapshot_download),
    ):
        cli.pull_command(args)
        resolved = _resolve_subfolder_checkpoint(RAW_REPO)

    escaped = ["quant[[]4[]][*][?]/*"]
    assert _download_gate.pulled_variant(RAW_REPO) == variant
    assert resolved == os.path.join(str(snapshot), variant)
    assert seen_patterns == [escaped, escaped]


def test_no_marker_resolves_repo_root_as_before(monkeypatch, tmp_path):
    """Without a recorded variant, a raw repo id is resolved unchanged.

    This is the historical behaviour for flat repos whose root IS the
    checkpoint; the marker fallback must never invent a subfolder for a repo
    that was not narrowed with ``--bits/--format``.
    """
    import huggingface_hub

    # snapshot_download must NOT be reached for a root-resolved id in a
    # cold-cache mirror of the prior behaviour (the offline-first path still
    # probes, but the resolution decision is the marker, which is empty).
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, **kw: str(tmp_path),
    )
    assert _download_gate.pulled_variant(RAW_REPO) is None
    assert _resolve_subfolder_checkpoint(RAW_REPO) == RAW_REPO


def test_incomplete_variant_raises_actionable_message(
    monkeypatch, tmp_path, marker_path
):
    """A present folder without weights must not be treated as a checkpoint."""
    snapshot = tmp_path / "snap"
    (snapshot / "4bit").mkdir(parents=True)
    # Only config.json — no weights. Incomplete, so it must be refused.
    (snapshot / "4bit" / "config.json").write_text("{}")

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, **kw: str(snapshot),
    )
    _download_gate.persist_pulled_variant(RAW_REPO, "4bit")

    with pytest.raises(RuntimeError, match="present but incomplete"):
        _resolve_subfolder_checkpoint(RAW_REPO)


def test_absent_variant_raises_actionable_message(monkeypatch, tmp_path, marker_path):
    """A marker whose folder is absent names a valid recovery command."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, **kw: str(snapshot),
    )
    _download_gate.persist_pulled_variant(RAW_REPO, "4bit")

    with pytest.raises(
        RuntimeError,
        match=(
            r"does not exist after download.*"
            r"rapid-mlx pull --format 4bit acme/MultiVariant-MLX"
        ),
    ):
        _resolve_subfolder_checkpoint(RAW_REPO)


# --- defensive branches that keep the marker off the hot path ---


def test_marker_path_falls_back_when_hub_constants_unavailable(monkeypatch):
    """If huggingface_hub's constants cannot be imported, the marker uses a
    best-effort default cache location instead of raising."""
    import sys

    class _NoConstants:
        def __getattr__(self, name):
            raise ImportError("huggingface_hub.constants unavailable")

    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", _NoConstants())
    path = _download_gate._variant_marker_path(RAW_REPO)
    assert path.endswith("models--acme--MultiVariant-MLX/.rapid-mlx/variant")


def test_marker_does_not_corrupt_hub_cache_scan(tmp_path, marker_path):
    """Rapid-MLX metadata must stay outside Hub's commit-ref namespace."""
    from huggingface_hub import scan_cache_dir

    snapshot = marker_path.parents[1] / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    _download_gate.persist_pulled_variant(RAW_REPO, "4bit")

    cache = scan_cache_dir(tmp_path)

    assert not cache.warnings
    assert {repo.repo_id for repo in cache.repos} == {RAW_REPO}


def test_persist_swallows_a_write_oserror(monkeypatch, marker_path):
    """A failed marker write must not break an already-valid pull."""
    import os as _os

    monkeypatch.setattr(
        _os, "makedirs", lambda *a, **k: (_ for _ in ()).throw(OSError("full"))
    )
    # Must not raise.
    assert _download_gate.persist_pulled_variant(RAW_REPO, "4bit") is False
    assert _download_gate.pulled_variant(RAW_REPO) is None


def test_failed_variant_update_invalidates_previous_marker(monkeypatch, marker_path):
    """A failed 8-bit update must never leave an authoritative 4-bit choice."""
    _download_gate.persist_pulled_variant(RAW_REPO, "4bit")
    assert _download_gate.pulled_variant(RAW_REPO) == "4bit"

    monkeypatch.setattr(
        os,
        "replace",
        lambda *args: (_ for _ in ()).throw(OSError("read-only cache")),
    )

    assert _download_gate.persist_pulled_variant(RAW_REPO, "8bit") is False
    assert _download_gate.pulled_variant(RAW_REPO) is None
    assert not marker_path.exists()


def test_persist_read_swallows_an_unreadable_marker(monkeypatch, marker_path):
    """An unreadable marker reads back None, never raises."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("4bit")
    with patch("builtins.open", side_effect=OSError("denied")):
        assert _download_gate.pulled_variant(RAW_REPO) is None


def test_pull_persist_failure_still_completes_the_pull(capsys):
    """The best-effort persist in ``pull_command`` must never fail the pull."""
    import argparse

    from vllm_mlx import cli

    args = argparse.Namespace(
        model="LiquidAI/LFM2.5-2.6B-MLX",
        bits="4",
        format=None,
        _original_alias="LiquidAI/LFM2.5-2.6B-MLX",
    )

    def fake_snapshot(*a, **kw):
        return "/cache/snapshot"

    def boom_persist(repo_id, variant):
        raise OSError("cannot write marker")

    with (
        patch(
            "huggingface_hub.HfApi.list_repo_tree", return_value=_multi_variant_tree()
        ),
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", fake_snapshot),
        patch(
            "vllm_mlx._download_gate.persist_pulled_variant",
            side_effect=boom_persist,
        ),
    ):
        cli.pull_command(args)  # must not raise
    out = capsys.readouterr().out
    assert "could not record that serving choice" in out
    assert "may select an older or default checkpoint" in out


def test_successful_ordinary_hf_pull_clears_previous_variant(marker_path):
    """Switching back to an ordinary pull must not leave a stale selector."""
    import argparse

    from vllm_mlx import cli

    _download_gate.persist_pulled_variant(RAW_REPO, "8bit")
    args = argparse.Namespace(
        model=RAW_REPO,
        bits=None,
        format=None,
        _original_alias=RAW_REPO,
    )

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", return_value="/cache/snapshot"),
    ):
        cli.pull_command(args)

    assert _download_gate.pulled_variant(RAW_REPO) is None


def test_ordinary_hf_pull_survives_marker_clear_failure(capsys):
    """Cleanup metadata is best-effort after a valid HF pull completes."""
    import argparse

    from vllm_mlx import cli

    args = argparse.Namespace(
        model=RAW_REPO,
        bits=None,
        format=None,
        _original_alias=RAW_REPO,
    )

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", return_value="/cache/snapshot"),
        patch(
            "vllm_mlx._download_gate.clear_pulled_variant",
            side_effect=OSError("read-only cache"),
        ),
    ):
        cli.pull_command(args)  # must not raise after a successful download

    assert "Cached at: /cache/snapshot" in capsys.readouterr().out


def test_successful_ordinary_mirror_pull_clears_previous_variant(marker_path):
    """The mirror-success early return applies the same marker transition."""
    import argparse

    from vllm_mlx import cli

    _download_gate.persist_pulled_variant(RAW_REPO, "8bit")
    args = argparse.Namespace(
        model=RAW_REPO,
        bits=None,
        format=None,
        _original_alias=RAW_REPO,
    )

    def mirror_success(repo_id, *, allow_patterns, out):
        assert allow_patterns is None
        out["network_fetch"] = False
        return True

    with patch.object(cli, "_try_mirror_prefetch", side_effect=mirror_success):
        cli.pull_command(args)

    assert _download_gate.pulled_variant(RAW_REPO) is None


def test_successful_variant_mirror_pull_persists_serving_choice(marker_path):
    """The default mirror path must retain the explicit variant for serve."""
    import argparse

    from vllm_mlx import cli

    args = argparse.Namespace(
        model=RAW_REPO,
        bits="4",
        format=None,
        _original_alias=RAW_REPO,
    )

    def mirror_success(repo_id, *, allow_patterns, out):
        assert repo_id == RAW_REPO
        assert allow_patterns == ["4bit/*"]
        out["network_fetch"] = True
        return True

    with (
        patch(
            "huggingface_hub.HfApi.list_repo_tree",
            return_value=_multi_variant_tree(),
        ),
        patch.object(cli, "_try_mirror_prefetch", side_effect=mirror_success),
    ):
        cli.pull_command(args)

    assert _download_gate.pulled_variant(RAW_REPO) == "4bit"


def test_runtime_asset_override_does_not_touch_model_variant(capsys, marker_path):
    """Dependency file filters neither rewrite metadata nor emit a false warning."""
    import argparse

    from vllm_mlx import cli

    _download_gate.persist_pulled_variant(RAW_REPO, "8bit")
    args = argparse.Namespace(
        model=RAW_REPO,
        bits=None,
        format=None,
        _original_alias=RAW_REPO,
    )

    def mirror_success(repo_id, *, allow_patterns, out):
        assert repo_id == RAW_REPO
        assert allow_patterns == ["runtime/*"]
        out["network_fetch"] = False
        return True

    with patch.object(cli, "_try_mirror_prefetch", side_effect=mirror_success):
        cli._pull_repository(args, allow_patterns_override=["runtime/*"])

    assert _download_gate.pulled_variant(RAW_REPO) == "8bit"
    assert "could not record that serving choice" not in capsys.readouterr().out


def test_ordinary_mirror_pull_survives_marker_clear_failure(capsys):
    """Cleanup metadata is best-effort on the mirror-success early return."""
    import argparse

    from vllm_mlx import cli

    args = argparse.Namespace(
        model=RAW_REPO,
        bits=None,
        format=None,
        _original_alias=RAW_REPO,
    )

    def mirror_success(repo_id, *, allow_patterns, out):
        assert allow_patterns is None
        out["network_fetch"] = False
        return True

    with (
        patch.object(cli, "_try_mirror_prefetch", side_effect=mirror_success),
        patch(
            "vllm_mlx._download_gate.clear_pulled_variant",
            side_effect=OSError("read-only cache"),
        ),
    ):
        cli.pull_command(args)  # must not raise after a successful download

    assert "Already cached" in capsys.readouterr().out
