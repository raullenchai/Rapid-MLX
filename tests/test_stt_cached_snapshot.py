"""Warm pinned STT loads must not require Hub metadata access."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

from rapid_mlx import model_downloads
from rapid_mlx.audio.stt import STTEngine
from rapid_mlx.model_aliases import resolve_revision

REPO = "mlx-community/whisper-tiny-mlx"


@pytest.mark.parametrize("name", [REPO, "whisper-tiny"])
@pytest.mark.parametrize("missing", [None, "weights.npz", "config.json"])
def test_stt_pinned_snapshot_uses_cache_before_hub(
    monkeypatch, tmp_path, missing, name
):
    import sys

    pin = resolve_revision(REPO)
    root = tmp_path / ("models--" + REPO.replace("/", "--"))
    snapshot = root / "snapshots" / pin
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "whisper"}))
    (snapshot / "weights.npz").write_bytes(b"weights")
    # A different mutable ref must never displace the catalog pin.
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("a" * 40)
    if missing:
        (snapshot / missing).unlink()
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    forbidden = Mock(side_effect=AssertionError("Hub network access on warm load"))
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", forbidden)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", forbidden)
    calls = []

    def download(repo, **kwargs):
        calls.append(kwargs)
        assert repo == REPO
        assert kwargs["revision"] == pin
        assert missing, "Hub snapshot resolver must not be called for a complete cache"
        assert not kwargs.get("local_files_only")
        return "/downloaded/checkpoint"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    loader = Mock(
        return_value=SimpleNamespace(config=SimpleNamespace(model_type="whisper"))
    )
    parts = ["whisper", "tiny"]
    monkeypatch.setitem(
        sys.modules, "mlx_audio.stt.utils", SimpleNamespace(load_model=loader)
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_audio.utils",
        SimpleNamespace(
            DEFAULT_ALLOW_PATTERNS=["*.json", "*.npz"],
            get_model_name_parts=lambda _: parts,
        ),
    )
    monkeypatch.setattr(STTEngine, "_ensure_whisper_processor", lambda _: None)
    engine = STTEngine(name)
    engine.load()
    loader.assert_called_once_with(
        "/downloaded/checkpoint" if missing else str(snapshot), model_name_parts=parts
    )
    assert len(calls) == (1 if missing else 0)
    forbidden.assert_not_called()
    engine.load()
    assert len(calls) == (
        1 if missing else 0
    )  # Already-loaded requests do no additional cache work.


@pytest.mark.parametrize("unsafe", ["snapshot-link", "weight-link", "wrong-revision"])
def test_whisper_pinned_resolver_rejects_unsafe_or_other_snapshot(
    monkeypatch, tmp_path, unsafe
):
    from rapid_mlx.model_metadata import resolve_pinned_cached_snapshot

    pin = resolve_revision(REPO)
    root = tmp_path / ("models--" + REPO.replace("/", "--"))
    snapshot = root / "snapshots" / pin
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "weights.npz").write_bytes(b"weights")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    if unsafe == "snapshot-link":
        moved = snapshot.rename(root / "moved")
        snapshot.symlink_to(moved, target_is_directory=True)
    elif unsafe == "weight-link":
        outside = tmp_path / "outside.npz"
        outside.write_bytes(b"weights")
        (snapshot / "weights.npz").unlink()
        (snapshot / "weights.npz").symlink_to(outside)
    else:
        snapshot.rename(root / "snapshots" / ("a" * 40))
    assert resolve_pinned_cached_snapshot(REPO) is None


@pytest.mark.parametrize("cached", [True, False])
def test_whisper_processor_local_first(monkeypatch, cached):
    import sys

    processor = object()
    hub = Mock(return_value=processor)
    calls = []

    def from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        if kwargs.get("local_files_only"):
            if cached:
                return processor
            raise OSError("Processor assets not found locally")
        return hub(name, **kwargs)

    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            WhisperProcessor=SimpleNamespace(from_pretrained=from_pretrained)
        ),
    )
    engine = STTEngine(REPO)
    engine.model = SimpleNamespace(_processor=None)
    engine._ensure_whisper_processor()
    assert engine.model._processor is processor
    expected = [("openai/whisper-tiny", {"local_files_only": True})]
    if cached:
        hub.assert_not_called()
    else:
        expected.append(("openai/whisper-tiny", {}))
        hub.assert_called_once_with("openai/whisper-tiny")
    assert calls == expected
    engine._ensure_whisper_processor()
    assert calls == expected


@pytest.mark.parametrize("error", [OSError("missing"), ValueError("invalid assets")])
@pytest.mark.parametrize("disabled", [True, False])
def test_whisper_processor_failure_preserves_policy(monkeypatch, error, disabled):
    import sys

    loader = Mock(side_effect=error)
    monkeypatch.setattr(model_downloads, "disabled", lambda: disabled)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(WhisperProcessor=SimpleNamespace(from_pretrained=loader)),
    )
    engine = STTEngine(REPO)
    engine.model = SimpleNamespace(_processor=None)
    engine._ensure_whisper_processor()
    assert engine.model._processor is None
    assert loader.call_args_list[0].kwargs == {"local_files_only": True}
    assert loader.call_count == (
        2 if isinstance(error, OSError) and not disabled else 1
    )
