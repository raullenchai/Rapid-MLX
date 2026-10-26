# SPDX-License-Identifier: Apache-2.0
"""STT consent is a local load boundary, never a transcription hot-path check."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.requires_mlx


@pytest.mark.parametrize("pinned", [False, True])
def test_stt_checks_resolved_snapshot_before_loading(tmp_path, monkeypatch, pinned):
    import mlx_audio.stt.utils
    import mlx_audio.utils

    from rapid_mlx import model_aliases, model_downloads
    from rapid_mlx.audio.stt import STTEngine

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_asr"}')
    metadata = tmp_path / "tokenizer_config.json"
    metadata.write_text(
        json.dumps({"auto_map": {"AutoTokenizer": ["custom.Tokenizer", None]}})
    )
    resolved = Mock(return_value=tmp_path)
    monkeypatch.setattr(model_downloads, "require_local", lambda name: None)
    monkeypatch.setattr(
        model_aliases, "resolve_revision", lambda name: "a" * 40 if pinned else None
    )
    monkeypatch.setattr(model_downloads, "snapshot_download", resolved)
    monkeypatch.setattr(mlx_audio.stt.utils, "get_model_path", resolved)
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="sensevoice"),
        generate=Mock(
            return_value=SimpleNamespace(text="hello", segments=[], language="en")
        ),
    )
    loader = Mock(return_value=model)
    monkeypatch.setattr(mlx_audio.stt.utils, "base_load_model", loader)
    engine = STTEngine("example/speech", enable_vad_pretrim=False)
    with pytest.raises(ValueError, match="explicitly opt in"):
        engine.load()
    loader.assert_not_called()
    assert not engine._loaded

    metadata.write_text("{}")
    engine.load()
    assert loader.call_args.kwargs["model_path"] == Path(tmp_path)
    assert loader.call_args.kwargs[
        "model_name_parts"
    ] == mlx_audio.utils.get_model_name_parts("example/speech")
    assert engine._loaded

    # Real transcribe dispatch after warmup must neither re-resolve nor read
    # metadata. A failed check is retried above; successful loads are latched.
    def forbidden(*args, **kwargs):
        raise AssertionError("consent I/O on a warm transcription path")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(mlx_audio.stt.utils, "get_model_path", forbidden)
    monkeypatch.setattr(model_downloads, "snapshot_download", forbidden)
    engine.load()
    for _ in range(3):
        assert engine.transcribe("speech.wav").text == "hello"
    assert loader.call_count == 1
    assert model.generate.call_count == 3
    assert resolved.call_count == 2


def test_native_archive_resolution_checks_final_metadata(tmp_path, monkeypatch):
    import mlx_audio.stt.utils as native
    from mlx_audio.stt.models.phonon import transport

    from rapid_mlx.utils.model_file_guard import checked_stt_loader

    outer = tmp_path / "outer"
    inner = tmp_path / "inner"
    outer.mkdir()
    inner.mkdir()
    (outer / "config.json").write_text("{}")
    (inner / "config.json").write_text('{"model_type":"qwen3_asr"}')
    (inner / "tokenizer_config.json").write_text(
        '{"auto_map":{"AutoTokenizer":["custom.Tokenizer",null]}}'
    )
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.setattr(native, "get_model_path", lambda *a, **kw: outer)
    monkeypatch.setattr(transport, "is_phonon_model", lambda *a: True)
    prepare = Mock(return_value=inner)
    monkeypatch.setattr(transport, "prepare_model_path", prepare)
    base = Mock(return_value="loaded")
    monkeypatch.setattr(native, "base_load_model", base)
    adapted = checked_stt_loader(native.load_model)
    with pytest.raises(ValueError, match="explicitly opt in"):
        adapted("example/archive")
    base.assert_not_called()
    assert prepare.call_args.kwargs["source"] == "example/archive"
    assert native.base_load_model is base
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "1")
    assert adapted("example/archive") == "loaded"
    assert base.call_args.kwargs["model_path"] == inner
