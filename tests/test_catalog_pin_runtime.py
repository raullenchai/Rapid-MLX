# SPDX-License-Identifier: Apache-2.0
"""Catalog pins reach download boundaries without fetching model bytes."""

import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx import model_aliases, model_downloads
from rapid_mlx.audio import registry


def test_all_catalog_aliases_resolve_exact_revision():
    root = Path(__file__).resolve().parents[1]
    repos = {}
    for file, field in [
        ("rapid_mlx/aliases.json", "hf_path"),
        ("rapid_mlx/audio/aliases.json", "hf_id"),
    ]:
        for alias, entry in json.loads((root / file).read_text()).items():
            if alias.startswith("_"):
                continue
            pin, repo = entry["revision"], entry[field]
            assert re.fullmatch("[0-9a-f]{40}", pin)
            assert model_aliases.resolve_revision(alias) == pin
            assert model_aliases.resolve_revision(repo) == pin
            assert repos.setdefault(repo.lower(), pin) == pin


@pytest.mark.parametrize("pin", ["main", "A" * 40, 123, []])
def test_audio_invalid_pin_rejected(pin):
    with pytest.raises(ValueError, match="40-hex"):
        registry._revision({"revision": pin})


@pytest.mark.parametrize(
    "name", ["kokoro", "mlx-community/Kokoro-82M-bf16", "prince-canuma/Kokoro-82M"]
)
@pytest.mark.parametrize("method", ["snapshot_download", "hf_hub_download"])
def test_audio_download_pin_reaches_hub(monkeypatch, name, method):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(
        huggingface_hub, method, lambda *a, **kw: calls.append((a, kw)) or "/snapshot"
    )
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    args = (name,) if method == "snapshot_download" else (name, "config.json")
    getattr(model_downloads, method)(*args)
    assert calls[0][0][0] == model_aliases.resolve_hub_model(name)
    assert calls[0][1]["revision"] == registry.resolve_audio_revision(name)
    with pytest.raises(ValueError, match="conflicting revision"):
        getattr(model_downloads, method)(*args, revision="main")


def test_stt_implicit_load_uses_pinned_local_snapshot(monkeypatch, tmp_path):
    pytest.importorskip("mlx_audio.utils")
    import huggingface_hub

    from rapid_mlx.audio.stt import STTEngine

    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **kw: calls.append((a, kw)) or str(tmp_path),
    )
    utils = ModuleType("mlx_audio.stt.utils")
    loaded = []
    utils.load_model = lambda path, **kwargs: (
        loaded.append((path, kwargs))
        or SimpleNamespace(config=SimpleNamespace(model_type="sensevoice"))
    )
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils)
    engine = STTEngine("mlx-community/SenseVoiceSmall")
    engine.load()
    from mlx_audio.utils import get_model_name_parts

    assert loaded == [
        (str(tmp_path), {"model_name_parts": get_model_name_parts(engine.model_name)})
    ]
    assert calls[0][1]["revision"] == registry.resolve_audio_revision(engine.model_name)


def test_f5_loader_uses_shared_pin_gate(monkeypatch):
    import huggingface_hub

    from rapid_mlx.audio.tts import TTSEngine

    utils = pytest.importorskip("f5_tts_mlx.utils")
    pytest.importorskip("f5_tts_mlx.cfm")
    original = utils.snapshot_download
    calls = []

    class BoundaryReachedError(Exception):
        pass

    def hub(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        raise BoundaryReachedError

    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)
    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    with pytest.raises(BoundaryReachedError):
        engine.load()
    assert calls[0][0] == engine.model_name
    assert calls[0][1]["revision"] == registry.resolve_audio_revision(engine.model_name)
    assert "model_v1.safetensors" in calls[0][1]["allow_patterns"]
    # The adapter is scoped to the load and must not leak process-globally.
    assert utils.snapshot_download is original


def test_kokoro_implicit_assets_use_pin_gate(monkeypatch, tmp_path):
    import huggingface_hub

    from rapid_mlx.audio import tts

    pipeline = pytest.importorskip("mlx_audio.tts.models.kokoro.pipeline")
    original = pipeline.snapshot_download
    generate = ModuleType("mlx_audio.tts.generate")
    generate.load_model = lambda path: path
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", generate)
    monkeypatch.setattr(tts, "_resolve_safe_tts_checkpoint", lambda name: tmp_path)
    calls = []

    def hub(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)
    marker = object()
    monkeypatch.setattr(pipeline, "load_voice_tensor", lambda path: marker)
    engine = tts.TTSEngine("mlx-community/Kokoro-82M-bf16")
    engine.load()
    # load_model itself does not fetch voices; the adapter must not leak.
    assert pipeline.snapshot_download is original
    voice_loader = pipeline.KokoroPipeline.__new__(pipeline.KokoroPipeline)
    voice_loader.repo_id = "prince-canuma/Kokoro-82M"
    voice_loader.voices = {}
    voice_loader.lang_code = "a"

    class _Result:
        audio = SimpleNamespace(tolist=lambda: [0.0])

    class _Model:
        def generate(self, **kwargs):
            # Voices are fetched lazily on generate(), under the pin adapter.
            assert pipeline.snapshot_download is model_downloads.snapshot_download
            assert voice_loader.load_single_voice("af_heart") is marker
            yield _Result()

    engine.model = _Model()
    engine._loaded = True
    engine.generate("hello", voice="af_heart")
    assert pipeline.snapshot_download is original
    assert len(calls) == 2  # local miss followed by the online request
    for repo, kwargs in calls:
        assert repo == voice_loader.repo_id
        assert kwargs["revision"] == registry.resolve_audio_revision(repo)
        assert kwargs["allow_patterns"] == ["voices/af_heart.safetensors"]


def test_catalog_agrees_with_runtime_artifact_pins():
    from rapid_mlx._download_gate import IMAGE_MODEL_REVISIONS
    from rapid_mlx.models.deepseek_v41_native.artifacts import PRODUCT_ARTIFACTS
    from rapid_mlx.video.wan import WAN_REVISIONS

    for repo, pin in {**IMAGE_MODEL_REVISIONS, **WAN_REVISIONS}.items():
        if model_aliases.resolve_profile(repo) is not None:
            assert model_aliases.resolve_revision(repo) == pin
    for artifact in PRODUCT_ARTIFACTS.values():
        assert (
            model_aliases.resolve_revision(artifact.alias) == artifact.target_revision
        )


def test_pinned_alias_resolved_to_configured_local_folder_stays_local(
    monkeypatch, tmp_path
):
    from rapid_mlx.utils.tokenizer import _resolve_subfolder_checkpoint

    monkeypatch.setattr(model_aliases, "resolve_model", lambda name: str(tmp_path))
    monkeypatch.setattr(
        model_aliases,
        "resolve_revision",
        lambda name: pytest.fail("local checkpoint must not resolve a Hub revision"),
    )
    assert _resolve_subfolder_checkpoint("flint-40b-a2.5b-4bit") == str(tmp_path)


def test_stt_pin_adapter_without_optional_backend(monkeypatch, tmp_path):
    """Exercise the adapter on CI hosts without the optional audio runtime."""
    import huggingface_hub

    from rapid_mlx.audio.stt import STTEngine

    common = ModuleType("mlx_audio.utils")
    common.DEFAULT_ALLOW_PATTERNS = ["*.json", "*.safetensors"]
    parts = ("sensevoice", "small")
    common.get_model_name_parts = lambda name: parts
    backend = ModuleType("mlx_audio.stt.utils")
    loaded = []
    model = SimpleNamespace(config=SimpleNamespace(model_type="sensevoice"))
    backend.load_model = lambda path, **kwargs: loaded.append((path, kwargs)) or model
    monkeypatch.setitem(sys.modules, "mlx_audio.utils", common)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", backend)
    calls = []
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *args, **kwargs: calls.append((args, kwargs)) or str(tmp_path),
    )
    engine = STTEngine("mlx-community/SenseVoiceSmall")
    engine.load()
    assert engine.model is model
    assert loaded == [(str(tmp_path), {"model_name_parts": parts})]
    assert calls == [
        (
            (engine.model_name,),
            {
                "allow_patterns": common.DEFAULT_ALLOW_PATTERNS,
                "revision": registry.resolve_audio_revision(engine.model_name),
            },
        )
    ]


def test_f5_pin_adapter_without_optional_backend(monkeypatch):
    """Keep adapter wiring covered independently of native loader availability."""
    import huggingface_hub

    from rapid_mlx.audio.tts import TTSEngine

    package = ModuleType("f5_tts_mlx")
    utils = ModuleType("f5_tts_mlx.utils")
    cfm = ModuleType("f5_tts_mlx.cfm")
    package.utils = utils
    original = lambda repo: pytest.fail("unpatched snapshot_download called")
    utils.snapshot_download = original
    cfm.F5TTS = SimpleNamespace(
        from_pretrained=lambda repo: utils.snapshot_download(repo)
    )
    for module in (package, utils, cfm):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []
    model = "/pinned-f5-snapshot"
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *args, **kwargs: calls.append((args, kwargs)) or model,
    )
    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    engine.load()
    assert engine.model == model
    # The adapter was installed during from_pretrained() and restored after.
    assert utils.snapshot_download is original
    assert calls == [
        (
            (engine.model_name,),
            {"revision": registry.resolve_audio_revision(engine.model_name)},
        )
    ]


def test_kokoro_scope_covers_only_first_result(monkeypatch):
    import types

    from rapid_mlx import model_downloads
    from rapid_mlx.audio import tts

    original = object()
    module = types.SimpleNamespace(snapshot_download=original)
    seen = []

    def results():
        seen.append(module.snapshot_download)
        yield 1
        seen.append(module.snapshot_download)
        yield 2

    stream = tts._first_result_scoped(results(), module)
    assert next(stream) == 1
    # Restored before the first result reaches the caller; lock is free.
    assert module.snapshot_download is original
    assert tts._SNAPSHOT_DOWNLOAD_LOCK.acquire(blocking=False)
    tts._SNAPSHOT_DOWNLOAD_LOCK.release()
    assert list(stream) == [2]
    assert seen == [model_downloads.snapshot_download, original]


def test_scoped_download_removes_attribute_it_added():
    import types

    from rapid_mlx.audio import tts

    module = types.SimpleNamespace()
    assert list(tts._first_result_scoped(iter(()), module)) == []
    assert not hasattr(module, "snapshot_download")


def test_kokoro_generate_results_scopes_backend_pipeline(monkeypatch):
    import sys
    import types

    from rapid_mlx import model_downloads
    from rapid_mlx.audio import tts

    original = object()
    pipeline = types.ModuleType("mlx_audio.tts.models.kokoro.pipeline")
    pipeline.snapshot_download = original
    kokoro = types.ModuleType("mlx_audio.tts.models.kokoro")
    kokoro.pipeline = pipeline
    for name, module in {
        "mlx_audio": types.ModuleType("mlx_audio"),
        "mlx_audio.tts": types.ModuleType("mlx_audio.tts"),
        "mlx_audio.tts.models": types.ModuleType("mlx_audio.tts.models"),
        "mlx_audio.tts.models.kokoro": kokoro,
        "mlx_audio.tts.models.kokoro.pipeline": pipeline,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    seen = []

    class Model:
        def generate(self, **_kwargs):
            seen.append(pipeline.snapshot_download)
            yield "chunk"

    engine = tts.TTSEngine.__new__(tts.TTSEngine)
    engine.model = Model()
    engine._model_family = "kokoro"
    assert list(engine._generate_results({})) == ["chunk"]
    assert seen == [model_downloads.snapshot_download]
    assert pipeline.snapshot_download is original
