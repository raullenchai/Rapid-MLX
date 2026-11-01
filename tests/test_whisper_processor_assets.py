"""Pinned Whisper processors are prepared during pull and used offline."""

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from rapid_mlx import cli, model_downloads
from rapid_mlx.audio import registry
from rapid_mlx.audio.stt import STTEngine

PIN = "e37978b90ca9030d5170a5c07aadb050351a65bb"
TURBO_PIN = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
PATTERNS = (
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "normalizer.json",
    "config.json",
)


@pytest.mark.parametrize(
    "processors,error",
    [
        pytest.param(
            [],
            "_whisper_processors must be an object",
            id="not-an-object",
        ),
        pytest.param(
            {"invalid": {"repo_id": "openai/whisper-base", "revision": PIN}},
            "invalid Whisper processor entry",
            id="hf-id-without-slash",
        ),
        pytest.param(
            {"owner/model/extra": {"repo_id": "openai/whisper-base", "revision": PIN}},
            "invalid Whisper processor entry",
            id="hf-id-with-extra-slash",
        ),
        pytest.param(
            {"mlx-community/whisper-base-mlx": []},
            "invalid Whisper processor entry",
            id="asset-not-an-object",
        ),
        pytest.param(
            {"mlx-community/whisper-base-mlx": {"revision": PIN}},
            "Whisper processors require a repo and revision",
            id="missing-repo-id",
        ),
        pytest.param(
            {"mlx-community/whisper-base-mlx": {"repo_id": "openai/whisper-base"}},
            "Whisper processors require a repo and revision",
            id="missing-revision",
        ),
        pytest.param(
            {
                "mlx-community/whisper-base-mlx": {
                    "repo_id": "openai/whisper-base",
                    "revision": PIN,
                },
                "MLX-COMMUNITY/WHISPER-BASE-MLX": {
                    "repo_id": "openai/whisper-base",
                    "revision": PIN,
                },
            },
            "duplicate Whisper processor entry",
            id="case-insensitive-duplicate",
        ),
    ],
)
def test_whisper_processor_registry_rejects_malformed_metadata(
    monkeypatch, tmp_path, processors, error
):
    raw = json.loads(Path(registry._registry_path()).read_text())
    raw["_whisper_processors"] = processors
    registry_file = tmp_path / "aliases.json"
    registry_file.write_text(json.dumps(raw))
    monkeypatch.setattr(registry, "_registry_path", lambda: str(registry_file))
    monkeypatch.setattr(registry, "_REGISTRY", None)
    monkeypatch.setattr(registry, "_HF_ID_INDEX", {})
    monkeypatch.setattr(registry, "_WHISPER_PROCESSORS", {})
    monkeypatch.setattr(registry, "_RUNTIME_ASSETS", {})
    monkeypatch.setattr(registry, "_RUNTIME_REQUIREMENTS", {})

    with pytest.raises(ValueError, match=re.escape(f"audio aliases.json: {error}")):
        registry._load_registry()


def test_whisper_aliases_declare_processors():
    for entry in registry.list_audio_aliases():
        if entry.family == "whisper":
            asset = registry.whisper_processor_asset(entry.hf_id.upper())
            assert asset is not None
            assert asset in registry.runtime_assets_for(entry.alias)
            assert asset.allow_patterns == PATTERNS
    assert (
        registry.whisper_processor_asset(
            "mlx-community/whisper-large-v3-turbo"
        ).revision
        == TURBO_PIN
    )
    assert registry.runtime_assets_for("parakeet") == ()


def test_pull_fetches_pinned_processor(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("rapid_mlx.audio.probe.require_audio_or_exit", lambda _: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_resolve_variant_allow_patterns", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_sync_pulled_variant_marker", lambda *a: None)
    monkeypatch.setattr(
        "rapid_mlx.model_integrity.verify_snapshot", lambda *a, **k: None
    )
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, **kw: calls.append((repo, kw)) or str(tmp_path),
    )
    entry = registry.resolve_audio_alias("whisper-large-v3-turbo")
    cli.pull_command(
        argparse.Namespace(
            model=entry.hf_id, _original_alias=entry.alias, bits=None, format=None
        )
    )
    assert (
        "openai/whisper-large-v3-turbo",
        {"revision": TURBO_PIN, "allow_patterns": list(PATTERNS)},
    ) in calls


def test_offline_processor_uses_pinned_revision(monkeypatch):
    processor = object()
    calls = []

    def load(repo, **kwargs):
        calls.append((repo, kwargs))
        if kwargs != {"revision": PIN, "local_files_only": True}:
            raise OSError("pinned snapshot required")
        return processor

    monkeypatch.setattr(model_downloads, "disabled", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(WhisperProcessor=SimpleNamespace(from_pretrained=load)),
    )
    engine = STTEngine("mlx-community/whisper-base-mlx")
    engine.model = SimpleNamespace(_processor=None)
    engine._ensure_whisper_processor()
    assert engine.model._processor is processor
    assert calls == [
        ("openai/whisper-base", {"revision": PIN, "local_files_only": True})
    ]


def test_offline_processor_falls_back_to_unpinned_cache(monkeypatch):
    processor = object()
    calls = []

    def load(repo, **kwargs):
        calls.append((repo, kwargs))
        if kwargs.get("local_files_only") and "revision" in kwargs:
            raise OSError("pinned snapshot missing")
        assert kwargs == {"local_files_only": True}
        return processor

    monkeypatch.setattr(model_downloads, "disabled", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(WhisperProcessor=SimpleNamespace(from_pretrained=load)),
    )
    engine = STTEngine("mlx-community/whisper-base-mlx")
    engine.model = SimpleNamespace(_processor=None)
    engine._ensure_whisper_processor()
    assert engine.model._processor is processor
    assert calls == [
        ("openai/whisper-base", {"revision": PIN, "local_files_only": True}),
        ("openai/whisper-base", {"local_files_only": True}),
    ]


@pytest.mark.parametrize("online_required", [False, True])
def test_processor_failed_composition_retries_with_downloads_enabled(
    monkeypatch, caplog, online_required
):
    processor = object()
    calls = []

    def load(source, **kwargs):
        calls.append(("processor", source, kwargs))
        if kwargs.get("local_files_only") and ("revision" in kwargs or online_required):
            raise OSError("original processor error")
        return processor

    def load_extractor(source, **kwargs):
        calls.append(("extractor", source, kwargs))
        raise ValueError("invalid cached extractor details")

    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            WhisperProcessor=SimpleNamespace(from_pretrained=load),
            WhisperFeatureExtractor=SimpleNamespace(from_pretrained=load_extractor),
            WhisperTokenizerFast=SimpleNamespace(),
        ),
    )
    engine = STTEngine("mlx-community/whisper-base-mlx")
    engine.model = SimpleNamespace(_processor=None)
    with caplog.at_level("DEBUG", logger="rapid_mlx.audio.stt"):
        engine._ensure_whisper_processor()

    assert engine.model._processor is processor
    expected = [
        (name, "openai/whisper-base", {"revision": PIN, "local_files_only": True})
        for name in ("processor", "extractor")
    ]
    expected.append(("processor", "openai/whisper-base", {"local_files_only": True}))
    if online_required:
        expected.extend(
            [
                ("extractor", "openai/whisper-base", {"local_files_only": True}),
                ("processor", "openai/whisper-base", {"revision": PIN}),
            ]
        )
    assert calls == expected
    assert "ValueError" in caplog.text
    assert "invalid cached extractor details" not in caplog.text


def test_offline_processor_composes_from_cached_parts(monkeypatch, caplog):
    processor, feature_extractor, tokenizer = object(), object(), object()
    calls = []
    constructed = []

    class WhisperProcessor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("processor", source, kwargs))
            raise OSError("processor metadata missing")

        def __new__(cls, **kwargs):
            constructed.append(kwargs)
            return processor

    def load_part(name, part):
        def load(source, **kwargs):
            calls.append((name, source, kwargs))
            return part

        return SimpleNamespace(from_pretrained=load)

    monkeypatch.setattr(model_downloads, "disabled", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            WhisperProcessor=WhisperProcessor,
            WhisperFeatureExtractor=load_part("extractor", feature_extractor),
            WhisperTokenizerFast=load_part("fast", tokenizer),
        ),
    )
    engine = STTEngine("mlx-community/whisper-base-mlx")
    engine.model = SimpleNamespace(_processor=None)
    with caplog.at_level("INFO", logger="rapid_mlx.audio.stt"):
        engine._ensure_whisper_processor()

    assert engine.model._processor is processor
    assert constructed == [
        {"feature_extractor": feature_extractor, "tokenizer": tokenizer}
    ]
    assert calls == [
        (name, "openai/whisper-base", {"revision": PIN, "local_files_only": True})
        for name in ("processor", "extractor", "fast")
    ]
    assert "cached feature extractor and tokenizer" in caplog.text


@pytest.mark.parametrize("fast_available", [True, False])
def test_offline_processor_uses_slow_tokenizer(monkeypatch, fast_available):
    processor, feature_extractor, tokenizer = object(), object(), object()
    calls = []
    constructed = []

    class WhisperProcessor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("processor", source, kwargs))
            raise OSError("processor metadata missing")

        def __new__(cls, **kwargs):
            constructed.append(kwargs)
            return processor

    def load_part(name, part=None):
        def load(source, **kwargs):
            calls.append((name, source, kwargs))
            if name == "fast":
                raise OSError("fast tokenizer unavailable")
            return part

        return SimpleNamespace(from_pretrained=load)

    transformers = SimpleNamespace(
        WhisperProcessor=WhisperProcessor,
        WhisperFeatureExtractor=load_part("extractor", feature_extractor),
        WhisperTokenizer=load_part("slow", tokenizer),
    )
    if fast_available:
        transformers.WhisperTokenizerFast = load_part("fast")
    monkeypatch.setattr(model_downloads, "disabled", lambda: True)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    engine = STTEngine("mlx-community/whisper-base-mlx")
    engine.model = SimpleNamespace(_processor=None)
    engine._ensure_whisper_processor()

    assert engine.model._processor is processor
    assert constructed == [
        {"feature_extractor": feature_extractor, "tokenizer": tokenizer}
    ]
    names = (
        ("processor", "extractor", "fast", "slow")
        if fast_available
        else ("processor", "extractor", "slow")
    )
    assert calls == [
        (name, "openai/whisper-base", {"revision": PIN, "local_files_only": True})
        for name in names
    ]


@pytest.mark.parametrize(
    "failing_part",
    ["extractor", "slow"],
    ids=["extractor-missing", "tokenizer-missing"],
)
def test_offline_processor_failed_composition_stays_offline(
    monkeypatch, caplog, failing_part
):
    calls = []

    def load_part(name):
        def load(source, **kwargs):
            calls.append((name, source, kwargs))
            if name == "processor":
                raise OSError("original processor error")
            if name in ("fast", failing_part):
                raise OSError("cached part missing")
            return object()

        return SimpleNamespace(from_pretrained=load)

    monkeypatch.setattr(model_downloads, "disabled", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            WhisperProcessor=load_part("processor"),
            WhisperFeatureExtractor=load_part("extractor"),
            WhisperTokenizerFast=load_part("fast"),
            WhisperTokenizer=load_part("slow"),
        ),
    )
    engine = STTEngine("mlx-community/whisper-base-mlx")
    engine.model = SimpleNamespace(_processor=None)
    engine._ensure_whisper_processor()

    assert engine.model._processor is None
    names = (
        ("processor", "extractor")
        if failing_part == "extractor"
        else ("processor", "extractor", "fast", "slow")
    )
    assert calls == [
        (name, "openai/whisper-base", kwargs)
        for kwargs in (
            {"revision": PIN, "local_files_only": True},
            {"local_files_only": True},
        )
        for name in names
    ]
    assert "original processor error" in caplog.text
    assert "cached part missing" not in caplog.text
