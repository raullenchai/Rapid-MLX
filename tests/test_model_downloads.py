"""--disable-model-downloads / RAPID_MLX_DISABLE_MODEL_DOWNLOADS."""

import asyncio
import importlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from huggingface_hub.errors import LocalEntryNotFoundError

from rapid_mlx import model_downloads
from rapid_mlx.model_downloads import ModelDownloadsDisabledError

_REVISION = "a" * 40


@pytest.fixture(autouse=True)
def _clean_switch(monkeypatch):
    for name in (model_downloads.ENV_VAR, "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    model_downloads.configure(False)
    yield
    model_downloads.configure(False)


# Other tests re-import rapid_mlx.cli; patch the module the code under test sees.
@pytest.fixture
def cli():
    return importlib.import_module("rapid_mlx.cli")


def _fail(*_args, **_kwargs):
    raise AssertionError("must not be called")


def test_downloads_allowed_by_default(monkeypatch, cli):
    monkeypatch.setattr(cli, "_cache_runnability", _fail)
    assert model_downloads.source() is None
    model_downloads.check("acme/model")
    model_downloads.require_local("acme/model")


def test_flag_wins_and_names_its_source(monkeypatch):
    monkeypatch.setenv(model_downloads.ENV_VAR, "1")
    model_downloads.configure(True)
    assert model_downloads.source() == "--disable-model-downloads"


def test_env_switch(monkeypatch):
    monkeypatch.setenv(model_downloads.ENV_VAR, "yes")
    assert model_downloads.source() == model_downloads.ENV_VAR
    monkeypatch.setenv(model_downloads.ENV_VAR, "0")
    assert model_downloads.disabled() is False


def test_check_names_the_model_and_the_setting():
    model_downloads.configure(True)
    with pytest.raises(ModelDownloadsDisabledError) as exc:
        model_downloads.check("acme/model")
    assert exc.value.model == "acme/model"
    assert exc.value.source == "--disable-model-downloads"
    assert str(exc.value) == (
        "acme/model is not available locally and model downloads are "
        "disabled by --disable-model-downloads"
    )


def test_require_local_accepts_a_local_path(monkeypatch, tmp_path, cli):
    model_downloads.configure(True)
    monkeypatch.setattr(cli, "_cache_runnability", _fail)
    model_downloads.require_local(str(tmp_path))


def test_require_local_accepts_a_runnable_cached_model(monkeypatch, cli):
    model_downloads.configure(True)
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: True)
    model_downloads.require_local("acme/model")


@pytest.mark.parametrize("runnable", [False, None])
def test_require_local_rejects_missing_and_unverifiable_models(
    monkeypatch, runnable, cli
):
    model_downloads.configure(True)
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: runnable)
    with pytest.raises(ModelDownloadsDisabledError):
        model_downloads.require_local("acme/model")


def test_pinned_download_uses_only_the_local_snapshot(monkeypatch):
    from rapid_mlx import _mirror

    model_downloads.configure(True)
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        assert kwargs["local_files_only"] is True
        return "/cache/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(_mirror, "download_with_mirror_fallback", _fail)
    path = _mirror.pinned_snapshot_download(
        "acme/model", _REVISION, allow_patterns=["*.json"]
    )
    assert path == "/cache/snapshot"
    assert calls == [
        (
            "acme/model",
            {
                "revision": _REVISION,
                "allow_patterns": ["*.json"],
                "local_files_only": True,
            },
        )
    ]


def test_pinned_download_refuses_a_missing_snapshot(monkeypatch):
    from rapid_mlx import _mirror

    model_downloads.configure(True)

    def fake_snapshot_download(repo_id, **kwargs):
        assert kwargs["local_files_only"] is True
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(_mirror, "download_with_mirror_fallback", _fail)
    with pytest.raises(ModelDownloadsDisabledError):
        _mirror.pinned_snapshot_download("acme/model", _REVISION)


def test_policy_snapshot_resolver_forces_local_only_and_fails_closed(monkeypatch):
    model_downloads.configure(True)
    calls = []

    def missing(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(ModelDownloadsDisabledError):
        model_downloads.snapshot_download("acme/sidecar", allow_patterns=["*.json"])
    assert calls == [
        (
            "acme/sidecar",
            {"allow_patterns": ["*.json"], "local_files_only": True},
        )
    ]


def test_policy_file_resolver_forces_local_only_and_fails_closed(monkeypatch):
    model_downloads.configure(True)
    calls = []

    def missing(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    with pytest.raises(ModelDownloadsDisabledError):
        model_downloads.hf_hub_download("acme/model", "config.json")
    assert calls == [("acme/model", "config.json", {"local_files_only": True})]


def test_policy_resolvers_preserve_default_online_behavior(monkeypatch):
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, **kwargs: calls.append((repo_id, kwargs)) or "/snapshot",
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo_id, filename, **kwargs: (
            calls.append((repo_id, filename, kwargs)) or "/file"
        ),
    )

    assert model_downloads.snapshot_download("acme/model", revision="main") == (
        "/snapshot"
    )
    assert model_downloads.hf_hub_download("acme/model", "config.json") == "/file"
    assert calls == [
        ("acme/model", {"revision": "main"}),
        ("acme/model", "config.json", {}),
    ]


def test_policy_resolvers_allow_complete_local_cache(monkeypatch):
    model_downloads.configure(True)

    def snapshot(_repo_id, **kwargs):
        assert kwargs == {"local_files_only": True}
        return "/snapshot"

    def file(_repo_id, _filename, **kwargs):
        assert kwargs == {"local_files_only": True}
        return "/file"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", file)
    assert model_downloads.snapshot_download("acme/model") == "/snapshot"
    assert model_downloads.hf_hub_download("acme/model", "config.json") == "/file"


def test_ddtree_draft_cache_miss_cannot_download(monkeypatch):
    from rapid_mlx.speculative.ddtree.runtime import _resolve_model_path

    model_downloads.configure(True)

    def missing(_repo_id, **kwargs):
        assert kwargs == {"local_files_only": True}
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(ModelDownloadsDisabledError):
        _resolve_model_path("acme/uncached-draft")


def test_indextts_partial_cache_cannot_retry_online(monkeypatch):
    from rapid_mlx.audio.tts import _resolve_indextts_snapshot

    model_downloads.configure(True)
    calls = []

    def missing(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        raise LocalEntryNotFoundError("partial cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(ModelDownloadsDisabledError):
        _resolve_indextts_snapshot("acme/partial-indextts")
    assert calls == [
        (
            "acme/partial-indextts",
            {
                "allow_patterns": [
                    "config.json",
                    "tokenizer.model",
                    "*.safetensors",
                    "*.safetensors.index.json",
                ],
                "local_files_only": True,
            },
        )
    ]


@pytest.mark.parametrize(
    "resolver",
    [
        "rapid_mlx.spec_decode.mtp.gemma4_inject._resolve_sidecar_dir",
        "rapid_mlx.spec_decode.mtp.hy3_inject._resolve_sidecar_file",
        "rapid_mlx.spec_decode.mtp.qwen3_5_inject._resolve_sidecar_file",
    ],
)
def test_remote_mtp_sidecars_propagate_download_policy(monkeypatch, resolver):
    module_name, function_name = resolver.rsplit(".", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    model_downloads.configure(True)

    def missing(_repo_id, **kwargs):
        assert kwargs.get("local_files_only") is True
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(ModelDownloadsDisabledError):
        function("acme/uncached-sidecar")


def test_server_reachable_hub_boundaries_use_the_policy_resolvers():
    root = Path(__file__).parents[1] / "rapid_mlx"
    direct_imports = {}
    for path in root.rglob("*.py"):
        text = path.read_text()
        names = {
            name
            for name in ("snapshot_download", "hf_hub_download")
            if f"from huggingface_hub import {name}" in text
        }
        if names:
            direct_imports[str(path.relative_to(root))] = names

    assert direct_imports == {
        # Central policy implementation.
        "model_downloads.py": {"snapshot_download", "hf_hub_download"},
        # Mirror internals are reached through pinned_snapshot_download's policy
        # guard or the explicit pull command.
        "_mirror.py": {"snapshot_download", "hf_hub_download"},
        # Explicit provisioning/import flows remain exempt by contract.
        "cli.py": {"snapshot_download"},
        "byom/imports.py": {"snapshot_download"},
        "byom/preflight.py": {"hf_hub_download"},
        # Music performs a per-file cache proof before this call.
        "audio/music.py": {"hf_hub_download"},
        # The server backend resolves the pinned release to a local path before
        # entering the integrity-pinned vendored loader.
        "clef/vendor/joint_schema_model.py": {"snapshot_download"},
    }


def test_prefetch_refuses_an_uncached_model_before_any_network_access(
    monkeypatch, capsys, cli
):
    model_downloads.configure(True)
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    monkeypatch.setattr(cli, "_check_disk_space", _fail)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", _fail)
    with pytest.raises(SystemExit) as exc:
        cli._ensure_model_downloaded("acme/model")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "acme/model is not cached" in err
    assert "disabled by --disable-model-downloads" in err
    assert "rapid-mlx pull acme/model" in err


def test_prefetch_keeps_a_cached_model(monkeypatch, cli):
    model_downloads.configure(True)
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: True)
    monkeypatch.setattr(cli, "_check_disk_space", _fail)
    cli._ensure_model_downloaded("acme/model")


def test_audio_engines_refuse_before_loading(monkeypatch, cli):
    from rapid_mlx.audio.stt import STTEngine
    from rapid_mlx.audio.tts import TTSEngine

    model_downloads.configure(True)
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    with pytest.raises(ModelDownloadsDisabledError):
        STTEngine("acme/whisper-uncached").load()
    with pytest.raises(ModelDownloadsDisabledError):
        TTSEngine("acme/tts-uncached").load()


def test_music_weights_come_only_from_the_cache(monkeypatch, tmp_path):
    from rapid_mlx.audio import music

    model_downloads.configure(True)
    lookups = []

    def fake_try_to_load_from_cache(repo_id, filename, *, revision):
        lookups.append((repo_id, filename, revision))

    monkeypatch.setattr(music, "_SA3_MLX_DIR", tmp_path)
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", fake_try_to_load_from_cache
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fail)
    with pytest.raises(ModelDownloadsDisabledError):
        music.MusicEngine()._ensure_weights()
    assert len(lookups) == 1
    assert lookups[0][2] == music._SA3_REVISION


def test_resident_load_refuses_an_uncached_model(monkeypatch, cli):
    from rapid_mlx import server

    model_downloads.configure(True)
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    with pytest.raises(ModelDownloadsDisabledError):
        asyncio.run(server._load_dynamic_resident_model("acme/uncached", None))


def test_models_load_route_answers_404_model_not_found(monkeypatch):
    from rapid_mlx.middleware.exception_handlers import install_exception_handlers
    from rapid_mlx.routes.residency import router

    class Manager:
        async def load(self, model_name, **_kwargs):
            model_downloads.check(model_name)

    model_downloads.configure(True)
    monkeypatch.setattr(
        "rapid_mlx.routes.residency.get_config",
        lambda: SimpleNamespace(residency_manager=Manager()),
    )
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/v1/models/load", json={"model": "acme/uncached"})
    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": (
                "The model `acme/uncached` is not available on this server, "
                "and model downloads are disabled."
            ),
            "type": "not_found_error",
            "code": "model_not_found",
            "param": "model",
        }
    }


def test_parser_accepts_the_flag():
    pytest.importorskip("websockets")
    from rapid_mlx.cli import build_parser

    parser = build_parser()
    for command in ("serve", "chat", "start", "share"):
        args = parser.parse_args([command, "m", "--disable-model-downloads"])
        assert args.disable_model_downloads is True


def test_chat_spawn_forwards_the_flag(monkeypatch, tmp_path, cli):
    captured = {}

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    proc, _ = cli._spawn_chat_server(
        "m", str(tmp_path / "chat.log"), disable_model_downloads=True
    )
    proc._rapid_mlx_log.close()
    assert "--disable-model-downloads" in captured["cmd"]
