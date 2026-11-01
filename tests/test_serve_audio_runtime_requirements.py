# SPDX-License-Identifier: Apache-2.0
"""Audio serve prepares environment requirements without loading model assets."""

import argparse
import re
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from huggingface_hub import constants

from rapid_mlx import cli, model_downloads
from rapid_mlx.audio import registry, runtime_requirements


@pytest.fixture
def audio_serve(monkeypatch):
    import rapid_mlx
    from rapid_mlx.config import get_config

    events = []
    server = ModuleType("rapid_mlx.server")
    server.app = object()
    server.configure_logging = lambda level: level
    server._sync_config = lambda: None
    server.register_audio_routes_if_enabled = lambda: events.append("routes")
    monkeypatch.setitem(sys.modules, "rapid_mlx.server", server)
    monkeypatch.setattr(rapid_mlx, "server", server, raising=False)
    monkeypatch.setattr(cli, "_configure_engineless_server_security", lambda *a: None)
    monkeypatch.setattr(cli, "_run_uvicorn", lambda *a: events.append("uvicorn"))
    monkeypatch.setattr(cli, "_hard_exit_after_serve", lambda: None)
    monkeypatch.setattr("rapid_mlx.update_refresh.start", lambda **kw: None)
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any", lambda **kw: False
    )
    monkeypatch.delenv("RAPID_MLX_DISABLE_MODEL_DOWNLOADS", raising=False)
    monkeypatch.setattr(model_downloads, "_cli_disabled", False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    cfg = get_config()
    for name in ("context_length", "bind_host", "bind_port", "bind_listen_fd"):
        monkeypatch.setattr(cfg, name, getattr(cfg, name))

    prepare = Mock(side_effect=lambda req: events.append(req))
    monkeypatch.setattr(runtime_requirements, "prepare_runtime_requirement", prepare)

    def serve(alias="kokoro", *, original_alias=None, entry=None):
        entry = entry or registry.resolve_audio_alias(alias)
        assert entry is not None
        args = argparse.Namespace(
            model=entry.hf_id,
            _original_alias=original_alias,
            log_level="info",
            host="127.0.0.1",
            port=8000,
        )
        cli._serve_audio_mode(args, entry)

    return serve, prepare, events


def test_kokoro_prepares_requirement_before_uvicorn(audio_serve, capsys):
    serve, prepare, events = audio_serve
    serve()
    requirement = registry.AudioRuntimeRequirement("spacy_pipeline", "en_core_web_sm")
    prepare.assert_called_once_with(requirement)
    assert events == ["routes", requirement, "uvicorn"]
    output = capsys.readouterr().out
    assert output.index(
        "Runtime requirement: spacy_pipeline en_core_web_sm"
    ) < output.index("Audio mode:")


@pytest.mark.parametrize("policy", ["disabled", "offline"])
def test_serve_skips_preparation_when_downloads_unavailable(
    audio_serve, monkeypatch, capsys, policy
):
    serve, prepare, events = audio_serve
    if policy == "disabled":
        monkeypatch.setenv("RAPID_MLX_DISABLE_MODEL_DOWNLOADS", "1")
    else:
        monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    serve()
    prepare.assert_not_called()
    assert events == ["routes", "uvicorn"]
    captured = capsys.readouterr()
    assert "Runtime requirement:" not in captured.out
    assert captured.err == ""


def test_whisper_serve_has_no_requirements(audio_serve, capsys):
    serve, prepare, events = audio_serve
    serve("whisper-small")
    prepare.assert_not_called()
    assert events == ["routes", "uvicorn"]
    assert "Runtime requirement:" not in capsys.readouterr().out


@pytest.mark.parametrize("original_alias", [None, "my-kokoro"])
def test_preparation_failure_warns_and_continues(audio_serve, capsys, original_alias):
    serve, prepare, events = audio_serve
    prepare.side_effect = runtime_requirements.AudioRuntimePreparationError(
        "unavailable"
    )
    serve(original_alias=original_alias)
    assert events == ["routes", "uvicorn"]
    shown = original_alias or "kokoro"
    assert capsys.readouterr().err == (
        f"  Warning: could not prepare the audio runtime for '{shown}': unavailable\n"
        "  Requests that need it will return 503 until you run "
        f"'rapid-mlx pull {shown}' while online.\n"
    )


def test_missing_audio_runtime_module_skips_silently(audio_serve, monkeypatch, capsys):
    serve, prepare, events = audio_serve
    monkeypatch.setitem(sys.modules, "rapid_mlx.audio.runtime_requirements", None)
    serve()
    prepare.assert_not_called()
    assert events == ["routes", "uvicorn"]
    captured = capsys.readouterr()
    assert "Runtime requirement:" not in captured.out
    assert captured.err == ""


def test_local_kokoro_uses_folder_requirement_lookup(
    audio_serve, monkeypatch, tmp_path
):
    serve, prepare, events = audio_serve
    folder = tmp_path / "local-kokoro"
    folder.mkdir()
    (folder / "config.json").write_text('{"model_type": "kokoro"}')
    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
    entry = registry.resolve_local_audio_folder(str(folder))
    assert entry is not None
    registry.register_served_local_audio_entry(entry, str(folder))
    serve(entry=entry)
    prepare.assert_called_once_with(
        registry.AudioRuntimeRequirement("spacy_pipeline", "en_core_web_sm")
    )
    assert events[-1] == "uvicorn"


def test_pull_preparation_failure_preserves_error_and_exit(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_pull_repository", lambda *a, **kw: None)
    monkeypatch.setattr("rapid_mlx.audio.probe.require_audio_or_exit", lambda *a: None)
    monkeypatch.setattr(
        runtime_requirements,
        "prepare_runtime_requirement",
        Mock(
            side_effect=runtime_requirements.AudioRuntimePreparationError("unavailable")
        ),
    )
    args = argparse.Namespace(
        model="mlx-community/Kokoro-82M-bf16", _original_alias="kokoro"
    )
    with pytest.raises(SystemExit) as error:
        cli.pull_command(args)
    assert error.value.code == 1
    assert capsys.readouterr().err == (
        "\n  Error: Could not prepare audio runtime for 'kokoro'.\n"
        "  unavailable\n"
        "  Install 'rapid-mlx[audio]' in this environment, then rerun "
        "'rapid-mlx pull kokoro'.\n"
    )


def test_readme_tts_curl_fails_on_http_error():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    block = readme.split("# Text to speech\n", 1)[1].split("# Transcription", 1)[0]
    command = re.search(r"curl\b[^\n]*?/v1/audio/speech\b[^\n]*", block)
    assert command is not None
    assert "--fail" in command.group()
