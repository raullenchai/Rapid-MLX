# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx serve --embedding-model X`` without a positional model."""

from __future__ import annotations

import sys
import types
from argparse import Namespace
from unittest import mock

import pytest

from rapid_mlx import cli

EMBED_ID = "mlx-community/all-MiniLM-L6-v2-4bit"


# main() must accept a model-less serve when --embedding-model is given.
def test_main_dispatches_serve_without_model_when_embedding_model_set(capsys):
    seen: list[Namespace] = []
    with (
        mock.patch.object(
            sys,
            "argv",
            ["rapid-mlx", "--no-banner", "serve", "--embedding-model", EMBED_ID],
        ),
        mock.patch.object(cli, "serve_command", side_effect=seen.append),
    ):
        cli.main()

    assert len(seen) == 1
    assert seen[0].model is None
    assert seen[0].embedding_model == EMBED_ID
    assert "a model is required" not in capsys.readouterr().err


# serve_command must fork before any primary-model resolution or download.
def test_serve_command_routes_to_embedding_only_mode(stub_serve_port_resolution):
    args = Namespace(
        model=None,
        embedding_model=EMBED_ID,
        host="127.0.0.1",
        port=None,
        listen_fd=None,
    )
    with (
        mock.patch.object(cli, "_serve_embedding_only_mode") as embed_only,
        mock.patch.object(cli, "_ensure_model_downloaded") as download,
        mock.patch.object(cli, "_serve_audio_mode") as audio,
    ):
        cli.serve_command(args)

    embed_only.assert_called_once_with(args)
    assert args.port == cli.DEFAULT_SERVE_PORT
    download.assert_not_called()
    audio.assert_not_called()


# Primary-model flags would be silently ignored, so they are a usage error.
def test_serve_command_rejects_primary_model_flags_in_embedding_only_mode(capsys):
    args = Namespace(
        model=None,
        embedding_model=EMBED_ID,
        served_model_name="embed",
        lazy_load=True,
        max_tokens=None,
    )
    with (
        mock.patch.object(cli, "_serve_embedding_only_mode") as embed_only,
        mock.patch.object(cli, "_resolve_serve_port") as resolve_port,
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.serve_command(args)

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "embeddings-only" in err
    assert "--served-model-name, --lazy-load" in err
    assert "--max-tokens" not in err
    embed_only.assert_not_called()
    resolve_port.assert_not_called()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max-tokens", "0"),
        ("--idle-unload-seconds", "0"),
        ("--served-model-name", ""),
        ("--mcp-config", ""),
    ],
)
def test_embedding_only_rejects_explicit_falsy_primary_options(flag, value, capsys):
    args = cli.build_parser().parse_args(
        ["serve", "--embedding-model", EMBED_ID, flag, value]
    )
    with (
        mock.patch.object(cli, "_resolve_serve_port") as resolve_port,
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.serve_command(args)
    assert exc_info.value.code == 2
    assert flag in capsys.readouterr().err
    resolve_port.assert_not_called()


def test_embedding_only_rejects_video_output_before_creating_directory(
    tmp_path, capsys
):
    output = tmp_path / "video-output"
    args = cli.build_parser().parse_args(
        ["serve", "--embedding-model", EMBED_ID, "--video-output-dir", str(output)]
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(args)
    assert exc_info.value.code == 2
    assert "--video-output-dir" in capsys.readouterr().err
    assert not output.exists()


@pytest.fixture
def embeddings_only_config():
    from rapid_mlx.config import reset_config

    cfg = reset_config()
    cfg.embedding_engine = object()
    cfg.embedding_model_locked = EMBED_ID
    cfg.ready = True
    yield cfg
    reset_config()


# Load-balancer probes must see the embedding model as the loaded model.
def test_health_probes_report_embedding_model_loaded(embeddings_only_config):
    import asyncio
    import json

    from rapid_mlx.middleware.probe_fastpath import _build_healthz_payload
    from rapid_mlx.routes import health

    full = asyncio.run(health.health())
    assert full["model_loaded"] is True
    assert full["model_type"] == "embedding"
    assert asyncio.run(health.healthz())["model_loaded"] is True
    assert asyncio.run(health.health_ready())["model_loaded"] is True
    status, body = _build_healthz_payload()
    assert status == 200
    assert json.loads(body)["model_loaded"] is True


# An audio server with an embedding sidecar keeps reporting its lazy primary.
def test_embedding_sidecar_does_not_mark_primary_loaded(embeddings_only_config):
    embeddings_only_config.model_name = "mlx-community/Kokoro-82M-bf16"
    assert embeddings_only_config.embeddings_only is False
    assert embeddings_only_config.model_loaded is False


# A positional model keeps the regular text path even with --embedding-model.
def test_serve_command_with_model_does_not_take_embedding_only_mode():
    args = Namespace(model="qwen3.5-4b-4bit", embedding_model=EMBED_ID)
    with (
        mock.patch.object(
            cli, "_serve_embedding_only_mode", side_effect=AssertionError
        ) as embed_only,
        mock.patch.object(
            cli, "_validate_primary_lifecycle_args", side_effect=SystemExit(0)
        ),
        pytest.raises(SystemExit),
    ):
        cli.serve_command(args)
    embed_only.assert_not_called()


# Security config, embedding load, bind stamping, uvicorn and hard exit.
def test_embedding_only_mode_configures_server_and_runs_uvicorn(monkeypatch):
    import rapid_mlx
    from rapid_mlx.config import get_config

    events: list[str] = []
    stub_server = types.ModuleType("rapid_mlx.server")
    stub_server.app = object()
    stub_server.configure_logging = lambda _level: "info"
    stub_server._resolve_api_key = lambda key: key
    stub_server.configure_cors_from_env = lambda _origins: events.append("cors")
    stub_server.configure_trusted_hosts = lambda _hosts: events.append("hosts")
    stub_server._sync_config = lambda: events.append("sync")
    stub_server.load_embedding_model = object()
    monkeypatch.setitem(sys.modules, "rapid_mlx.server", stub_server)
    monkeypatch.setattr(rapid_mlx, "server", stub_server, raising=False)

    def _load(args, load_fn):
        assert load_fn is stub_server.load_embedding_model
        events.append(f"load:{args.embedding_model}")

    uvicorn_kwargs: dict = {}

    def _run(app, _args, _level, **kwargs):
        assert app is stub_server.app
        uvicorn_kwargs.update(kwargs)
        events.append("uvicorn")

    monkeypatch.setattr(cli, "_load_embedding_model_or_exit", _load)
    monkeypatch.setattr(cli, "_run_uvicorn", _run)
    monkeypatch.setattr(cli, "_hard_exit_after_serve", lambda: events.append("exit"))
    cfg = get_config()
    monkeypatch.setattr(cfg, "bind_host", None)
    monkeypatch.setattr(cfg, "bind_port", None)
    monkeypatch.setattr(cfg, "bind_listen_fd", None)

    args = types.SimpleNamespace(
        embedding_model=EMBED_ID,
        host="0.0.0.0",
        port=8123,
        listen_fd=None,
        log_level="info",
        api_key="secret",
        timeout=300,
        max_request_bytes=1024,
        rate_limit=0,
        cors_origins=None,
    )
    cli._serve_embedding_only_mode(args)

    assert events == ["cors", "hosts", "sync", f"load:{EMBED_ID}", "uvicorn", "exit"]
    assert stub_server._api_key == "secret"
    assert stub_server._default_timeout == 300
    assert stub_server._max_request_bytes == 1024
    assert (cfg.bind_host, cfg.bind_port) == ("localhost", 8123)
    assert uvicorn_kwargs["on_server_accepting"]() is None
