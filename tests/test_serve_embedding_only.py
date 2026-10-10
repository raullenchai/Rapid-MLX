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


@pytest.mark.parametrize(
    ("options", "explicit", "seconds"),
    [
        ([], False, 0.0),
        (["--idle-unload-seconds=0"], True, 0.0),
        (["--idle-unload-seconds", "60"], True, 60.0),
    ],
)
def test_embedding_only_parser_preserves_idle_default_and_provenance(
    options, explicit, seconds
):
    args = cli.build_parser().parse_args(
        ["serve", "--embedding-model", EMBED_ID, *options]
    )
    assert args.idle_unload_seconds == seconds
    assert ("--idle-unload-seconds" in args._serve_explicit_options) is explicit


@pytest.mark.parametrize(
    "options",
    [
        ["--max-num-seqs", "16"],
        ["--max-num-seqs=16"],
        ["--default-temperature", "0"],
        ["--default-top-p", "1"],
        ["--disable-prefix-cache"],
        ["--no-thinking"],
        ["--no-think"],
        ["--text-only"],
        ["--simple-engine"],
        ["--force-hybrid"],
        ["--vision-min-pixels", "0"],
        ["--gpu-memory-utilization", "0.9"],
    ],
)
def test_embedding_only_rejects_explicit_generation_tuning(options, capsys):
    args = cli.build_parser().parse_args(
        ["serve", "--embedding-model", EMBED_ID, *options]
    )
    with (
        mock.patch.object(cli, "_resolve_serve_port") as resolve_port,
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.serve_command(args)
    assert exc_info.value.code == 2
    assert "embeddings-only" in capsys.readouterr().err
    resolve_port.assert_not_called()


def test_embedding_only_accepts_shared_and_embedding_options():
    args = cli.build_parser().parse_args(
        [
            "--no-banner",
            "serve",
            "--embedding-model",
            EMBED_ID,
            "--embedding-max-length=256",
            "--embedding-overflow-policy",
            "error",
            "--host=127.0.0.1",
            "--port=8123",
            "--api-key=local-test",
            "--rate-limit=30",
            "--max-request-bytes=4096",
            "--timeout=30",
            "--cors-origins=http://localhost",
            "--trusted-hosts=localhost",
            "--log-level=info",
            "--disable-disk-caches",
            "--disable-model-downloads",
            "-y",
        ]
    )
    assert cli._embedding_only_incompatible_options(args) == []


@pytest.mark.parametrize("flag", ["--lazy", "--lazy-l", "--embedding-mo=other"])
def test_embedding_only_parser_rejects_abbreviated_options(flag, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["serve", "--embedding-model", EMBED_ID, flag])
    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


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
    assert full["model_name"] is None
    assert full["model_type"] == "embedding"
    assert asyncio.run(health.healthz())["model_loaded"] is True
    ready = asyncio.run(health.health_ready())
    assert ready["model_loaded"] is True
    assert ready["model"] is None
    status, body = _build_healthz_payload()
    assert status == 200
    assert json.loads(body)["model_loaded"] is True
    assert json.loads(body)["model_name"] is None


@pytest.mark.requires_mlx
@pytest.mark.asyncio
async def test_embedding_only_lifespan_sets_and_clears_readiness(monkeypatch):
    from contextlib import asynccontextmanager

    from rapid_mlx import server
    from rapid_mlx.config import reset_config
    from rapid_mlx.routes import audio, video

    cfg = reset_config()
    cfg.embedding_engine = object()
    cfg.embedding_model_locked = EMBED_ID
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_primary_model_lifecycle", None)
    monkeypatch.setattr(server, "_mcp_manager", None)
    monkeypatch.setattr(
        server,
        "_residency_manager",
        types.SimpleNamespace(start=mock.AsyncMock(), shutdown=mock.AsyncMock()),
    )
    monkeypatch.setattr(video, "start_video_jobs", mock.Mock())
    monkeypatch.setattr(video, "shutdown_video_jobs", mock.AsyncMock())
    monkeypatch.setattr(audio, "shutdown_audio_lanes", mock.AsyncMock())
    assert cfg.ready is False
    try:
        async with asynccontextmanager(server.lifespan)(server.app):
            assert cfg.ready is True
            assert cfg.model_loaded is True
        assert cfg.ready is False
        assert cfg.draining is True
    finally:
        reset_config()


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
@pytest.mark.parametrize("listen_fd", [None, 7])
def test_embedding_only_mode_configures_server_and_runs_uvicorn(monkeypatch, listen_fd):
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
        listen_fd=listen_fd,
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
    assert (cfg.bind_host, cfg.bind_port, cfg.bind_listen_fd) == (
        ("localhost", 8123, None) if listen_fd is None else (None, None, listen_fd)
    )
    assert uvicorn_kwargs["on_server_accepting"]() is None


@pytest.mark.parametrize(
    ("body_limit", "expected"),
    [("4096", 4096), ("-1", 0), ("invalid", 8 * 1024 * 1024)],
)
def test_engineless_security_applies_environment_limit_and_rate_limiter(
    monkeypatch, body_limit, expected
):
    from rapid_mlx.middleware import auth

    monkeypatch.setenv("RAPID_MLX_MAX_REQUEST_BYTES", body_limit)
    limiter = object()
    configure_limiter = mock.Mock(return_value=limiter)
    monkeypatch.setattr(auth, "configure_rate_limiter", configure_limiter)
    server = types.SimpleNamespace(
        _resolve_api_key=lambda key: key,
        configure_cors_from_env=mock.Mock(),
        configure_trusted_hosts=mock.Mock(),
    )
    args = Namespace(
        api_key="test-key",
        timeout=30,
        max_request_bytes=None,
        cors_origins=None,
        trusted_hosts=None,
        rate_limit=30,
    )
    cli._configure_engineless_server_security(args, server)
    assert server._max_request_bytes == expected
    assert server._rate_limiter is limiter
    configure_limiter.assert_called_once_with(30, enabled=True)
