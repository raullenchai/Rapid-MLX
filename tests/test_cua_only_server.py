from __future__ import annotations

from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.config import get_config, reset_config


class _ServerStartedError(RuntimeError):
    pass


def _args(**overrides):
    values = {
        "model": None,
        "cua_only": True,
        "host": "127.0.0.1",
        "port": 8123,
        "_port_was_explicit": True,
        "listen_fd": None,
        "log_level": "warning",
        "api_key": None,
        "timeout": 60,
        "max_request_bytes": None,
        "cors_origins": ["http://127.0.0.1"],
        "trusted_hosts": None,
        "rate_limit": 0,
        "watchdog_ppid": None,
        "yes": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cua_only_dispatches_before_model_preflight(monkeypatch):
    dispatched = []

    def start(args):
        dispatched.append(args)

    monkeypatch.setattr(cli, "_serve_cua_only_mode", start)
    cli.serve_command(_args())
    assert len(dispatched) == 1


@pytest.mark.parametrize(
    ("field", "value", "flag"),
    [
        ("model", "qwen", "--model"),
        ("embedding_model", "embed", "--embedding-model"),
        ("enable_audio", True, "--enable-audio"),
        ("mcp_config", "mcp.json", "--mcp-config"),
        ("lazy_load", True, "--lazy-load"),
        ("enable_dflash", True, "--enable-dflash"),
    ],
)
def test_cua_only_rejects_model_lane_options(field, value, flag):
    incompatible = cli._cua_only_incompatible_options(_args(**{field: value}))
    assert flag in incompatible


def test_cua_only_requires_api_key(monkeypatch, capsys):
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_resolve_serve_port", lambda *args, **kwargs: 8123)
    with pytest.raises(SystemExit) as exc:
        cli._serve_cua_only_mode(_args())
    assert exc.value.code == 2
    assert "requires an API key" in capsys.readouterr().err


def test_cua_only_configures_model_free_authenticated_app(monkeypatch):
    from rapid_mlx.cua import server as cua_server

    cfg = reset_config()
    monkeypatch.setattr(cua_server, "_configured", False)
    monkeypatch.setenv("RAPID_MLX_API_KEY", "cua-secret")
    monkeypatch.setattr(cli, "_resolve_serve_port", lambda *args, **kwargs: 8123)

    def started(app, args, log_level, **kwargs):
        assert app is cua_server.app
        assert args.port == 8123
        assert callable(kwargs["on_server_accepting"])
        assert kwargs["proxy_headers"] is False
        raise _ServerStartedError

    monkeypatch.setattr(cli, "_run_uvicorn", started)
    with pytest.raises(_ServerStartedError):
        cli._serve_cua_only_mode(_args())

    cfg = get_config()
    assert cfg.api_key == "cua-secret"
    assert cfg.engine is None
    assert cfg.model_name is None
    assert cfg.model_path is None
    assert cfg.enable_audio_lane is False
    assert cfg.bind_host == "127.0.0.1"
    assert cfg.bind_port == 8123
    assert cfg.cua_permission_requests_enabled is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"host": "0.0.0.0"},
        {"host": "192.0.2.10"},
        {"listen_fd": 7},
    ],
)
def test_cua_permission_prompts_require_explicit_loopback_listener(
    monkeypatch, overrides
):
    from rapid_mlx.cua import server as cua_server

    reset_config()
    monkeypatch.setattr(cua_server, "_configured", False)
    monkeypatch.setenv("RAPID_MLX_API_KEY", "cua-secret")
    monkeypatch.setattr(cli, "_resolve_serve_port", lambda *args, **kwargs: 8123)
    monkeypatch.setattr(
        cli,
        "_run_uvicorn",
        lambda *args, **kwargs: (_ for _ in ()).throw(_ServerStartedError()),
    )

    with pytest.raises(_ServerStartedError):
        cli._serve_cua_only_mode(_args(**overrides))

    assert get_config().cua_permission_requests_enabled is False
