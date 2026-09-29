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


def test_cua_only_rejects_model_options_before_binding(monkeypatch, capsys):
    monkeypatch.setenv("RAPID_MLX_API_KEY", "cua-secret")
    monkeypatch.setattr(
        cli,
        "_resolve_serve_port",
        lambda *args, **kwargs: pytest.fail("model options must be rejected first"),
    )
    with pytest.raises(SystemExit) as exc:
        cli._serve_cua_only_mode(_args(model="unexpected"))
    assert exc.value.code == 2
    assert "--model" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("environment", "overrides", "expected_bytes", "expected_timeout"),
    [
        ({}, {"max_request_bytes": 1024}, 1024, None),
        ({"RAPID_MLX_MAX_REQUEST_BYTES": "2048"}, {}, 2048, None),
        ({"RAPID_MLX_MAX_REQUEST_BYTES": "bad"}, {}, 8 * 1024 * 1024, None),
        ({"RAPID_MLX_BODY_RECEIVE_TIMEOUT_SECONDS": "0.5"}, {}, None, 0.5),
        ({"RAPID_MLX_BODY_RECEIVE_TIMEOUT_SECONDS": "bad"}, {}, None, 15.0),
    ],
)
def test_cua_only_request_limits_use_validated_cli_or_environment(
    monkeypatch, environment, overrides, expected_bytes, expected_timeout
):
    from rapid_mlx.cua import server as cua_server

    reset_config()
    monkeypatch.setattr(cua_server, "_configured", False)
    monkeypatch.setenv("RAPID_MLX_API_KEY", "cua-secret")
    for name in (
        "RAPID_MLX_MAX_REQUEST_BYTES",
        "RAPID_MLX_BODY_RECEIVE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cli, "_resolve_serve_port", lambda *args, **kwargs: 8123)
    monkeypatch.setattr(cli, "_run_uvicorn", lambda *args, **kwargs: None)
    exited = []
    monkeypatch.setattr(cli, "_hard_exit_after_serve", lambda: exited.append(True))

    cli._serve_cua_only_mode(_args(**overrides))
    cfg = get_config()
    if expected_bytes is not None:
        assert cfg.max_request_bytes == expected_bytes
    if expected_timeout is not None:
        assert cfg.body_receive_timeout_seconds == expected_timeout
    assert exited == [True]


def test_cua_only_rate_limit_is_applied_and_announced(monkeypatch, capsys):
    from rapid_mlx.cua import server as cua_server

    reset_config()
    monkeypatch.setattr(cua_server, "_configured", False)
    monkeypatch.setenv("RAPID_MLX_API_KEY", "cua-secret")
    monkeypatch.setattr(cli, "_resolve_serve_port", lambda *args, **kwargs: 8123)
    applied = []
    monkeypatch.setattr(
        "rapid_mlx.middleware.auth.configure_rate_limiter",
        lambda limit, enabled: applied.append((limit, enabled)),
    )
    monkeypatch.setattr(
        cli,
        "_run_uvicorn",
        lambda *args, **kwargs: (_ for _ in ()).throw(_ServerStartedError()),
    )
    with pytest.raises(_ServerStartedError):
        cli._serve_cua_only_mode(_args(rate_limit=7))
    assert applied == [(7, True)]
    assert "rate-limit: 7/min" in capsys.readouterr().out


def test_cua_only_invalid_listener_hostname_disables_permission_prompt(monkeypatch):
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
        cli._serve_cua_only_mode(_args(host="example.invalid"))
    assert get_config().cua_permission_requests_enabled is False
