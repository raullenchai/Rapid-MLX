"""Stable controller contracts for process state and the cached-model alias."""

import json
from types import SimpleNamespace

import psutil
import pytest

from rapid_mlx import cli


@pytest.mark.parametrize(
    "host,url",
    [
        ("192.168.1.4", "http://192.168.1.4:8123"),
        ("::1", "http://[::1]:8123"),
        ("0.0.0.0", "http://127.0.0.1:8123"),
    ],
)
def test_ps_json_schema(monkeypatch, capsys, host, url):
    proc = SimpleNamespace(
        environ=lambda: {},
        net_connections=lambda **_: [],
        info={
            "pid": 123,
            "create_time": 1234.5,
            "cmdline": [
                "rapid-mlx",
                "serve",
                "tiny",
                "--host=" + host,
                "--port",
                "8123",
                "--served-model-name",
                "public-model",
                "--api-key=SECRET",
                "--log-file",
                "/tmp/server.log",
            ],
        },
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *_: [proc])
    monkeypatch.setattr(cli, "_ps_ready", lambda base_url: base_url == url)
    cli.ps_command(cli.build_parser().parse_args(["ps", "--json"]))
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "SECRET" not in captured.out
    assert json.loads(captured.out) == {
        "schema_version": 1,
        "servers": [
            {
                "pid": 123,
                "bind_host": host,
                "port": 8123,
                "base_url": url,
                "model_ids": ["public-model"],
                "auth_required": True,
                "ready": True,
                "started_at": 1234.5,
                "log_path": "/tmp/server.log",
            }
        ],
    }


def test_ps_json_empty(monkeypatch, capsys):
    monkeypatch.setattr(psutil, "process_iter", lambda *_: [])
    cli.ps_command(cli.build_parser().parse_args(["ps", "--json"]))
    assert json.loads(capsys.readouterr().out) == {"schema_version": 1, "servers": []}


def test_ls_json_uses_cached_payload(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "ls", "--json"])
    monkeypatch.setattr(cli, "_cached_models_json_payload", lambda: {"models": []})
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"models": []}


@pytest.mark.parametrize(
    "status,expected", [(200, True), (503, False), (401, None), (302, None)]
)
def test_readiness_probe(monkeypatch, status, expected):
    import requests

    session = SimpleNamespace(trust_env=True)

    def get(url, **kwargs):
        assert url == "http://127.0.0.1:8123/health/ready"
        assert kwargs == {"timeout": 0.25, "allow_redirects": False}
        assert session.trust_env is False
        return SimpleNamespace(status_code=status)

    session.get = get

    class Session:
        def __enter__(self):
            return session

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(requests, "Session", Session)
    assert cli._ps_ready("http://127.0.0.1:8123") is expected


def test_readiness_connection_failure(monkeypatch):
    import requests

    def failed(*args, **kwargs):
        raise requests.ConnectionError()

    monkeypatch.setattr(requests.Session, "get", failed)
    assert cli._ps_ready("http://127.0.0.1:8123") is None


@pytest.mark.parametrize(
    "env,auth,log",
    [
        ({}, False, None),
        (
            {"RAPID_MLX_API_KEY": "SECRET", "RAPID_MLX_LOG_FILE": "/tmp/server.log"},
            True,
            "/tmp/server.log",
        ),
        (None, None, None),
    ],
)
def test_unknown_and_environment_state(monkeypatch, env, auth, log):
    def environ():
        if env is None:
            raise psutil.AccessDenied()
        return env

    def connections(**_):
        raise psutil.NoSuchProcess(123)

    proc = SimpleNamespace(
        info={"pid": 123}, environ=environ, net_connections=connections
    )
    state = cli._ps_server_state(proc, {"--port": "0"}, "(unknown)")
    assert state == {
        "pid": 123,
        "bind_host": "127.0.0.1",
        "port": None,
        "base_url": None,
        "model_ids": [],
        "auth_required": auth,
        "ready": None,
        "started_at": None,
        "log_path": log,
    }
    assert "SECRET" not in json.dumps(state)


@pytest.mark.parametrize("requested", ["0", "8123"])
def test_listener_overrides_argv(monkeypatch, requested):
    address = SimpleNamespace(ip="::", port=8123)
    proc = SimpleNamespace(
        info={"pid": 123},
        environ=lambda: {},
        net_connections=lambda **_: [
            SimpleNamespace(status=psutil.CONN_LISTEN, laddr=address)
        ],
    )
    monkeypatch.setattr(cli, "_ps_ready", lambda _: False)
    state = cli._ps_server_state(proc, {"--port": requested}, "tiny")
    assert state["bind_host"] == "::"
    assert state["port"] == 8123
    assert state["base_url"] == "http://[::1]:8123"
    assert state["ready"] is False


def test_separate_api_key_and_port_before_model(monkeypatch, capsys):
    proc = SimpleNamespace(
        info={
            "pid": 123,
            "create_time": 1234.5,
            "cmdline": [
                "rapid-mlx",
                "serve",
                "--port=8123",
                "--api-key",
                "SECRET",
                "tiny",
                "--served-model-name=public-model",
            ],
        },
        environ=lambda: {},
        net_connections=lambda **_: [],
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *_: [proc])
    monkeypatch.setattr(cli, "_ps_ready", lambda _: None)
    cli.ps_command(SimpleNamespace(json=True))
    output = capsys.readouterr().out
    assert "SECRET" not in output
    assert json.loads(output)["servers"][0]["model_ids"] == ["public-model"]


@pytest.mark.parametrize("key_args", [["--api-k", "SECRET"], ["--api-k=SECRET"]])
def test_unsupported_abbreviated_key_fails_closed(monkeypatch, capsys, key_args):
    proc = SimpleNamespace(
        info={
            "pid": 123,
            "create_time": 1234.5,
            "cmdline": ["rapid-mlx", "serve", *key_args, "tiny"],
        },
        environ=lambda: {},
        net_connections=lambda **_: [],
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *_: [proc])
    monkeypatch.setattr(cli, "_ps_ready", lambda _: None)
    cli.ps_command(SimpleNamespace(json=True))
    output = capsys.readouterr().out
    assert "SECRET" not in output
    state = json.loads(output)["servers"][0]
    assert state["auth_required"] is None
    assert state["model_ids"] == []


@pytest.mark.parametrize("json_output", [True, False])
@pytest.mark.parametrize(
    "argv,parsed",
    [
        (["--kv-cache-turboquant", "--api-key", "SECRET", "tiny"], True),
        (["tiny", "--port", "SECRET", "--api-key=SECRET"], False),
        (["--api-key=SECRET", "--", "tiny"], True),
        (["tiny", "--trusted-hosts", "one", "two", "--api-key=SECRET"], True),
        (["tiny", "--api-key=SECRET", "--host"], False),
        (["tiny", "--api-key=SECRET", "--help"], False),
        (["--future-option", "SECRET", "--api-key=SECRET"], False),
        (["--api-key=SECRET", "--served-model-name=SECRET", "SECRET"], True),
        (["--api-key=SECRET", "--api-key=OTHER", "SECRET"], True),
    ],
)
def test_process_credentials_never_reach_output(
    monkeypatch, capsys, json_output, argv, parsed
):
    proc = SimpleNamespace(
        info={
            "pid": 123,
            "create_time": 1234.5,
            "cmdline": ["rapid-mlx", "serve", *argv],
        },
        environ=lambda: {"RAPID_MLX_API_KEY": "ENV_SECRET"},
        net_connections=lambda **_: [],
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *_: [proc])
    monkeypatch.setattr(cli, "_ps_ready", lambda _: None)
    monkeypatch.setattr(
        "rapid_mlx._version_check.print_staleness_warning_if_any", lambda: None
    )
    cli.ps_command(SimpleNamespace(json=json_output))
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "SECRET" not in captured.out
    assert "OTHER" not in captured.out
    if json_output:
        state = json.loads(captured.out)["servers"][0]
        assert state["auth_required"] is (True if parsed else None)
        if not parsed:
            assert state["model_ids"] == []
            assert state["bind_host"] == "127.0.0.1"
            assert state["port"] == 8000
            assert state["log_path"] is None
        elif "tiny" in argv:
            assert state["model_ids"] == ["tiny"]
    elif not parsed:
        assert "(unknown)" in captured.out


@pytest.mark.parametrize("parse_failure", [True, False])
def test_credential_redaction_in_all_metadata(monkeypatch, capsys, parse_failure):
    argv = [
        "tiny",
        "--api-key=SECRET",
        "--host=SECRET",
        "--log-file=/tmp/SECRET-ENV_SECRET.log",
        "--port=0",
    ]
    if parse_failure:
        argv.append("--port=invalid")
    proc = SimpleNamespace(
        info={
            "pid": 123,
            "create_time": 1234.5,
            "cmdline": ["rapid-mlx", "serve", *argv],
        },
        environ=lambda: {
            "RAPID_MLX_API_KEY": "ENV_SECRET",
            "RAPID_MLX_LOG_FILE": "/tmp/SECRET-ENV_SECRET.log",
        },
        net_connections=lambda **_: [],
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *_: [proc])
    monkeypatch.setattr(cli, "_ps_ready", lambda _: None)
    cli.ps_command(SimpleNamespace(json=True))
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "SECRET" not in captured.out
    state = json.loads(captured.out)["servers"][0]
    assert state["auth_required"] is (None if parse_failure else True)
    assert state["port"] == (8000 if parse_failure else None)
    assert state["log_path"] == "/tmp/[redacted]-[redacted].log"
    if not parse_failure:
        assert state["bind_host"] == "[redacted]"
