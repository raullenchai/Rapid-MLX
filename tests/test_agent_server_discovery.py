# SPDX-License-Identifier: Apache-2.0
"""Setup discovery and startup lifetime contracts, with a fake clock."""

import json
import urllib.error
from io import BytesIO
from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.agents import adapter, setup, telemetry
from rapid_mlx.agents import server_discovery as discovery
from rapid_mlx.agents.server_hint import ServerNotRunningError
from rapid_mlx.cli_parser import build_parser


@pytest.fixture(autouse=True)
def no_live_http(monkeypatch):
    def refuse_http(*_args, **_kwargs):
        raise ConnectionRefusedError()

    monkeypatch.setattr(discovery.urllib.request, "urlopen", refuse_http)


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(setup.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        setup.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    monkeypatch.setattr(discovery.Progress, "show", lambda *_a: None)
    monkeypatch.setattr(discovery, "scan_servers", lambda: [])
    return now


@pytest.fixture
def events(monkeypatch):
    recorded = []
    for module in (setup, adapter, telemetry):
        monkeypatch.setattr(
            module,
            "track_agent_configure_failed",
            lambda *v: recorded.append(("failed", *v)),
        )
        if hasattr(module, "track_agent_configured"):
            monkeypatch.setattr(
                module,
                "track_agent_configured",
                lambda *v: recorded.append(("configured", *v)),
            )
    return recorded


def args(agent, model=None, port=8462):
    return SimpleNamespace(
        agent_name=agent,
        model=model,
        port=port,
        base_url=None,
        test=False,
        setup=True,
        dry_run=False,
        no_check=False,
        yes=True,
        agent_version=None,
    )


@pytest.fixture
def configs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        setup.claude_code,
        "current_config_path",
        lambda: tmp_path / ".claude/settings.json",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path


def refuse(*_a, **_k):
    raise ServerNotRunningError("no server is running at http://localhost:8462")


@pytest.mark.parametrize("agent", ["claude-code", "opencode", "codex"])
@pytest.mark.parametrize("model", [None, "tiny-model"])
def test_offline_setup(clock, events, configs, monkeypatch, capsys, agent, model):
    monkeypatch.setattr(setup, "verify_server", refuse)
    monkeypatch.setattr("rapid_mlx.run.cli._cached_context_window", lambda _m: 8192)
    if model:
        cli.agents_command(args(agent, model))
        paths = list(configs.rglob("*.json" if agent != "codex" else "config.toml"))
        assert paths
        contents = "\n".join(p.read_text() for p in paths)
        assert "localhost:8462" in contents
        assert "tiny-model" in contents
        assert events == [("configured", agent)]
        assert "--port 8462" in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit) as exc:
            cli.agents_command(args(agent))
        assert exc.value.code == 1
        assert not list(configs.rglob("config.toml"))
        assert not list(configs.rglob("opencode.json"))
        assert not (configs / ".claude/settings.json").exists()
        assert events == [("failed", "server_not_running", agent)]
        assert "http://localhost:8462" in capsys.readouterr().out
    assert clock[0] < 3


@pytest.mark.parametrize("agent", ["claude-code", "opencode", "codex"])
@pytest.mark.parametrize("outcome", ["ready", "died", "cancel"])
def test_startup_lifetime(clock, events, configs, monkeypatch, capsys, agent, outcome):
    server = discovery.Server(123, 8462, "tiny-model")
    monkeypatch.setattr(discovery, "scan_servers", lambda: [server])
    monkeypatch.setattr(
        discovery, "pid_alive", lambda _pid: outcome != "died" or clock[0] < 70
    )

    def probe(*_a, **_k):
        if clock[0] >= 75:
            if outcome == "cancel":
                raise KeyboardInterrupt
            return "tiny-model"
        if clock[0] < 2:
            refuse()
        raise setup.ServerLoadingError("http://localhost:8462/health returned HTTP 503")

    monkeypatch.setattr(setup, "verify_server", probe)
    monkeypatch.setattr(adapter, "fetch_context_window", lambda *_a: 8192)
    if outcome == "ready":
        cli.agents_command(args(agent))
        assert events == [("configured", agent)]
        assert clock[0] > 60
    else:
        with pytest.raises(SystemExit) as exc:
            cli.agents_command(args(agent))
        assert exc.value.code == (130 if outcome == "cancel" else 1)
        assert events == (
            [] if outcome == "cancel" else [("failed", "server_not_ready", agent)]
        )
        assert not list(configs.rglob("config.toml"))
        assert not list(configs.rglob("opencode.json"))
        assert not (configs / ".claude/settings.json").exists()
        output = capsys.readouterr().out
        assert (
            "nothing was written" in output
            if outcome == "cancel"
            else ":8462" in output
        )


@pytest.mark.parametrize("agent", ["claude-code", "opencode", "codex"])
def test_discovered_nondefault_port_is_written(
    clock, events, configs, monkeypatch, agent
):
    monkeypatch.setattr(
        discovery.urllib.request,
        "urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(ConnectionRefusedError()),
    )
    monkeypatch.setattr(
        discovery, "scan_servers", lambda: [discovery.Server(123, 8462, "tiny-model")]
    )
    monkeypatch.setattr(discovery, "pid_alive", lambda _pid: True)
    monkeypatch.setattr(setup, "verify_server", lambda url, *_a, **_k: "tiny-model")
    monkeypatch.setattr(adapter, "fetch_context_window", lambda *_a: 8192)
    cli.agents_command(args(agent, port=None))
    paths = list(configs.rglob("*.json" if agent != "codex" else "config.toml"))
    assert any("localhost:8462" in p.read_text() for p in paths)


def test_ambiguous_servers_and_model_selection(monkeypatch):
    monkeypatch.setattr(
        discovery.urllib.request,
        "urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(ConnectionRefusedError()),
    )
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: [
            discovery.Server(1, 8462, "a"),
            discovery.Server(2, 8463, "b", "public-b"),
        ],
    )
    with pytest.raises(RuntimeError, match="Pass --port") as exc:
        discovery.resolve_endpoint()
    assert all(s in str(exc.value) for s in ["8462", "8463", "pid 1", "pid 2"])
    assert discovery.resolve_endpoint("public-b").base_url == "http://localhost:8463/v1"


def test_port_parser():
    parser = build_parser()
    assert parser.parse_args(["agents", "codex", "--port", "8462"]).port == 8462
    for options in (
        ["--port", "0"],
        ["--port", "8462", "--base-url", "http://localhost:8000"],
    ):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["agents", "codex", *options])
        assert exc.value.code == 2


def test_default_ready_endpoint_takes_priority(monkeypatch):
    monkeypatch.setattr(discovery, "_default_endpoint_ready", lambda: (True, True))

    class Response(BytesIO):
        status = 200

    monkeypatch.setattr(
        discovery.urllib.request, "urlopen", lambda *_a, **_k: Response(b"{}")
    )
    monkeypatch.setattr(
        discovery, "scan_servers", lambda: pytest.fail("default endpoint already ready")
    )
    endpoint = discovery.resolve_endpoint("tiny-model")
    assert endpoint == discovery.Endpoint(
        "http://localhost:8000/v1", follow=True, model="tiny-model"
    )
    endpoint.refresh()
    assert endpoint == discovery.Endpoint(
        "http://localhost:8000/v1", follow=True, model="tiny-model"
    )


@pytest.mark.parametrize("requested", [None, "model-a", "public-a", "model-b"])
def test_loading_default_respects_requested_model(monkeypatch, requested):
    default = discovery.Server(1, 8000, "model-a", "public-a")
    other = discovery.Server(2, 8462, "model-b")
    monkeypatch.setattr(discovery, "_default_endpoint_ready", lambda: (False, True))
    monkeypatch.setattr(discovery, "scan_servers", lambda: [default, other])

    endpoint = discovery.resolve_endpoint(requested)

    expected = other if requested == "model-b" else default
    assert endpoint.server == expected
    assert endpoint.base_url == f"http://localhost:{expected.port}/v1"


@pytest.mark.parametrize(
    "failure,path",
    [
        (401, "/health"),
        (403, "/v1/models"),
        (500, "/health"),
        (503, "/v1/models"),
        ("timeout", "/health"),
        ("bad-json", "/v1/models"),
        ("bad-shape", "/v1/models"),
    ],
)
def test_live_serve_does_not_unbound_verification_errors(
    clock, events, monkeypatch, failure, path
):
    server = discovery.Server(123, 8462, "tiny-model")
    monkeypatch.setattr(discovery, "scan_servers", lambda: [server])
    monkeypatch.setattr(discovery, "pid_alive", lambda _pid: True)

    class Response(BytesIO):
        status = 200

    def response(request, **_kwargs):
        # Fail promptly if the bounded wait regresses into an endless loop.
        assert clock[0] <= 60
        url = request if isinstance(request, str) else request.full_url
        if url.endswith(path):
            if isinstance(failure, int):
                raise urllib.error.HTTPError(url, failure, "failed", {}, BytesIO())
            if failure == "timeout":
                raise TimeoutError("request timed out")
            return Response(
                b"invalid JSON" if failure == "bad-json" else b'{"data": null}'
            )
        return Response(b"{}")

    monkeypatch.setattr(setup.urllib.request, "urlopen", response)
    if failure == "bad-shape":
        with pytest.raises(setup.ServerNoModelsError, match="reported no models"):
            setup.wait_for_server(
                "http://localhost:8462/v1", "tiny-model", agent="codex"
            )
        assert clock[0] == 0
        assert events == [("failed", "server_no_models", "codex")]
        return
    with pytest.raises(RuntimeError, match="http://localhost:8462/v1") as raised:
        setup.wait_for_server("http://localhost:8462/v1", "tiny-model", agent="codex")
    assert clock[0] == 60
    assert events == [("failed", "server_not_ready", "codex")]
    if failure in (401, 403):
        assert "authentication failed" in str(raised.value)
        assert "RAPID_MLX_API_KEY" in str(raised.value)
    elif isinstance(failure, int):
        assert f"HTTP {failure}" in str(raised.value)


@pytest.mark.parametrize("local", [False, True])
def test_verification_allows_two_second_responses(clock, events, monkeypatch, local):
    base_url = "http://localhost:8462/v1" if local else "https://example.com/v1"
    if local:
        monkeypatch.setattr(
            discovery,
            "scan_servers",
            lambda: [discovery.Server(123, 8462, "tiny-model")],
        )
        monkeypatch.setattr(discovery, "pid_alive", lambda _pid: True)
    calls = []

    class Response(BytesIO):
        status = 200

    def response(request, *, timeout):
        url = request if isinstance(request, str) else request.full_url
        calls.append(url)
        # Simulate a server that needs more than 0.5 s for each response.
        if timeout < 1.0:
            raise TimeoutError("response needs one second")
        clock[0] += 1
        return Response(
            b'{"data": [{"id": "tiny-model"}]}' if url.endswith("/models") else b"{}"
        )

    monkeypatch.setattr(setup.urllib.request, "urlopen", response)
    assert setup.wait_for_server(base_url, "tiny-model", agent="codex") == "tiny-model"
    assert len(calls) == 2
    assert clock[0] == 2
    assert events == []


@pytest.mark.parametrize("startup", ["refused", "503", "503-response"])
def test_only_recognized_startup_can_outlast_deadline(
    clock, events, monkeypatch, startup
):
    server = discovery.Server(123, 8462, "tiny-model")
    endpoint = discovery.Endpoint("http://localhost:8462/v1", server)
    monkeypatch.setattr(discovery, "pid_alive", lambda _pid: True)

    class Response(BytesIO):
        status = 200

    def response(request, *, timeout):
        assert timeout == 2.0
        url = request if isinstance(request, str) else request.full_url
        if clock[0] < 75:
            assert url.endswith("/health")
            if startup == "refused":
                raise urllib.error.URLError(ConnectionRefusedError())
            if startup == "503":
                raise urllib.error.HTTPError(url, 503, "loading", {}, BytesIO())
            loading = Response(b"{}")
            loading.status = 503
            return loading
        return Response(
            b'{"data": [{"id": "tiny-model"}]}' if url.endswith("/models") else b"{}"
        )

    monkeypatch.setattr(setup.urllib.request, "urlopen", response)
    assert (
        setup.wait_for_server(
            endpoint.base_url, "tiny-model", agent="codex", endpoint=endpoint
        )
        == "tiny-model"
    )
    assert clock[0] == 75
    assert events == []


def test_progress_reports_phase_and_measured_bytes(monkeypatch, capsys):
    payload = {"phase": "loading_weights", "progress": {"done": 42, "total": 100}}

    def response(*_a, **_k):
        return BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(discovery.urllib.request, "urlopen", response)
    progress = discovery.Progress()
    progress.show("http://localhost:8462/v1", 0)
    progress.show("http://localhost:8462/v1", 1)
    assert capsys.readouterr().out.count("loading_weights 42%") == 1
    payload["phase"] = "initializing"
    progress.show("http://localhost:8462/v1", 2)
    assert "initializing" in capsys.readouterr().out


def test_scan_prefers_socket_and_ignores_caffeinate(
    monkeypatch, agent_discovery_defaults
):
    import psutil

    proc = SimpleNamespace(
        pid=123,
        info={
            "cmdline": ["python", "/tmp/rapid-mlx", "serve", "tiny", "--port", "8000"]
        },
        net_connections=lambda **_k: [
            SimpleNamespace(
                status=psutil.CONN_LISTEN,
                laddr=SimpleNamespace(ip="127.0.0.1", port=8462),
            )
        ],
    )
    wrapper = SimpleNamespace(
        info={"cmdline": ["/usr/bin/caffeinate", "rapid-mlx", "serve", "tiny"]}
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *_a: [proc, wrapper])
    assert agent_discovery_defaults() == [discovery.Server(123, 8462, "tiny")]


@pytest.mark.parametrize("port_arg", ["0", "not-a-port"])
def test_scan_keeps_server_with_unusable_argv_port(
    monkeypatch, port_arg, agent_discovery_defaults
):
    """``serve --port 0`` must not crash setup or hide the server (dogfood)."""
    import psutil

    proc = SimpleNamespace(
        pid=321,
        info={"cmdline": ["python", "rapid-mlx", "serve", "tiny", "--port", port_arg]},
        net_connections=lambda **_k: [
            SimpleNamespace(
                status=psutil.CONN_LISTEN,
                laddr=SimpleNamespace(ip="0.0.0.0", port=59001),
            )
        ],
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *_a: [proc])
    assert agent_discovery_defaults() == [discovery.Server(321, 59001, "tiny")]


def test_foreign_unhealthy_listener_keeps_bounded_failure(clock, events, monkeypatch):
    def unhealthy(*_a, **_k):
        raise RuntimeError("health at http://localhost:8462 returned HTTP 503")

    monkeypatch.setattr(setup, "verify_server", unhealthy)
    with pytest.raises(RuntimeError, match="8462"):
        setup.wait_for_server("http://localhost:8462/v1", "tiny-model", agent="codex")
    assert clock[0] == 60
    assert events == [("failed", "server_not_ready", "codex")]


def test_unknown_port_follows_listener_only_for_implicit_endpoint(monkeypatch):
    bound = discovery.Server(123, 8007, "tiny")
    monkeypatch.setattr(discovery, "scan_servers", lambda: [bound])
    implicit = discovery.Endpoint(
        "http://localhost:8000/v1", discovery.Server(123, None, "tiny"), follow=True
    )
    explicit = discovery.Endpoint(
        "http://localhost:8000/v1", discovery.Server(123, None, "tiny")
    )
    implicit.refresh()
    explicit.refresh()
    assert implicit.base_url == "http://localhost:8007/v1"
    assert explicit.base_url == "http://localhost:8000/v1"
    assert explicit.server is None


@pytest.mark.parametrize("port_args,expected", [([], None), (["--port=8462"], 8462)])
def test_scan_access_denied_uses_argv_or_unknown(
    monkeypatch, port_args, expected, agent_discovery_defaults
):
    import psutil

    def denied(**_k):
        raise psutil.AccessDenied(123)

    proc = SimpleNamespace(
        pid=123,
        info={
            "cmdline": [
                "python",
                "-m",
                "rapid_mlx.cli",
                "serve",
                "--host",
                "localhost",
                "tiny",
                "--served-model-name=public",
                *port_args,
            ]
        },
        net_connections=denied,
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *_a: [proc])
    assert agent_discovery_defaults() == [
        discovery.Server(123, expected, "tiny", "public")
    ]


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
def test_real_health_checks_through_download_and_loading(
    events, configs, monkeypatch, capsys, agent
):
    now = [0.0]
    monkeypatch.setattr(setup.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        setup.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    monkeypatch.setattr(
        discovery, "scan_servers", lambda: [discovery.Server(123, 8462, "tiny-model")]
    )
    monkeypatch.setattr(discovery, "pid_alive", lambda _pid: True)
    monkeypatch.setattr(adapter, "fetch_context_window", lambda *_a: 8192)

    class Response(BytesIO):
        status = 200

    def open_url(request, **_kwargs):
        url = request if isinstance(request, str) else request.full_url
        assert url.startswith("http://localhost:8462/")
        if now[0] < 2:
            raise urllib.error.URLError(ConnectionRefusedError())
        if now[0] < 75:
            body = json.dumps(
                {"phase": "loading_weights", "progress": {"done": 42, "total": 100}}
            ).encode()
            raise urllib.error.HTTPError(url, 503, "loading", {}, BytesIO(body))
        return Response(
            b'{"data": [{"id": "tiny-model"}]}' if url.endswith("/models") else b"{}"
        )

    monkeypatch.setattr(discovery.urllib.request, "urlopen", open_url)
    cli.agents_command(args(agent))
    assert now[0] >= 75
    assert events == [("configured", agent)]
    output = capsys.readouterr().out
    assert "loading_weights 42%" in output
    assert "downloading or preparing" in output
    assert output.count("Waiting for") <= 10


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
def test_ambiguous_cli_does_not_write(configs, events, monkeypatch, capsys, agent):
    def ambiguous(_model):
        raise RuntimeError(
            "Multiple local servers: pid 1 port 8462 model a; pid 2 port 8463 model b. Pass --port."
        )

    monkeypatch.setattr(discovery, "resolve_endpoint", ambiguous)
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(args(agent, port=None))
    assert exc.value.code == 1
    assert not (configs / ".claude/settings.json").exists()
    assert not list(configs.rglob("config.toml"))
    assert events == []
    assert "--port" in capsys.readouterr().out


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
@pytest.mark.parametrize("explicit", [False, True])
def test_ready_default_and_explicit_endpoint(
    configs, events, monkeypatch, agent, explicit
):
    monkeypatch.setattr(discovery, "_default_endpoint_ready", lambda: (True, True))

    class Response(BytesIO):
        status = 200

    monkeypatch.setattr(
        discovery.urllib.request, "urlopen", lambda *_a, **_k: Response(b"{}")
    )
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: pytest.fail("ready server needs no process scan"),
    )
    monkeypatch.setattr(adapter, "fetch_context_window", lambda *_a: 8192)
    options = args(agent, port=None)
    expected = "http://localhost:8000/v1"
    if explicit:
        expected = "http://127.0.0.1:8462/v1"
        options.base_url = expected
        monkeypatch.setattr(
            discovery,
            "resolve_endpoint",
            lambda *_a: pytest.fail("explicit URL must bypass discovery"),
        )

    def ready(url, *_a, **_k):
        assert url == expected
        return "tiny-model"

    monkeypatch.setattr(setup, "verify_server", ready)
    cli.agents_command(options)
    paths = list(configs.rglob("*.json" if agent == "claude-code" else "config.toml"))
    assert any(expected.removesuffix("/v1") in p.read_text() for p in paths)
    assert events == [("configured", agent)]


@pytest.mark.parametrize("served_alias", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_discovery_matches_alias_and_checkpoint(monkeypatch, served_alias, reverse):
    alias = "qwen3.5-9b-4bit"
    checkpoint = "mlx-community/Qwen3.5-9B-4bit"
    requested, running = (checkpoint, alias) if reverse else (alias, checkpoint)
    server = discovery.Server(
        123,
        8462,
        "other" if served_alias else running,
        running if served_alias else None,
    )
    monkeypatch.setattr(discovery, "scan_servers", lambda: [server])
    assert discovery.resolve_endpoint(requested).server == server


@pytest.mark.parametrize(
    "requested,running,match",
    [
        ("/models/tiny/../tiny", "/models/tiny", True),
        ("tiny", "/models/tiny", True),
        ("/a/tiny", "/b/tiny", False),
        ("org-a/tiny", "org-b/tiny", False),
    ],
)
def test_discovery_local_model_identity(monkeypatch, requested, running, match):
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_model", lambda value: value)
    server = discovery.Server(123, 8462, running)
    monkeypatch.setattr(discovery, "scan_servers", lambda: [server])
    assert (discovery.resolve_endpoint(requested).server == server) is match


def test_discovery_guards_alias_resolution_errors(monkeypatch):
    def resolve(value):
        raise ValueError("alias unavailable")

    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_model", resolve)
    server = discovery.Server(123, 8462, "tiny", "public")
    monkeypatch.setattr(discovery, "scan_servers", lambda: [server])
    assert discovery.resolve_endpoint("public").server == server
    assert discovery.resolve_endpoint("unknown").server is None


@pytest.mark.parametrize("count", [1, 8])
def test_no_model_match_ignores_servers_with_bounded_notes(monkeypatch, capsys, count):
    servers = [
        discovery.Server(57783 + i, 18825 + i, "whisper-base") for i in range(count)
    ]
    monkeypatch.setattr(discovery, "scan_servers", lambda: servers)
    endpoint = discovery.resolve_endpoint("tiny-model")
    assert endpoint.server is None
    assert endpoint.base_url == "http://localhost:8000/v1"
    notes = capsys.readouterr().out
    assert "on :18825 (pid 57783) serves whisper-base, not tiny-model" in notes
    assert "pass --port 18825 to use it" in notes
    assert len(notes.splitlines()) == min(count, 5) + (count > 5)
    if count > 5:
        assert "and 3 more" in notes
    endpoint.refresh()
    assert capsys.readouterr().out == ""


def test_multiple_model_matches_are_ambiguous(monkeypatch):
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: [
            discovery.Server(1, 8462, "qwen3.5-9b-4bit"),
            discovery.Server(2, 8463, "mlx-community/Qwen3.5-9B-4bit"),
            discovery.Server(3, 8464, "whisper-base"),
        ],
    )
    with pytest.raises(RuntimeError, match="Pass --port") as exc:
        discovery.resolve_endpoint("qwen3.5-9b-4bit")
    message = str(exc.value)
    assert all(
        value in message
        for value in ["pid 1", "pid 2", "8462", "8463", "qwen3.5-9b-4bit"]
    )
    assert "whisper-base" not in message
    assert "http://localhost:8000/v1" in message


@pytest.mark.parametrize("agent", ["claude-code", "opencode", "codex"])
def test_unrelated_server_still_allows_offline_model_config(
    clock, configs, events, monkeypatch, capsys, agent
):
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: [discovery.Server(57783, 18825, "whisper-base")],
    )
    monkeypatch.setattr(setup, "verify_server", refuse)
    monkeypatch.setattr("rapid_mlx.run.cli._cached_context_window", lambda _m: 8192)
    cli.agents_command(args(agent, "tiny-model", port=None))
    paths = list(configs.rglob("*.json" if agent != "codex" else "config.toml"))
    assert any("localhost:8000" in p.read_text() for p in paths)
    assert events == [("configured", agent)]
    assert capsys.readouterr().out.count("Note: rapid-mlx serve") == 1


@pytest.mark.parametrize("agent", ["claude-code", "opencode", "codex"])
def test_ready_default_wrong_model_remains_an_error(
    configs, events, monkeypatch, capsys, agent
):
    monkeypatch.setattr(discovery, "_default_endpoint_ready", lambda: (True, True))
    monkeypatch.setattr(
        discovery, "scan_servers", lambda: pytest.fail("healthy default wins")
    )

    class Response(BytesIO):
        status = 200

    def response(request, **_kwargs):
        url = request if isinstance(request, str) else request.full_url
        assert url.startswith("http://localhost:8000/")
        return Response(
            b'{"data": [{"id": "whisper-base"}]}' if url.endswith("/models") else b"{}"
        )

    monkeypatch.setattr(discovery.urllib.request, "urlopen", response)
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(args(agent, "tiny-model", port=None))
    assert exc.value.code == 1
    assert events == [("failed", "model_not_advertised", agent)]
    assert not list(configs.rglob("config.toml"))
    assert not list(configs.rglob("opencode.json"))
    assert not (configs / ".claude/settings.json").exists()
    assert "does not advertise" in capsys.readouterr().out


def test_argv_port_is_rechecked_until_the_process_listens(monkeypatch):
    """An argv ``--port`` guess must not pin setup once serve binds elsewhere."""
    scans = [
        [discovery.Server(123, 8000, "tiny")],
        [discovery.Server(123, 8001, "tiny", listening=True)],
    ]
    calls = []

    def scan():
        calls.append(1)
        return scans[min(len(calls), len(scans)) - 1]

    now = [0.0]
    monkeypatch.setattr(discovery.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(discovery, "scan_servers", scan)
    endpoint = discovery.Endpoint(
        "http://localhost:8000/v1", discovery.Server(123, 8000, "tiny"), follow=True
    )
    endpoint.refresh()
    assert endpoint.base_url == "http://localhost:8000/v1"
    endpoint.refresh()  # throttled: no rescan within 2 s
    assert len(calls) == 1
    now[0] = 2.5
    endpoint.refresh()
    assert endpoint.base_url == "http://localhost:8001/v1"
    assert endpoint.server.listening
    now[0] = 10.0
    endpoint.refresh()  # a confirmed listener is never rescanned
    assert len(calls) == 2


def test_env_only_profile_still_detects_running_model(monkeypatch, capsys):
    """Env-only profiles skip the readiness wait but keep model detection."""
    seen = {}

    def detect(base_url):
        seen["detect"] = base_url
        return "served-model", 32768

    def fail_wait(*_a, **_k):
        raise AssertionError("env-only setup must not wait for a server")

    def fake_setup(profile, base_url, model_id, **_k):
        seen["model"] = model_id
        return "Set OPENAI_BASE_URL"

    monkeypatch.setattr(adapter, "_detect_running_model", detect)
    monkeypatch.setattr(setup, "wait_for_server", fail_wait)
    monkeypatch.setattr(adapter, "setup_agent_config", fake_setup)
    args = SimpleNamespace(
        agent_name="openhands",
        base_url=None,
        port=None,
        test=False,
        setup=True,
        model=None,
        agent_version=None,
        dry_run=False,
        yes=True,
        no_check=False,
    )
    cli.agents_command(args)
    assert seen["detect"] == "http://localhost:8000/v1"
    assert seen["model"] == "served-model"


def test_uninspectable_unknown_port_server_owns_no_endpoint(monkeypatch):
    """AccessDenied + no argv port must not make setup wait on any port."""
    hidden = discovery.Server(9, None, "other", inspected=False)
    unbound = discovery.Server(10, None, "mine")
    monkeypatch.setattr(discovery, "scan_servers", lambda: [hidden])
    assert discovery.server_for_url("http://localhost:8003/v1") is None
    monkeypatch.setattr(discovery, "scan_servers", lambda: [hidden, unbound])
    assert discovery.server_for_url("http://localhost:8003/v1") == unbound


def test_scan_skips_servers_bound_only_to_a_lan_address(
    monkeypatch, agent_discovery_defaults
):
    """Setup writes a localhost URL; a LAN-only listener is never a candidate."""
    import psutil

    def proc(pid, cmdline, ip):
        return SimpleNamespace(
            pid=pid,
            info={"cmdline": cmdline},
            net_connections=lambda **_k: [
                SimpleNamespace(
                    status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip=ip, port=8100)
                )
            ],
        )

    lan_argv = proc(
        1, ["rapid-mlx", "serve", "a", "--host", "192.168.1.10"], "192.168.1.10"
    )
    lan_socket = proc(2, ["rapid-mlx", "serve", "b"], "192.168.1.10")
    local = proc(3, ["rapid-mlx", "serve", "c"], "127.0.0.1")
    other_loopback = proc(4, ["rapid-mlx", "serve", "d"], "127.0.0.2")
    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda *_a: [lan_argv, lan_socket, local, other_loopback],
    )
    # Only the 127.0.0.1 listener is reachable at the localhost URL setup
    # writes; the others must not linger as "unbound, still starting".
    assert [s.pid for s in agent_discovery_defaults()] == [3]


def _listen(port, ip="127.0.0.1"):
    import psutil

    return SimpleNamespace(
        status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip=ip, port=port)
    )


def test_scan_edge_cases(monkeypatch, agent_discovery_defaults):
    import psutil

    def denied_iter(*_a):
        raise psutil.AccessDenied(1)

    monkeypatch.setattr(psutil, "process_iter", denied_iter)
    assert agent_discovery_defaults() == []

    class Vanished:
        pid = 9

        @property
        def info(self):
            raise psutil.NoSuchProcess(9)

    foreign = SimpleNamespace(pid=7, info={"cmdline": ["python", "other", "serve"]})
    pinned = SimpleNamespace(
        pid=8,
        info={"cmdline": ["rapid-mlx", "serve", "tiny", "--port", "8462"]},
        net_connections=lambda **_k: [_listen(8462), _listen(9999)],
    )
    monkeypatch.setattr(
        psutil, "process_iter", lambda *_a: [foreign, Vanished(), pinned]
    )
    servers = agent_discovery_defaults()
    assert servers == [discovery.Server(8, 8462, "tiny")]
    assert servers[0].listening is True


def test_startup_loading_and_pid_alive(monkeypatch):
    import contextlib
    import os

    import psutil

    def answer(code):
        def urlopen(url, **_k):
            if code == 200:
                return contextlib.nullcontext()
            raise urllib.error.HTTPError(url, code, "x", None, None)

        return urlopen

    for code, loading in [(200, False), (503, True), (500, False)]:
        monkeypatch.setattr(discovery.urllib.request, "urlopen", answer(code))
        assert discovery.startup_loading("http://localhost:8462/v1") is loading

    assert discovery.pid_alive(os.getpid()) is True
    for exc, alive in [
        (psutil.NoSuchProcess(1), False),
        (psutil.AccessDenied(1), True),
    ]:

        def process(_pid, exc=exc):
            raise exc

        monkeypatch.setattr(psutil, "Process", process)
        assert discovery.pid_alive(1) is alive


def test_default_endpoint_ready_states(monkeypatch):
    import contextlib

    # Drop the autouse stub so the real probe runs against a fake urlopen.
    monkeypatch.undo()
    monkeypatch.setattr(discovery, "scan_servers", lambda: [])

    def respond(outcome):
        def urlopen(url, **_k):
            if outcome == "http-error":
                raise urllib.error.HTTPError(url, 503, "x", None, None)
            if outcome == "refused":
                raise urllib.error.URLError("refused")
            return contextlib.nullcontext(SimpleNamespace(status=outcome))

        return urlopen

    for outcome, expected in [
        (200, (True, True)),
        (204, (False, True)),
        ("http-error", (False, True)),
        ("refused", (False, False)),
    ]:
        monkeypatch.setattr(discovery.urllib.request, "urlopen", respond(outcome))
        assert discovery._default_endpoint_ready() == expected


def test_refresh_skips_other_pids_and_unbound_rescans(monkeypatch):
    selected = discovery.Server(5, 8001, "tiny")
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: [discovery.Server(4, 8002, "other"), discovery.Server(5, None, "tiny")],
    )
    endpoint = discovery.Endpoint("http://localhost:8001/v1", selected)
    endpoint.refresh()
    assert endpoint.server is selected
    assert endpoint.base_url == "http://localhost:8001/v1"


def test_remote_url_has_no_local_server_and_bad_progress_payload(monkeypatch, capsys):
    assert discovery.server_for_url("http://192.0.2.10:8000/v1") is None
    monkeypatch.setattr(
        discovery.urllib.request, "urlopen", lambda *_a, **_k: BytesIO(b"[1, 2]")
    )
    discovery.Progress().show("http://localhost:8462/v1", 0)
    assert "downloading or preparing" in capsys.readouterr().out


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("port", [8000, 62801])
@pytest.mark.parametrize("protected_path", ["/health", "/v1/models"])
def test_auto_setup_skips_inaccessible_server(
    clock, events, configs, monkeypatch, capsys, status, port, protected_path
):
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    server = discovery.Server(123, port, "tiny-model", listening=True)
    monkeypatch.setattr(discovery, "scan_servers", lambda: [server])
    record = configs / "Library/Application Support/Rapid/owned-server.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"pid": 123, "port": port}))

    class Response(BytesIO):
        status = 200

    def response(request, **_kwargs):
        url = request if isinstance(request, str) else request.full_url
        if f":{port}/" not in url:
            raise ConnectionRefusedError()
        if url.endswith(protected_path):
            raise urllib.error.HTTPError(url, status, "auth", {}, BytesIO())
        return Response(b"{}")

    monkeypatch.setattr(discovery.urllib.request, "urlopen", response)
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(args("codex", port=None))
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "No Rapid-MLX server" in output
    assert "setup failed" not in output
    assert f"Server on :{port} belongs to Rapid-MLX Desktop" in output
    hint_lines = [line for line in output.splitlines() if f"Server on :{port}" in line]
    assert len(hint_lines) == 1
    assert "Agent > Connection details > API key" in hint_lines[0]
    assert "Connect Tools" not in output
    assert (
        f"RAPID_MLX_API_KEY=<key> rapid-mlx agents codex --setup --port {port}"
        in hint_lines[0]
    )
    assert "rapid-mlx serve" in output
    assert "Using the Rapid-MLX server" not in output
    assert not list(configs.rglob("config.toml"))
    assert events == [("failed", "server_not_running", "codex")]


@pytest.mark.parametrize("explicit", ["port", "url"])
def test_explicit_auth_failure_retry_includes_key(
    clock, events, configs, monkeypatch, capsys, explicit
):
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)

    class Response(BytesIO):
        status = 200

    def response(request, **_kwargs):
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/models"):
            raise urllib.error.HTTPError(url, 401, "auth", {}, BytesIO())
        return Response(b"{}")

    monkeypatch.setattr(discovery.urllib.request, "urlopen", response)
    options = args("codex", port=62801)
    if explicit == "url":
        options.port = None
        options.base_url = "http://localhost:62801/v1"
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(options)
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "authentication failed" in output
    assert "Retry: RAPID_MLX_API_KEY=<key> rapid-mlx agents codex --setup" in output
    assert "Agent > Connection details > API key" in output
    assert "Connect Tools" not in output
    assert not list(configs.rglob("config.toml"))


@pytest.mark.parametrize("port", [8000, 62801])
def test_valid_key_auto_setup_uses_protected_server(
    clock, events, configs, monkeypatch, capsys, port
):
    monkeypatch.setenv("RAPID_MLX_API_KEY", "test-key")
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: [discovery.Server(123, port, "tiny-model", listening=True)],
    )
    monkeypatch.setattr(discovery, "pid_alive", lambda _pid: True)

    class Response(BytesIO):
        status = 200

    def response(request, **_kwargs):
        url = request if isinstance(request, str) else request.full_url
        if f":{port}/" not in url:
            raise ConnectionRefusedError()
        if url.endswith("/models"):
            assert request.get_header("Authorization") == "Bearer test-key"
            return Response(b'{"data": [{"id": "tiny-model"}]}')
        return Response(b"{}")

    monkeypatch.setattr(discovery.urllib.request, "urlopen", response)
    cli.agents_command(args("codex", port=None))
    config = configs / "codex/config.toml"
    assert f"localhost:{port}" in config.read_text()
    assert "tiny-model" in config.read_text()
    assert "Connect Tools" not in capsys.readouterr().out
    assert events == [("configured", "codex")]


def test_auto_discovery_continues_after_auth_rejection(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: [discovery.Server(1, 62801, "a"), discovery.Server(2, 8462, "b")],
    )

    def response(url, **_kwargs):
        if ":62801/" in url:
            raise urllib.error.HTTPError(url, 401, "auth", {}, BytesIO())
        raise ConnectionRefusedError()

    monkeypatch.setattr(discovery.urllib.request, "urlopen", response)
    assert discovery.resolve_endpoint().base_url == "http://localhost:8462/v1"


@pytest.mark.parametrize(
    "record", [None, [], {"pid": 999, "port": 62801}, {"pid": 123, "port": 8000}]
)
def test_desktop_recognition_requires_matching_record(configs, record):
    path = configs / "Library/Application Support/Rapid/owned-server.json"
    path.parent.mkdir(parents=True)
    if record is not None:
        path.write_text(json.dumps(record))
    hint = discovery.skipped_auth_hint(discovery.Server(123, 62801, "tiny"), "codex")
    assert "requires an API key" in hint
    assert "belongs to" not in hint


@pytest.mark.parametrize("discovery_times_out", [False, True])
def test_auto_setup_reselects_when_starting_server_begins_requiring_auth(
    clock, events, configs, monkeypatch, capsys, discovery_times_out
):
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: [discovery.Server(123, 62801, "tiny-model", listening=True)],
    )
    monkeypatch.setattr(discovery, "pid_alive", lambda _pid: True)
    protected = False

    class Response(BytesIO):
        status = 200

    def response(request, **_kwargs):
        url = request if isinstance(request, str) else request.full_url
        if ":62801/" not in url:
            raise ConnectionRefusedError()
        if protected and url.endswith("/models"):
            if discovery_times_out and isinstance(request, str):
                raise TimeoutError()
            raise urllib.error.HTTPError(url, 401, "auth", {}, BytesIO())
        return Response(b'{"data": [{"id": "tiny-model"}]}')

    original_verify = setup.verify_server

    def verify(*a, **kw):
        nonlocal protected
        protected = True
        return original_verify(*a, **kw)

    monkeypatch.setattr(discovery.urllib.request, "urlopen", response)
    monkeypatch.setattr(setup, "verify_server", verify)
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(args("codex", port=None))
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "No Rapid-MLX server" in output
    assert "Server on :62801 requires an API key" in output
    assert events == [("failed", "server_not_running", "codex")]


def test_public_default_health_does_not_bypass_auth_filter(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    monkeypatch.setattr(discovery, "_default_endpoint_ready", lambda: (True, True))
    monkeypatch.setattr(discovery, "scan_servers", lambda: [])

    class Response(BytesIO):
        status = 200

    def response(url, **_kwargs):
        if url.endswith("/models"):
            raise urllib.error.HTTPError(url, 401, "auth", {}, BytesIO())
        return Response(b"{}")

    monkeypatch.setattr(discovery.urllib.request, "urlopen", response)
    endpoint = discovery.resolve_endpoint()
    assert endpoint.server is None
    assert [s.port for s in endpoint.skipped_auth] == [8000]
    with pytest.raises(ServerNotRunningError):
        setup.wait_for_server(
            endpoint.base_url, "default", agent="codex", endpoint=endpoint
        )


NON_CHAT_MODELS = [
    {"id": "cogvideox-fun-5b-q4", "modality": "video-gen"},
    {"id": "picture", "modality": "image-gen"},
    {"id": "speech", "modality": "tts"},
    {"id": "transcribe", "modality": "stt"},
    {"id": "vectors", "modality": "embedding"},
    {"id": "kokoro", "modality": "audio", "capabilities": ["audio.speech"]},
    {"id": "embed", "modality": "text", "capabilities": ["embedding"]},
    {"id": "rank", "modality": "text", "capabilities": ["rerank"]},
]


@pytest.mark.parametrize(
    "payload",
    [{}, [], None, "invalid", {"data": None}, {"data": 42}, {"data": {"id": "chat"}}],
)
def test_verify_server_malformed_model_body_has_no_models(monkeypatch, events, payload):
    class Response(BytesIO):
        status = 200

    monkeypatch.setattr(
        setup.urllib.request,
        "urlopen",
        lambda *_a, **_kw: Response(json.dumps(payload).encode()),
    )
    with pytest.raises(setup.ServerNoModelsError, match="reported no models"):
        setup.verify_server("http://localhost:18403/v1", "default", agent="codex")
    assert events == [("failed", "server_no_models", "codex")]


def fake_model_server(monkeypatch, listings):
    class Response(BytesIO):
        status = 200

    def response(request, **_kwargs):
        from urllib.parse import urlsplit

        url = request if isinstance(request, str) else request.full_url
        models = listings.get(urlsplit(url).port)
        if models is None:
            raise ConnectionRefusedError()
        return Response(json.dumps({"data": models}).encode())

    monkeypatch.setattr(discovery.urllib.request, "urlopen", response)


@pytest.mark.parametrize("model", NON_CHAT_MODELS)
@pytest.mark.parametrize("port", [8000, 18403])
def test_auto_setup_skips_non_chat_server(
    clock, events, configs, monkeypatch, capsys, model, port
):
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: [discovery.Server(123, port, model["id"], listening=True)],
    )
    monkeypatch.setattr(discovery, "pid_alive", lambda _pid: True)
    fake_model_server(monkeypatch, {port: [model]})
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(args("codex", port=None))
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "No Rapid-MLX server" in output
    hints = [line for line in output.splitlines() if "Skipping" in line]
    assert len(hints) == 1
    assert f":{port}" in hints[0] and model["id"] in hints[0]
    from rapid_mlx.agents.chat_models import non_chat_kind

    kind = non_chat_kind(model)
    article = "an" if kind[0] in "aeiou" else "a"
    assert f"it serves {article} {kind} model" in hints[0]
    assert hints[0].endswith(", not a chat model.")
    assert "rapid-mlx serve" not in hints[0]
    assert "configured!" not in output
    assert not list(configs.rglob("config.toml"))


@pytest.mark.parametrize("explicit", ["port", "url"])
@pytest.mark.parametrize("model", NON_CHAT_MODELS)
@pytest.mark.parametrize("requested", [None, "wanted-chat"])
def test_explicit_non_chat_server_fails_before_writing(
    clock, events, configs, monkeypatch, capsys, explicit, model, requested
):
    fake_model_server(monkeypatch, {18403: [model]})
    options = args("codex", model=requested, port=18403)
    if explicit == "url":
        options.port = None
        options.base_url = "http://localhost:18403/v1"
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(options)
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "server on :18403 is serving" in output
    assert model["id"] in output
    assert "coding agents need a chat model" in output
    assert "rapid-mlx serve " in output
    assert not list(configs.rglob("config.toml"))


@pytest.mark.parametrize(
    "chat",
    [
        {"id": "chat", "modality": "text"},
        {"id": "chat", "modality": "image", "capabilities": ["text", "vision"]},
        {"id": "chat", "modality": "mllm"},
        {"id": "chat"},
    ],
)
def test_discovery_continues_from_video_to_chat(monkeypatch, capsys, chat):
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: [
            discovery.Server(123, 18403, "video"),
            discovery.Server(124, 8462, "chat"),
        ],
    )
    fake_model_server(monkeypatch, {18403: [NON_CHAT_MODELS[0]], 8462: [chat]})
    assert discovery.resolve_endpoint().base_url == "http://localhost:8462/v1"
    assert capsys.readouterr().out.splitlines() == [
        "  Skipping the server on :18403: it serves a video model "
        "(cogvideox-fun-5b-q4), not a chat model.",
        "  Using the Rapid-MLX server on :8462 (pid 124).",
    ]
    assert (
        setup.verify_server("http://localhost:8462/v1", "chat", agent="codex") == "chat"
    )


@pytest.mark.parametrize("model", NON_CHAT_MODELS)
def test_sibling_launch_start_discovery_and_connection_reject_non_chat(
    monkeypatch, model
):
    from rapid_mlx.agents import isolated

    fake_model_server(monkeypatch, {18403: [model]})
    with pytest.raises(ValueError, match="chat model"):
        isolated.running_model(18403, "key")
    assert isolated.connection_status(18403, "key", model["id"]) != "ready"


@pytest.mark.parametrize("port", [8000, 18403])
def test_setup_reselects_when_modality_appears_after_discovery(
    clock, events, configs, monkeypatch, capsys, port
):
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: [discovery.Server(123, port, "cogvideox-fun-5b-q4", listening=True)],
    )
    monkeypatch.setattr(discovery, "pid_alive", lambda _pid: True)
    fake_model_server(monkeypatch, {port: [{"id": "cogvideox-fun-5b-q4"}]})
    original_verify = setup.verify_server

    def verify(*a, **kw):
        fake_model_server(monkeypatch, {port: [NON_CHAT_MODELS[0]]})
        return original_verify(*a, **kw)

    monkeypatch.setattr(setup, "verify_server", verify)
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(args("codex", port=None))
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "No Rapid-MLX server" in output
    assert output.count("Skipping") == 1
    hints = [line.strip() for line in output.splitlines() if "Skipping" in line]
    assert hints == [
        f"Skipping the server on :{port}: it serves a video model "
        "(cogvideox-fun-5b-q4), not a chat model."
    ]
    assert not list(configs.rglob("config.toml"))


@pytest.mark.parametrize("model", NON_CHAT_MODELS)
def test_start_refuses_non_chat_server_before_attach(monkeypatch, capsys, model):
    from rapid_mlx.agents import get_profile
    from rapid_mlx.run import cli as start

    fake_model_server(monkeypatch, {18403: [model]})
    monkeypatch.setattr(
        start, "_attach_and_configure", lambda *_a: pytest.fail("must not attach")
    )
    assert (
        start._reuse_or_refuse(
            "http://localhost:18403",
            model["id"],
            model["id"],
            get_profile("codex"),
            SimpleNamespace(port=18403, dry_run=False),
        )
        == 1
    )
    assert "coding agents need a chat model" in capsys.readouterr().out


def test_mixed_model_listing_selects_chat_and_rejects_explicit_video(monkeypatch):
    from rapid_mlx.agents import isolated

    fake_model_server(
        monkeypatch, {18403: [NON_CHAT_MODELS[0], {"id": "chat", "modality": "text"}]}
    )
    assert adapter._detect_running_model("http://localhost:18403/v1")[0] == "chat"
    assert (
        setup.verify_server("http://localhost:18403/v1", "default", agent="codex")
        == "chat"
    )
    assert isolated.running_model(18403, "key") == "chat"
    with pytest.raises(RuntimeError, match="video model"):
        setup.verify_server(
            "http://localhost:18403/v1", NON_CHAT_MODELS[0]["id"], agent="codex"
        )


def test_late_modality_reselection_adopts_second_servers_model(
    clock, events, configs, monkeypatch, capsys
):
    monkeypatch.setattr(discovery, "_default_endpoint_ready", lambda: (True, True))
    monkeypatch.setattr(
        discovery,
        "scan_servers",
        lambda: [
            discovery.Server(123, 8000, "video", listening=True),
            discovery.Server(124, 8462, "chat", listening=True),
        ],
    )
    monkeypatch.setattr(discovery, "pid_alive", lambda _pid: True)
    listings = {8000: [{"id": "video"}], 8462: [{"id": "chat", "modality": "text"}]}
    fake_model_server(monkeypatch, listings)
    original_verify = setup.verify_server

    def verify(*a, **kw):
        listings[8000] = [NON_CHAT_MODELS[0]]
        return original_verify(*a, **kw)

    monkeypatch.setattr(setup, "verify_server", verify)
    cli.agents_command(args("codex", port=None))
    config = (configs / "codex/config.toml").read_text()
    assert 'model = "chat"' in config
    assert "localhost:8462" in config
    assert capsys.readouterr().out.count("Skipping") == 1


@pytest.mark.parametrize(
    "port,busy,expected",
    [(18403, set(), 8000), (8000, set(), 8001), (8000, {8001}, 8002)],
)
@pytest.mark.parametrize("explicit", ["port", "url"])
def test_non_chat_recovery_changes_endpoint_and_model(
    clock, events, configs, monkeypatch, capsys, port, busy, expected, explicit
):
    from rapid_mlx.run import cli as start

    fake_model_server(monkeypatch, {port: [NON_CHAT_MODELS[5]]})
    monkeypatch.setattr(cli, "_port_is_busy", lambda _host, p: p in busy)
    monkeypatch.setattr(start, "_select_model", lambda **_kw: "fitting-chat")
    options = args("codex", model="kokoro", port=port)
    options._original_alias = "kokoro"
    options.agent_version = "1.2"
    if explicit == "url":
        options.port = None
        options.base_url = f"http://localhost:{port}/v1"
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(options)
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "serving an audio model" in output
    assert "Retry:" not in output
    assert (
        f"Start a chat model: rapid-mlx serve fitting-chat --port {expected}" in output
    )
    retry = output.split("then run: ")[1].strip()
    expected_retry = "rapid-mlx agents codex --setup --yes"
    if expected != 8000:
        expected_retry += f" --port {expected}"
    assert retry == expected_retry + " --agent-version 1.2"
    assert not list(configs.rglob("config.toml"))


@pytest.mark.parametrize("ram,cached", [(16.0, False), (16.0, True), (64.0, True)])
def test_non_chat_recommendation_uses_host_fit_and_cache(
    monkeypatch, capsys, ram, cached
):
    from rapid_mlx.agents import get_profile
    from rapid_mlx.agents.chat_models import NonChatModelError
    from rapid_mlx.agents.server_hint import start_server_command
    from rapid_mlx.run import cli as start

    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: ram)
    monkeypatch.setattr(
        "rapid_mlx._download_gate.is_repo_cached", lambda repo: cached and "9B" in repo
    )
    monkeypatch.setattr(cli, "_port_is_busy", lambda *_a: False)
    profile = get_profile("codex")
    expected = start._select_model(explicit=None, profile=profile, no_download=False)
    assert expected and expected != profile.recommended_models[0]
    error = NonChatModelError("http://localhost:18403/v1", NON_CHAT_MODELS[5])
    assert f"rapid-mlx serve {expected} --port 8000" in str(error)
    assert (
        start_server_command(profile, "http://localhost:8000/v1", "default")
        == f"rapid-mlx serve {expected} --port 8000"
    )
    assert capsys.readouterr().out == ""


def test_non_chat_recommendation_does_not_suggest_an_oversized_model(
    monkeypatch, capsys
):
    from rapid_mlx.agents.chat_models import NonChatModelError

    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 1.0)
    monkeypatch.setattr(cli, "_port_is_busy", lambda *_a: False)
    error = NonChatModelError("http://localhost:18403/v1", NON_CHAT_MODELS[5])
    assert "rapid-mlx serve <chat-model> --port 8000" in str(error)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "modality,phrase",
    [
        ("audio", "an audio"),
        ("embedding", "an embedding"),
        ("image-gen", "an image generation"),
        ("video-gen", "a video"),
        ("tts", "a speech synthesis"),
    ],
)
def test_non_chat_articles_and_short_alias_across_entry_points(
    monkeypatch, capsys, modality, phrase
):
    from rapid_mlx.agents import get_profile, isolated
    from rapid_mlx.agents.chat_models import NonChatModelError
    from rapid_mlx.run import cli as start

    hf = {"id": "mlx-community/non-chat", "modality": modality}
    alias = {"id": "short-alias", "modality": modality}
    fake_model_server(monkeypatch, {18403: [hf, alias]})
    monkeypatch.setattr(cli, "_port_is_busy", lambda *_a: False)
    monkeypatch.setattr(start, "_select_model", lambda **_kw: "fitting-chat")
    error = discovery.non_chat_server(18403, hf["id"])
    assert f"it serves {phrase} model (short-alias)" in error.skip_hint
    with pytest.raises(NonChatModelError) as exc:
        setup.verify_server("http://localhost:18403/v1", "default", agent="codex")
    assert f"is serving {phrase} model (short-alias)" in str(exc.value)
    with pytest.raises(NonChatModelError) as exc:
        isolated.running_model(18403, "key")
    assert "(short-alias)" in str(exc.value)
    assert "then run: rapid-mlx agents codex --setup --yes" in str(exc.value)
    assert (
        start._reuse_or_refuse(
            "http://localhost:18403",
            hf["id"],
            hf["id"],
            get_profile("codex"),
            SimpleNamespace(port=18403, dry_run=False),
        )
        == 1
    )
    output = capsys.readouterr().out
    assert f"is serving {phrase} model (short-alias)" in output
    assert (
        "rapid-mlx serve fitting-chat --port 8000, then run: rapid-mlx agents codex --setup --yes"
        in output
    )


@pytest.mark.parametrize("agent", ["codex", "claude-code"])
def test_launch_non_chat_failure_provides_recovery_for_same_agent(
    monkeypatch, capsys, agent
):
    from rapid_mlx.agents import isolated
    from rapid_mlx.launch import cli as launch
    from rapid_mlx.run import cli as start

    fake_model_server(monkeypatch, {8000: [NON_CHAT_MODELS[5]]})
    monkeypatch.setattr(cli, "_port_is_busy", lambda *_a: False)
    monkeypatch.setattr(start, "_select_model", lambda **_kw: "fitting-chat")
    monkeypatch.setattr(isolated, "stable_key", lambda **_kw: "key")
    monkeypatch.setattr(
        "rapid_mlx.agents.cli_preflight.missing_agent_cli", lambda _p: None
    )
    with pytest.raises(SystemExit) as exc:
        launch.launch_command(
            SimpleNamespace(
                client=agent,
                isolated=True,
                all=False,
                remove=False,
                dry_run=False,
                check=False,
                port=8000,
                model=None,
                run=True,
            )
        )
    assert exc.value.code == 1
    output = capsys.readouterr().err
    assert "serving an audio model" in output
    assert "rapid-mlx serve fitting-chat --port 8001" in output
    assert f"then run: rapid-mlx agents {agent} --setup --yes --port 8001" in output
    assert "Retry:" not in output


def test_non_chat_recovery_when_no_suggested_port_is_free(monkeypatch):
    from rapid_mlx.agents.chat_models import NonChatModelError
    from rapid_mlx.run import cli as start

    monkeypatch.setattr(cli, "_port_is_busy", lambda *_a: True)
    monkeypatch.setattr(start, "_select_model", lambda **_kw: "fitting-chat")
    error = NonChatModelError("http://localhost:8000/v1", NON_CHAT_MODELS[5])
    assert "on a free port" in str(error)
    assert "re-run setup with --port <port>" in str(error)
