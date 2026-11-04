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
    assert discovery.resolve_endpoint().base_url == "http://localhost:8000/v1"


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
            SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(port=8462))
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
                status=psutil.CONN_LISTEN, laddr=SimpleNamespace(port=59001)
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
