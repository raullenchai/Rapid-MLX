"""Closed readiness causes survive retries without leaking exception details."""

import json
import socket
import urllib.error
from io import BytesIO
from types import SimpleNamespace

import pytest

from rapid_mlx.agents import server_discovery, setup, telemetry
from rapid_mlx.telemetry import registry, track


class Response(BytesIO):
    status = 200


@pytest.mark.parametrize("stage", ["health", "models"])
@pytest.mark.parametrize(
    "error,local_cause,remote_cause",
    [
        (
            urllib.error.HTTPError("private", 401, "secret", {}, None),
            "http_auth",
            "http_auth",
        ),
        (
            urllib.error.HTTPError("private", 403, "secret", {}, None),
            "http_auth",
            "http_auth",
        ),
        (
            urllib.error.HTTPError("private", 500, "secret", {}, None),
            "http_error",
            "http_error",
        ),
        (
            urllib.error.URLError(ConnectionRefusedError("secret")),
            "connection_refused",
            "remote_unreachable",
        ),
        (ConnectionRefusedError("secret"), "connection_refused", "remote_unreachable"),
        (TimeoutError("secret"), "startup_timeout", "remote_unreachable"),
        (
            urllib.error.URLError(TimeoutError("secret")),
            "startup_timeout",
            "remote_unreachable",
        ),
        (
            urllib.error.URLError(socket.gaierror("secret")),
            "other",
            "remote_unreachable",
        ),
        (ValueError("secret"), "other", "other"),
    ],
)
@pytest.mark.parametrize("remote", [False, True])
def test_probe_and_wait_causes(
    monkeypatch, stage, error, local_cause, remote_cause, remote
):
    base = "http://remote.invalid/v1" if remote else "http://localhost:8000/v1"
    expected = remote_cause if remote else local_cause
    emitted = []

    def open_url(request, **_kwargs):
        if stage == "health" or not isinstance(request, str):
            raise error
        return Response()

    monkeypatch.setattr(setup.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(
        setup, "track_agent_configure_failed", lambda *v: emitted.append(v)
    )
    monkeypatch.setattr(server_discovery, "server_for_url", lambda *_: None)
    for wait in (False, True):
        emitted.clear()
        with pytest.raises(RuntimeError) as caught:
            if wait:
                setup.wait_for_server(base, "default", agent="claude-code", timeout=0)
            else:
                setup.verify_server(base, "default", agent="claude-code")
        assert caught.value.cause == expected
        assert emitted == (
            []
            if expected == "connection_refused"
            else [("server_not_ready", "claude-code", expected)]
        )


@pytest.mark.parametrize(
    "stage,expected", [("health", "model_not_loaded"), ("models", "http_error")]
)
@pytest.mark.parametrize("wait", [False, True])
def test_503_depends_on_probe_stage(monkeypatch, stage, expected, wait):
    clock = [0.0]
    monkeypatch.setattr(setup.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        setup.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    def open_url(request, **_kwargs):
        if stage == "health" or not isinstance(request, str):
            raise urllib.error.HTTPError("private", 503, "secret", {}, None)
        return Response()

    emitted = []
    monkeypatch.setattr(setup.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(
        setup, "track_agent_configure_failed", lambda *v: emitted.append(v)
    )
    monkeypatch.setattr(server_discovery, "server_for_url", lambda *_: None)
    with pytest.raises(setup.ServerReadinessError) as caught:
        if wait:
            # Health 503 means loading; only an exhausted readiness wait
            # changes that probe cause to startup_timeout.
            setup.wait_for_server(
                "http://localhost:8000/v1", "default", agent="claude-code", timeout=1
            )
        else:
            setup.verify_server(
                "http://localhost:8000/v1", "default", agent="claude-code"
            )
    if wait:
        assert clock[0] == 1
        if stage == "health":
            expected = "startup_timeout"
    else:
        assert clock[0] == 0
    assert caught.value.cause == expected
    assert emitted == [("server_not_ready", "claude-code", expected)]


@pytest.mark.parametrize(
    "body,expected",
    [
        (b"not json", ("server_not_ready", "codex", "other")),
        (b"[]", ("server_no_models", "codex")),
        (b'{"data": {}}', ("server_no_models", "codex")),
    ],
)
def test_malformed_models_preserves_failure_class(monkeypatch, body, expected):
    monkeypatch.setattr(
        setup.urllib.request,
        "urlopen",
        lambda request, **_: Response() if isinstance(request, str) else Response(body),
    )
    emitted = []
    monkeypatch.setattr(
        setup, "track_agent_configure_failed", lambda *v: emitted.append(v)
    )
    with pytest.raises(RuntimeError):
        setup.verify_server("http://localhost:8000/v1", "default", agent="codex")
    assert emitted == [expected]


def test_dead_starting_process(monkeypatch):
    endpoint = SimpleNamespace(
        base_url="http://localhost:8000/v1",
        server=SimpleNamespace(pid=123),
        refresh=lambda: None,
    )
    monkeypatch.setattr(server_discovery, "pid_alive", lambda _: False)
    emitted = []
    monkeypatch.setattr(
        setup, "track_agent_configure_failed", lambda *v: emitted.append(v)
    )
    with pytest.raises(RuntimeError, match="exited during startup"):
        setup.wait_for_server(
            endpoint.base_url, "default", agent="codex", endpoint=endpoint
        )
    assert emitted == [("server_not_ready", "codex", "process_exited")]


@pytest.mark.parametrize(
    "cause",
    [
        None,
        *registry.load_registry()["enums"]["agent_failure_cause"]["values"],
        "https://private/secret",
    ],
)
@pytest.mark.parametrize("error_class", ["server_not_ready", "server_not_running"])
def test_telemetry_optional_closed_cause(monkeypatch, cause, error_class):
    emitted = []
    monkeypatch.setattr(
        track,
        "track",
        lambda name, props: emitted.append((name, registry.validate(name, props))),
    )
    telemetry.track_agent_configure_failed(error_class, "claude-code", cause)
    props = {"error_class": error_class, "agent": "claude-code"}
    if cause is not None:
        props["cause"] = "other" if cause.startswith("https:") else cause
    assert emitted == [("agent_configure_failed", props)]
    assert "private" not in json.dumps(emitted)


def test_cause_not_allowed_for_other_error_classes():
    assert (
        registry.validate(
            "agent_configure_failed", {"error_class": "other", "cause": "http_auth"}
        )
        is None
    )


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("follow", [False, True])
def test_auth_cause_survives_reselection_deadline(monkeypatch, status, follow):
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    endpoint = server_discovery.Endpoint("http://localhost:18403/v1", follow=follow)
    clock = [0.0]
    emitted = []
    monkeypatch.setattr(setup.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        setup.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(server_discovery.Endpoint, "refresh", lambda _: None)
    monkeypatch.setattr(server_discovery, "server_for_url", lambda *_: None)
    monkeypatch.setattr(
        setup, "track_agent_configure_failed", lambda *values: emitted.append(values)
    )

    def reject(*_args, **_kwargs):
        raise urllib.error.HTTPError("private", status, "secret", {}, None)

    monkeypatch.setattr(setup.urllib.request, "urlopen", reject)
    with pytest.raises(setup.ServerAuthenticationError) as caught:
        setup.wait_for_server(
            endpoint.base_url, "default", agent="codex", timeout=1, endpoint=endpoint
        )
    assert caught.value.cause == "http_auth"
    assert clock[0] == 1
    assert emitted == [("server_not_ready", "codex", "http_auth")]
