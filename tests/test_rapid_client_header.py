# SPDX-License-Identifier: Apache-2.0
"""``X-Rapid-Client`` — every Rapid-owned client sends it, the server trusts
it only for our own closed label set.

45% of callers bucketed to ``other`` because a Rapid client's User-Agent is
whatever HTTP library it happens to use. The header fixes attribution
without ever putting caller-controlled free text on a payload: an off-list
value is ignored outright, never echoed.

The "does the client send it?" tests drive the REAL request-construction
code with the HTTP library stubbed, so deleting the header from a call site
turns them red — asserting on the constant alone would not.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from rapid_mlx.client_header import (
    RAPID_CLIENT_HEADER,
    RAPID_CLIENT_LABELS,
    rapid_client_headers,
)
from rapid_mlx.telemetry.redact import normalize_caller_agent

# --------------------------------------------------------------- the helper


def test_helper_rejects_off_list_labels():
    with pytest.raises(ValueError):
        rapid_client_headers("rapid-something-new")


def test_helper_label_wins_over_extra_headers():
    got = rapid_client_headers(
        "rapid-bench", {RAPID_CLIENT_HEADER: "spoofed", "Authorization": "Bearer k"}
    )
    assert got[RAPID_CLIENT_HEADER] == "rapid-bench"
    assert got["Authorization"] == "Bearer k"


def test_auth_helper_merges_bearer_and_client(monkeypatch):
    from rapid_mlx.http_auth import rapid_mlx_client_headers

    monkeypatch.setenv("RAPID_MLX_API_KEY", "sekret")
    got = rapid_mlx_client_headers("rapid-cli-chat")
    assert got[RAPID_CLIENT_HEADER] == "rapid-cli-chat"
    assert got["Authorization"] == "Bearer sekret"


# ------------------------------------------------------------ server buckets


def test_header_beats_user_agent():
    assert (
        normalize_caller_agent("python-httpx/0.27", "rapid-desktop") == "rapid-desktop"
    )


def test_unknown_header_value_is_ignored_and_never_echoed():
    got = normalize_caller_agent("claude-code/1.0", "acme-internal-tool/9")
    assert got == "claude-code"
    assert "acme" not in got


def test_unknown_header_with_no_user_agent_falls_all_the_way_back():
    assert normalize_caller_agent(None, "acme-internal-tool/9") == "unknown"
    assert normalize_caller_agent("wat/1", "acme-internal-tool/9") == "other"


@pytest.mark.parametrize("label", sorted(RAPID_CLIENT_LABELS))
def test_every_label_round_trips(label):
    assert normalize_caller_agent("curl/8", label) == label


def test_header_is_whitespace_tolerant_but_not_prefix_tolerant():
    assert normalize_caller_agent("curl/8", "  rapid-bench  ") == "rapid-bench"
    assert normalize_caller_agent("curl/8", "rapid-bench-evil") == "curl"


# ------------------------------------------------- end-to-end over the route


@pytest.fixture
def telemetry_on(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)

    import rapid_mlx.telemetry.emit as emit
    import rapid_mlx.telemetry.state as state

    importlib.reload(state)
    importlib.reload(emit)
    emit._reset_for_tests()
    state.record_consent(True, rapid_mlx_version="0.0.0+test")
    monkeypatch.setenv("RAPID_MLX_TELEMETRY_REQUEST_SAMPLE", "1")
    return emit


@pytest.fixture
def captured(telemetry_on, monkeypatch):
    events: list[dict] = []

    class _StubQueue:
        def enqueue(self, payload):
            events.append(payload)

    monkeypatch.setattr(telemetry_on, "get_queue", lambda: _StubQueue())
    return events


class _Engine:
    preserve_native_tool_format = False
    tokenizer = SimpleNamespace(
        chat_template=None,
        apply_chat_template=lambda *a, **k: "templated",
        decode=lambda *a, **k: "",
        encode=lambda *a, **k: [1, 2, 3],
    )

    async def chat(self, messages, **kwargs):
        return SimpleNamespace(
            text="hello there",
            raw_text="hello there",
            prompt_tokens=9,
            completion_tokens=7,
            finish_reason="stop",
            tool_calls=None,
            matched_stop=None,
            reasoning_text=None,
            model="test-model",
        )


def _messages_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.routes.anthropic import router

    cfg = reset_config()
    cfg.engine = _Engine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.reasoning_parser_name = None

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _post(client, headers):
    return client.post(
        "/v1/messages",
        headers=headers,
        json={
            "model": "test-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )


def test_route_prefers_the_rapid_client_header(captured):
    pytest.importorskip("mlx")
    client = _messages_client()
    resp = _post(
        client,
        {"user-agent": "python-httpx/0.27", "x-rapid-client": "rapid-desktop"},
    )
    assert resp.status_code == 200, resp.text
    events = [p["request"] for p in captured if "request" in p]
    assert events, captured
    assert events[-1]["caller_agent"] == "rapid-desktop"


def test_route_ignores_an_off_list_header(captured):
    pytest.importorskip("mlx")
    client = _messages_client()
    resp = _post(
        client,
        {"user-agent": "claude-code/1.0", "x-rapid-client": "acme-internal-tool"},
    )
    assert resp.status_code == 200, resp.text
    events = [p["request"] for p in captured if "request" in p]
    assert events, captured
    assert events[-1]["caller_agent"] == "claude-code"
    assert "acme" not in repr(captured)


# ------------------------------------------- every Rapid-owned Python client


def _header_of(recorded) -> str | None:
    headers = recorded.get("headers") or {}
    return headers.get(RAPID_CLIENT_HEADER)


def test_cli_chat_stream_sends_the_header(monkeypatch):
    """``rapid-mlx chat``'s streaming POST to /v1/chat/completions."""
    import requests

    from rapid_mlx import cli

    recorded: dict = {}

    def _fake_post(url, **kwargs):
        recorded["url"] = url
        recorded.update(kwargs)
        raise RuntimeError("stop here — the request is already built")

    monkeypatch.setattr(requests, "post", _fake_post)
    with pytest.raises(RuntimeError):
        cli._stream_chat_response("http://127.0.0.1:8000", {"model": "m"}, 30)
    assert recorded["url"].endswith("/v1/chat/completions")
    assert _header_of(recorded) == "rapid-cli-chat"


def test_agents_client_sends_the_header(monkeypatch):
    import httpx

    from rapid_mlx.agents import testing as agents_testing

    recorded: dict = {}

    def _fake_post(url, **kwargs):
        recorded["url"] = url
        recorded.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(httpx, "post", _fake_post)
    with pytest.raises(RuntimeError):
        agents_testing._api_call("http://127.0.0.1:8000/v1", "m", [])
    assert _header_of(recorded) == "rapid-agents"


def test_bench_speed_client_sends_the_header(monkeypatch):
    import httpx

    from rapid_mlx.bench import tier_runner

    recorded: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def __enter__(self):
            raise RuntimeError("stop here")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    tier_runner._run_speed("m", "http://127.0.0.1:8000/v1")
    assert _header_of(recorded) == "rapid-bench"


def test_bench_health_probe_sends_the_header(monkeypatch):
    import urllib.request

    from rapid_mlx.bench import tier_runner

    recorded: dict = {}

    def _fake_urlopen(req, timeout=None):
        recorded["req"] = req
        raise OSError("stop here")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    assert tier_runner._health_check("http://127.0.0.1:8000/v1") is False
    # urllib capitalizes header names it stores.
    assert recorded["req"].get_header(RAPID_CLIENT_HEADER.capitalize()) == "rapid-bench"


def test_community_bench_runner_sends_the_header(monkeypatch):
    pytest.importorskip("jsonschema")
    import requests

    from rapid_mlx.community_bench import local_runner

    recorded: dict = {}

    def _fake_get(url, **kwargs):
        recorded["url"] = url
        recorded.update(kwargs)
        raise requests.RequestException("stop here")

    monkeypatch.setattr(requests, "get", _fake_get)
    assert local_runner._peak_memory_mib("http://127.0.0.1:8000") is None
    assert _header_of(recorded) == "rapid-bench"


def test_gradio_chat_sends_the_header(monkeypatch):
    import requests

    from rapid_mlx import gradio_app

    recorded: dict = {}

    def _fake_post(url, **kwargs):
        recorded["url"] = url
        recorded.update(kwargs)
        raise requests.exceptions.ConnectionError("stop here")

    monkeypatch.setattr(requests, "post", _fake_post)
    chat = gradio_app.create_chat_function("http://127.0.0.1:8000", 64, 0.7)
    chat({"text": "hi", "files": []}, [])
    assert _header_of(recorded) == "rapid-gradio"


def test_cli_server_ready_probe_sends_the_header(monkeypatch):
    """``rapid-mlx chat`` polls the server it spawned; that poll is a Rapid
    client too."""
    import requests

    from rapid_mlx import cli

    recorded: dict = {}

    class _Resp:
        status_code = 200

    def _fake_get(url, **kwargs):
        recorded["url"] = url
        recorded.update(kwargs)
        return _Resp()

    monkeypatch.setattr(requests, "get", _fake_get)
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: False, raising=False)

    class _Proc:
        returncode = None

        def poll(self):
            return None

    cli._wait_for_chat_server("http://127.0.0.1:8000", _Proc(), timeout_s=5)
    assert recorded["url"].endswith("/health/ready")
    assert _header_of(recorded) == "rapid-cli-chat"
