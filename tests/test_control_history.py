# SPDX-License-Identifier: Apache-2.0
"""In-process lifecycle, delivery, privacy, storage and event contracts."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.control import RUNNER_ENV, SECRET_ENV, SOCKET_ENV, commands, observer
from rapid_mlx.control.history import History
from rapid_mlx.control.supervisor import Supervisor


@pytest.fixture
def history(tmp_path):
    return History(tmp_path / "instance", cap=4096, event_limit=3)


def record(rid="r", **fields):
    return dict(
        ts=1,
        request_id=rid,
        client="127.0.0.1 sdk",
        app="forged",
        model="tiny",
        route="/v1/chat/completions",
        http_status=200,
        outcome="ok",
        prompt_tokens=3,
        completion_tokens=2,
        ttft_ms=1,
        duration_ms=2,
        **fields,
    )


def test_storage_privacy_filters_rotation_and_settings(history):
    history.append(
        [record(prompt="SECRET PROMPT", Authorization="Bearer secret"), record()], 1, 2
    )
    assert history.status() == dict(
        enabled=True, records=1, bytes=history.logs.stat().st_size, dropped=2
    )
    text = history.logs.read_text()
    assert all(
        s not in text for s in ("SECRET PROMPT", "Authorization", "Bearer", "forged")
    )
    assert json.loads(text)["app"] is None
    assert history.root.stat().st_mode & 0o777 == 0o700
    assert history.logs.stat().st_mode & 0o777 == 0o600
    assert (
        history.api(
            "GET",
            "/rapid/v1/log?client=127.0.0.1+sdk&model=tiny&route=/v1/chat/completions&outcome=ok&limit=10000",
            {},
        )[1][0]["request_id"]
        == "r"
    )
    assert history.api("GET", "/rapid/v1/log?model=other", {}) == (200, [])
    assert history.api("GET", "/rapid/v1/log?since=1", {}) == (200, [])
    for i in range(30):
        history.append([record(str(i))], 1)
    assert history.status()["bytes"] <= 4096
    assert history.old.exists()
    loaded = History(history.root)
    before = loaded.status()["records"]
    loaded.append([record("29")], 1)
    assert loaded.status()["records"] == before
    assert (
        history.api("PUT", "/rapid/v1/log/settings", {"enabled": False})[1]["enabled"]
        is False
    )
    history.append([record("disabled")], 1)
    assert History(history.root).enabled is False
    assert history.settings_path.stat().st_mode & 0o777 == 0o600
    assert history.api("GET", "/rapid/v1/log/settings", {})[1]["max_bytes"] == 4096
    assert history.api("PATCH", "/rapid/v1/log/settings", {}) == (404, {})
    assert history.api("POST", "/other", {}) == (404, {})
    assert history.api("DELETE", "/rapid/v1/log", {}) == (200, {})
    assert history.status()["records"] == 0


@pytest.mark.parametrize(
    "settings",
    [{"enabled": "false"}, {"content": True}, {"max_bytes": 0}, {"max_bytes": "10000"}],
)
def test_invalid_settings(history, settings):
    with pytest.raises(ValueError):
        history.api("PUT", "/rapid/v1/log/settings", settings)


def test_disabled_invalid_and_oversize_records(history):
    disabled = History(history.root, disabled=True)
    disabled.append([record()], 2)
    assert not disabled.logs.exists()
    history.append([{"request_id": None}, {"request_id": "x" * 129}], 1)
    r = record("large")
    r.update(
        client="c" * 1000,
        model="m" * 1000,
        route="r" * 1000,
        outcome="o" * 1000,
        ts="bad",
    )
    history.settings["max_bytes"] = 200
    history.append([r], 1)
    assert history.dropped == 3


def test_events_persistence_monotonic_and_bounded_ring(history):
    history.event("runner_starting")
    history.event("runner_ready")
    for _ in range(4):
        history.event("model_loaded", model="tiny")
    assert len(history.events) == 3
    assert [e["id"] for e in history.api("GET", "/rapid/v1/events?since=4", {})[1]] == [
        5,
        6,
    ]
    resumed = History(history.root, event_limit=3)
    assert resumed.event("runner_stopped")["id"] == 7


@pytest.mark.parametrize(
    "route",
    [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/audio/transcriptions",
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_one_record_per_route_and_internal_dispatch(route, stream):
    sink = observer.Delivery(capacity=2)
    receive = AsyncMock()
    send = AsyncMock()

    async def adapter(scope, original_receive, wrapped_send):
        assert original_receive is receive
        scope["route"] = SimpleNamespace(path=route)
        observer.metadata(model="tiny")
        observer.usage(
            SimpleNamespace(
                prompt_tokens=3, completion_tokens=2, time_to_first_token_ms=1
            )
        )
        await wrapped_send({"type": "http.response.start", "status": 200})
        await wrapped_send(
            {
                "type": "http.response.body",
                "body": b"SECRET RESPONSE",
                "more_body": stream,
            }
        )
        if stream:
            await wrapped_send(
                {"type": "http.response.body", "body": b"", "more_body": False}
            )

    app = observer.CallObserver(observer.CallObserver(adapter, sink), sink)
    scope = dict(
        type="http",
        client=("127.0.0.1", 42),
        headers=[
            (b"authorization", b"Bearer SECRET"),
            (b"x-rapid-app", b"forged"),
            (b"x-rapid-client", b"sdk" * 100),
        ],
    )
    await app(scope, receive, send)
    assert sink.queue.qsize() == 1
    item = sink.queue.get_nowait()
    assert item["route"] == route and item["model"] == "tiny"
    assert (item["prompt_tokens"], item["completion_tokens"], item["ttft_ms"]) == (
        3,
        2,
        1,
    )
    assert item["outcome"] == "ok" and item["app"] is None
    assert len(item["client"]) < 256
    assert "SECRET" not in json.dumps(item)
    assert scope["state"]["rapid_call"]["model"] == "tiny"
    assert observer._current.get() is None


@pytest.mark.parametrize(
    "failure,expected,status",
    [
        (ValueError("stream"), "error", 200),
        (asyncio.CancelledError(), "cancelled", 200),
        (BrokenPipeError(), "cancelled", 200),
        (None, "cancelled", 200),
        ("explicit", "cancelled", 200),
        ("encoded_error", "error", 200),
        ("preheader", "error", 500),
        ("http_error", "error", 403),
        ("499", "cancelled", 499),
    ],
)
async def test_outcomes(failure, expected, status):
    sink = observer.Delivery()

    async def app(scope, receive, send):
        if failure == "preheader":
            raise ValueError("before start")
        await send({"type": "http.response.start", "status": status})
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        if isinstance(failure, BaseException):
            raise failure
        if failure in ("explicit", "encoded_error"):
            observer.metadata(outcome="cancelled" if failure == "explicit" else "error")
        if failure:
            await send({"type": "http.response.body", "body": b""})

    try:
        await observer.CallObserver(app, sink)(
            {"type": "http"}, AsyncMock(), AsyncMock()
        )
    except (ValueError, asyncio.CancelledError, OSError):
        pass
    item = sink.queue.get_nowait()
    assert item["outcome"] == expected and item["http_status"] == status
    assert item["ttft_ms"] is None if failure == "preheader" else item["ttft_ms"] >= 0


async def test_non_http_and_single_process(monkeypatch):
    app = AsyncMock()
    monkeypatch.delenv(RUNNER_ENV, raising=False)
    assert observer.install(app) is app
    sink = observer.Delivery()
    await observer.CallObserver(app, sink)({"type": "lifespan"}, None, None)
    assert sink.queue.empty()
    observer.metadata(model="outside-request")
    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setattr(observer, "delivery", lambda: sink)
    assert isinstance(observer.install(app), observer.CallObserver)


def test_delivery_full_down_batch_and_singleton(monkeypatch):
    sink = observer.Delivery(capacity=10)
    for i in range(11):
        sink.enqueue({"i": i})
    assert sink.dropped == 1
    monkeypatch.setenv(SOCKET_ENV, "/tmp/control.sock")
    monkeypatch.setenv(SECRET_ENV, "secret")
    transport = Mock(side_effect=OSError)
    monkeypatch.setattr(observer, "request", transport)
    sink.flush([{}])
    assert sink.dropped == 2
    transport.side_effect = None
    sink.closed = Mock()
    sink.closed.wait.side_effect = [False, False, False, True]
    sink.run()
    assert [len(c.args[3]["records"]) for c in transport.call_args_list[1:]] == [
        8,
        2,
        0,
    ]
    monkeypatch.setattr(observer.atexit, "register", Mock())
    monkeypatch.setattr(observer, "_delivery", None)
    monkeypatch.setattr(observer.threading.Thread, "start", lambda _: None)
    assert observer.delivery() is observer.delivery()


async def test_sse_resume_heartbeat_gap_and_transport(monkeypatch):
    from rapid_mlx.routes import control_status as routes

    transport = Mock(return_value=[{"id": 3, "event": "runner_ready"}])
    monkeypatch.setattr(routes, "control_request", transport)
    stream = routes.event_stream("/tmp/control.sock", "secret", 2)
    assert "id: 3\n" in await anext(stream)
    assert await anext(stream) == ": heartbeat\n\n"
    transport.return_value = [{"id": 9}]
    monkeypatch.setattr(routes.asyncio, "sleep", AsyncMock())
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert transport.call_args.kwargs["target"].endswith("since=3")
    transport.side_effect = OSError
    with pytest.raises(StopAsyncIteration):
        await anext(routes.event_stream("/tmp/control.sock", "secret", 0))
    send = AsyncMock()
    response = routes.BoundedEventResponse(iter(["a"]))
    await response.stream_response(send)
    assert send.call_count == 3


def test_control_routes_auth_settings_and_sse_header(monkeypatch):
    from rapid_mlx.routes import control_status as routes

    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(SOCKET_ENV, "/tmp/control.sock")
    monkeypatch.setenv(SECRET_ENV, "private")
    app = FastAPI()
    routes.mount_status(app)
    app.dependency_overrides[routes.verify_api_key] = lambda: None
    transport = Mock(return_value={"enabled": False})
    monkeypatch.setattr(routes, "control_request", transport)
    with TestClient(app, client=("127.0.0.1", 1)) as client:
        assert client.put("/rapid/v1/log/settings", json={"enabled": False}).json() == {
            "enabled": False
        }
        assert client.get("/rapid/v1/log?limit=2").status_code == 200
        assert transport.call_args.kwargs["target"] == "/rapid/v1/log?limit=2"
        assert client.delete("/rapid/v1/log").status_code == 200
        assert (
            client.put("/rapid/v1/log/settings", content="a" * 17000).status_code == 413
        )
        transport.side_effect = OSError
        assert client.get("/rapid/v1/log").status_code == 503
        assert (
            client.get(
                "/rapid/v1/events", headers={"Last-Event-ID": "invalid"}
            ).status_code
            == 400
        )
        captured = Mock(return_value=routes.StreamingResponse(iter([])))
        monkeypatch.setattr(routes, "BoundedEventResponse", captured)
        assert (
            client.get("/rapid/v1/events", headers={"Last-Event-ID": "7"}).status_code
            == 200
        )
        stream = captured.call_args.args[0]
        assert stream.ag_frame.f_locals["cursor"] == 7
    with TestClient(app, client=("192.0.2.1", 1)) as client:
        for path in ("/rapid/v1/log", "/rapid/v1/log/settings", "/rapid/v1/events"):
            assert client.get(path).status_code == 403


async def test_parent_record_and_model_reports(history):
    supervisor = Supervisor([], port=8000, mode="lazy")
    supervisor.history = history
    supervisor.child = SimpleNamespace(pid=42)
    supervisor.spawn_ready.set()

    async def call(target, data, pid=42):
        reader = asyncio.StreamReader()
        body = json.dumps({**data, "pid": pid}).encode()
        reader.feed_data(
            f"POST {target} HTTP/1.1\r\nX-Rapid-Control-Secret: {supervisor.secret}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        reader.feed_eof()
        writer = SimpleNamespace(
            write=Mock(), drain=AsyncMock(), close=Mock(), wait_closed=AsyncMock()
        )
        await supervisor.handle(reader, writer)
        return writer.write.call_args.args[0]

    assert b"200" in await call("/records", {"records": [record()], "dropped": 2})
    assert supervisor.status()["log"]["records"] == 1
    assert b"409" in await call("/records", {"records": [], "dropped": 0}, pid=43)
    for event in ("model_loaded", "model_unloaded", "bound"):
        assert b"200" in await call(
            "/startup", {"event": event, "model": "tiny", "port": 8000}
        )
    assert [e["event"] for e in history.events] == [
        "model_loaded",
        "model_unloaded",
        "runner_ready",
    ]


@pytest.mark.parametrize(
    "command,json_output,follow",
    [("log", True, False), ("events", False, False), ("events", True, True)],
)
def test_cli_history(monkeypatch, capsys, command, json_output, follow):
    monkeypatch.setattr(
        commands,
        "connection",
        lambda _: dict(socket="/tmp/control.sock", secret="private"),
    )
    transport = Mock(return_value=[{"id": 1, "ts": 2}])
    monkeypatch.setattr(commands, "request", transport)
    monkeypatch.setattr(commands.time, "sleep", Mock(side_effect=KeyboardInterrupt))
    args = SimpleNamespace(
        command=command, port=None, limit=3, json=json_output, follow=follow
    )
    assert commands.history_command(args) == 0
    assert "1" in capsys.readouterr().out
    args.action = "clear"
    assert commands.history_command(args) == 0
    assert transport.call_args.args[2] == "DELETE"
    transport.side_effect = OSError("down")
    assert commands.history_command(args) == 1
    assert "down" in capsys.readouterr().err


def test_connection_discovery(monkeypatch, tmp_path):
    monkeypatch.setattr(commands, "discovery_root", lambda: tmp_path)
    monkeypatch.setattr(commands, "discover", lambda _: [])
    with pytest.raises(OSError, match="No supervised"):
        commands.connection()
    monkeypatch.setattr(commands, "discover", lambda _: [{"supervisor_pid": 42}])
    for i in range(3):
        (tmp_path / f"{i}.json").write_text(
            json.dumps({"socket": f"/tmp/{i}.sock", "secret": "s"})
        )
    transport = Mock(
        side_effect=[OSError(), {"supervisor_pid": 43}, {"supervisor_pid": 42}]
    )
    monkeypatch.setattr(commands, "request", transport)
    assert commands.connection()["socket"] == "/tmp/2.sock"
    transport.side_effect = OSError()
    with pytest.raises(OSError, match="unavailable"):
        commands.connection()


async def test_slow_sse_client_disconnects(monkeypatch):
    from rapid_mlx.routes import control_status as routes

    timeout = asyncio.timeout
    monkeypatch.setattr(routes.asyncio, "timeout", lambda _: timeout(0.001))

    async def blocked_send(_):
        await asyncio.Event().wait()

    response = routes.BoundedEventResponse(iter(["event"]))
    with pytest.raises(TimeoutError):
        await response.stream_response(blocked_send)


def test_settings_cap_shrink(history):
    for i in range(100):
        history.append([record(str(i))], 1)
    history.settings["max_bytes"] = 50000
    for i in range(100, 150):
        history.append([record(str(i))], 1)
    assert history.status()["bytes"] > 4096
    history.api("PUT", "/rapid/v1/log/settings", {"max_bytes": 4096, "enabled": False})
    assert 0 < history.status()["bytes"] <= 4096


def test_cli_dispatch_and_flags(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.cli_parser import build_parser

    for command in ("log", "events"):
        args = build_parser().parse_args([command, "--follow", "--json"])
        assert args.follow and args.json
        monkeypatch.setattr(cli.sys, "argv", ["rapid-mlx", command])
        monkeypatch.setattr(commands, "history_command", lambda _: 0)
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
    assert build_parser().parse_args(["serve", "tiny", "--no-call-log"]).no_call_log


async def test_parent_log_query_and_put(history):
    from rapid_mlx.control import SECRET_HEADER

    supervisor = Supervisor([], port=8000, mode="lazy")
    supervisor.history = history
    for method, target, payload in [
        ("GET", "/rapid/v1/log?limit=2", None),
        ("PUT", "/rapid/v1/log/settings", {"enabled": False}),
    ]:
        body = json.dumps(payload).encode() if payload else b""
        reader = asyncio.StreamReader()
        reader.feed_data(
            f"{method} {target} HTTP/1.1\r\n{SECRET_HEADER}: {supervisor.secret}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        reader.feed_eof()
        writer = SimpleNamespace(
            write=Mock(), drain=AsyncMock(), close=Mock(), wait_closed=AsyncMock()
        )
        await supervisor.handle(reader, writer)
        assert writer.write.call_args.args[0].startswith(b"HTTP/1.1 200")


def test_runner_observer_is_outermost(monkeypatch):
    from rapid_mlx import _uvicorn

    sink = observer.Delivery()
    monkeypatch.setattr(observer, "delivery", lambda: sink)
    monkeypatch.setenv(RUNNER_ENV, "1")
    config = _uvicorn._TransportPeerConfig(FastAPI(), proxy_headers=False)
    config.load()
    assert isinstance(config.loaded_app, observer.CallObserver)
    asyncio.run(
        config.loaded_app(
            {
                "type": "http",
                "path": "/",
                "method": "GET",
                "headers": [],
                "query_string": b"",
                "client": ("127.0.0.1", 1),
                "server": ("localhost", 1),
                "scheme": "http",
                "http_version": "1.1",
            },
            AsyncMock(),
            AsyncMock(),
        )
    )
    assert sink.queue.get_nowait()["http_status"] == 404


@pytest.mark.parametrize("route", ["chat", "completions", "anthropic", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_real_text_adapters_emit_usage(route, stream):
    from rapid_mlx.config import reset_config
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.routes import anthropic, chat, completions, responses
    from tests.test_stream_include_usage_honored import (
        _PlainChatEngine,
        _PlainCompletionsEngine,
    )

    class Chat(_PlainChatEngine):
        async def chat(self, *args, **kwargs):
            return GenerationOutput(
                text="answer",
                prompt_tokens=4,
                completion_tokens=3,
                finish_reason="stop",
            )

    class Completion(_PlainCompletionsEngine):
        async def generate(self, *args, **kwargs):
            return GenerationOutput(
                text="answer",
                prompt_tokens=4,
                completion_tokens=3,
                finish_reason="stop",
            )

    cfg = reset_config()
    cfg.engine = Completion() if route == "completions" else Chat()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    app.include_router(
        dict(
            chat=chat, completions=completions, anthropic=anthropic, responses=responses
        )[route].router
    )
    sink = observer.Delivery()
    client = TestClient(observer.CallObserver(app, sink))
    payload = dict(model="test-model", stream=stream, max_tokens=32)
    if route == "completions":
        path = "/v1/completions"
        payload["prompt"] = "SECRET PROMPT"
    elif route == "responses":
        path = "/v1/responses"
        payload["input"] = "SECRET PROMPT"
    else:
        path = "/v1/messages" if route == "anthropic" else "/v1/chat/completions"
        payload["messages"] = [{"role": "user", "content": "SECRET PROMPT"}]
    result = client.post(path, json=payload)
    assert result.status_code == 200, result.text
    assert sink.queue.qsize() == 1
    item = sink.queue.get_nowait()
    assert item["outcome"] == "ok", result.text
    assert item["model"] == "test-model"
    assert item["prompt_tokens"] == (3 if route == "completions" and stream else 4)
    assert item["completion_tokens"] == 3
    assert "SECRET PROMPT" not in json.dumps(item)


@pytest.mark.parametrize("stream", [False, True])
async def test_adapter_disconnect_and_task_cancel(monkeypatch, stream):
    import sys

    from rapid_mlx.service import helpers

    engine_context = SimpleNamespace(get=lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.engine.batched",
        SimpleNamespace(_admission_engine_context=engine_context),
    )
    data = {}
    token = observer._current.set(data)
    raw_request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))

    async def pending():
        await asyncio.Event().wait()

    async def generate():
        await pending()
        yield "unreachable"

    try:
        if stream:
            assert [
                x
                async for x in helpers._disconnect_guard(
                    generate(), raw_request, poll_interval=0.001
                )
            ] == []
        else:
            assert (
                await helpers._wait_with_disconnect(
                    pending(), raw_request, timeout=1, poll_interval=0.001
                )
                is None
            )
        assert data["outcome"] == "cancelled"
        if stream:

            async def cancelled():
                raise asyncio.CancelledError()
                yield "unreachable"

            with pytest.raises(asyncio.CancelledError):
                async for _ in helpers._disconnect_guard(
                    cancelled(), raw_request, poll_interval=100
                ):
                    pass
    finally:
        observer._current.reset(token)


def test_real_embeddings_adapter(monkeypatch):
    import sys

    from rapid_mlx.config import reset_config
    from rapid_mlx.routes import embeddings

    # Optional tensor implementation is replaced at its import boundary; the
    # actual route still performs admission, counters and response rendering.
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.embedding",
        SimpleNamespace(
            EMBEDDINGS_EXTRA_HTTP_INSTALL_HINT="",
            EmbeddingInputTooLongError=type("TooLong", (Exception,), {}),
            EmbeddingUnsupportedMediaError=type("Unsupported", (Exception,), {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.server",
        SimpleNamespace(load_embedding_model=lambda *a, **kw: None),
    )
    cfg = reset_config()
    cfg.embedding_engine = SimpleNamespace(
        count_tokens=lambda _: 4, embed=lambda _: [[1.0, 0.0]], effective_max_length=512
    )
    cfg.embedding_model_locked = "test-model"
    app = FastAPI()
    app.include_router(embeddings.router)
    sink = observer.Delivery()
    with TestClient(observer.CallObserver(app, sink)) as client:
        result = client.post(
            "/v1/embeddings", json={"model": "test-model", "input": "SECRET PROMPT"}
        )
    assert result.status_code == 200, result.text
    item = sink.queue.get_nowait()
    assert (item["model"], item["prompt_tokens"], item["completion_tokens"]) == (
        "test-model",
        4,
        0,
    )


def test_real_audio_speech_adapter(monkeypatch):
    from rapid_mlx.audio import probe
    from rapid_mlx.config import reset_config
    from rapid_mlx.routes import audio

    reset_config()
    monkeypatch.setattr(probe, "require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr(probe, "require_kokoro_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(
        audio, "_generate_speech_blocking", lambda *a: (b"RIFF0", 24000, 1)
    )
    app = FastAPI()
    app.include_router(audio.router)
    sink = observer.Delivery()
    with TestClient(observer.CallObserver(app, sink)) as client:
        result = client.post(
            "/v1/audio/speech",
            json={"model": "kokoro", "input": "SECRET PROMPT", "voice": "af_heart"},
        )
    assert result.status_code == 200, result.text
    item = sink.queue.get_nowait()
    assert "Kokoro" in item["model"] and item["prompt_tokens"] is None
    assert item["completion_tokens"] is None and item["outcome"] == "ok"


async def test_audio_music_model_metadata(monkeypatch):
    from fastapi import HTTPException

    from rapid_mlx.api.models import AudioMusicRequest
    from rapid_mlx.routes import audio

    data = {}
    token = observer._current.set(data)
    try:
        with pytest.raises(HTTPException):
            await audio.create_music(AudioMusicRequest(model="unknown", input="sound"))
        assert data["model"] == "unknown"
    finally:
        observer._current.reset(token)


def test_actual_stream_failure_after_200():
    from rapid_mlx.config import reset_config
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.routes import chat
    from tests.test_stream_include_usage_honored import _PlainChatEngine

    class Failed(_PlainChatEngine):
        async def stream_chat(self, *args, **kwargs):
            yield GenerationOutput(
                text="first", new_text="first", prompt_tokens=4, completion_tokens=1
            )
            raise RuntimeError("PRIVATE failure")

    cfg = reset_config()
    cfg.engine = Failed()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    app.include_router(chat.router)
    sink = observer.Delivery()
    with TestClient(observer.CallObserver(app, sink)) as client:
        result = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert result.status_code == 200
    item = sink.queue.get_nowait()
    assert item["http_status"] == 200 and item["outcome"] == "error"
    assert "PRIVATE" not in json.dumps(item)


def test_event_disk_failure_is_best_effort(history, monkeypatch):
    from rapid_mlx.control import history as storage

    monkeypatch.setattr(storage, "write_private", Mock(side_effect=OSError))
    assert history.event("runner_stopped")["id"] == 1
    assert history.dropped == 1


@pytest.mark.parametrize("route", ["chat", "completions"])
def test_real_stream_explicit_cancellation(route):
    from rapid_mlx.config import reset_config
    from rapid_mlx.routes import chat, completions
    from tests.test_cancellation_protocol import (
        _CancelledChatEngine,
        _CancelledCompletionEngine,
    )

    cfg = reset_config()
    cfg.engine = (
        _CancelledChatEngine(before_first_token=False)
        if route == "chat"
        else _CancelledCompletionEngine()
    )
    cfg.engine.build_prompt = lambda *a, **kw: "prompt"
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    app.include_router(chat.router if route == "chat" else completions.router)
    sink = observer.Delivery()
    payload = {"model": "test-model", "stream": True}
    if route == "chat":
        payload["messages"] = [{"role": "user", "content": "hi"}]
    else:
        payload["prompt"] = "hi"
    with TestClient(observer.CallObserver(app, sink)) as client:
        result = client.post(
            "/v1/chat/completions" if route == "chat" else "/v1/completions",
            json=payload,
        )
    assert result.status_code == 200
    assert sink.queue.get_nowait()["outcome"] == "cancelled"


async def test_nonstream_task_cancellation_metadata(monkeypatch):
    import sys

    from rapid_mlx.service import helpers

    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.engine.batched",
        SimpleNamespace(_admission_engine_context=SimpleNamespace(get=lambda: None)),
    )

    async def cancelled():
        raise asyncio.CancelledError()

    data = {}
    token = observer._current.set(data)
    try:
        with pytest.raises(asyncio.CancelledError):
            await helpers._wait_with_disconnect(
                cancelled(),
                SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
                timeout=1,
            )
        assert data["outcome"] == "cancelled"
    finally:
        observer._current.reset(token)


@pytest.mark.parametrize(
    "alive,expired", [(False, False), (False, True), (True, False)]
)
def test_delivery_graceful_flush_is_bounded(monkeypatch, alive, expired):
    sink = observer.Delivery()
    sink.thread = Mock()
    sink.thread.is_alive.return_value = alive
    sink.enqueue({"request_id": "last"})
    flush = Mock()
    monkeypatch.setattr(sink, "flush", flush)
    monkeypatch.setattr(
        observer.time, "monotonic", Mock(side_effect=[0, 1] if expired else [0, 0])
    )
    sink.close()
    assert flush.call_count == int(not alive and not expired)
    assert sink.dropped == int(not alive and expired)


def test_discovery_preserves_user_settings(history, monkeypatch):
    from rapid_mlx.control import protocol

    history.api("PUT", "/rapid/v1/log/settings", {"enabled": False})
    monkeypatch.setattr(protocol, "discovery_root", lambda: history.root.parent)
    assert protocol.discover() == []
    assert history.settings_path.exists()
    assert not History(history.root).enabled


def test_lifecycle_events_are_optional_and_best_effort(monkeypatch, caplog):
    supervisor = Supervisor([], port=8000, mode="lazy")
    supervisor.emit("runner_starting")
    supervisor.history = SimpleNamespace(event=Mock(side_effect=OSError("disk full")))
    supervisor.emit("runner_ready")
    supervisor.child = SimpleNamespace(pid=42)
    supervisor.emit("runner_stopped")
    assert len(caplog.records) == 1
    assert "event emission failed (OSError)" in caplog.text


async def test_history_setup_failure_does_not_prevent_control_startup(
    tmp_path, monkeypatch
):
    from rapid_mlx.control import history
    from rapid_mlx.control import supervisor as module

    monkeypatch.setattr(module, "runtime_root", lambda: tmp_path / "runtime")
    monkeypatch.setattr(module, "discovery_root", lambda: tmp_path / "control")
    monkeypatch.setattr(history, "History", Mock(side_effect=OSError("unavailable")))
    socket_server = SimpleNamespace()

    async def start(handler, *, path, **kwargs):
        from pathlib import Path

        Path(path).touch()
        return socket_server

    monkeypatch.setattr(module.asyncio, "start_unix_server", start)
    supervisor = Supervisor([], port=8000, mode="lazy")
    await supervisor.open()
    assert supervisor.server is socket_server
    assert supervisor.history is None
    supervisor.emit("runner_starting")
    assert supervisor.status()["log"] == {}
    supervisor.child = SimpleNamespace(pid=42)
    reader = asyncio.StreamReader()
    body = json.dumps({"pid": 42, "records": [], "dropped": 0}).encode()
    reader.feed_data(
        f"POST /records HTTP/1.1\r\nX-Rapid-Control-Secret: {supervisor.secret}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    writer = SimpleNamespace(
        write=Mock(), drain=AsyncMock(), close=Mock(), wait_closed=AsyncMock()
    )
    await supervisor.handle(reader, writer)
    assert writer.write.call_args.args[0].startswith(b"HTTP/1.1 503")
