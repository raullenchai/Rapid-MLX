# SPDX-License-Identifier: Apache-2.0
"""In-process lifecycle, delivery, privacy, storage and event contracts."""

import asyncio
import contextvars
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
        enabled=True,
        records=1,
        bytes=history.logs.stat().st_size,
        dropped=2,
        skipped=0,
        malformed=0,
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
        scope["route"] = SimpleNamespace(path="/v1/chat/completions")
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
    sink.closed.is_set.side_effect = [False, False, False, False, False, False, True]
    sink.pending = Mock()
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


def test_idle_delivery_periodic_intervals_without_subscribers(monkeypatch):
    """P2 must not wake the supervisor with 10 empty socket reports per second."""
    sink = observer.Delivery()
    intervals = []
    flush = Mock()

    def wait(event, timeout=None):
        intervals.append(timeout)
        if len(intervals) == 3:
            sink.closed.set()
            return True
        return False  # Advance an idle reporting interval without wall-clock delay.

    monkeypatch.setattr(observer.threading.Event, "wait", wait)
    monkeypatch.setattr(sink, "flush", flush)
    sink.run()
    assert intervals == [5.0, 5.0, 5.0]
    assert flush.call_args_list == [(([],),), (([],),)]


def test_delivery_wakes_for_records_and_close(monkeypatch):
    sink = observer.Delivery(capacity=20)
    idle = observer.threading.Event()
    delivered = observer.threading.Event()
    real_wait = sink.pending.wait
    batches = []

    def wait(timeout):
        idle.set()
        return real_wait(timeout)

    def flush(batch):
        batches.append(batch)
        if sum(map(len, batches)) == 20:
            delivered.set()

    monkeypatch.setattr(sink.pending, "wait", wait)
    monkeypatch.setattr(sink, "flush", flush)
    sink.thread.start()
    try:
        assert idle.wait(1)
        assert batches == []  # No report on startup with no subscribers/requests.
        for i in range(20):
            sink.enqueue({"i": i})
        assert delivered.wait(1)  # Records must not wait for the five-second timer.
        assert [item["i"] for batch in batches for item in batch] == list(range(20))
        assert all(len(batch) <= 8 for batch in batches)
    finally:
        sink.close()
        sink.thread.join(timeout=1)
    assert not sink.thread.is_alive()  # Closing an idle worker wakes it as well.


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

    assert b"200" in await call(
        "/records", {"records": [record()], "dropped": 2, "skipped": 4}
    )
    assert supervisor.status()["log"]["records"] == 1
    assert supervisor.status()["log"]["skipped"] == 4
    assert b"200" in await call(
        "/records",
        {
            "records": [],
            "dropped": 2,
            "events": [
                {"event": "model_loaded", "model": "tiny"},
                {"event": "forged", "model": "ignored"},
                {"event": "model_unloaded", "model": "tiny"},
            ],
        },
    )
    assert [e["event"] for e in history.events] == ["model_loaded", "model_unloaded"]
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
    assert history.status()["bytes"] <= 4096


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
    assert sink.queue.empty()
    assert sink.skipped == 1


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


def _fake_batched():
    # The conftest admission guard resets both contexts at teardown, before
    # monkeypatch restores sys.modules, so the stub must expose both.
    return SimpleNamespace(
        _admission_engine_context=contextvars.ContextVar(
            "fake_admission_engine", default=None
        ),
        _admission_token_context=contextvars.ContextVar(
            "fake_admission_token", default=None
        ),
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_adapter_disconnect_and_task_cancel(monkeypatch, stream):
    import sys

    from rapid_mlx.service import helpers

    monkeypatch.setitem(sys.modules, "rapid_mlx.engine.batched", _fake_batched())
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


@pytest.mark.parametrize("route", ["chat", "completions", "anthropic", "responses"])
def test_real_stream_explicit_cancellation(route):
    from rapid_mlx.config import reset_config
    from rapid_mlx.routes import anthropic, chat, completions, responses
    from tests.test_cancellation_protocol import (
        _CancelledChatEngine,
        _CancelledCompletionEngine,
    )

    cfg = reset_config()
    cfg.engine = (
        _CancelledChatEngine(before_first_token=False)
        if route != "completions"
        else _CancelledCompletionEngine()
    )
    cfg.engine.build_prompt = lambda *a, **kw: "prompt"
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    app.include_router(
        {
            "chat": chat,
            "completions": completions,
            "anthropic": anthropic,
            "responses": responses,
        }[route].router
    )
    sink = observer.Delivery()
    payload = {"model": "test-model", "stream": True, "max_tokens": 32}
    if route in ("chat", "anthropic"):
        payload["messages"] = [{"role": "user", "content": "hi"}]
    elif route == "responses":
        payload["input"] = "hi"
    else:
        payload["prompt"] = "hi"
    with TestClient(observer.CallObserver(app, sink)) as client:
        result = client.post(
            {
                "chat": "/v1/chat/completions",
                "anthropic": "/v1/messages",
                "responses": "/v1/responses",
                "completions": "/v1/completions",
            }[route],
            json=payload,
        )
    assert result.status_code == 200
    assert sink.queue.get_nowait()["outcome"] == "cancelled"


async def test_nonstream_task_cancellation_metadata(monkeypatch):
    import sys

    from rapid_mlx.service import helpers

    monkeypatch.setitem(sys.modules, "rapid_mlx.engine.batched", _fake_batched())

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


async def test_residency_does_not_wait_for_blocked_control_socket(monkeypatch):
    import threading
    import time

    from rapid_mlx.control import protocol
    from tests.test_resident_models import manager_fixture

    entered = threading.Event()
    release = threading.Event()
    payloads = []

    class BlockedSocket:
        def __init__(self, path):
            pass

        def request(self, method, target, *, body, headers):
            payloads.append(json.loads(body))
            entered.set()
            assert release.wait(2)

        def getresponse(self):
            return SimpleNamespace(status=200, read=lambda _: b"{}")

        def close(self):
            pass

    sink = observer.Delivery(capacity=2)
    monkeypatch.setattr(observer, "delivery", lambda: sink)
    monkeypatch.setattr(protocol, "UnixConnection", BlockedSocket)
    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(SOCKET_ENV, "/fake/control.sock")
    monkeypatch.setenv(SECRET_ENV, "private")
    monkeypatch.setattr(
        "rapid_mlx.runtime.resident_models._release_allocator_cache", lambda: None
    )
    sink.thread.start()
    try:
        manager, _, _, _ = manager_fixture()
        assert await asyncio.to_thread(entered.wait, 1)
        # The first report is now stuck inside the fake socket. Neither the
        # residency lock nor the event loop may depend on that thread.
        beats = []
        loop = asyncio.get_running_loop()
        loop.call_soon(lambda: beats.append(time.monotonic()))
        started = time.monotonic()
        await manager.load("secondary", estimated_bytes=1)
        await manager.unload("secondary")
        await asyncio.sleep(0)
        assert beats and beats[0] - started < 0.1
        assert time.monotonic() - started < 0.1
        # A saturated control plane drops and counts rather than waiting.
        observer.residency_event("model_loaded", model="overflow")
        assert sink.queue.qsize() == 2 and sink.dropped == 1
    finally:
        release.set()
        sink.close()
        sink.thread.join(1)
    events = [event for payload in payloads for event in payload["events"]]
    assert [e["event"] for e in events] == [
        "model_loaded",
        "model_loaded",
        "model_unloaded",
    ]
    assert all(payload["records"] == [] for payload in payloads)
    monkeypatch.delenv(RUNNER_ENV)
    observer.residency_event("model_loaded", model="standalone")
    assert sink.queue.empty()


def test_shrink_enforces_each_file_even_below_total_cap(tmp_path):
    mib = 1024 * 1024
    history = History(tmp_path / "instance")

    # Valid metadata rows with exactly the reviewed 25 MiB + 1 MiB sizes.
    def row(seq):
        item = record(str(seq)) | {"seq": seq}
        encoded = json.dumps(item)
        return encoded + " " * (1024 - len(encoded) - 1) + "\n"

    history.old.write_text("".join(row(i) for i in range(25 * 1024)))
    history.logs.write_text("".join(row(i) for i in range(25 * 1024, 26 * 1024)))
    history.call_sequence = 26 * 1024
    assert history.status()["bytes"] == 26 * mib
    history.api("PUT", "/rapid/v1/log/settings", {"max_bytes": 30 * mib})
    assert not history.old.exists()
    assert history.logs.stat().st_size == mib
    retained = list(history.records())
    assert retained[0]["seq"] == 25 * 1024
    assert retained[-1]["seq"] == 26 * 1024 - 1
    # Growth up to the active-file bound cannot exceed the new total cap.
    for i in range(40000, 40100):
        history.append([record(str(i))], 1)
    assert history.status()["bytes"] <= 30 * mib


def test_sequence_pagination_survives_ties_rollback_clear_and_restart(history):
    history.settings["max_bytes"] = 50000
    for i, ts in enumerate([3, 3, 2, 1, 4, 0, 0]):
        history.append([record(str(i)) | {"ts": ts, "seq": 999999}], 1)
    cursor = 0
    seen = []
    for _ in range(5):
        status, page = history.api("GET", f"/rapid/v1/log?after={cursor}&limit=2", {})
        assert status == 200
        seen.extend(item["request_id"] for item in page)
        if page:
            cursor = page[-1]["seq"]
    assert seen == [str(i) for i in range(7)]
    assert cursor == 7
    restarted = History(history.root)
    assert [r["seq"] for r in restarted.records()] == list(range(1, 8))
    restarted.api("DELETE", "/rapid/v1/log", {})
    restarted = History(history.root)
    restarted.append([record("next")], 1)
    assert restarted.api("GET", "/rapid/v1/log?after=7", {})[1][0]["seq"] == 8


def test_legacy_records_receive_stable_cursors(history):
    history.old.write_text(json.dumps(record("old")) + "\n")
    history.logs.write_text(json.dumps(record("active")) + "\n")
    migrated = History(history.root)
    assert [r["seq"] for r in migrated.records()] == [1, 2]
    assert [r["seq"] for r in History(history.root).records()] == [1, 2]


def test_follow_drains_pages_before_polling(history, monkeypatch, capsys):
    history.settings["max_bytes"] = 50000

    def append(start, stop):
        for i in range(start, stop):
            history.append([record(str(i)) | {"ts": 7 - i}], 1)

    append(0, 7)
    monkeypatch.setattr(
        commands, "connection", lambda _: {"socket": "fake", "secret": "s"}
    )
    targets = []

    def request(*args, target):
        targets.append(target)
        return history.api("GET", target, {})[1]

    sleeps = []

    def sleep(_):
        sleeps.append(len(targets))
        if len(sleeps) == 1:
            append(7, 14)  # More than a page arrives between polls, with older times.
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(commands, "request", request)
    monkeypatch.setattr(commands.time, "sleep", sleep)
    args = SimpleNamespace(command="log", port=None, limit=3, json=True, follow=True)
    assert commands.history_command(args) == 0
    assert [
        json.loads(line)["request_id"] for line in capsys.readouterr().out.splitlines()
    ] == [str(i) for i in range(14)]
    assert sleeps == [3, 6]
    assert all("after=" in target for target in targets)


@pytest.mark.parametrize("route", ["chat", "anthropic", "responses"])
@pytest.mark.parametrize("failure", ["window", "forced_tool", "tool_schema"])
def test_wire_errors_mark_call_failed(route, failure):
    from rapid_mlx.config import reset_config
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.routes import anthropic, chat, responses
    from tests.test_stream_include_usage_honored import _PlainChatEngine

    class Engine(_PlainChatEngine):
        supports_tool_calls = True

        async def stream_chat(self, *args, **kwargs):
            text = (
                '<tool_call>{"name":"lookup","arguments":{"value":"bad"}}</tool_call>'
                if failure == "tool_schema"
                else "answer"
            )
            yield GenerationOutput(
                text=text,
                raw_text=text,
                new_text=text,
                prompt_tokens=70,
                completion_tokens=10,
                finished=True,
                finish_reason="length" if failure == "window" else "stop",
            )

    cfg = reset_config()
    cfg.engine = Engine()
    cfg.model_name = "test-model"
    cfg.no_thinking = True
    cfg.tool_call_parser = "hermes"
    cfg.context_length = 80 if failure == "window" else None
    if failure == "window":
        from tests.test_context_overflow_400 import _StubTokenizer

        cfg.engine.tokenizer = _StubTokenizer()
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    function = {"name": "lookup", "parameters": schema}
    payload = {"model": "test-model", "stream": True, "max_tokens": 10}
    if route == "responses":
        payload["input"] = "hi"
        path = "/v1/responses"
    else:
        payload["messages"] = [{"role": "user", "content": "hi"}]
        path = "/v1/messages" if route == "anthropic" else "/v1/chat/completions"
    if failure != "window":
        if route == "anthropic":
            payload["tools"] = [{"name": "lookup", "input_schema": schema}]
            payload["tool_choice"] = {"type": "tool", "name": "lookup"}
        else:
            payload["tools"] = [{"type": "function", "function": function}]
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": "lookup"},
            }
    app = FastAPI()
    app.include_router(
        {"chat": chat, "anthropic": anthropic, "responses": responses}[route].router
    )
    sink = observer.Delivery()
    with TestClient(observer.CallObserver(app, sink)) as client:
        result = client.post(path, json=payload)
    assert result.status_code == 200, result.text
    assert (
        "event: response.failed"
        if route == "responses"
        else "event: error"
        if route == "anthropic"
        else '"error"'
    ) in result.text
    item = sink.queue.get_nowait()
    assert (item["http_status"], item["outcome"]) == (200, "error"), result.text
    assert sink.queue.empty()


def test_strict_stream_buffer_overflow_logs_error(monkeypatch):
    from tests.test_response_format_json_schema_strict import (
        _Engine,
        _make_client,
        _payload,
    )

    monkeypatch.setenv("RAPID_MLX_STRICT_BUFFER_BYTES", "8")
    app = _make_client(
        _Engine(supports_guided=False, chat_text='{"value":42}' * 50)
    ).app
    sink = observer.Delivery()
    with TestClient(observer.CallObserver(app, sink)) as client:
        result = client.post(
            "/v1/chat/completions", json=_payload(strict=True, stream=True)
        )
    assert result.status_code == 200, result.text
    assert "buffer_overflow" in result.text
    item = sink.queue.get_nowait()
    assert item["http_status"] == 200 and item["outcome"] == "error"


@pytest.mark.parametrize("alignment", [False, True])
def test_audio_rejected_capability_retains_resolved_model(monkeypatch, alignment):
    from rapid_mlx.audio import probe
    from rapid_mlx.config import reset_config
    from rapid_mlx.routes import audio

    reset_config()
    monkeypatch.setattr(probe, "require_mlx_audio_stt", lambda: None)
    model = (
        "mlx-community/whisper-tiny"
        if alignment
        else "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
    )
    app = FastAPI()
    app.include_router(audio.router)
    sink = observer.Delivery()
    data = {"model": model}
    if alignment:
        data["text"] = "hi"
    with TestClient(observer.CallObserver(app, sink)) as client:
        result = client.post(
            "/v1/audio/transcriptions",
            data=data,
            files={"file": ("sample.wav", b"invalid", "audio/wav")},
        )
    assert result.status_code == 400, result.text
    item = sink.queue.get_nowait()
    assert item["model"] == model and item["outcome"] == "error"


def test_user_history_concurrent_writers_restart_and_global_cap(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    root = tmp_path / "history"
    a = History(root, instance="first", port=8000, cap=8192)
    b = History(root, instance="second", port=8001, cap=8192)

    def write(store, prefix):
        for i in range(25):
            store.append([record(f"{prefix}{i}")], 1)
            store.event("runner_ready")

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(write, a, "a"), pool.submit(write, b, "b")]
        for future in futures:
            future.result()
    rows = list(a.records())
    assert len({r["seq"] for r in rows}) == len(rows)
    assert [r["seq"] for r in rows] == sorted(r["seq"] for r in rows)
    assert {r["port"] for r in rows} <= {8000, 8001}
    assert all(r["instance"] in {"first", "second"} for r in rows)
    assert json.loads(a.cursor_path.read_text()) == 50
    assert json.loads(a.event_cursor.read_text()) == 50
    assert sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) <= 8192
    resumed = History(root, instance="third", port=8002)
    resumed.append([record("restart")], 2)
    assert list(resumed.records())[-1]["seq"] == 51
    assert resumed.event("runner_ready")["id"] == 51
    assert (
        resumed.api("GET", "/rapid/v1/log?port=8002", {})[1][0]["instance"] == "third"
    )
    assert resumed.api("GET", "/rapid/v1/log?port=8003", {})[1] == []
    # Settings are per user, including servers already running.
    a.api("PUT", "/rapid/v1/log/settings", {"enabled": False, "max_bytes": 4096})
    b.append([record("disabled-on-other-server")], 1)
    assert all(r["request_id"] != "disabled-on-other-server" for r in b.records())
    assert sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) <= 4096


def test_legacy_cleanup_only_recognized_private_owned_directories(history):
    import os

    valid = history.root / "i-abcd1234"
    valid.mkdir(mode=0o700)
    (valid / "calls.jsonl").write_text("old")
    retained = []
    for name, mode, filename in [
        ("i-abcdefgh", 0o755, "calls.jsonl"),
        ("i-12345678", 0o700, "other"),
        ("i-short", 0o700, "calls.jsonl"),
        ("i-symlink1", 0o700, "calls.jsonl"),
    ]:
        path = history.root / name
        path.mkdir(mode=mode)
        os.chmod(path, mode)
        if name == "i-symlink1":
            (path / filename).symlink_to(history.root / "outside")
        else:
            (path / filename).touch()
        retained.append(path)
    (history.root / "i-link0000").symlink_to(valid, target_is_directory=True)
    history.cleanup_legacy()
    assert not valid.exists()
    assert all(p.exists() for p in retained)
    assert (history.root / "i-link0000").is_symlink()


def test_offline_log_read_clear_and_empty_without_creating_files(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(commands, "connection", Mock(side_effect=OSError("offline")))
    monkeypatch.setattr(commands, "discovery_root", lambda: tmp_path / "control")
    args = SimpleNamespace(command="log", port=None, limit=100, json=True, follow=False)
    assert commands.history_command(args) == 0
    assert not (tmp_path / "history").exists()
    store = History(tmp_path / "history", instance="a", port=8080)
    store.append([record("before")], 1)
    before = {
        p: (p.stat().st_mtime_ns, p.read_bytes())
        for p in store.root.rglob("*")
        if p.is_file()
    }
    assert commands.history_command(args) == 0
    assert json.loads(capsys.readouterr().out)["request_id"] == "before"
    assert before == {
        p: (p.stat().st_mtime_ns, p.read_bytes())
        for p in store.root.rglob("*")
        if p.is_file()
    }
    args.action = "clear"
    assert commands.history_command(args) == 0
    assert list(store.records()) == []
    store.append([record("after")], 1)
    assert list(store.records())[0]["seq"] == 2


@pytest.mark.parametrize(
    "path",
    [
        "/healthz",
        "/ready",
        "/startup",
        "/rapid/v1/status",
        "/rapid/v1/log",
        "/rapid/v1/events",
        "/v1/models",
        "/v1/models/",
        "/v1/scanner",
        "/.env",
    ],
)
def test_noise_is_not_recorded(path):
    app = FastAPI()
    for known in [
        "/healthz",
        "/ready",
        "/startup",
        "/rapid/v1/status",
        "/rapid/v1/log",
        "/rapid/v1/events",
        "/v1/models",
        "/v1/models/",
    ]:
        app.get(known)(lambda: {})
    sink = observer.Delivery()
    with TestClient(observer.CallObserver(app, sink)) as client:
        client.get(path)
    assert sink.queue.empty()
    assert sink.skipped == 1


def test_table_local_time_nulls_short_model_and_follow(history, monkeypatch, capsys):
    import time
    from datetime import datetime

    item = record() | {
        "ts": time.time(),
        "model": "organization/model",
        "ttft_ms": 2.9,
        "duration_ms": 7.8,
    }
    row = commands.table_row(item, "log")
    assert datetime.fromtimestamp(item["ts"]).strftime("%H:%M:%S") in row
    assert "organization" not in row and "model" in row
    assert "3→2" in row and row.endswith("      2        7")
    assert "1970-01-01" in commands.table_row(item | {"ts": 86400}, "log")
    assert commands.table_row({"ts": None}, "log").split() == ["-"] * 6 + [
        "-→-",
        "-",
        "-",
    ]
    assert "runner_ready" in commands.table_row(
        {"ts": time.time(), "event": "runner_ready", "port": 8000}, "events"
    )
    history.append([item | {"request_id": "a"}, item | {"request_id": "b"}], 1)
    monkeypatch.setattr(
        commands, "connection", lambda _: {"socket": "s", "secret": "x"}
    )
    monkeypatch.setattr(
        commands, "request", lambda *a, target: history.api("GET", target, {})[1]
    )
    monkeypatch.setattr(commands.time, "sleep", Mock(side_effect=KeyboardInterrupt))
    args = SimpleNamespace(command="log", port=8000, json=False, follow=True, limit=1)
    # Explicit port filters; matching records follow in persisted sequence order.
    args.port = None
    assert commands.history_command(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("TIME") and len(lines) == 3
    assert all("3→2" in line for line in lines[1:])


def test_history_rejects_symlinks_and_foreign_files(tmp_path, monkeypatch):
    import os

    from rapid_mlx.control.history import read_private

    outside = tmp_path / "outside"
    outside.write_text("secret")
    link = tmp_path / "link"
    link.symlink_to(outside)
    with pytest.raises(OSError):
        read_private(link)
    root = tmp_path / "history"
    root.mkdir(mode=0o755)
    with pytest.raises(OSError, match="0700"):
        History(root, readonly=True)
    root.chmod(0o700)
    (root / "history.lock").symlink_to(outside)
    with pytest.raises(OSError):
        History(root)
    (root / "history.lock").unlink()
    monkeypatch.setattr(os, "getuid", lambda: -1)
    with pytest.raises(OSError):
        read_private(outside)
    with pytest.raises(OSError):
        History(root, readonly=True)


def test_oversized_event_tail_and_restart_cursor(history):
    first = history.event("x" * 10000)
    assert not history.events
    resumed = History(history.root, cap=4096)
    assert resumed.event("runner_ready")["id"] == first["id"] + 1


def test_existing_store_rejects_foreign_lock(history, monkeypatch):
    import os

    monkeypatch.setattr(os, "getuid", lambda: -1)
    with pytest.raises(OSError, match="lock"), history.locked():
        pass


async def test_supervisor_open_uses_shared_user_history(tmp_path, monkeypatch):
    from rapid_mlx.control import supervisor as module

    runtime = tmp_path / "runtime"
    index = tmp_path / "user" / "control"
    monkeypatch.setattr(module, "runtime_root", lambda: runtime)
    monkeypatch.setattr(module, "discovery_root", lambda: index)

    async def listen(handler, *, path, limit):
        from pathlib import Path

        Path(path).touch()
        return SimpleNamespace(close=Mock(), wait_closed=AsyncMock())

    monkeypatch.setattr(module.asyncio, "start_unix_server", listen)
    stores = []
    for port in (8000, 8001):
        supervisor = Supervisor([], port=port, mode="lazy")
        await supervisor.open()
        try:
            assert supervisor.history is not None
            assert supervisor.history.root == index.parent / "history"
            supervisor.history.append([record(str(port))], 1)
            stores.append(supervisor.history)
        finally:
            supervisor.server.close()
            await supervisor.server.wait_closed()
    rows = (
        await asyncio.to_thread(stores[-1].api, "GET", "/rapid/v1/log?after=0", {})
    )[1]
    for store in stores:
        store.close()
    assert [r["port"] for r in rows] == [8000, 8001]
    assert len({r["instance"] for r in rows}) == 2
    assert not list(stores[-1].root.glob("i-*"))


@pytest.mark.parametrize("tail", ['{"request_id":', "{broken}\n", "null\n"])
def test_damaged_history_recovers_reads_appends_and_clear(history, tail):
    history.append([record("before")], 1)
    with history.logs.open("a") as stream:
        stream.write(tail)
    reopened = History(history.root)
    assert [r["request_id"] for r in reopened.records()] == ["before"]
    assert reopened.malformed > 0
    reopened.append([record("after")], 1)
    assert [r["request_id"] for r in reopened.records()] == ["before", "after"]
    with reopened.logs.open("a") as stream:
        stream.write('{"broken":')
    assert reopened.api("DELETE", "/rapid/v1/log", {}) == (200, {})
    assert list(reopened.records()) == []
    reopened.append([record("recovered")], 1)
    assert list(reopened.records())[0]["request_id"] == "recovered"


def test_sequence_replace_fsyncs_file_then_directory(history, monkeypatch):
    import os
    import stat

    from rapid_mlx.control.history import write_private

    synced = []
    real_sync = os.fsync

    def sync(fd):
        synced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        real_sync(fd)

    monkeypatch.setattr(os, "fsync", sync)
    write_private(history.cursor_path, "42")
    assert synced == [False, True]
    assert json.loads(history.cursor_path.read_text()) == 42


async def test_populated_locked_history_keeps_supervisor_responsive(tmp_path):
    import fcntl
    import signal
    import threading
    import time

    from rapid_mlx.control.history import HistoryWriter

    root = tmp_path / "history"
    store = History(root)
    row = json.dumps(record("fake") | {"seq": 1})
    for path in (store.old, store.logs):
        with path.open("w") as stream:
            stream.write((row + " " * (1023 - len(row)) + "\n") * (16 * 1024))
        path.chmod(0o600)
    writer = await asyncio.to_thread(HistoryWriter, root)
    supervisor = Supervisor([], port=8000, mode="lazy")
    supervisor.history = writer
    assert writer.status()["bytes"] >= 30 * 1024 * 1024
    offline = await asyncio.to_thread(History, root, readonly=True)
    acquired = threading.Event()
    release = threading.Event()

    def hold_lock():
        with (root / "history.lock").open("r+") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired.set()
            release.wait(5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert await asyncio.to_thread(acquired.wait, 1)
    try:
        # An independent descriptor holds the cooperative lock, like another process.
        writer.append([], 1, dropped=3)
        assert writer.queue.empty()
        writer.append([record("queued")], 1)
        supervisor.emit("runner_ready")
        query = asyncio.create_task(
            asyncio.to_thread(writer.api, "GET", "/rapid/v1/log", {})
        )
        offline_query = asyncio.create_task(
            asyncio.to_thread(offline.api, "GET", "/rapid/v1/log", {})
        )
        delays = []
        for _ in range(15):
            start = time.monotonic()
            supervisor.status()
            await asyncio.sleep(0.005)
            delays.append(time.monotonic() - start)
        assert max(delays) < 0.05
        supervisor.stop(signal.SIGTERM)
        assert supervisor.stopping.is_set()
        with pytest.raises(TimeoutError, match="history lock timed out"):
            await asyncio.wait_for(offline_query, 2.5)
        code, payload = await asyncio.wait_for(query, 2.5)
        assert code == 503 and "timed out" in payload["error"]
        start = time.monotonic()
        writer.close()
        assert time.monotonic() - start < 0.05
    finally:
        release.set()
        await asyncio.to_thread(holder.join, 1)
        writer.close()
        await asyncio.to_thread(writer.thread.join, 3)
    assert not writer.thread.is_alive()


async def test_history_writer_queue_drop_idle_and_shutdown(tmp_path, monkeypatch):
    import threading

    from rapid_mlx.control.history import HistoryWriter

    entered = threading.Event()
    release = threading.Event()
    original = History.append

    def append(self, *args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(History, "append", append)
    writer = await asyncio.to_thread(HistoryWriter, tmp_path / "history", queue_size=1)
    try:
        writer.append([], 1, 2)
        assert not entered.is_set()
        writer.append([record("one")], 1)
        assert await asyncio.to_thread(entered.wait, 1)
        writer.append([record("two")], 1)
        writer.append([record("three"), record("four")], 1)
        assert writer.status()["dropped"] == 4
        assert writer.queue.qsize() == 1
        assert writer.api("GET", "/rapid/v1/log", {})[0] == 503
        writer.close()
        assert writer.status()["dropped"] == 6
        writer.event("closed")
    finally:
        release.set()
        writer.close()
        await asyncio.to_thread(writer.thread.join, 3)
    assert not writer.thread.is_alive()


def test_history_writes_never_parse_retained_records(history, monkeypatch):
    history.append([record("existing")], 1)
    monkeypatch.setattr(history, "read_rows", Mock(side_effect=AssertionError("scan")))
    history.append([], 1)
    history.append([record("new")], 1)
    history.event("runner_ready")
    assert history.status()["records"] == 2
    history.api("PUT", "/rapid/v1/log/settings", {"max_bytes": 4096})
    assert history.status()["records"] == 2


def test_history_repair_long_incomplete_tail_and_bounded_dedup(history):
    history.logs.write_bytes(b"{" + b"x" * 20000)
    history.append([record("repaired")], 1)
    assert list(history.records())[0]["request_id"] == "repaired"
    assert history.malformed == 1
    for i in range(9000):
        history.remember(str(i))
    assert len(history.seen) == 8192
    assert "0" not in history.seen and "8999" in history.seen


async def test_writer_close_handles_concurrent_consumer(tmp_path, monkeypatch):
    import queue

    from rapid_mlx.control.history import HistoryWriter

    writer = await asyncio.to_thread(HistoryWriter, tmp_path / "history")
    put = writer.queue.put_nowait
    calls = 0

    def race(job):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise queue.Full  # The consumer empties it before get_nowait.
        return put(job)

    monkeypatch.setattr(writer.queue, "put_nowait", race)
    writer.close()
    writer.close()
    writer.append([record()], 1)
    await asyncio.to_thread(writer.thread.join, 1)
    assert not writer.thread.is_alive()
    assert writer.status()["dropped"] == 1


def test_history_rejects_foreign_data_descriptors(history, monkeypatch):
    import os

    history.append([record()], 1)
    original = os.fstat

    def foreign(fd):
        result = original(fd)
        import stat

        # Preserve lock-file checks; spoof only a nonempty history data file.
        if stat.S_ISREG(result.st_mode) and result.st_size > 100:
            values = list(result)
            values[4] = os.getuid() + 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(os, "fstat", foreign)
    with pytest.raises(OSError, match="owned regular file"):
        History(history.root)
    with pytest.raises(OSError, match="owned regular file"):
        list(history.read_rows(history.logs))


@pytest.mark.parametrize("shutdown", [False, True])
@pytest.mark.parametrize("event_first", [False, True])
def test_writer_retries_contended_appends(tmp_path, monkeypatch, shutdown, event_first):
    import fcntl
    import threading
    import time

    from rapid_mlx.control.history import HistoryWriter

    original = History.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.lock_timeout = 0.05

    monkeypatch.setattr(History, "__init__", initialize)
    batches = []
    append = History.append

    def track(self, records, *args, **kwargs):
        batches.append(len(records))
        return append(self, records, *args, **kwargs)

    monkeypatch.setattr(History, "append", track)
    root = tmp_path / "history"
    writer = HistoryWriter(root)
    acquired = threading.Event()
    release = threading.Event()

    def hold():
        with (root / "history.lock").open("r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            acquired.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    assert acquired.wait(1)
    try:
        futures = [writer.submit("event", "runner_ready")] if event_first else []
        futures.append(writer.submit("append", [record("first")], 1))
        time.sleep(0.2)  # Longer than the store's lock timeout.
        futures.extend(writer.submit("append", [record(str(i))], 1) for i in range(4))
        if not event_first:
            futures.append(writer.submit("event", "runner_ready"))
        if shutdown:
            start = time.monotonic()
            writer.close()
            writer.thread.join(2)
            assert not writer.thread.is_alive()
            assert time.monotonic() - start < 2
            assert writer.status()["dropped"] == 6
            assert all(f.exception(timeout=0) for f in futures)
        else:
            assert writer.status()["dropped"] == 0
            release.set()
            for future in futures:
                future.result(timeout=3)
            assert writer.status()["dropped"] == 0
            assert batches == [5]
            store = History(root)
            assert [r["request_id"] for r in store.records()] == [
                "first",
                "0",
                "1",
                "2",
                "3",
            ]
            assert [e["event"] for e in store.events] == ["runner_ready"]
    finally:
        release.set()
        holder.join(1)
        writer.close()
        writer.thread.join(3)
