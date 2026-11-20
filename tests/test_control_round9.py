# SPDX-License-Identifier: Apache-2.0
"""Mutation deadlines, safe rendering, batch acknowledgements and event scope."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.control import RUNNER_ENV, SECRET_ENV, SOCKET_ENV, observer, protocol
from rapid_mlx.control.commands import table_row
from rapid_mlx.control.history import APIJob, History, HistoryWriter
from rapid_mlx.control.supervisor import Supervisor
from rapid_mlx.control.text import client_identity, display, unsafe
from rapid_mlx.routes import control_status as routes


async def call(supervisor, method, target, data=None, headers=None):
    reader = asyncio.StreamReader()
    body = json.dumps(data).encode() if data is not None else b""
    fields = {
        "X-Rapid-Control-Secret": supervisor.secret,
        "Content-Length": str(len(body)),
        **(headers or {}),
    }
    reader.feed_data(
        (
            f"{method} {target} HTTP/1.1\r\n"
            + "".join(f"{k}: {v}\r\n" for k, v in fields.items())
            + "\r\n"
        ).encode()
        + body
    )
    reader.feed_eof()
    writer = SimpleNamespace(
        write=Mock(), drain=AsyncMock(), close=Mock(), wait_closed=AsyncMock()
    )
    await supervisor.handle(reader, writer)
    head, body = writer.write.call_args.args[0].split(b"\r\n\r\n", 1)
    return head, json.loads(body)


@pytest.mark.parametrize("method", ["PUT", "DELETE"])
def test_route_passes_deadline_and_does_not_timeout_claimed_mutation(
    monkeypatch, method
):
    monkeypatch.setenv(RUNNER_ENV, "1")
    monkeypatch.setenv(SOCKET_ENV, "/tmp/socket")
    monkeypatch.setenv(SECRET_ENV, "private")
    now = [100.0]
    monkeypatch.setattr(routes, "time", SimpleNamespace(monotonic=lambda: now[0]))
    # A fake body timeout verifies forwarding runs outside that timeout.
    active = []

    class BodyTimeout:
        async def __aenter__(self):
            active.append(True)

        async def __aexit__(self, *args):
            active.pop()

    monkeypatch.setattr(routes.asyncio, "timeout", lambda _: BodyTimeout())

    def request(*args, **kwargs):
        assert not active
        assert kwargs["deadline"] == 100 + protocol.JOB_TIMEOUT
        now[0] += protocol.TIMEOUT * 10
        return {"done": True}

    monkeypatch.setattr(routes, "control_request", request)
    app = FastAPI()
    routes.mount_status(app)
    app.dependency_overrides[routes.verify_api_key] = lambda: None
    with TestClient(app, client=("127.0.0.1", 1)) as client:
        assert client.request(
            method, "/rapid/v1/log/settings" if method == "PUT" else "/rapid/v1/log"
        ).json() == {"done": True}


def test_socket_mutation_deadline_and_claimed_response_wait(monkeypatch):
    conn = Mock()
    conn.getresponse.return_value.status = 200
    conn.getresponse.return_value.read.return_value = b"{}"
    monkeypatch.setattr(protocol, "UnixConnection", lambda _: conn)
    monkeypatch.setattr(protocol, "time", SimpleNamespace(monotonic=lambda: 40))
    protocol.request("socket", "secret", "DELETE", target="/rapid/v1/log")
    assert (
        float(conn.request.call_args.kwargs["headers"][protocol.DEADLINE_HEADER])
        == 40 + protocol.JOB_TIMEOUT
    )
    conn.sock.settimeout.assert_called_with(None)
    protocol.request(
        "socket", "secret", "PUT", {}, target="/rapid/v1/log/settings", deadline=41
    )
    assert conn.request.call_args.kwargs["headers"][protocol.DEADLINE_HEADER] == "41"


async def test_supervisor_passes_expired_outer_deadline_to_writer(
    tmp_path, monkeypatch
):
    supervisor = Supervisor([], port=8000, mode="lazy")
    writer = HistoryWriter(tmp_path / "history")
    supervisor.history = writer
    writer.append([{"request_id": "retained"}], 1)
    assert writer.api("GET", "/rapid/v1/log", {})[1]
    monkeypatch.setattr(protocol, "time", SimpleNamespace(monotonic=lambda: 0))
    # Absolute monotonic deadline 0 is already expired on the real worker clock.
    head, _ = await call(
        supervisor, "DELETE", "/rapid/v1/log", headers={protocol.DEADLINE_HEADER: "0"}
    )
    assert b"503" in head
    assert writer.api("GET", "/rapid/v1/log", {})[1][0]["request_id"] == "retained"
    writer.close()
    writer.thread.join(3)


def test_writer_deadline_claim_fake_clock(monkeypatch):
    from rapid_mlx.control import history

    now = [10.0]
    monkeypatch.setattr(history, "time", SimpleNamespace(monotonic=lambda: now[0]))
    job = APIJob()
    assert job.deadline == 10 + protocol.JOB_TIMEOUT
    now[0] = job.deadline
    assert not job.claim()
    assert job.cancelled()
    claimed = APIJob()
    assert claimed.claim()
    now[0] += 20
    assert not claimed.cancel()
    claimed.set_result((200, {}))
    assert claimed.result() == (200, {})


@pytest.mark.parametrize("command", ["log", "events"])
def test_every_display_string_escapes_terminal_controls(command):
    attack = "\x1b[31m\x1b]52;c;clipboard\x07\r\n\t\x7f\x85\u202e\u2066"
    row = table_row(
        {
            "client": attack,
            "model": attack,
            "route": attack,
            "http_status": attack,
            "outcome": attack,
            "prompt_tokens": attack,
            "completion_tokens": attack,
            "event": attack,
            attack: attack,
        },
        command,
    )
    assert not any(unsafe(c) for c in row)
    assert "\\x1b" in row
    assert "\\u202e" in display(attack)
    assert "\\u2066" in display(attack)
    assert display("plain") == "plain"
    assert len(display("\x1b" * 1000)) <= 256
    assert len(display("a" * 1000)) <= 256
    assert client_identity(attack + "a" * 1000) == "[31m]52;c;clipboard" + "a" * (
        256 - len("[31m]52;c;clipboard")
    )


async def test_retry_after_lost_ack_is_idempotent_across_rotation_and_clear(
    tmp_path, monkeypatch
):
    supervisor = Supervisor([], port=8000, mode="lazy")
    store = History(tmp_path / "history", cap=4096)
    supervisor.history = store
    supervisor.child = SimpleNamespace(pid=42)
    supervisor.spawn_ready.set()
    batch = {
        "pid": 42,
        "runner_instance": "runner-uuid",
        "batch_seq": 1,
        "records": [{"request_id": "once"}],
        "events": [{"event": "model_loaded", "model": "tiny"}],
        "dropped": 0,
    }
    first = await call(supervisor, "POST", "/records", batch)
    # The first reply is lost. Retrying must acknowledge without inserting again.
    assert await call(supervisor, "POST", "/records", batch) == first
    assert first[1] == {"batch_seq": 1}
    assert [r["request_id"] for r in store.records()] == ["once"]
    assert len(store.events) == 1
    store.api("DELETE", "/rapid/v1/log", {})
    assert (await call(supervisor, "POST", "/records", batch))[1] == {"batch_seq": 1}
    assert list(store.records()) == []
    batch["batch_seq"] = 2
    assert b"200" in (await call(supervisor, "POST", "/records", batch))[0]
    assert len(list(store.records())) == 1


def test_delivery_retries_same_batch_id(monkeypatch):
    monkeypatch.setenv(SOCKET_ENV, "socket")
    monkeypatch.setenv(SECRET_ENV, "secret")
    sink = observer.Delivery()
    payloads = []

    def request(*args, **kwargs):
        payloads.append(dict(args[3]))
        if len(payloads) == 1:
            raise TimeoutError("ack lost")
        return {"batch_seq": args[3]["batch_seq"]}

    monkeypatch.setattr(observer, "request", request)
    sink.flush([{"request_id": "once"}])
    assert len(payloads) == 2 and payloads[0] == payloads[1]
    assert sink.dropped == 0
    sink.flush([])
    assert payloads[-1]["batch_seq"] == 2
    assert payloads[-1]["runner_instance"] == sink.instance


async def test_events_filters_online_offline_and_sse(tmp_path, monkeypatch):
    root = tmp_path / "history"
    a = History(root, instance="a", port=8000)
    b = History(root, instance="b", port=8001)
    a.event("runner_ready")
    b.event("runner_ready")
    a.event("model_loaded")
    target = "/rapid/v1/events?port=8000&instance=a&since=1"
    offline = History(root, readonly=True)
    assert [e["id"] for e in offline.api("GET", target, {})[1]] == [3]
    assert offline.api("GET", "/rapid/v1/events?instance=missing", {})[1] == []
    supervisor = Supervisor([], port=8000, mode="lazy")
    supervisor.history = a
    assert [e["id"] for e in (await call(supervisor, "GET", target))[1]] == [3]
    seen = []

    def request(*args, target):
        seen.append(target)
        return offline.api("GET", target, {})[1]

    monkeypatch.setattr(routes, "control_request", request)
    stream = routes.event_stream(
        "socket", "secret", 1, {"port": "8000", "instance": "a"}
    )
    assert "id: 3" in await anext(stream)  # Filtered gaps must not terminate SSE.
    assert "port=8000" in seen[0] and "instance=a" in seen[0]
    await stream.aclose()


def test_delivery_rejects_unacknowledged_batch(monkeypatch):
    monkeypatch.setenv(SOCKET_ENV, "socket")
    monkeypatch.setenv(SECRET_ENV, "secret")
    request = Mock(return_value={})
    monkeypatch.setattr(observer, "request", request)
    sink = observer.Delivery()
    sink.flush([{"request_id": "r"}])
    assert request.call_count == 2
    assert sink.dropped == 1


async def test_supervisor_deadline_clamps_to_local_budget(monkeypatch):
    from rapid_mlx.control import supervisor as module

    supervisor = Supervisor([], port=8000, mode="lazy")
    api = Mock(return_value=(200, {}))
    supervisor.history = SimpleNamespace(api=api)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 100.0))
    for outer, expected in [(100.1, 100.1), (999, 100 + protocol.JOB_TIMEOUT)]:
        head, _ = await call(
            supervisor,
            "DELETE",
            "/rapid/v1/log",
            headers={protocol.DEADLINE_HEADER: str(outer)},
        )
        assert b"200" in head
        assert api.call_args.kwargs["deadline"] == expected


async def test_supervisor_rejects_invalid_batch_sequence():
    supervisor = Supervisor([], port=8000, mode="lazy")
    supervisor.child = SimpleNamespace(pid=42)
    supervisor.history = Mock()
    body = json.dumps({"pid": 42, "runner_instance": "uuid", "batch_seq": -1}).encode()
    reader = asyncio.StreamReader()
    reader.feed_data(
        f"POST /records HTTP/1.1\r\nX-Rapid-Control-Secret: {supervisor.secret}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    reader.feed_eof()
    writer = SimpleNamespace(
        write=Mock(), drain=AsyncMock(), close=Mock(), wait_closed=AsyncMock()
    )
    await supervisor.handle(reader, writer)
    writer.write.assert_not_called()
    supervisor.history.append.assert_not_called()


async def test_client_controls_are_removed_at_capture():
    sink = observer.Delivery()

    async def app(scope, receive, send):
        scope["route"] = SimpleNamespace(path="/v1/chat/completions")
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"ok"})

    await observer.CallObserver(app, sink)(
        {
            "type": "http",
            "client": ("127.0.0.1", 1),
            "headers": [
                (b"x-rapid-client", b"\x1b]52;c;test\x07\x7f" + "\u202e".encode())
            ],
        },
        AsyncMock(),
        AsyncMock(),
    )
    identity = sink.queue.get_nowait()["client"]
    assert identity == "127.0.0.1 ]52;c;test"
