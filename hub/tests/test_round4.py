"""Regressions for persistence disconnects, cleanup shutdown, and overload."""

import asyncio
import threading

import httpx
import pytest
from rapid_hub.app import create_app
from rapid_hub.storage import Journal
from test_hub import client, make_hub


def request_task(app):
    messages = asyncio.Queue()
    messages.put_nowait({"type": "http.request", "body": b"{}", "more_body": False})
    scope = dict(
        type="http",
        asgi={"version": "3.0"},
        http_version="1.1",
        method="POST",
        scheme="http",
        path="/hub/v1/engines/chat/v1/chat/completions",
        raw_path=b"/",
        query_string=b"",
        server=("127.0.0.1", 1234),
        client=("127.0.0.1", 1235),
        headers=[
            (b"host", b"hub.localhost"),
            (b"authorization", b"Bearer one"),
            (b"x-rapid-priority", b"background"),
        ],
    )

    async def send(message):
        pass

    return asyncio.create_task(app(scope, messages.get, send)), messages


async def until(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0.005)


def test_disconnect_during_initial_journal_wait_never_sends_upstream(
    tmp_path, monkeypatch
):
    async def run():
        sent = []
        hub = make_hub(
            tmp_path, lambda r: sent.append(r) or httpx.Response(200, json={})
        )
        hub.admission = "hub"
        entered, release = threading.Event(), threading.Event()
        original = hub.journal._append

        def append(entry):
            entered.set()
            assert release.wait(3)
            return original(entry)

        monkeypatch.setattr(hub.journal, "_append", append)
        task, messages = request_task(create_app(hub, supervise=False))
        try:
            await until(entered.is_set)
            assert hub.gates["chat"].locked()
            messages.put_nowait({"type": "http.disconnect"})
            await until(lambda: not hub.gates["chat"].locked())
            assert not sent
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            rows = await hub.journal.call("read")
            await hub.client.aclose()
            await hub.journal.close()
        assert not sent
        assert rows[0]["cancelled"] is True
        assert rows[0]["status"] == 499

    asyncio.run(run())


def test_shutdown_waits_for_detached_cleanup_terminal_record(tmp_path):
    async def run():
        closing, release, streaming = asyncio.Event(), asyncio.Event(), asyncio.Event()
        closes = []

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                streaming.set()
                yield b"data: {}\n\n"
                await asyncio.Event().wait()

            async def aclose(self):
                closes.append(1)
                closing.set()
                await release.wait()

        hub = make_hub(
            tmp_path,
            lambda r: httpx.Response(
                200, stream=Stream(), headers={"content-type": "text/event-stream"}
            ),
        )
        hub.admission = "hub"

        async def poll():
            await asyncio.Event().wait()

        hub.poll = poll
        app = create_app(hub, supervise=False)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        task, _ = request_task(app)
        shutdown = None
        try:
            await asyncio.wait_for(streaming.wait(), 1)
            task.cancel()
            await asyncio.wait_for(closing.wait(), 1)
            await asyncio.sleep(0.02)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await asyncio.sleep(0.05)
            assert not shutdown.done(), (
                "shutdown closed persistence before cleanup finished"
            )
            assert not hub.client.is_closed
        finally:
            release.set()
            if shutdown is None:
                shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await shutdown
            await asyncio.sleep(0.02)
        assert closes == [1]
        reopened = Journal(tmp_path / "journal")
        try:
            row = reopened.read()[0]
            assert row["status"] == 200
            assert row["cancelled"] is True
        finally:
            await reopened.close()
        assert not hub.gates["chat"].locked()

    asyncio.run(run())


@pytest.mark.parametrize("limit", ["ops", "bytes"])
def test_journal_backlog_bounds_drop_payload_then_record(tmp_path, monkeypatch, limit):
    async def run():
        journal = Journal(tmp_path / "journal")
        journal.configure("one", "p", True, 24)
        # Set the same configurable limits after construction to exercise the
        # unmodified implementation as a failing negative control too.
        journal.queue_max_ops = 3 if limit == "ops" else 100
        journal.queue_max_bytes = 600
        entered, release = threading.Event(), threading.Event()
        original = journal._append
        seen = []

        def append(entry):
            seen.append(entry)
            entered.set()
            assert release.wait(3)
            return original(entry)

        monkeypatch.setattr(journal, "_append", append)
        tasks = [
            asyncio.create_task(
                journal.call("append", dict(plugin="one", request_id="first"))
            )
        ]
        try:
            await until(entered.is_set)
            for i in range(20):
                tasks.append(
                    asyncio.create_task(
                        journal.call(
                            "append",
                            dict(
                                plugin="one",
                                project="p",
                                request_id=str(i),
                                status=200,
                                request={"text": "x" * 2000},
                                response="y" * 2000,
                            ),
                        )
                    )
                )
            await asyncio.sleep(0.05)
            assert journal.dropped_records == 20
            assert journal.pending_ops <= journal.queue_max_ops
            assert journal.pending_bytes <= journal.queue_max_bytes
            assert sum(task.done() for task in tasks) > 0
        finally:
            release.set()
            await asyncio.gather(*tasks)
        rows = await journal.call("read", limit=50)
        assert 1 < len(rows) <= (3 if limit == "ops" else 20)
        assert all("request" not in row and "response" not in row for row in rows)
        assert all("request" not in entry and "response" not in entry for entry in seen)
        await journal.close()

    asyncio.run(run())


@pytest.mark.parametrize("priority", ["interactive", "background", "rejected"])
def test_overloaded_journal_does_not_delay_or_fail_gateway(
    tmp_path, monkeypatch, priority
):
    async def run():
        hub = make_hub(tmp_path, lambda r: httpx.Response(200, json={"ok": True}))
        hub.admission = "hub"
        hub.journal.queue_max_ops = 1
        if priority == "rejected":
            hub.queue_max_length = 0
            await hub.gates["chat"].acquire()
        entered, release = threading.Event(), threading.Event()
        original = hub.journal._append

        def append(entry):
            entered.set()
            assert release.wait(3)
            return original(entry)

        monkeypatch.setattr(hub.journal, "_append", append)
        blocked = asyncio.create_task(hub.journal.call("append", dict(plugin="one")))
        try:
            await until(entered.is_set)
            async with client(hub) as c:
                response = await asyncio.wait_for(
                    c.post(
                        "/hub/v1/engines/chat/v1/chat/completions",
                        json={},
                        headers={
                            "X-Rapid-Priority": "interactive"
                            if priority == "interactive"
                            else "background"
                        },
                    ),
                    0.5,
                )
                assert response.status_code == (429 if priority == "rejected" else 200)
            assert hub.journal.dropped_records == (1 if priority == "rejected" else 2)
        finally:
            release.set()
            await blocked
            await hub.client.aclose()
            await hub.journal.close()

    asyncio.run(run())


def test_cancelled_journal_waits_remain_bounded_and_fifo(tmp_path, monkeypatch):
    async def run():
        journal = Journal(tmp_path / "journal", queue_max_ops=2, queue_max_bytes=500)
        entered, release = threading.Event(), threading.Event()
        original = journal._append
        written = []

        def append(entry):
            entered.set()
            assert release.wait(3)
            written.append(entry["request_id"])
            return original(entry)

        monkeypatch.setattr(journal, "_append", append)
        first = asyncio.create_task(
            journal.call("append", dict(plugin="one", request_id="first"))
        )
        second = None
        try:
            await until(entered.is_set)
            second = asyncio.create_task(
                journal.call("append", dict(plugin="one", request_id="second"))
            )
            await until(lambda: journal.pending_ops == 2)
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)
            for _ in range(20):
                await journal.call("append", dict(plugin="one", request_id="dropped"))
            assert journal.pending_ops == 2
            assert journal.dropped_records == 20
        finally:
            release.set()
            await asyncio.gather(
                first, *([second] if second else []), return_exceptions=True
            )
            await journal.close()
        assert written == ["first", "second"]
        assert journal.pending_ops == journal.pending_bytes == 0

    asyncio.run(run())


def test_byte_overload_keeps_metadata_without_delaying_gateway(tmp_path, monkeypatch):
    async def run():
        hub = make_hub(tmp_path, lambda r: httpx.Response(200, json={}))
        hub.admission = "hub"
        hub.journal.configure("one", "p", True, 24)
        hub.journal.queue_max_bytes = 2000
        entered, release = threading.Event(), threading.Event()
        original = hub.journal._append

        def append(entry):
            entered.set()
            assert release.wait(3)
            return original(entry)

        monkeypatch.setattr(hub.journal, "_append", append)
        blocked = asyncio.create_task(hub.journal.call("append", dict(plugin="one")))
        try:
            await until(entered.is_set)
            async with client(hub) as c:
                response = await asyncio.wait_for(
                    c.post(
                        "/hub/v1/engines/chat/v1/chat/completions",
                        json={"text": "x" * 4000},
                        headers={"X-Rapid-Project": "p"},
                    ),
                    0.5,
                )
                assert response.status_code == 200
            assert hub.journal.dropped_records == 2
            assert hub.journal.pending_ops == 3
            assert hub.journal.pending_bytes <= 2000
        finally:
            release.set()
            await blocked
            rows = await hub.journal.call("read", "one", "p")
            await hub.client.aclose()
            await hub.journal.close()
        assert len(rows) == 1 and rows[0]["status"] == 200
        assert "request" not in rows[0] and "response" not in rows[0]

    asyncio.run(run())
