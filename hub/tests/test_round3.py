"""Regressions for the five round-three review scenarios."""

import asyncio
import json
import threading

import httpx
import pytest
from rapid_hub.app import create_app
from rapid_hub.manifest import load
from rapid_hub.storage import Journal, private_write
from rapid_hub.supervisor import Supervisor
from test_hub import MANIFEST, client, make_hub


@pytest.mark.parametrize("phase", ["headers", "json"])
def test_admitted_disconnect_cancels_upstream(tmp_path, phase):
    async def run():
        entered, cancelled, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                yield b"{}"

            async def aclose(self):
                closed.set()

        async def upstream(request):
            if phase == "headers":
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return httpx.Response(
                200, stream=Stream(), headers={"content-type": "application/json"}
            )

        hub = make_hub(tmp_path, upstream)
        hub.admission = "hub"
        messages = asyncio.Queue()
        await messages.put({"type": "http.request", "body": b"{}", "more_body": False})
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

        task = asyncio.create_task(
            create_app(hub, supervise=False)(scope, messages.get, send)
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert hub.gates["chat"].locked()
            await messages.put({"type": "http.disconnect"})
            await asyncio.wait_for(cancelled.wait(), 0.5)
            await asyncio.gather(task, return_exceptions=True)
            assert not hub.gates["chat"].locked()
            if phase == "json":
                assert closed.is_set()
            assert hub.journal.read()[0]["cancelled"] is True
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await hub.client.aclose()

    asyncio.run(run())


def test_overlapping_restart_and_stop_track_every_generation(tmp_path):
    async def run():
        hub = make_hub(tmp_path, lambda r: httpx.Response(200))
        supervisor = Supervisor(hub, {"one": {"name": "One"}})
        live, tasks, records = set(), [], []
        entered, release = asyncio.Event(), asyncio.Event()

        async def runner(ident):
            task = asyncio.current_task()
            tasks.append(task)
            records.append(supervisor.items[ident])
            live.add(task)
            try:
                await asyncio.Event().wait()
            finally:
                entered.set()
                try:
                    await release.wait()
                finally:
                    live.discard(task)

        supervisor.run = runner
        supervisor.tasks["one"] = asyncio.create_task(runner("one"))
        await asyncio.sleep(0)
        first = asyncio.create_task(supervisor.restart("one"))
        await entered.wait()
        second = asyncio.create_task(supervisor.restart("one"))
        await asyncio.sleep(0)
        release.set()
        try:
            await asyncio.gather(first, second)
            await asyncio.sleep(0)
            assert len(live) == 1
            assert len({id(r) for r in records}) == len(records)
            await asyncio.gather(supervisor.restart("one"), supervisor.stop())
            await asyncio.sleep(0)
            assert not live
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await hub.client.aclose()

    asyncio.run(run())


def test_admin_manifest_reserved_and_identity_cannot_grant_admin(tmp_path):
    path = tmp_path / "rapid-plugin.toml"
    path.write_text(MANIFEST.replace('id = "research-loop"', 'id = "admin"'))
    with pytest.raises(ValueError, match="reserved"):
        load(path)


def test_admin_plugin_token_has_no_administrative_authority(tmp_path):
    async def run():
        hub = make_hub(tmp_path, lambda r: httpx.Response(200, json={"data": []}))
        hub.tokens["plugin-admin"] = "admin"
        async with client(hub, "plugin-admin") as c:
            assert (await c.get("/hub/v1/status")).status_code == 403
            assert (await c.get("/hub/v1/journal?plugin=two")).status_code == 403
            assert (await c.post("/hub/v1/jobs/missing/cancel")).status_code == 403
        async with client(hub, hub.admin) as c:
            assert (await c.get("/hub/v1/status")).status_code == 200
        await hub.client.aclose()

    asyncio.run(run())


def test_journal_append_and_recent_read_do_not_scan_history(tmp_path, monkeypatch):
    journal = Journal(tmp_path / "journal")
    journal.configure("one", "grokking", True, 24)
    # Accumulate roughly 64 MiB of Research Loop-style payload history.
    for i in range(520):
        journal.append(
            dict(
                plugin="one",
                project="grokking",
                request_id=str(i),
                request={"content": "x" * 65000},
                response={"content": "y" * 65000},
            )
        )
    parsed = []
    original = json.loads

    def loads(value, *args, **kwargs):
        parsed.append(1)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", loads)
    journal.append(dict(plugin="one", request_id="last"))
    rows = journal.read(limit=5)
    assert rows[0]["request_id"] == "last"
    assert len(parsed) <= 10


def test_gateway_journal_persistence_runs_off_event_loop(tmp_path, monkeypatch):
    async def run():
        hub = make_hub(tmp_path, lambda r: httpx.Response(200, json={}))
        hub.admission = "hub"
        threads = []
        original = private_write

        def write(*args, **kwargs):
            threads.append(threading.get_ident())
            return original(*args, **kwargs)

        monkeypatch.setattr("rapid_hub.storage.private_write", write)
        async with client(hub) as c:
            assert (
                await c.post("/hub/v1/engines/chat/v1/chat/completions", json={})
            ).status_code == 200
        assert threads and threading.get_ident() not in threads
        await hub.client.aclose()

    asyncio.run(run())
