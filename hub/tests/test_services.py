"""Round-2 contracts: metadata, origin boundaries, routing and failure recovery."""

import asyncio
import hashlib
import json
import time
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import uvicorn
from rapid_hub.app import Hub, create_app
from rapid_hub.storage import Journal, private_write
from test_hub import client, make_hub


@asynccontextmanager
async def live(hub):
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(hub, supervise=False),
            log_level="critical",
            access_log=False,
            timeout_graceful_shutdown=2,
            timeout_keep_alive=30,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(500):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        assert server.started
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await task
        sock.close()


def reported(request):
    if request.url.path == "/health":
        return httpx.Response(200, json={"ready": True})
    if request.url.path.endswith("/models"):
        return httpx.Response(
            200,
            json={
                "data": [{"id": "m", "capabilities": ["chat", "embeddings", "rerank"]}]
            },
        )
    return httpx.Response(200, json={"ok": True, "api_key": "response-secret"})


def test_journal_policy_redaction_retention_delete_and_failure(tmp_path, monkeypatch):
    async def run():
        hub = make_hub(tmp_path, reported)
        async with client(hub) as c:
            path = "/hub/v1/chat/completions"
            body = {
                "model": "m",
                "messages": [{"content": "private"}],
                "nested": {"access_token": "secret", "password": "secret"},
            }
            assert (
                await c.post(path, json=body, headers={"X-Rapid-Project": "plain"})
            ).status_code == 200
            row = hub.journal.read()[0]
            assert set(row) == {
                "ts",
                "request_id",
                "client",
                "plugin",
                "project",
                "job_id",
                "engine",
                "model",
                "path",
                "priority",
                "status",
                "latency_ms",
                "queue_wait_ms",
                "usage",
                "request_bytes",
                "response_bytes",
                "request_sha256",
                "cancelled",
            }
            assert "private" not in json.dumps(row)
            assert len(row["request_sha256"]) == 64
            assert (
                await c.put(
                    "/hub/v1/projects/grokking/capture",
                    json={"payload": True, "plugin": "two"},
                )
            ).status_code == 403
            assert (
                await c.put("/hub/v1/projects/grokking/capture", json={"payload": True})
            ).status_code == 200
            await c.post(path, json=body, headers={"X-Rapid-Project": "grokking"})
            row = hub.journal.read()[0]
            assert json.loads(row["request"])["nested"] == {
                "access_token": "[REDACTED]",
                "password": "[REDACTED]",
            }
            assert "response-secret" not in row["response"]
            await c.post(
                path,
                json={"model": "m", "text": "a" * 100000},
                headers={"X-Rapid-Project": "grokking"},
            )
            assert len(hub.journal.read()[0]["request"].encode()) <= 65536
            paths = list((tmp_path / "journal").glob("*/*.jsonl"))
            for file in paths:
                rows = [json.loads(line) for line in file.read_text().splitlines()]
                for item in rows:
                    if "payload_expires_at" in item:
                        item["payload_expires_at"] = time.time() - 1
                private_write(file, "".join(json.dumps(r) + "\n" for r in rows))
            assert all("request" not in r for r in hub.journal.read())
            await c.delete("/hub/v1/journal?project=grokking")
            assert len(hub.journal.read()) == 1

            def fail(*args, **kwargs):
                raise OSError("disk full")

            monkeypatch.setattr("rapid_hub.storage.private_write", fail)
            assert (await c.post(path, json=body)).status_code == 200
            assert hub.journal.dropped_records > 0
            assert (
                await c.get(
                    "/hub/v1/status", headers={"Authorization": "Bearer " + hub.admin}
                )
            ).json()["dropped_records"] > 0
        await hub.client.aclose()

    asyncio.run(run())


def test_journal_disk_cap_and_old_metadata(tmp_path):
    journal = Journal(tmp_path / "journal", max_mb=0.001)
    for i in range(20):
        journal.append(
            dict(plugin="one", project="p", request_id=str(i), text="x" * 300)
        )
        assert (
            sum(p.stat().st_size for p in journal.directory.glob("*/*.jsonl")) <= 1048
        )
    path = journal.directory / "one/2000-01-01.jsonl"
    private_write(
        path, json.dumps(dict(ts="2000-01-01T00:00:00+00:00", plugin="one")) + "\n"
    )
    Journal(journal.directory)
    assert not path.exists()


def test_hostname_cookie_isolation_and_single_use_expiring_codes(tmp_path):
    async def run():
        hub = make_hub(tmp_path, reported)
        hub.supervisor.items["one"] = dict(
            id="one", state="ready", port=1234, token="one"
        )
        async with client(hub, "") as c:
            login = await c.get("/login?nonce=" + hub.login)
            cookie = login.headers["set-cookie"]
            assert (
                "Domain=" not in cookie
                and "HttpOnly" in cookie
                and "SameSite=strict" in cookie
            )
            assert (await c.get("/")).status_code == 200
            cross = c.build_request("GET", "http://one.localhost:1234/")
            assert "cookie" not in cross.headers
            assert (
                await c.get(
                    "http://127.0.0.1:4800/", headers={"Cookie": cookie.split(";")[0]}
                )
            ).status_code == 403
            assert (
                await c.get(
                    "http://127.0.0.1:4800/hub/v1/status",
                    headers={"Cookie": cookie.split(";")[0]},
                )
            ).status_code == 401
            response = await c.get("/hub/v1/plugins/one/open")
            assert response.status_code == 303
            target = urlsplit(response.headers["location"])
            assert target.hostname == "one.localhost"
            code = parse_qs(target.query)["code"][0]
            assert (
                await c.post(
                    "/hub/v1/ui-codes/redeem",
                    json={"code": code},
                    headers={"Authorization": "Bearer two"},
                )
            ).status_code == 401
            for expected in (200, 401):
                assert (
                    await c.post(
                        "/hub/v1/ui-codes/redeem",
                        json={"code": code},
                        headers={"Authorization": "Bearer one"},
                    )
                ).status_code == expected
            response = await c.get("/hub/v1/plugins/one/open")
            code = parse_qs(urlsplit(response.headers["location"]).query)["code"][0]
            hub.ui_codes[code] = ("one", time.monotonic() - 1, "one")
            assert (
                await c.post(
                    "/hub/v1/ui-codes/redeem",
                    json={"code": code},
                    headers={"Authorization": "Bearer one"},
                )
            ).status_code == 401
        await hub.client.aclose()

    asyncio.run(run())


def test_model_routing_ambiguity_identity_and_status(tmp_path):
    async def run():
        seen = []

        def upstream(request):
            if request.method == "GET":
                return reported(request)
            seen.append(request)
            return httpx.Response(
                503, content=b"busy exactly", headers={"Retry-After": "17"}
            )

        hub = Hub(
            {"a": "http://a/v1", "b": "http://b/v1"},
            tmp_path,
            tmp_path / "empty",
            transport=httpx.MockTransport(upstream),
            discovery_file="discovery.json",
        )
        hub.tokens["one"] = "one"
        async with client(hub) as c:
            response = await c.post("/hub/v1/chat/completions", json={"model": "m"})
            assert response.status_code == 409
            assert response.json()["detail"]["candidates"] == ["a/m", "b/m"]
            for endpoint in ("chat/completions", "embeddings", "rerank"):
                response = await c.post(
                    "/hub/v1/" + endpoint,
                    json={"model": "b/m"},
                    headers={
                        "X-Rapid-Client": "forged",
                        "X-Rapid-Plugin": "forged",
                        "X-Request-Id": "forged",
                        "X-Rapid-Project": "p",
                        "X-Rapid-Priority": "background",
                    },
                )
                assert (
                    response.status_code == 503
                    and response.content == b"busy exactly"
                    and response.headers["retry-after"] == "17"
                )
                req = seen[-1]
                assert req.url.host == "b" and json.loads(req.content)["model"] == "m"
                assert (
                    req.headers["x-rapid-client"]
                    == req.headers["x-rapid-plugin"]
                    == "one"
                )
                assert (
                    req.headers["x-rapid-project"] == "p"
                    and req.headers["x-rapid-priority"] == "background"
                )
                assert (
                    req.headers["x-request-id"] != "forged"
                    and "authorization" not in req.headers
                )
                row = hub.journal.read()[0]
                assert row["request_sha256"] == hashlib.sha256(req.content).hexdigest()
        await hub.client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode,advertised,queued",
    [
        ("auto", False, True),
        ("auto", True, False),
        ("hub", True, True),
        ("engine", False, False),
    ],
)
def test_admission_capability_and_overrides(tmp_path, mode, advertised, queued):
    async def run():
        def upstream(request):
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "ready": True,
                        "data": [],
                        "capabilities": {"admission": advertised},
                    },
                )
            return httpx.Response(200, json={})

        hub = make_hub(tmp_path, upstream)
        hub.admission, hub.queue_max_wait = mode, 0.05
        await hub.gates["chat"].acquire()
        async with client(hub) as c:
            response = await c.post(
                "/hub/v1/engines/chat/v1/chat/completions",
                json={},
                headers={"X-Rapid-Priority": "background"},
            )
            assert response.status_code == (503 if queued else 200)
            if queued:
                assert response.headers["retry-after"] == "1"
            assert hub.waiters["chat"] == 0
        hub.gates["chat"].release()
        await hub.client.aclose()

    asyncio.run(run())


def test_queue_bound_and_disconnect_removes_waiter(tmp_path):
    async def run():
        hub = make_hub(tmp_path, reported)
        hub.admission, hub.queue_max_length = "hub", 1
        await hub.gates["chat"].acquire()
        async with (
            live(hub) as url,
            httpx.AsyncClient(
                base_url=url,
                trust_env=False,
                headers={
                    "Authorization": "Bearer one",
                    "X-Rapid-Priority": "background",
                },
            ) as c,
        ):
            path = "/hub/v1/engines/chat/v1/chat/completions"
            waiting = asyncio.create_task(c.post(path, json={}))
            for _ in range(100):
                if hub.waiters["chat"] == 1:
                    break
                await asyncio.sleep(0.01)
            assert hub.waiters["chat"] == 1
            full = await c.post(path, json={})
            assert full.status_code == 429 and full.headers["retry-after"] == "1"
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            for _ in range(100):
                if hub.waiters["chat"] == 0:
                    break
                await asyncio.sleep(0.02)
            assert hub.waiters["chat"] == 0
            hub.gates["chat"].release()
            assert (await c.post(path, json={})).status_code == 200

    asyncio.run(run())


def test_sse_tool_calls_byte_identical(tmp_path):
    chunks = [
        b': hello\r\ndata: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call1","type":"function","function":{"name":"lookup","arguments":""}}]}}]}\r\n\r\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\":\\"abc\\"}"}}]}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in chunks:
                yield chunk

    async def run():
        def upstream(request):
            if request.method == "GET":
                return reported(request)
            return httpx.Response(
                200, stream=Stream(), headers={"Content-Type": "text/event-stream"}
            )

        hub = make_hub(tmp_path, upstream)
        async with client(hub) as c:
            response = await c.post(
                "/hub/v1/chat/completions", json={"model": "m", "stream": True}
            )
            assert response.content == b"".join(chunks)
        await hub.client.aclose()

    asyncio.run(run())


def test_discovery_lifecycle_jobs_restart_stale_and_ack(tmp_path):
    async def run():
        hub = make_hub(tmp_path, reported)
        app = create_app(hub, supervise=False)
        async with app.router.lifespan_context(app):
            discovery = json.loads(hub.discovery_file.read_text())
            assert discovery == hub.instance and "token" not in json.dumps(discovery)
            assert hub.discovery_file.stat().st_mode & 0o777 == 0o600
            async with client(hub) as c:
                assert (
                    await c.get("/hub/v1/instance", headers={"Authorization": ""})
                ).json()["instance_id"] == discovery["instance_id"]
                await c.post("/hub/v1/jobs", json={"id": "stale", "project": "p"})
                hub.jobs["stale"]["updated_at"] -= 601
                assert (await c.get("/hub/v1/jobs/stale")).json()[
                    "state"
                ] == "interrupted"
                await c.post("/hub/v1/jobs", json={"id": "previous"})
        assert (
            not hub.discovery_file.exists()
            and not (tmp_path / "connection.json").exists()
        )
        next_hub = make_hub(tmp_path, reported)
        assert next_hub.instance["instance_id"] != discovery["instance_id"]
        assert next_hub.jobs["previous"]["state"] == "interrupted"
        await next_hub.client.aclose()

    asyncio.run(run())


def test_engine_restart_readiness_catalog_and_job_events(tmp_path):
    async def run():
        up = True

        def upstream(request):
            if not up:
                raise httpx.ConnectError("offline", request=request)
            return reported(request)

        hub = make_hub(tmp_path, upstream)
        async with (
            live(hub) as url,
            httpx.AsyncClient(
                base_url=url,
                trust_env=False,
                timeout=12,
                headers={"Authorization": "Bearer " + hub.admin},
            ) as c,
            c.stream("GET", "/hub/v1/events") as response,
        ):
            lines = response.aiter_lines()

            async def next_event(kind):
                async for line in lines:
                    if line == "event: " + kind:
                        data = await anext(lines)
                        return json.loads(data.removeprefix("data: "))

            assert (await next_event("instance"))["instance_id"] == hub.instance[
                "instance_id"
            ]
            up = False
            assert (
                await c.post("/hub/v1/chat/completions", json={"model": "chat/m"})
            ).status_code == 503
            event = await asyncio.wait_for(next_event("engine_readiness"), 8)
            assert event == {"engine": "chat", "ready": False}
            assert not (await next_event("catalog"))["engines"][0]["ready"]
            up = True
            assert (
                await c.post("/hub/v1/chat/completions", json={"model": "chat/m"})
            ).status_code == 200
            assert (await asyncio.wait_for(next_event("engine_readiness"), 8))[
                "ready"
            ] is True
            await c.post("/hub/v1/jobs", json={"id": "event-job"})
            assert (await next_event("job_state"))["id"] == "event-job"
            hub.emit("plugin_state", {"id": "one", "state": "ready"})
            assert (await next_event("plugin_state"))["state"] == "ready"

    asyncio.run(run())
