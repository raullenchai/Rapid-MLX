"""Authenticated model gateway and local control plane."""

import asyncio
import fcntl
import hashlib
import json
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from .manifest import discover
from .storage import Journal, JournalOverloadedError, atomic_write, private_write
from .supervisor import Supervisor


class Identity(str):
    """A serializable caller name with authority independent of that name."""

    def __new__(cls, name, *, is_admin=False):
        value = super().__new__(cls, name)
        value.is_admin = is_admin
        return value


async def await_connected(operation, request):
    task = asyncio.create_task(operation)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.05)
            if await request.is_disconnected():
                raise asyncio.CancelledError()
            if done:
                return task.result()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def supports_admission(payload):
    caps = payload.get("capabilities", [])
    return payload.get("admission") is True or (
        bool(caps.get("admission")) if isinstance(caps, dict) else "admission" in caps
    )


async def acquire_connected(gate, request, timeout):
    task = asyncio.create_task(gate.acquire())
    deadline = time.monotonic() + timeout
    transferred = False
    try:
        while True:
            done, _ = await asyncio.wait(
                {task}, timeout=min(0.05, max(0, deadline - time.monotonic()))
            )
            if await request.is_disconnected():
                raise asyncio.CancelledError()
            if done:
                transferred = True
                return
            if time.monotonic() >= deadline:
                raise TimeoutError()
    finally:
        if not transferred:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if not task.cancelled() and task.exception() is None and task.result():
                gate.release()


class OriginGuard:
    """ASGI guard preserves disconnect notifications without receive wrappers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        host = urlsplit("http://" + request.headers.get("host", "")).hostname
        dashboard = request.url.path in ("/", "/login") or request.url.path.endswith(
            "/open"
        )
        if host not in {"hub.localhost", "localhost", "127.0.0.1"} or (
            dashboard and host != "hub.localhost"
        ):
            return await Response("Host rejected", status_code=403)(
                scope, receive, send
            )
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return await Response("Origin rejected", status_code=403)(
                scope, receive, send
            )

        async def guarded_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message["headers"]) + [
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-content-type-options", b"nosniff"),
                ]
            await send(message)

        await self.app(scope, receive, guarded_send)


class Hub:
    def __init__(
        self,
        engines,
        data_dir,
        plugins_dir="apps",
        host="127.0.0.1",
        port=4800,
        background_concurrency=1,
        transport=None,
        journal_max_mb=500,
        admission="auto",
        queue_max_length=32,
        queue_max_wait=120,
        job_stale_seconds=600,
        discovery_file=None,
        journal_queue_max_ops=256,
        journal_queue_max_bytes=8 * 1024 * 1024,
        plugin_ports=None,
    ):
        if journal_max_mb <= 0 or job_stale_seconds <= 0:
            raise ValueError("Journal cap and job stale interval must be positive")
        if background_concurrency < 1:
            raise ValueError("background concurrency must be positive")
        self.engines, self.data_dir = engines, Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if host != "127.0.0.1":
            raise ValueError("Hub must bind to 127.0.0.1")
        if (
            admission not in ("auto", "hub", "engine")
            or queue_max_length < 0
            or queue_max_wait <= 0
        ):
            raise ValueError("Invalid admission configuration")
        self.admission, self.queue_max_length, self.queue_max_wait = (
            admission,
            queue_max_length,
            queue_max_wait,
        )
        self.waiters = {name: 0 for name in engines}
        self.engine_admission = {name: False for name in engines}
        self.job_stale_seconds = job_stale_seconds
        self.ui_codes = {}
        self.subscribers = set()
        self.cleanup_tasks = set()
        self.last_catalog = None
        self.known_models = {}
        self.host, self.port = host, port
        self.plugin_host = "127.0.0.1"
        self.url = f"http://{host}:{port}"
        self.dashboard_url = f"http://hub.localhost:{port}"
        self.instance = dict(
            url=self.url,
            instance_id=secrets.token_hex(16),
            pid=os.getpid(),
            api_versions=["1"],
            capabilities=["models", "journal", "jobs", "events"],
            started_at=time.time(),
        )
        self.discovery_file = (
            self.data_dir / discovery_file
            if discovery_file
            else Path.home() / "Library/Application Support/Rapid Hub/connection.json"
        )
        self.admin = secrets.token_urlsafe(32)
        self.login = secrets.token_urlsafe(32)
        self.tokens = {}
        self.sessions = set()
        manifests, self.errors = discover(plugins_dir)
        self.supervisor = Supervisor(self, manifests, ports=plugin_ports)
        # Hold an OS lock for this Hub's lifetime. Never unlink the lock file:
        # another process may already hold the inode open while waiting.
        self.data_lock = (self.data_dir / ".owner.lock").open("a+")
        try:
            fcntl.flock(self.data_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.data_lock.close()
            raise ValueError(
                "Hub data directory is already owned by another instance"
            ) from exc
        self.client = httpx.AsyncClient(
            transport=transport, timeout=httpx.Timeout(1800, connect=5), trust_env=False
        )
        self.gates = {
            name: asyncio.Semaphore(background_concurrency) for name in engines
        }
        self.journal = Journal(
            self.data_dir / "journal",
            journal_max_mb,
            journal_queue_max_ops,
            journal_queue_max_bytes,
        )
        self.jobs_path = self.data_dir / "jobs.json"
        self.jobs = (
            json.loads(self.jobs_path.read_text()) if self.jobs_path.exists() else {}
        )
        for job in self.jobs.values():
            if job["state"] in ("running", "cancel_requested", "cancelling"):
                job["state"] = "interrupted"
        self.save_jobs()
        private_write(self.data_dir / "admin-token", self.admin)

    def release_ownership(self):
        self.data_lock.close()

    async def catalog(self):
        data, engines = [], []
        for name, base in self.engines.items():
            ready, models, error, health_data = False, [], None, {}
            self.engine_admission[name] = False
            try:
                health = await self.client.get(
                    base.removesuffix("/v1") + "/health", timeout=3
                )
                health_data = health.json() if health.is_success else {}
                ready = health.is_success and health_data.get("ready") is True
                self.engine_admission[name] = supports_admission(health_data)
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                error = type(exc).__name__
            try:
                response = await self.client.get(base + "/models", timeout=3)
                response.raise_for_status()
                catalog_data = response.json()
                self.engine_admission[name] |= supports_admission(catalog_data)
                for model in catalog_data.get("data", []):
                    self.engine_admission[name] |= supports_admission(model)
                    reported = model.get("capabilities") or []
                    if isinstance(reported, dict):
                        reported = [k for k, v in reported.items() if v]
                    capabilities = []
                    if any(c in reported for c in ("chat", "text", "completions")):
                        capabilities.append("chat")
                    if "rerank" in reported:
                        capabilities.append("rerank")
                    if any(c in reported for c in ("embedding", "embeddings")):
                        capabilities.append("embeddings")
                    row = dict(
                        model, engine=name, capabilities=capabilities, ready=ready
                    )
                    models.append(row)
                    data.append(row)
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                error = type(exc).__name__
            if models:
                self.known_models[name] = models
            elif error and name in self.known_models:
                models = [dict(m, ready=False) for m in self.known_models[name]]
                data.extend(models)
            engines.append(dict(name=name, ready=ready, models=models, error=error))
        default = next((m for m in data if "chat" in m["capabilities"]), None)
        return dict(data=data, engines=engines, default=default)

    def identity(self, request, admin=False):
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        identity = Identity(self.tokens[token]) if token in self.tokens else None
        if secrets.compare_digest(token, self.admin):
            identity = Identity("admin", is_admin=True)
        if (
            not identity
            and request.url.hostname == "hub.localhost"
            and request.cookies.get("rapid_hub") in self.sessions
        ):
            identity = Identity("admin", is_admin=True)
        if not identity:
            raise HTTPException(401, "Bearer token required")
        if admin and not identity.is_admin:
            raise HTTPException(403, "Admin required")
        return identity

    def emit(self, event, data):
        for queue in tuple(self.subscribers):
            if queue.full():
                self.subscribers.discard(queue)
                continue
            queue.put_nowait(dict(event=event, data=data))

    def stale_jobs(self):
        for job in self.jobs.values():
            if (
                job["state"] in ("running", "cancel_requested", "cancelling")
                and time.time() - job.get("updated_at", 0) > self.job_stale_seconds
            ):
                job["state"] = "interrupted"
                self.emit("job_state", job.copy())
                self.save_jobs()

    async def poll(self):
        previous_plugins = None
        last_prune = time.monotonic()
        while True:
            catalog = await self.catalog()
            if self.last_catalog != catalog:
                old = {
                    e["name"]: e["ready"]
                    for e in (self.last_catalog or {}).get("engines", [])
                }
                for engine in catalog["engines"]:
                    if old.get(engine["name"]) != engine["ready"]:
                        self.emit(
                            "engine_readiness",
                            dict(engine=engine["name"], ready=engine["ready"]),
                        )
                self.emit("catalog", catalog)
                self.last_catalog = catalog
            plugins = [
                {k: row.get(k) for k in ("id", "state", "port", "restarts")}
                for row in self.supervisor.public()
            ]
            if plugins != previous_plugins:
                self.emit("plugin_state", plugins)
                previous_plugins = plugins
            self.stale_jobs()
            if time.monotonic() - last_prune >= 3600:
                try:
                    await self.journal.call("prune")
                except JournalOverloadedError:
                    pass
                last_prune = time.monotonic()
            await asyncio.sleep(5)

    def save_jobs(self):
        temporary = self.jobs_path.with_suffix(".tmp")
        private_write(temporary, json.dumps(self.jobs))
        temporary.replace(self.jobs_path)


def create_app(hub, supervise=True):
    @asynccontextmanager
    async def lifespan(app):
        atomic_write(hub.discovery_file, json.dumps(hub.instance))
        atomic_write(hub.data_dir / "connection.json", json.dumps(hub.instance))
        poller = asyncio.create_task(hub.poll())
        try:
            if supervise:
                await hub.supervisor.start()
            yield
        finally:
            poller.cancel()
            await asyncio.gather(poller, return_exceptions=True)
            await hub.supervisor.stop()
            for path in {hub.discovery_file, hub.data_dir / "connection.json"}:
                if (
                    path.exists()
                    and json.loads(path.read_text()).get("instance_id")
                    == hub.instance["instance_id"]
                ):
                    path.unlink()
            while hub.cleanup_tasks:
                await asyncio.gather(*hub.cleanup_tasks)
            await hub.client.aclose()
            await hub.journal.close()
            hub.release_ownership()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.hub = hub

    app.add_middleware(OriginGuard)

    @app.exception_handler(JournalOverloadedError)
    async def journal_overloaded(request, exc):
        return Response("Journal busy", status_code=503, headers={"Retry-After": "1"})

    @app.get("/login")
    async def login(request: Request):
        # The URL contains a single-use login nonce, never the admin credential.
        nonce = request.query_params.get("nonce", "")
        if not hub.login or not secrets.compare_digest(nonce, hub.login):
            raise HTTPException(401, "Invalid or consumed login")
        hub.login = None
        session = secrets.token_urlsafe(32)
        hub.sessions.add(session)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie("rapid_hub", session, httponly=True, samesite="strict")
        return response

    @app.post("/hub/v1/login-nonce")
    async def login_nonce(request: Request):
        hub.identity(request, admin=True)
        hub.login = secrets.token_urlsafe(32)
        return {"url": f"{hub.dashboard_url}/login?nonce={hub.login}"}

    @app.get("/hub/v1/instance")
    async def instance():
        return hub.instance

    @app.get("/hub/v1/events")
    async def events(request: Request):
        identity = hub.identity(request)
        queue = asyncio.Queue(maxsize=256)
        hub.subscribers.add(queue)

        async def stream():
            try:
                yield "event: instance\ndata: " + json.dumps(hub.instance) + "\n\n"
                while queue in hub.subscribers:
                    try:
                        event = await asyncio.wait_for(queue.get(), 15)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    if (
                        not identity.is_admin
                        and event["event"] == "job_state"
                        and event["data"].get("plugin") != identity
                    ):
                        continue
                    yield (
                        "event: "
                        + event["event"]
                        + "\ndata: "
                        + json.dumps(event["data"])
                        + "\n\n"
                    )
            finally:
                hub.subscribers.discard(queue)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/hub/v1/plugins/{ident}/open")
    async def open_ui(ident: str, request: Request):
        hub.identity(request, admin=True)
        item = hub.supervisor.items.get(ident)
        if not item or item["state"] != "ready":
            raise HTTPException(409, "Plugin is not ready")
        hub.ui_codes = {
            k: v for k, v in hub.ui_codes.items() if v[1] > time.monotonic()
        }
        code = secrets.token_urlsafe(32)
        hub.ui_codes[code] = (ident, time.monotonic() + 60, item["token"])
        return RedirectResponse(
            f"http://{ident}.localhost:{item['port']}/hub-auth?code={code}",
            status_code=303,
        )

    @app.post("/hub/v1/ui-codes/redeem")
    async def redeem(request: Request):
        identity = hub.identity(request)
        code = (await request.json()).get("code", "")
        value = hub.ui_codes.get(code)
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if (
            not value
            or value[0] != identity
            or value[1] <= time.monotonic()
            or value[2] != token
        ):
            raise HTTPException(401, "Invalid or expired UI code")
        del hub.ui_codes[code]
        return {"plugin": identity, "authenticated": True}

    @app.put("/hub/v1/projects/{project}/capture")
    async def capture(project: str, request: Request):
        identity = hub.identity(request)
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", project):
            raise HTTPException(422, "Invalid project")
        body = await request.json()
        owner = body.get("plugin", "*" if identity.is_admin else identity)
        if not identity.is_admin and owner != identity:
            raise HTTPException(403, "Own project namespace only")
        ttl = body.get("ttl_hours", 24)
        if (
            type(body.get("payload")) is not bool
            or type(ttl) not in (float, int)
            or not 0 < ttl <= 168
        ):
            raise HTTPException(
                422, "payload boolean and ttl_hours in (0,168] required"
            )
        await hub.journal.call("configure", owner, project, body["payload"], ttl)
        return hub.journal.policy(owner, project)

    @app.delete("/hub/v1/journal")
    async def delete_journal(request: Request, project: str | None = None):
        identity = hub.identity(request)
        await hub.journal.call(
            "prune",
            plugin=None if identity.is_admin else identity,
            project=project,
            delete=True,
        )
        return {"ok": True}

    @app.get("/hub/v1/models")
    async def models(request: Request):
        hub.identity(request)
        return await hub.catalog()

    @app.get("/hub/v1/status")
    async def status(request: Request):
        hub.identity(request, admin=True)
        return dict(
            **await hub.catalog(),
            plugins=hub.supervisor.public(),
            errors=hub.errors,
            dropped_records=hub.journal.dropped_records,
        )

    @app.post("/hub/v1/plugins/{ident}/restart")
    async def restart(ident: str, request: Request):
        hub.identity(request, admin=True)
        if ident not in hub.supervisor.items:
            raise HTTPException(404)
        await hub.supervisor.restart(ident)
        return {"ok": True}

    @app.get("/hub/v1/journal")
    async def journal(
        request: Request,
        plugin: str | None = None,
        project: str | None = None,
        limit: int = 50,
    ):
        identity = hub.identity(request)
        if not identity.is_admin:
            if plugin and plugin != identity:
                raise HTTPException(403, "Own journal only")
            plugin = identity
        return {
            "entries": await hub.journal.call(
                "read", plugin, project, min(max(limit, 1), 1000)
            )
        }

    def own_job(request, ident):
        identity = hub.identity(request)
        hub.stale_jobs()
        job = hub.jobs.get(ident)
        if not job:
            raise HTTPException(404)
        if not identity.is_admin and job["plugin"] != identity:
            raise HTTPException(403)
        return job

    @app.get("/hub/v1/jobs")
    async def jobs(request: Request):
        identity = hub.identity(request)
        hub.stale_jobs()
        return {
            "jobs": [
                j
                for j in hub.jobs.values()
                if identity.is_admin or j["plugin"] == identity
            ]
        }

    async def write_job(request, ident=None):
        identity = hub.identity(request)
        body = await request.json()
        ident = ident or body.get("id")
        if not isinstance(ident, str) or not ident or len(ident) > 200:
            raise HTTPException(422, "Job id required")
        old = hub.jobs.get(ident)
        if old:
            own_job(request, ident)
        elif request.method == "PATCH":
            raise HTTPException(404)
        values = {
            k: body[k] for k in ("title", "project", "state", "progress") if k in body
        }
        if any(not isinstance(v, str) or len(v) > 16000 for v in values.values()):
            raise HTTPException(422, "Job fields must be bounded text")
        job = dict(
            old or {"id": ident, "plugin": identity, "state": "running"}, **values
        )
        transitions = {
            "running": {"running", "cancel_requested", "done", "failed"},
            "cancel_requested": {"cancel_requested", "cancelling"},
            "cancelling": {"cancelling", "cancelled", "failed", "done"},
        }
        if old and old["state"] == "cancel_requested" and job["state"] == "running":
            job["state"] = "cancel_requested"
        allowed = transitions.get(old["state"], {old["state"]}) if old else {"running"}
        if job["state"] not in allowed:
            raise HTTPException(
                409, "Invalid job transition; cancellation requires acknowledgement"
            )
        job["updated_at"] = time.time()
        hub.jobs[ident] = job
        hub.save_jobs()
        hub.emit("job_state", job.copy())
        return job

    @app.api_route("/hub/v1/jobs", methods=["POST", "PATCH"])
    async def upsert(request: Request):
        return await write_job(request)

    @app.patch("/hub/v1/jobs/{ident}")
    async def patch(ident: str, request: Request):
        return await write_job(request, ident)

    @app.get("/hub/v1/jobs/{ident}")
    async def job(ident: str, request: Request):
        return own_job(request, ident)

    @app.post("/hub/v1/jobs/{ident}/cancel")
    async def cancel(ident: str, request: Request):
        hub.identity(request, admin=True)
        job = own_job(request, ident)
        if job["state"] == "running":
            job["state"] = "cancel_requested"
            job["updated_at"] = time.time()
            hub.save_jobs()
        hub.emit("job_state", job.copy())
        return job

    @app.post("/hub/v1/chat/completions")
    @app.post("/hub/v1/embeddings")
    @app.post("/hub/v1/rerank")
    async def routed(request: Request):
        hub.identity(request)
        try:
            payload = await request.json()
            model = payload["model"]
            if not isinstance(model, str):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise HTTPException(422, "model is required")
        catalog = await hub.catalog()
        candidates = [
            m for m in catalog["data"] if model == m["engine"] + "/" + m["id"]
        ]
        if not candidates:
            candidates = [m for m in catalog["data"] if model == m["id"]]
        if len(candidates) > 1:
            raise HTTPException(
                409, {"candidates": [m["engine"] + "/" + m["id"] for m in candidates]}
            )
        if not candidates:
            raise HTTPException(404, "Unknown model")
        chosen = candidates[0]
        payload["model"] = chosen["id"]
        return await gateway(
            chosen["engine"],
            request.url.path.removeprefix("/hub/v1/"),
            request,
            json.dumps(payload).encode(),
        )

    @app.api_route("/hub/v1/engines/{engine}/v1/{path:path}", methods=["POST", "GET"])
    async def gateway_route(engine: str, path: str, request: Request):
        return await gateway(engine, path, request)

    async def gateway(engine, path, request, routed_body=None):
        identity = hub.identity(request)
        if engine not in hub.engines:
            raise HTTPException(404, "Unknown engine")
        if not path or any(p in ("..", ".") for p in path.split("/")) or "\\" in path:
            raise HTTPException(400, "Invalid upstream path")
        priority = request.headers.get("x-rapid-priority", "interactive")
        if priority not in ("interactive", "background"):
            raise HTTPException(422, "Invalid priority")
        body = routed_body if routed_body is not None else await request.body()
        try:
            payload = json.loads(body) if body else {}
        except ValueError:
            payload = {}
        started = time.monotonic()
        request_id = secrets.token_hex(16)
        project = request.headers.get("x-rapid-project", "")[:200]
        entry = dict(
            request_id=request_id,
            client=identity,
            plugin=identity,
            project=project,
            job_id=request.headers.get("x-rapid-job-id", "")[:200] or None,
            engine=engine,
            path=path,
            model=payload.get("model") if isinstance(payload, dict) else None,
            priority=priority,
            status=499,
            latency_ms=0,
            queue_wait_ms=0,
            usage=None,
            request_bytes=len(body),
            response_bytes=0,
            request_sha256=hashlib.sha256(body).hexdigest(),
            cancelled=False,
            request=payload,
            response="",
        )
        capture_payload = hub.journal.policy(identity, project).get("payload", False)
        acquired, upstream = False, None
        cleanup_task = None

        async def cleanup():
            nonlocal acquired
            try:
                if upstream is not None:
                    await upstream.aclose()
            finally:
                if acquired:
                    hub.gates[engine].release()
                    acquired = False
                entry["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                await hub.journal.call("append", entry)

        async def finish():
            nonlocal cleanup_task
            # Repeated cancellation must reuse the same cleanup and permit release.
            if cleanup_task is None:
                cleanup_task = asyncio.create_task(cleanup())
                hub.cleanup_tasks.add(cleanup_task)
                cleanup_task.add_done_callback(hub.cleanup_tasks.discard)
            await asyncio.shield(cleanup_task)

        async def send_upstream():
            nonlocal upstream
            if await request.is_disconnected():
                raise asyncio.CancelledError()
            upstream = await hub.client.send(req, stream=True)

        try:
            if hub.admission == "auto" and hub.last_catalog is None:
                # Capability discovery is read-only and never retries a model call.
                await hub.catalog()
            queue_here = hub.admission == "hub" or (
                hub.admission == "auto" and not hub.engine_admission[engine]
            )
            if priority == "background" and queue_here:
                gate = hub.gates[engine]
                if not gate.locked():
                    # An available semaphore acquires synchronously, so a burst
                    # cannot race past the waiter limit before taking a permit.
                    await gate.acquire()
                    acquired = True
                else:
                    if hub.waiters[engine] >= hub.queue_max_length:
                        entry["status"] = 429
                        raise HTTPException(
                            429, "Hub queue full", headers={"Retry-After": "1"}
                        )
                    hub.waiters[engine] += 1
                    try:
                        await acquire_connected(gate, request, hub.queue_max_wait)
                        acquired = True
                    except TimeoutError:
                        entry["status"] = 503
                        raise HTTPException(
                            503, "Hub queue wait expired", headers={"Retry-After": "1"}
                        )
                    finally:
                        hub.waiters[engine] -= 1
            entry["queue_wait_ms"] = round((time.monotonic() - started) * 1000, 3)
            headers = {
                "X-Rapid-Plugin": identity,
                "X-Rapid-Client": identity,
                "X-Rapid-Priority": priority,
                "X-Rapid-Project": project,
                "X-Request-Id": request_id,
            }
            for name in ("content-type", "accept"):
                if name in request.headers:
                    headers[name] = request.headers[name]
            url = hub.engines[engine] + "/" + path
            req = hub.client.build_request(
                request.method,
                url,
                params=request.query_params,
                content=body,
                headers=headers,
            )
            await await_connected(
                hub.journal.call("append", dict(entry, status=None)), request
            )
            await await_connected(send_upstream(), request)
            entry["status"] = upstream.status_code
            headers = {
                k: v
                for k, v in upstream.headers.items()
                if k.lower()
                in ("content-type", "retry-after", "cache-control", "x-request-id")
            }
            if "text/event-stream" not in upstream.headers.get("content-type", ""):
                data = await await_connected(upstream.aread(), request)
                entry["response_bytes"] = len(data)
                entry["response"] = data if capture_payload else ""
                try:
                    entry["usage"] = json.loads(data).get("usage")
                except (ValueError, AttributeError):
                    pass
                await finish()
                return Response(data, status_code=upstream.status_code, headers=headers)

            async def stream():
                pending = b""
                captured = bytearray()
                try:
                    async for chunk in upstream.aiter_bytes():
                        entry["response_bytes"] += len(chunk)
                        if capture_payload:
                            captured.extend(chunk[: max(0, 65536 - len(captured))])
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if line.startswith(b"data:"):
                                try:
                                    event = json.loads(line[5:])
                                    if event.get("usage"):
                                        entry["usage"] = event["usage"]
                                except (ValueError, AttributeError):
                                    pass
                        # Bound malformed upstream frames without buffering the response.
                        if len(pending) > 1048576:
                            pending = b""
                        yield chunk
                except asyncio.CancelledError:
                    entry["cancelled"] = True
                    raise
                finally:
                    entry["response"] = captured.decode(errors="replace")
                    await finish()

            return StreamingResponse(
                stream(), status_code=upstream.status_code, headers=headers
            )
        except BaseException as exc:
            entry["cancelled"] = isinstance(exc, asyncio.CancelledError)
            if isinstance(exc, httpx.HTTPError):
                entry["status"] = 503
            await finish()
            if isinstance(exc, httpx.HTTPError):
                raise HTTPException(
                    503, "Engine unavailable", headers={"Retry-After": "1"}
                ) from exc
            raise

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        hub.identity(request, admin=True)
        return HTMLResponse((Path(__file__).parent / "home.html").read_text())

    return app
