"""Authenticated model gateway and local control plane."""

import asyncio
import json
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
from .storage import Journal, private_write
from .supervisor import Supervisor


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
    ):
        if background_concurrency < 1:
            raise ValueError("background concurrency must be positive")
        self.engines, self.data_dir = engines, Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.host, self.port = host, port
        self.plugin_host = "127.0.0.1"
        self.url = f"http://{host}:{port}"
        self.admin = secrets.token_urlsafe(32)
        self.login = secrets.token_urlsafe(32)
        self.tokens = {self.admin: "admin"}
        self.sessions = set()
        self.client = httpx.AsyncClient(
            transport=transport, timeout=httpx.Timeout(1800, connect=5), trust_env=False
        )
        self.gates = {
            name: asyncio.Semaphore(background_concurrency) for name in engines
        }
        self.journal = Journal(self.data_dir / "journal")
        self.jobs_path = self.data_dir / "jobs.json"
        self.jobs = (
            json.loads(self.jobs_path.read_text()) if self.jobs_path.exists() else {}
        )
        private_write(self.data_dir / "admin-token", self.admin)
        private_write(self.data_dir / "connection.json", json.dumps({"url": self.url}))
        manifests, self.errors = discover(plugins_dir)
        self.supervisor = Supervisor(self, manifests)

    async def catalog(self):
        data, engines = [], []
        for name, base in self.engines.items():
            ready, models, error = False, [], None
            try:
                health = await self.client.get(
                    base.removesuffix("/v1") + "/health", timeout=3
                )
                ready = health.is_success and health.json().get("ready") is True
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                error = type(exc).__name__
            try:
                response = await self.client.get(base + "/models", timeout=3)
                response.raise_for_status()
                for model in response.json().get("data", []):
                    reported = model.get("capabilities") or []
                    if isinstance(reported, dict):
                        reported = [k for k, v in reported.items() if v]
                    capabilities = []
                    if any(c in reported for c in ("chat", "text", "completions")):
                        capabilities.append("chat")
                    if any(c in reported for c in ("embedding", "embeddings")):
                        capabilities.append("embeddings")
                    row = dict(
                        model, engine=name, capabilities=capabilities, ready=ready
                    )
                    models.append(row)
                    data.append(row)
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                error = type(exc).__name__
            engines.append(dict(name=name, ready=ready, models=models, error=error))
        default = next((m for m in data if "chat" in m["capabilities"]), None)
        return dict(data=data, engines=engines, default=default)

    def identity(self, request, admin=False):
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        identity = self.tokens.get(token)
        if not identity and request.cookies.get("rapid_hub") in self.sessions:
            identity = "admin"
        if not identity:
            raise HTTPException(401, "Bearer token required")
        if admin and identity != "admin":
            raise HTTPException(403, "Admin required")
        return identity

    def save_jobs(self):
        temporary = self.jobs_path.with_suffix(".tmp")
        private_write(temporary, json.dumps(self.jobs))
        temporary.replace(self.jobs_path)


def create_app(hub, supervise=True):
    @asynccontextmanager
    async def lifespan(app):
        try:
            if supervise:
                await hub.supervisor.start()
            yield
        finally:
            await hub.supervisor.stop()
            await hub.client.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.hub = hub

    @app.middleware("http")
    async def origin_guard(request, call_next):
        allowed = {hub.host, "localhost", "127.0.0.1", "::1"}
        host = urlsplit("http://" + request.headers.get("host", "")).hostname
        origin = request.headers.get("origin")
        if host not in allowed:
            return Response("Host rejected", status_code=403)
        if origin and origin != str(request.base_url).rstrip("/"):
            return Response("Origin rejected", status_code=403)
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

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

    @app.get("/hub/v1/models")
    async def models(request: Request):
        hub.identity(request)
        return await hub.catalog()

    @app.get("/hub/v1/status")
    async def status(request: Request):
        hub.identity(request, admin=True)
        return dict(
            **await hub.catalog(), plugins=hub.supervisor.public(), errors=hub.errors
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
        if identity != "admin":
            if plugin and plugin != identity:
                raise HTTPException(403, "Own journal only")
            plugin = identity
        return {"entries": hub.journal.read(plugin, project, min(max(limit, 1), 1000))}

    def own_job(request, ident):
        identity = hub.identity(request)
        job = hub.jobs.get(ident)
        if not job:
            raise HTTPException(404)
        if identity != "admin" and job["plugin"] != identity:
            raise HTTPException(403)
        return job

    @app.get("/hub/v1/jobs")
    async def jobs(request: Request):
        identity = hub.identity(request)
        return {
            "jobs": [
                j
                for j in hub.jobs.values()
                if identity == "admin" or j["plugin"] == identity
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
        if job["state"] not in ("running", "done", "failed", "cancel_requested"):
            raise HTTPException(422, "Invalid job state")
        if old and old["state"] == "cancel_requested" and job["state"] == "running":
            job["state"] = "cancel_requested"
        hub.jobs[ident] = job
        hub.save_jobs()
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
            hub.save_jobs()
        return job

    @app.api_route("/hub/v1/engines/{engine}/v1/{path:path}", methods=["POST", "GET"])
    async def gateway(engine: str, path: str, request: Request):
        identity = hub.identity(request)
        if engine not in hub.engines:
            raise HTTPException(404, "Unknown engine")
        if not path or any(p in ("..", ".") for p in path.split("/")) or "\\" in path:
            raise HTTPException(400, "Invalid upstream path")
        priority = request.headers.get("x-rapid-priority", "interactive")
        if priority not in ("interactive", "background"):
            raise HTTPException(422, "Invalid priority")
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except ValueError:
            payload = {}
        started = time.monotonic()
        entry = dict(
            id=secrets.token_hex(16),
            phase="started",
            plugin=identity,
            project=request.headers.get("x-rapid-project"),
            engine=engine,
            path=path,
            model=payload.get("model") if isinstance(payload, dict) else None,
            priority=priority,
            request=payload if payload else body.decode(errors="replace"),
            status=499,
            latency_ms=0,
            queue_wait_ms=0,
            usage=None,
            response="",
        )
        acquired, upstream = False, None

        async def finish():
            nonlocal acquired
            try:
                if upstream is not None:
                    await upstream.aclose()
            finally:
                if acquired:
                    hub.gates[engine].release()
                    acquired = False
                entry["phase"] = "finished"
                entry["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                hub.journal.append(entry)

        try:
            if priority == "background":
                await hub.gates[engine].acquire()
                acquired = True
            entry["queue_wait_ms"] = round((time.monotonic() - started) * 1000, 3)
            headers = {"X-Rapid-Plugin": identity, "X-Rapid-Priority": priority}
            for name in ("content-type", "accept", "x-rapid-project"):
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
            # Persist before exposing any request text to the model. A hard crash
            # leaves a started record; completion appends another immutable row.
            hub.journal.append(dict(entry, status=None))
            upstream = await hub.client.send(req, stream=True)
            entry["status"] = upstream.status_code
            headers = {
                k: v
                for k, v in upstream.headers.items()
                if k.lower()
                in ("content-type", "retry-after", "cache-control", "x-request-id")
            }
            if "text/event-stream" not in upstream.headers.get("content-type", ""):
                data = await upstream.aread()
                entry["response"] = data[:65536].decode(errors="replace")
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
                finally:
                    entry["response"] = captured.decode(errors="replace")
                    await finish()

            return StreamingResponse(
                stream(), status_code=upstream.status_code, headers=headers
            )
        except BaseException as exc:
            if isinstance(exc, httpx.HTTPError):
                entry["status"] = 502
            await finish()
            if isinstance(exc, httpx.HTTPError):
                raise HTTPException(502, "Engine unavailable") from exc
            raise

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        hub.identity(request, admin=True)
        return HTMLResponse((Path(__file__).parent / "home.html").read_text())

    return app
