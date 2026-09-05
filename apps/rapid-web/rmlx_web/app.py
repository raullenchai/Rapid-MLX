# SPDX-License-Identifier: Apache-2.0
"""HTTP surface served to the phone.

``GET /`` and the static assets are unauthenticated — the page is what the
user opens in order to enter the token. Everything under ``/api`` and
``/v1`` requires the bearer plus the browser-origin checks in :mod:`.auth`,
including read-only ``/api/status``, which reveals the loaded model and the
engine's log tail.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile

from . import auth, connectors, proxy, tools
from .catalog import CatalogError, ModelCatalog, RemovalError
from .connectors import ConnectorError, ConnectorStore
from .downloads import (
    DownloadError,
    DownloadManager,
    check_disk_budget,
)
from .images import ImageJobError, ImageJobManager, ImageJobState
from .lifecycle import EngineLifecycle, TransitionStart
from .supervisor import (
    AttachedEngine,
    ChildState,
    EngineSupervisor,
    ResidencyOutcome,
    SupervisorError,
)
from .uploads import UploadBodyLimitMiddleware

STATIC_DIR = Path(__file__).parent / "static"
ASSETS_DIR = STATIC_DIR / "assets"

_IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"

# Matches the engine's own ceiling (``MAX_AUDIO_UPLOAD_SIZE``), so an upload
# that would be refused there is refused here instead of after another hop.
MAX_AUDIO_BYTES = 25 * 1024 * 1024

# The engine's ``_MAX_EDIT_IMAGE_BYTES``, for the same reason.
MAX_IMAGE_BYTES = 25 * 1024 * 1024

# Multipart framing and the small metadata fields share the request body with
# the file. One MiB is ample headroom without turning the file cap into a late
# parser-time check.
MAX_UPLOAD_REQUEST_BYTES = max(MAX_AUDIO_BYTES, MAX_IMAGE_BYTES) + 1024 * 1024
_UPLOAD_PATHS = frozenset({"/api/images/jobs", "/api/audio/transcriptions"})

# Filenames reaching a multipart part. Restricted rather than sanitised: the
# name is advisory (the engine spools every upload to a ``.wav`` temp file
# regardless), so there is nothing to gain by accepting a caller's arbitrary
# string in a header.
_UPLOAD_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# What ``/api/residency`` answers when the engine is not reachable. A limit
# of 0 is the engine's own "no ceiling" spelling, which the page reads as
# "nothing to show" rather than "0 bytes used".
_EMPTY_RESIDENCY = {
    "memory_limit_bytes": 0,
    "memory_used_bytes": 0,
    "models": [],
}


def _upload_filename(candidate: object) -> str:
    if isinstance(candidate, str) and _UPLOAD_NAME_RE.match(candidate):
        return candidate
    return "recording.wav"


async def _read_upload(upload: UploadFile, limit: int) -> bytes | JSONResponse:
    content = bytearray()
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > limit:
            return _json_error(
                413,
                f"upload is larger than {limit // (1024 * 1024)} MB",
                "payload_too_large",
            )
    if not content:
        return _json_error(400, "the uploaded file was empty", "invalid_body")
    return bytes(content)


class _HashedAssets(StaticFiles):
    """Serves the build's content-hashed assets as immutable."""

    def file_response(self, *args: object, **kwargs: object) -> Response:
        response = super().file_response(*args, **kwargs)  # type: ignore[arg-type]
        response.headers["Cache-Control"] = _IMMUTABLE_CACHE_CONTROL
        return response


@dataclass
class WebConfig:
    """Everything the HTTP layer needs that is decided at startup."""

    # Required even on loopback: a user can attach a tunnel to a loopback
    # listener, so the bind address cannot decide whether authentication is
    # necessary.
    token: str
    engine: EngineSupervisor | AttachedEngine
    # Loaded once the event loop is running: an asyncio subprocess is bound
    # to the loop that created it, so spawning under a throwaway
    # ``asyncio.run`` leaves a dead output drain and eventually a full pipe.
    initial_model: str | None = None
    # ``None`` in --attach mode: listing aliases needs the CLI, and an
    # attached engine may be the only rapid-mlx on the machine.
    catalog: ModelCatalog | None = None
    # ``None`` when downloads are disabled — also the single source of
    # truth for whether they are allowed.
    downloads: DownloadManager | None = None
    # Owns the MCP config file and the switches around it. Always present:
    # unlike downloads, there is no mode in which connectors cannot be
    # configured — the master switch is the off state.
    connectors: ConnectorStore = field(default_factory=ConnectorStore)

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("a non-empty web access token is required")


# Catalog kind -> the engine's own modality vocabulary. `audio` never
# reaches a resident load (its lane rides on the served model), so it is
# absent and the caller's `.get(..., "text")` default is never exercised
# for it.
_ENGINE_MODALITY = {"text": "text", "image": "image-gen"}


def _json_error(status: int, message: str, code: str) -> JSONResponse:
    """Uniform error envelope matching the engine's ``{"error": {...}}``."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code}},
    )


async def _boot(config: WebConfig) -> None:
    """Load the initial model, recording failure in the supervisor.

    Failures are swallowed because this is a detached task: raising only
    reaches the user as an asyncio warning on stderr, whereas the
    supervisor's ``FAILED`` state is what ``/api/status`` surfaces.
    """
    with contextlib.suppress(SupervisorError):
        await config.engine.start(config.initial_model)


async def _switch(config: WebConfig, alias: str, entry=None) -> None:
    """Make ``alias`` usable, hot if the engine allows it.

    A hot ``POST /v1/models/load`` is tried FIRST because it is the only
    way two models are usable at once: the engine keeps text/vision in one
    single-slot group and gives each media modality its own, so loading an
    image model beside a chat model leaves the chat model running. A
    respawn, by contrast, can only ever serve the one model it was started
    for.

    Every failure falls back to the respawn this package did
    unconditionally before, so the worst case is the old behaviour.
    Failures are swallowed as in :func:`_boot` — this is a detached task,
    and ``/api/status`` is what the page reads.
    """
    # Duck-typed rather than an isinstance check: the engine is the one seam
    # this package mocks, and `--attach` mode never reaches here (the route
    # refuses on `can_switch` first).
    hot = entry is not None and hasattr(config.engine, "residency_load")
    modality = _ENGINE_MODALITY.get(entry.kind, "text") if entry is not None else "text"
    if hot:
        outcome, _refusal = await config.engine.residency_load(
            alias,
            modality=modality,
            size_bytes=entry.size_bytes,
            image_mode="generation" if entry.kind == "image" else None,
        )
        if outcome is ResidencyOutcome.LOADED:
            if config.catalog is not None:
                config.catalog.invalidate_cache()
            return

    with contextlib.suppress(SupervisorError):
        await config.engine.start(alias, modality=modality)
    # The engine's own downloader may have just pulled these weights.
    if config.catalog is not None:
        config.catalog.invalidate_cache()


def create_app(config: WebConfig) -> FastAPI:
    lifecycle = EngineLifecycle()
    # Renders run detached from the request that started them, so they are
    # counted here for their whole life rather than by a route.
    image_jobs = ImageJobManager(lifecycle)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One client per process: per-request clients lose connection reuse
        # and leak connections when a streaming response is abandoned.
        app.state.http = httpx.AsyncClient()

        # Background, not awaited: a cold start is minutes, and blocking
        # startup would leave the port unbound for all of it — the phone
        # would see "connection refused" instead of a page saying "loading".
        if config.initial_model and isinstance(config.engine, EngineSupervisor):
            lifecycle.start_transition(
                kind="initial-load",
                model=config.initial_model,
                work=lambda: _boot(config),
            )

        try:
            yield
        finally:
            await lifecycle.shutdown()
            with contextlib.suppress(Exception):
                await app.state.http.aclose()
            # A pull left running would keep writing to the cache with
            # nothing watching it, and no way to stop it short of the PID.
            if config.downloads is not None:
                with contextlib.suppress(Exception):
                    await config.downloads.shutdown()
            with contextlib.suppress(Exception):
                await image_jobs.shutdown()
            # Always stop the child, including when startup itself failed:
            # a half-started engine still holds GPU memory.
            await config.engine.stop()

    app = FastAPI(
        title="rmlx-web",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        UploadBodyLimitMiddleware,
        limits={path: MAX_UPLOAD_REQUEST_BYTES for path in _UPLOAD_PATHS},
    )
    app.state.config = config
    app.state.lifecycle = lifecycle

    def _transition_unavailable() -> JSONResponse:
        pending = lifecycle.pending
        model = pending.model if pending is not None else "the requested model"
        return _json_error(
            503,
            f"{model} is still loading; wait for it to finish",
            "engine_unavailable",
        )

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        path = request.url.path

        # ``/api/config`` is open so the page can learn that it must present
        # the login gate before it has a token. It reveals only that one bit.
        if path == "/" or path == "/api/config" or path.startswith("/static"):
            response = await call_next(request)
            _apply_security_headers(response)
            return response

        if not auth.origin_is_allowed(
            request.headers.get("origin"),
            request.headers.get("host"),
            request.headers.get("sec-fetch-site"),
        ):
            return _json_error(403, "cross-origin request refused", "origin_refused")

        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("content-type")
            multipart_upload = (
                path in _UPLOAD_PATHS
                and auth.content_type_is_multipart(content_type)
                and request.headers.get("x-rapid-upload") == "1"
            )
            # Multipart is accepted only on the two upload routes and only
            # with a non-safelisted header. A cross-origin browser must
            # preflight that header, and this server never grants CORS.
            if not multipart_upload and not auth.content_type_is_json(content_type):
                return _json_error(
                    415,
                    "requests must be JSON or an approved multipart upload",
                    "unsupported_media_type",
                )

        presented = auth.extract_bearer(request.headers.get("authorization"))
        if not auth.token_matches(config.token, presented):
            return _json_error(401, "missing or invalid token", "unauthorized")

        response = await call_next(request)
        _apply_security_headers(response)
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Response:
        raw = (STATIC_DIR / "index.html").read_bytes()
        etag = f'"{hashlib.sha256(raw).hexdigest()[:32]}"'

        # If-None-Match may carry a list, so match on membership.
        if etag in [
            candidate.strip()
            for candidate in (request.headers.get("if-none-match") or "").split(",")
            if candidate.strip()
        ]:
            return Response(status_code=304, headers={"ETag": etag})

        return HTMLResponse(
            raw.decode("utf-8"),
            headers={"ETag": etag, "Cache-Control": "no-cache"},
        )

    @app.get("/api/config")
    async def public_config() -> JSONResponse:
        """Unauthenticated boot contract: this server requires a token.

        The only unauthenticated JSON endpoint. The page needs the answer
        before it can decide whether to show a login prompt.
        """
        return JSONResponse({"auth_required": True})

    @app.post("/api/auth")
    async def check_auth() -> JSONResponse:
        """Token probe for the login screen.

        Reaching this handler already means the middleware accepted the
        bearer; it exists so the page can validate a pasted token without
        sending a chat turn.
        """
        engine = config.engine
        return JSONResponse(
            {
                "ok": True,
                "can_switch": engine.can_switch,
                "allow_downloads": config.downloads is not None,
            }
        )

    @app.get("/api/status")
    async def status() -> JSONResponse:
        engine = config.engine
        snapshot = engine.status()
        body = snapshot.to_dict()
        if lifecycle.pending is not None:
            body.update(
                state=ChildState.STARTING.value,
                model=lifecycle.pending.model,
                detail="model transition is pending",
            )
        body["can_switch"] = engine.can_switch
        # The log tail is only useful when something went wrong, and it
        # can contain file paths; withhold it otherwise.
        if lifecycle.pending is None and snapshot.state is ChildState.FAILED:
            body["recent_output"] = snapshot.recent_output[-20:]
        return JSONResponse(body)

    @app.get("/api/models")
    async def list_models(refresh: bool = False) -> JSONResponse:
        """Every alias, tagged with its kind.

        Image and audio rows are included so the model manager can show
        them, and each carries ``loadable`` — audio has no ``serve`` lane
        here, so the picker must not offer to start one.
        """
        if config.catalog is None:
            return _json_error(
                501,
                "model listing is unavailable in --attach mode",
                "catalog_unavailable",
            )
        try:
            entries = await config.catalog.list_models(force=refresh)
        except CatalogError as exc:
            return _json_error(503, str(exc), "catalog_error")

        snapshot = config.engine.status()
        pending = lifecycle.pending
        return JSONResponse(
            {
                "models": [entry.to_dict() for entry in entries],
                "loaded": pending.model if pending is not None else snapshot.model,
                "state": (
                    ChildState.STARTING.value
                    if pending is not None
                    else snapshot.state.value
                ),
                "can_switch": config.engine.can_switch,
                "allow_downloads": config.downloads is not None,
            }
        )

    @app.post("/api/models/load")
    async def load_model(request: Request):
        engine = config.engine
        if not engine.can_switch or config.catalog is None:
            return _json_error(
                409,
                "this server does not own the engine, so it cannot switch models",
                "switch_unavailable",
            )

        try:
            payload = await request.json()
        except (ValueError, json.JSONDecodeError):
            return _json_error(400, "request body was not valid JSON", "invalid_json")

        alias = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(alias, str) or not alias.strip():
            return _json_error(
                400, "`model` must be a non-empty string", "invalid_body"
            )
        alias = alias.strip()

        # Validate against the catalog before the alias reaches a subprocess
        # argument: an arbitrary string would let a remote caller name any
        # `org/repo`, turning a model picker into a general-purpose fetch.
        try:
            entry = await config.catalog.profile(alias)
        except CatalogError as exc:
            return _json_error(503, str(exc), "catalog_error")
        if entry is None:
            return _json_error(404, f"unknown model alias: {alias}", "unknown_model")
        if not entry.loadable:
            # Only `video` today: its lane needs extras a plain install
            # does not ship, which is also why the catalog omits it.
            return _json_error(
                409,
                f"{alias} cannot be loaded as the served model.",
                "kind_not_loadable",
            )

        snapshot = engine.status()
        if snapshot.model == alias and snapshot.state is ChildState.READY:
            # Already there; a double-tap on a phone is easy and a restart
            # would cost minutes of reload for no change.
            return JSONResponse({"ok": True, "model": alias, "state": "ready"})

        if snapshot.state is ChildState.STARTING:
            return _json_error(
                409,
                f"{snapshot.model or 'a model'} is still loading; wait for it to finish",
                "busy_loading",
            )

        # Already resident from an earlier hot load — the engine routes by
        # the request's `model` field, so there is nothing to do.
        if alias in snapshot.resident and snapshot.state is ChildState.READY:
            return JSONResponse({"ok": True, "model": alias, "state": "ready"})

        # Admission and pending-state publication happen synchronously before
        # the response. No second request can enqueue another transition or
        # start engine work in the detached task's scheduling window.
        started = lifecycle.start_transition(
            kind="model-load",
            model=alias,
            work=lambda: _switch(config, alias, entry),
        )
        if started is TransitionStart.BUSY_ACTIVITY:
            return _json_error(
                409,
                "a response is still running; try again once it finishes",
                "busy_streaming",
            )
        if started is TransitionStart.BUSY_TRANSITION:
            pending = lifecycle.pending
            return _json_error(
                409,
                f"{pending.model if pending else 'a model'} is still loading; "
                "wait for it to finish",
                "busy_loading",
            )
        return JSONResponse({"ok": True, "model": alias, "state": "starting"})

    @app.post("/api/models/pull")
    async def pull_model(request: Request):
        if config.downloads is None or config.catalog is None:
            return _json_error(
                403,
                "downloads are disabled on this server "
                "(start it with --allow-downloads)",
                "downloads_disabled",
            )

        try:
            payload = await request.json()
        except (ValueError, json.JSONDecodeError):
            return _json_error(400, "request body was not valid JSON", "invalid_json")

        alias = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(alias, str) or not alias.strip():
            return _json_error(
                400, "`model` must be a non-empty string", "invalid_body"
            )
        alias = alias.strip()

        # Same reasoning as the switch route: an unvalidated alias reaching
        # a subprocess argument is a remote fetch primitive.
        try:
            entry = await config.catalog.profile(alias)
        except CatalogError as exc:
            return _json_error(503, str(exc), "catalog_error")
        if entry is None:
            return _json_error(404, f"unknown model alias: {alias}", "unknown_model")

        # Fails closed when the size is unknown — see check_disk_budget.
        reason = check_disk_budget(entry.size_bytes)
        if reason is not None:
            return _json_error(507, reason, "insufficient_storage")

        try:
            job = await config.downloads.start(alias, total_bytes=entry.size_bytes)
        except DownloadError as exc:
            return _json_error(409, str(exc), "download_conflict")

        return JSONResponse({"ok": True, **job.to_dict()})

    # POST, not DELETE: the middleware's CSRF control (reject CORS-simple
    # content types) runs on POST/PUT/PATCH, so routing the one destructive
    # operation through it avoids a second policy to keep correct.
    @app.post("/api/models/remove")
    async def remove_model(request: Request):
        if config.catalog is None:
            return _json_error(
                501,
                "model removal is unavailable in --attach mode",
                "catalog_unavailable",
            )

        try:
            payload = await request.json()
        except (ValueError, json.JSONDecodeError):
            return _json_error(400, "request body was not valid JSON", "invalid_json")

        alias = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(alias, str) or not alias.strip():
            return _json_error(
                400, "`model` must be a non-empty string", "invalid_body"
            )
        alias = alias.strip()

        # Refuse to delete what the engine is running: the weights are
        # mmap'd by the child. READY and STARTING only — a FAILED child has
        # exited and holds nothing, and deleting a checkpoint that just
        # failed to load is precisely what a user does next.
        snapshot = config.engine.status()
        if snapshot.model == alias and snapshot.state in (
            ChildState.READY,
            ChildState.STARTING,
        ):
            return _json_error(
                409,
                f"{alias} is the model this server is running. "
                "Switch to another model first, then delete it.",
                "model_in_use",
            )

        # A pull writing into the snapshot being unlinked leaves a
        # half-materialised repo: present enough to look downloaded to a
        # stale page, broken enough to fail inside the engine.
        running = config.downloads.job if config.downloads is not None else None
        if (
            running is not None
            and running.alias == alias
            and config.downloads.is_running()
        ):
            return _json_error(
                409,
                f"{alias} is still downloading. Cancel the download first.",
                "model_in_use",
            )

        try:
            freed = await config.catalog.remove(alias)
        except CatalogError as exc:
            return _json_error(503, str(exc), "catalog_error")
        except RemovalError as exc:
            return _json_error(409, str(exc), "removal_failed")

        return JSONResponse({"ok": True, "model": alias, "freed_bytes": freed})

    @app.post("/api/downloads/cancel")
    async def cancel_download():
        if config.downloads is None:
            return _json_error(
                403, "downloads are disabled on this server", "downloads_disabled"
            )
        cancelled = await config.downloads.cancel()
        if not cancelled:
            return _json_error(409, "no download is running", "no_download")
        return JSONResponse({"ok": True})

    @app.get("/api/downloads/status")
    async def download_status():
        """Current download job, polled by the page.

        A poll rather than the SSE feed this replaced: measured against a
        real ``trycloudflare`` tunnel, a sparse feed delivered headers in
        1.8 s and then no body byte in 65 s (loopback: 0.0 s). Cloudflare
        strips ``X-Accel-Buffering`` and padding the first frame did not
        help. Chat streaming survives the same tunnel because it emits
        tokens continuously — sparseness is the variable, not SSE.
        """
        if config.downloads is None:
            return _json_error(
                403, "downloads are disabled on this server", "downloads_disabled"
            )
        job = config.downloads.job
        return JSONResponse(job.to_dict() if job is not None else {"state": "idle"})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            payload = await request.json()
        except (ValueError, json.JSONDecodeError):
            return _json_error(400, "request body was not valid JSON", "invalid_json")

        if not isinstance(payload, dict):
            return _json_error(
                400, "request body must be a JSON object", "invalid_json"
            )

        lease = lifecycle.acquire_activity()
        if lease is None:
            return _transition_unavailable()

        engine = config.engine
        base_url = engine.base_url
        if base_url is None:
            lease.release()
            snapshot = engine.status()
            # 503 rather than 502: the engine is not broken, it is not
            # there yet. The page retries on 503 and gives up on 502.
            return _json_error(
                503,
                _unavailable_message(snapshot.state, snapshot.detail),
                "engine_unavailable",
            )

        if proxy.is_streaming_request(payload):
            # Counted for the whole life of the relay, so a concurrent
            # /api/models/load refuses rather than killing the engine.
            async def tracked() -> AsyncIterator[bytes]:
                try:
                    async for chunk in proxy.proxy_streaming(
                        app.state.http,
                        base_url=base_url,
                        path="/v1/chat/completions",
                        payload=payload,
                        api_key=engine.api_key,
                    ):
                        yield chunk
                finally:
                    lease.release()

            return StreamingResponse(
                tracked(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    # nginx and several tunnels honour this to disable the
                    # buffering that would deliver the stream all at once.
                    "X-Accel-Buffering": "no",
                },
                background=BackgroundTask(lease.release),
            )

        with lease:
            try:
                upstream = await proxy.proxy_unary(
                    app.state.http,
                    base_url=base_url,
                    path="/v1/chat/completions",
                    payload=payload,
                    api_key=engine.api_key,
                )
            except httpx.HTTPError as exc:
                return _json_error(
                    502, f"connection to the engine failed: {exc}", "engine_transport"
                )

        return JSONResponse(
            status_code=upstream.status_code,
            content=_decode_json_body(upstream),
            headers=proxy.filtered_response_headers(upstream.headers),
        )

    @app.post("/api/images/jobs")
    async def start_image_job(request: Request):
        """Start a render or an edit, and answer with its id immediately.

        The connection is NOT held for the render. The engine replies only
        once the whole image is finished, so relaying inline left a socket
        with no bytes flowing for minutes and Cloudflare cut it at 100 s
        with a 524. The page polls ``/api/images/jobs/{id}`` instead.

        Generation uses JSON. An edit uses a bounded multipart upload so the
        browser and server never materialise Base64 copies of the source.
        """
        is_edit = auth.content_type_is_multipart(request.headers.get("content-type"))
        payload: dict = {}
        content: bytes | None = None
        if is_edit:
            async with request.form(max_files=1, max_fields=8) as form:
                image = form.get("image")
                if not isinstance(image, UploadFile):
                    return _json_error(400, "`image` must be a file", "invalid_body")
                read = await _read_upload(image, MAX_IMAGE_BYTES)
                if isinstance(read, JSONResponse):
                    return read
                content = read
                payload = {key: form.get(key) for key in ("prompt", "model")}
        else:
            try:
                decoded = await request.json()
            except (ValueError, json.JSONDecodeError):
                return _json_error(
                    400, "request body was not valid JSON", "invalid_json"
                )
            if not isinstance(decoded, dict):
                return _json_error(
                    400, "request body must be a JSON object", "invalid_json"
                )
            payload = decoded
            if payload.get("mode", "generation") != "generation":
                return _json_error(
                    415,
                    "image edits must use multipart upload",
                    "unsupported_media_type",
                )

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return _json_error(400, "`prompt` must not be empty", "invalid_body")
        prompt = prompt.strip()

        model = payload.get("model")
        model = model if isinstance(model, str) and model else None

        engine = config.engine
        base_url = engine.base_url
        if base_url is None:
            snapshot = engine.status()
            return _json_error(
                503,
                _unavailable_message(snapshot.state, snapshot.detail),
                "engine_unavailable",
            )

        if is_edit:
            assert content is not None
            fields = {"prompt": prompt, "n": "1", "response_format": "b64_json"}
            if model:
                fields["model"] = model

            async def work():
                upstream = await proxy.proxy_multipart(
                    app.state.http,
                    base_url=base_url,
                    path="/v1/images/edits",
                    api_key=engine.api_key,
                    field="image",
                    # The engine sniffs the real format from the bytes; the
                    # name and type here only have to be well-formed.
                    filename="input.png",
                    content_type="image/png",
                    content=content,
                    fields=fields,
                )
                return upstream.status_code, _decode_json_body(upstream)
        else:
            body = {
                "prompt": prompt,
                "n": 1,
                "response_format": "b64_json",
                **({"model": model} if model else {}),
                # `size` is forwarded only for generation: the edit backends
                # derive their canvas from the source image and discard it.
                **(
                    {"size": payload["size"]}
                    if isinstance(payload.get("size"), str)
                    else {}
                ),
            }

            async def work():
                upstream = await proxy.proxy_unary(
                    app.state.http,
                    base_url=base_url,
                    path="/v1/images/generations",
                    payload=body,
                    api_key=engine.api_key,
                )
                return upstream.status_code, _decode_json_body(upstream)

        try:
            job = image_jobs.start(work, model=model)
        except ImageJobError as exc:
            return _json_error(409, str(exc), "image_busy")
        return JSONResponse(job.to_dict())

    @app.get("/api/images/jobs/{job_id}")
    async def image_job(job_id: str):
        """The job's progress while it runs, and its result when it ends.

        One poll answers both, so a render occupies a single connection at
        a time. Polled rather than streamed: a sparse SSE body is buffered
        indefinitely by a tunnel, which is what removed the download feed.

        An unknown id is 404 — only the last job is kept, and reporting a
        vanished one as idle would leave the page waiting on it forever.
        """
        job = image_jobs.get(job_id)
        if job is None:
            return _json_error(404, "no such render", "unknown_image_job")

        snapshot = job.to_dict()
        if job.state is not ImageJobState.RUNNING:
            return JSONResponse(snapshot)

        # Denoise steps come from the engine, and a dropped read is not a
        # render failure — the job's own state is what reports one.
        engine = config.engine
        base_url = engine.base_url
        step, total = 0, 0
        if base_url is not None:
            with contextlib.suppress(httpx.HTTPError):
                upstream = await proxy.proxy_get(
                    app.state.http,
                    base_url=base_url,
                    path="/v1/images/progress",
                    api_key=engine.api_key,
                    params={"model": job.model} if job.model else None,
                )
                if upstream.status_code < 400:
                    progress = _decode_json_body(upstream)
                    step = progress.get("step") or 0
                    total = progress.get("total") or 0
        return JSONResponse({**snapshot, "step": step, "total": total})

    @app.post("/api/images/cancel")
    async def image_cancel(request: Request):
        engine = config.engine
        base_url = engine.base_url
        if base_url is None:
            return _json_error(503, "no model is loaded", "engine_unavailable")

        try:
            payload = await request.json()
        except (ValueError, json.JSONDecodeError):
            payload = {}
        model = payload.get("model") if isinstance(payload, dict) else None

        try:
            upstream = await proxy.proxy_post_query(
                app.state.http,
                base_url=base_url,
                path="/v1/images/cancel",
                api_key=engine.api_key,
                params={"model": model} if isinstance(model, str) and model else None,
            )
        except httpx.HTTPError as exc:
            return _json_error(
                502, f"connection to the engine failed: {exc}", "engine_transport"
            )
        return JSONResponse(
            status_code=upstream.status_code, content=_decode_json_body(upstream)
        )

    @app.get("/api/residency")
    async def residency():
        """Resident models and process memory against the engine's ceiling.

        Polled while the page is open, so an unreachable engine answers an
        EMPTY snapshot rather than an error: the panel's job is to describe
        the machine, and a dropped poll during a model switch is not a
        failure worth putting a banner over.
        """
        engine = config.engine
        base_url = engine.base_url
        if base_url is None:
            return JSONResponse(_EMPTY_RESIDENCY)
        try:
            upstream = await proxy.proxy_get(
                app.state.http,
                base_url=base_url,
                path="/v1/models/residency",
                api_key=engine.api_key,
            )
        except httpx.HTTPError:
            return JSONResponse(_EMPTY_RESIDENCY)
        if upstream.status_code >= 400:
            return JSONResponse(_EMPTY_RESIDENCY)
        return JSONResponse(_decode_json_body(upstream))

    # ---------------------------------------------------------------- tools
    #
    # The tool loop lives in the page — it is what streams the answer — but
    # the tools run here: a browser cannot fetch an arbitrary origin, and none
    # of these providers send the CORS headers that would let it.

    @app.get("/api/tools")
    async def list_tools() -> JSONResponse:
        """Every tool this server can run, as OpenAI tool definitions.

        The page sends the enabled subset back on the chat request, so this
        is the source of truth for the schemas the model is shown.
        """
        return JSONResponse(
            {
                "tools": tools.DEFINITIONS,
                "approval_required": sorted(tools.APPROVAL_REQUIRED),
            }
        )

    @app.post("/api/tools/call")
    async def call_tool(request: Request):
        try:
            payload = await request.json()
        except (ValueError, json.JSONDecodeError):
            return _json_error(400, "request body was not valid JSON", "invalid_json")
        if not isinstance(payload, dict):
            return _json_error(
                400, "request body must be a JSON object", "invalid_json"
            )

        name = payload.get("name")
        if not isinstance(name, str) or not name:
            return _json_error(400, "`name` must be a non-empty string", "invalid_body")

        arguments = payload.get("arguments")
        if not isinstance(arguments, str):
            return _json_error(
                400, "`arguments` must be a JSON-encoded string", "invalid_body"
            )

        # What the model was actually shown this round. The load-bearing gate:
        # leaving a tool out of the request body does not stop a malformed
        # model emitting a call for it.
        advertised = payload.get("advertised")
        if not isinstance(advertised, list) or not all(
            isinstance(item, str) for item in advertised
        ):
            return _json_error(
                400, "`advertised` must be an array of tool names", "invalid_body"
            )

        origins = payload.get("approved_origins") or []
        if not isinstance(origins, list) or not all(
            isinstance(item, str) for item in origins
        ):
            return _json_error(
                400, "`approved_origins` must be an array of strings", "invalid_body"
            )

        result = await tools.run_tool(
            app.state.http,
            name=name,
            arguments=arguments,
            advertised=set(advertised),
            approved_origins=set(origins),
        )
        return JSONResponse(result.to_dict())

    # ----------------------------------------------------------- connectors
    #
    # MCP servers are programs on this Mac that expose tools. The engine
    # spawns and validates them; this owns the config file it reads
    # (``~/.config/rapid-mlx/mcp.json``) and relays the engine's read-only
    # view of what actually connected.
    #
    # ``--mcp-config`` is read ONCE at spawn, so arming connectors on a
    # running child is impossible — hence ``needs_restart`` below and the
    # restart route, rather than a silent no-op.

    async def _engine_mcp(path: str, *, method: str = "GET") -> dict | None:
        """One engine MCP read, or None when it could not be reached.

        None rather than an exception: an unreachable engine is most of a
        model switch, and the panel still has a config to render.
        """
        engine = config.engine
        base_url = engine.base_url
        if base_url is None:
            return None
        try:
            if method == "POST":
                upstream = await proxy.proxy_post_query(
                    app.state.http,
                    base_url=base_url,
                    path=path,
                    api_key=engine.api_key,
                    timeout=30.0,
                )
            else:
                upstream = await proxy.proxy_get(
                    app.state.http,
                    base_url=base_url,
                    path=path,
                    api_key=engine.api_key,
                    timeout=15.0,
                )
        except httpx.HTTPError:
            return None
        # 404 is an engine predating these routes, not a failure to report.
        if upstream.status_code >= 400:
            return None
        body = _decode_json_body(upstream)
        return body if isinstance(body, dict) else None

    def _connector_snapshot(servers_body: dict | None, tools_body: dict | None) -> dict:
        """The whole panel's state, composed from config plus engine truth.

        One response rather than three: every field the panel renders is
        derived from the same instant, and a rail that fetched them
        separately would show a server as connected beside a tool list that
        no longer contains its tools.
        """
        store = config.connectors
        engine_servers = []
        subsystem_error = None
        configured = False
        if servers_body is not None:
            raw = servers_body.get("servers")
            engine_servers = raw if isinstance(raw, list) else []
            subsystem_error = servers_body.get("error")
            configured = servers_body.get("configured") is True

        # Only tools whose namespaced name is a legal OpenAI function name.
        # Advertising one that would 400 on the wire reads as "that tool
        # silently does nothing"; not advertising it is honest.
        engine_tools = []
        if tools_body is not None:
            raw_tools = tools_body.get("tools")
            for tool in raw_tools if isinstance(raw_tools, list) else []:
                name = tool.get("name") if isinstance(tool, dict) else None
                if isinstance(name, str) and _is_legal_function_name(name):
                    engine_tools.append(
                        {
                            "name": name,
                            "description": tool.get("description") or "",
                            "server": tool.get("server") or "",
                            "parameters": tool.get("parameters"),
                        }
                    )

        snapshot = config.engine.status()
        # Connectors are on, there is something to connect, a child is
        # running, and that child reports no MCP config — which is exactly
        # what a spawn that predates the master switch looks like. Derived
        # every time rather than recorded, so it cannot survive the condition
        # it describes.
        needs_restart = (
            store.is_enabled
            and any(server.enabled for server in store.servers)
            and snapshot.state is ChildState.READY
            and not configured
        )

        return {
            "enabled": store.is_enabled,
            "servers": [server.to_dict() for server in store.servers],
            "load_error": store.load_error,
            "config_path": str(store.path),
            "engine_servers": engine_servers,
            "engine_reachable": servers_body is not None,
            "subsystem_error": subsystem_error,
            "configured": configured,
            "needs_restart": needs_restart,
            "engine_running": snapshot.state is ChildState.READY,
            "tools": engine_tools,
            "disabled_tools": sorted(store.disabled_tools),
            "granted_tools": sorted(store.granted_tools),
            "auto_approve_all": store.auto_approve_all,
        }

    async def _connector_state() -> JSONResponse:
        store = config.connectors
        # The panel is where a user comes to ask "did it work?", so read the
        # file back rather than trusting the copy in memory: it is
        # hand-editable and other tools on this Mac read it too.
        store.reload_from_disk()
        store.reconcile_grants()
        servers_body = None
        tools_body = None
        if store.is_enabled:
            servers_body = await _engine_mcp("/v1/mcp/servers")
            tools_body = await _engine_mcp("/v1/mcp/tools")
        return JSONResponse(_connector_snapshot(servers_body, tools_body))

    @app.get("/api/connectors")
    async def get_connectors() -> JSONResponse:
        return await _connector_state()

    @app.post("/api/connectors/settings")
    async def set_connector_settings(request: Request):
        """The switches: master, auto-approve, per-tool, grant reset."""
        payload = await _json_object(request)
        if isinstance(payload, JSONResponse):
            return payload
        store = config.connectors

        try:
            if isinstance(payload.get("enabled"), bool):
                store.set_enabled(payload["enabled"])
            if isinstance(payload.get("auto_approve_all"), bool):
                store.set_auto_approve_all(payload["auto_approve_all"])
            tool = payload.get("tool")
            if isinstance(tool, str) and isinstance(payload.get("tool_enabled"), bool):
                store.set_tool_enabled(tool, payload["tool_enabled"])
            if isinstance(tool, str) and payload.get("grant") is True:
                store.grant_tool(tool)
            if payload.get("reset_grants") is True:
                store.reset_grants()
        except ConnectorError as exc:
            return _json_error(500, str(exc), "connector_write_failed")

        return await _connector_state()

    @app.post("/api/connectors/servers")
    async def upsert_connector(request: Request):
        """Add or edit one server.

        The command reaches a file the ENGINE spawns from, and the engine's
        allowlist (``mcp/security.py``: npx/uv/python/…) plus its argument
        and environment scrubbing are the gate. Nothing here second-guesses
        it, because a second, different allowlist would only disagree with
        the one that actually runs.
        """
        payload = await _json_object(request)
        if isinstance(payload, JSONResponse):
            return payload

        replacing = payload.get("replacing")
        if replacing is not None and not isinstance(replacing, str):
            return _json_error(400, "`replacing` must be a string", "invalid_body")

        try:
            server = connectors.server_from_payload(payload.get("server"))
            reconfigured = config.connectors.upsert(server, replacing=replacing)
        except ConnectorError as exc:
            return _json_error(400, str(exc), "invalid_connector")

        # Only after the write is durable: a failed save must not strand a
        # connector with its consent deleted.
        if reconfigured and isinstance(replacing, str):
            config.connectors.revoke_grants_for_server(replacing)

        await _apply_connector_change()
        return await _connector_state()

    @app.post("/api/connectors/servers/remove")
    async def remove_connector(request: Request):
        payload = await _json_object(request)
        if isinstance(payload, JSONResponse):
            return payload
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            return _json_error(400, "`name` must be a non-empty string", "invalid_body")

        try:
            config.connectors.remove(name)
        except ConnectorError as exc:
            return _json_error(404, str(exc), "unknown_connector")

        config.connectors.revoke_grants_for_server(name)
        await _apply_connector_change()
        return await _connector_state()

    @app.post("/api/connectors/servers/enabled")
    async def set_connector_enabled(request: Request):
        payload = await _json_object(request)
        if isinstance(payload, JSONResponse):
            return payload
        name = payload.get("name")
        enabled = payload.get("enabled")
        if not isinstance(name, str) or not isinstance(enabled, bool):
            return _json_error(400, "`name` and `enabled` are required", "invalid_body")

        try:
            config.connectors.set_server_enabled(name, enabled)
        except ConnectorError as exc:
            return _json_error(404, str(exc), "unknown_connector")

        await _apply_connector_change()
        return await _connector_state()

    async def _apply_connector_change() -> None:
        """Push a config edit into the running child, best effort.

        The engine's reload route re-reads the file and rebuilds every
        connection, which is what makes an edit take effect without a model
        restart. When it cannot — no child, or a child spawned without
        ``--mcp-config`` — the snapshot's ``configured`` stays false and
        ``needs_restart`` raises the banner instead. Nothing is recorded, so
        the banner cannot outlive the condition.
        """
        if config.engine.base_url is None:
            return
        await _engine_mcp("/v1/mcp/reload", method="POST")

    @app.post("/api/connectors/restart")
    async def restart_for_connectors():
        """Respawn the current model so the child gets ``--mcp-config``.

        A real button rather than an instruction: telling the user to go find
        the model picker and cycle it themselves is asking them to do the
        app's job.
        """
        engine = config.engine
        if not engine.can_switch:
            return _json_error(
                409, "this server cannot restart the engine", "switching_disabled"
            )
        snapshot = engine.status()
        alias = snapshot.model
        if alias is None:
            return _json_error(409, "no model is loaded", "no_model")

        started = lifecycle.start_transition(
            kind="connector-restart",
            model=alias,
            work=lambda: _boot_alias(config, alias),
        )
        if started is TransitionStart.BUSY_ACTIVITY:
            return _json_error(
                409,
                "a response is still running; stop it before restarting",
                "engine_busy",
            )
        if started is TransitionStart.BUSY_TRANSITION:
            return _json_error(
                409,
                "the engine is already changing models",
                "busy_loading",
            )
        return JSONResponse({"restarting": True, "model": alias})

    @app.post("/api/connectors/execute")
    async def execute_connector_tool(request: Request):
        """Run one MCP tool through the engine.

        The consent gate is in the PAGE — by the time a call reaches here the
        user has approved it. This still refuses a tool the user switched off,
        because the switch has to hold against a malformed model emitting the
        name anyway, exactly as ``/api/tools/call`` checks ``advertised``.
        """
        payload = await _json_object(request)
        if isinstance(payload, JSONResponse):
            return payload

        name = payload.get("name")
        if not isinstance(name, str) or not name:
            return _json_error(400, "`name` must be a non-empty string", "invalid_body")
        arguments = payload.get("arguments")
        if arguments is not None and not isinstance(arguments, str):
            return _json_error(
                400, "`arguments` must be a JSON-encoded string", "invalid_body"
            )

        store = config.connectors
        if not store.is_enabled:
            return _json_error(409, "connectors are turned off", "connectors_disabled")
        if name in store.disabled_tools:
            return _json_error(409, f"tool '{name}' is turned off", "tool_disabled")

        engine = config.engine
        base_url = engine.base_url
        if base_url is None:
            snapshot = engine.status()
            return _json_error(
                503,
                _unavailable_message(snapshot.state, snapshot.detail),
                "engine_unavailable",
            )

        # An empty string means a no-arg tool, which is legal.
        text = (arguments or "").strip()
        try:
            parsed = json.loads(text) if text else {}
        except (ValueError, json.JSONDecodeError):
            parsed = None
        if not isinstance(parsed, dict):
            return JSONResponse(
                {
                    "content": f"tool '{name}' error: arguments must be a JSON object",
                    "is_error": True,
                }
            )

        try:
            upstream = await app.state.http.post(
                f"{base_url.rstrip('/')}/v1/mcp/execute",
                json={"tool_name": name, "arguments": parsed},
                headers=proxy.upstream_headers(engine.api_key),
                # A connector tool can legitimately take a while (a query, a
                # fetch); the engine applies the per-server timeout from the
                # config file.
                timeout=httpx.Timeout(connect=10.0, read=180.0, write=60.0, pool=10.0),
            )
        except httpx.HTTPError as exc:
            return _json_error(502, f"the engine did not answer: {exc}", "engine_error")

        if upstream.status_code >= 400:
            return _json_error(
                upstream.status_code,
                _describe_connector_failure(upstream),
                "tool_failed",
            )

        body = _decode_json_body(upstream)
        body = body if isinstance(body, dict) else {}
        return JSONResponse(
            {
                "content": _flatten_tool_content(body),
                "is_error": body.get("is_error") is True,
            }
        )

    # ---------------------------------------------------------------- audio
    #
    # The audio lane rides on WHATEVER model the engine is serving: the
    # child is spawned with ``--enable-audio``, and the engine's gate
    # short-circuits on that flag before it looks at the model. So speech
    # works while a chat model is loaded, and no model switch is needed.

    @app.get("/api/audio/voices")
    async def audio_voices(model: str = ""):
        engine = config.engine
        base_url = engine.base_url
        if base_url is None:
            snapshot = engine.status()
            return _json_error(
                503,
                _unavailable_message(snapshot.state, snapshot.detail),
                "engine_unavailable",
            )
        try:
            upstream = await proxy.proxy_get(
                app.state.http,
                base_url=base_url,
                path="/v1/audio/voices",
                api_key=engine.api_key,
                params={"model": model} if model else None,
                # The first call loads the TTS registry, not the weights,
                # but a cold import is still slower than a status poll.
                timeout=60.0,
            )
        except httpx.HTTPError as exc:
            return _json_error(
                502, f"connection to the engine failed: {exc}", "engine_transport"
            )
        return JSONResponse(
            status_code=upstream.status_code, content=_decode_json_body(upstream)
        )

    @app.post("/api/audio/speech")
    async def audio_speech(request: Request):
        """Synthesise speech, answering with the audio bytes.

        Counted as a stream: a cold Kokoro request measured 47 s, and a
        model switch mid-synthesis would kill the engine doing it.
        """
        try:
            payload = await request.json()
        except (ValueError, json.JSONDecodeError):
            return _json_error(400, "request body was not valid JSON", "invalid_json")
        if not isinstance(payload, dict):
            return _json_error(
                400, "request body must be a JSON object", "invalid_json"
            )

        lease = lifecycle.acquire_activity()
        if lease is None:
            return _transition_unavailable()

        engine = config.engine
        base_url = engine.base_url
        if base_url is None:
            lease.release()
            snapshot = engine.status()
            return _json_error(
                503,
                _unavailable_message(snapshot.state, snapshot.detail),
                "engine_unavailable",
            )

        with lease:
            try:
                upstream = await proxy.proxy_audio_json(
                    app.state.http,
                    base_url=base_url,
                    path="/v1/audio/speech",
                    payload=payload,
                    api_key=engine.api_key,
                )
            except httpx.HTTPError as exc:
                return _json_error(
                    502, f"connection to the engine failed: {exc}", "engine_transport"
                )

        # A failure is JSON; a success is audio. Branch on the status, not
        # on the content type, so an engine that mislabels still surfaces
        # its error rather than handing the page unplayable bytes.
        if upstream.status_code >= 400:
            return JSONResponse(
                status_code=upstream.status_code, content=_decode_json_body(upstream)
            )
        return Response(
            content=upstream.content,
            media_type=upstream.headers.get("content-type", "audio/wav"),
            headers=proxy.filtered_response_headers(upstream.headers),
        )

    @app.post("/api/audio/transcriptions")
    async def audio_transcriptions(request: Request):
        """Transcribe a bounded multipart upload without Base64 expansion."""
        if not auth.content_type_is_multipart(request.headers.get("content-type")):
            return _json_error(
                415, "audio must use multipart upload", "unsupported_media_type"
            )
        async with request.form(max_files=1, max_fields=8) as form:
            audio = form.get("file")
            if not isinstance(audio, UploadFile):
                return _json_error(400, "`file` must be an audio file", "invalid_body")
            read = await _read_upload(audio, MAX_AUDIO_BYTES)
            if isinstance(read, JSONResponse):
                return read
            content = read
            filename = _upload_filename(audio.filename)
            metadata = {key: form.get(key) for key in ("model", "language", "context")}

        lease = lifecycle.acquire_activity()
        if lease is None:
            return _transition_unavailable()

        engine = config.engine
        base_url = engine.base_url
        if base_url is None:
            lease.release()
            snapshot = engine.status()
            return _json_error(
                503,
                _unavailable_message(snapshot.state, snapshot.detail),
                "engine_unavailable",
            )

        fields = {"response_format": "json"}
        for key in ("model", "language", "context"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                fields[key] = value

        with lease:
            try:
                upstream = await proxy.proxy_multipart(
                    app.state.http,
                    base_url=base_url,
                    path="/v1/audio/transcriptions",
                    api_key=engine.api_key,
                    # Advisory: the engine spools to a ``.wav`` temp file and
                    # decodes the CONTAINER, so a name cannot make an
                    # undecodable upload readable. The page transcodes to WAV
                    # before sending — libsndfile reads neither mp4 nor webm.
                    filename=filename,
                    content=content,
                    fields=fields,
                )
            except httpx.HTTPError as exc:
                return _json_error(
                    502, f"connection to the engine failed: {exc}", "engine_transport"
                )

        return JSONResponse(
            status_code=upstream.status_code, content=_decode_json_body(upstream)
        )

    # After the API routes: a mount matches on prefix and swallows
    # everything beneath it. check_dir=False so a checkout that never ran the
    # frontend build still starts.
    app.mount(
        "/static/assets",
        _HashedAssets(directory=ASSETS_DIR, check_dir=False),
        name="assets",
    )

    return app


def _decode_json_body(response: httpx.Response) -> dict:
    try:
        return response.json()
    except ValueError:
        return {
            "error": {
                "message": response.text[:400],
                "type": "engine_malformed_response",
            }
        }


async def _json_object(request: Request) -> dict | JSONResponse:
    """The request body as a dict, or the error response to return instead."""
    try:
        payload = await request.json()
    except (ValueError, json.JSONDecodeError):
        return _json_error(400, "request body was not valid JSON", "invalid_json")
    if not isinstance(payload, dict):
        return _json_error(400, "request body must be a JSON object", "invalid_json")
    return payload


def _unavailable_message(state: ChildState, detail: str | None) -> str:
    if state is ChildState.STARTING:
        return "the model is still loading; retry shortly"
    if state is ChildState.FAILED:
        return f"the engine failed to start: {detail or 'unknown error'}"
    return "no model is loaded"


_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _is_legal_function_name(name: str) -> bool:
    """Whether a namespaced ``server__tool`` can travel as a function name.

    A connector names its own tools, so nothing bounds the composite's length
    or characters. Advertising one the model cannot emit — or that 400s on the
    wire — reads as "that tool silently does nothing".
    """
    return bool(_FUNCTION_NAME_RE.match(name))


def _flatten_tool_content(body: dict) -> str:
    """The engine's tool result as the string the model reads.

    An empty body from a SUCCESSFUL call is not an error, but handing the
    model "" invites it to invent the answer — say plainly that nothing came
    back.
    """
    if body.get("is_error") and isinstance(body.get("error_message"), str):
        return body["error_message"]
    content = body.get("content")
    if isinstance(content, str):
        return content or "(the tool returned no content)"
    if content is None:
        return "(the tool returned no content)"
    return json.dumps(content, sort_keys=True)


def _describe_connector_failure(response: httpx.Response) -> str:
    """The engine's own reason for refusing a tool call.

    Its 503 says "MCP not configured. Start server with --mcp-config", which
    is operator language for a state a phone user reaches without ever seeing
    a command line — so that one is replaced. Everything else is passed
    through: the sandbox's refusals name the pattern that blocked the tool,
    which nothing composed here would know.
    """
    if response.status_code == 503:
        return (
            "the running model was started without connectors — restart it "
            "from Settings → Connectors"
        )
    body = _decode_json_body(response)
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, str):
            return detail
        if isinstance(detail, dict):
            error = detail.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return error["message"]
            if isinstance(error, str):
                return error
    return f"the engine returned HTTP {response.status_code}"


async def _boot_alias(config: WebConfig, alias: str) -> None:
    """Respawn the child for ``alias``. Detached, so failure is recorded in
    the supervisor rather than raised at a caller that has already answered."""
    with contextlib.suppress(SupervisorError):
        await config.engine.start(alias)


def _apply_security_headers(response) -> None:
    """Headers applied to every response.

    ``'unsafe-inline'`` stays on style-src: Radix's scroll lock injects a
    ``<style>`` tag at runtime and is silently ignored without it.

    ``media-src`` must name ``blob:`` explicitly. Synthesised speech is
    handed to ``<audio>`` as an object URL, and without its own directive
    ``media-src`` falls back to ``default-src 'self'`` — which does not
    cover ``blob:``. The element then fails with ``MediaError`` code 4 and
    a player stuck at ``0:00 / 0:00``, while the identical URL still
    downloads fine (a download is not governed by a fetch directive), so
    the bytes look correct and the fault appears to be in the audio.
    """
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "frame-ancestors 'none'; "
        "base-uri 'none'",
    )
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
