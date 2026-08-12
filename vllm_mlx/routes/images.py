# SPDX-License-Identifier: Apache-2.0
"""Image generation endpoints (OpenAI ``/v1/images/*`` compatible)."""

import base64
import io
import logging
import math
import os
import secrets
import tempfile
import time

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from starlette.responses import JSONResponse

from ..api.models import ImageGenerationRequest, parse_image_size
from ._async_utils import run_to_completion

logger = logging.getLogger(__name__)

router = APIRouter()

# Cap the uploaded init image so a single edit request can't buffer an
# unbounded body into memory before the size validators run.
_MAX_EDIT_IMAGE_BYTES = 25 * 1024 * 1024
_MAX_EDIT_IMAGE_DIMENSION = 8192
_MAX_EDIT_IMAGE_PIXELS = 40_000_000
# Whole-request cap for the middleware (init image + a little multipart framing
# slack), enforced BEFORE FastAPI spools the body to disk.
_IMAGE_EDIT_REQUEST_BYTES = _MAX_EDIT_IMAGE_BYTES + 1024 * 1024
_OVERSIZE_DETAIL = "image edit request body exceeds the 25 MB limit"


class _ImageBodyTooLargeError(Exception):
    """Signals the bounded receive tripped the body cap mid-stream."""


class ImageBodyLimitMiddleware:
    """Cap /v1/images/edits multipart bodies before FastAPI spools them.

    FastAPI resolves the ``UploadFile``/``Form`` params by parsing (and
    spooling) the whole multipart body before the endpoint runs, so an
    in-handler size check is too late to stop an oversized upload from
    consuming disk. This ASGI middleware rejects on the advertised
    Content-Length and, absent that, caps the streamed bytes — mirroring the
    video lane's ``VideoBodyLimitMiddleware``.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/v1/images/edits"
        ):
            return await self.app(scope, receive, send)

        headers = {name.lower(): value for name, value in scope.get("headers", ())}
        advertised = headers.get(b"content-length")
        if advertised is not None:
            try:
                advertised_bytes = int(advertised.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                advertised_bytes = None
            if (
                advertised_bytes is not None
                and advertised_bytes > _IMAGE_EDIT_REQUEST_BYTES
            ):
                response = JSONResponse(
                    status_code=413, content={"detail": _OVERSIZE_DETAIL}
                )
                return await response(scope, receive, send)

        total = 0
        limit_tripped = False
        replacement_sent = False
        downstream_started = False
        downstream_completed = False

        async def bounded_receive():
            nonlocal total, limit_tripped
            message = await receive()
            if message.get("type") == "http.request":
                total += len(message.get("body", b"") or b"")
                if total > _IMAGE_EDIT_REQUEST_BYTES:
                    limit_tripped = True
                    raise _ImageBodyTooLargeError
            return message

        async def guarded_send(message):
            nonlocal replacement_sent, downstream_started, downstream_completed
            if limit_tripped:
                if not downstream_started and not replacement_sent:
                    response = JSONResponse(
                        status_code=413, content={"detail": _OVERSIZE_DETAIL}
                    )
                    await response(scope, receive, send)
                    replacement_sent = True
                return
            message_type = message.get("type")
            if message_type == "http.response.start":
                downstream_started = True
            elif message_type == "http.response.body" and not message.get(
                "more_body", False
            ):
                downstream_completed = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, guarded_send)
        except _ImageBodyTooLargeError:
            if replacement_sent or downstream_completed:
                return
            if downstream_started:
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )
                return
            response = JSONResponse(
                status_code=413, content={"detail": _OVERSIZE_DETAIL}
            )
            await response(scope, receive, send)


def install_image_body_limit_middleware(app) -> None:
    app.add_middleware(ImageBodyLimitMiddleware)


# Compatibility fallback for edit engines that do not advertise a family-aware
# default. Current built-in FLUX.2 Klein advertises four distilled steps.
_DEFAULT_EDIT_STEPS = 20


def _supports_generation(img_engine) -> bool:
    """Capability probe with compatibility for older/fake image engines."""
    return bool(
        getattr(
            img_engine,
            "supports_generation",
            not getattr(img_engine, "is_edit", False),
        )
    )


def _supports_editing(img_engine) -> bool:
    """Capability probe with compatibility for older/fake image engines."""
    return bool(
        getattr(img_engine, "supports_editing", getattr(img_engine, "is_edit", False))
    )


def _image_engine(model_name: str = ""):
    """Resolve an image engine exactly by request model, with single-model fallback."""
    from ..config import get_config

    cfg = get_config()
    registry = getattr(cfg, "model_registry", None)
    img_engine = None

    if model_name and registry:
        try:
            registry.validate_model_name(model_name)
            img_engine = registry.get_engine(model_name)
        except KeyError:
            img_engine = None
    elif model_name:
        accepted = {
            getattr(cfg, "model_name", None),
            getattr(cfg, "model_alias", None),
            getattr(cfg, "model_path", None),
        }
        if model_name in accepted:
            img_engine = getattr(cfg, "engine", None)
    elif registry:
        image_entries = [
            entry
            for entry in registry.list_entries()
            if getattr(entry.engine, "is_image_gen", False)
        ]
        # Backward compatibility for clients that omit model: it remains
        # unambiguous when exactly one image model is resident.
        if len(image_entries) == 1:
            img_engine = image_entries[0].engine
    else:
        img_engine = getattr(cfg, "engine", None)

    if img_engine is None or not getattr(img_engine, "is_image_gen", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "message": (
                        "This server is not running an image model. Start it with "
                        "`rapid-mlx serve flux-schnell` (or another image alias)."
                    ),
                    "type": "invalid_request_error",
                    "code": "image_model_not_loaded",
                    "param": "model",
                }
            },
        )
    manager = getattr(cfg, "residency_manager", None)
    if manager is not None:
        manager.touch(model_name or getattr(img_engine, "model_name", None))
    return img_engine


def _generate_one(img_engine, request: ImageGenerationRequest, seed: int) -> bytes:
    """Blocking single-image render — runs off the event loop."""
    width, height = request.dimensions()
    # Step count is family-aware: a distilled model (Klein/schnell, 4 steps)
    # would waste wall-clock at 20 and a non-distilled one (Qwen, 20) would be
    # noise at 4. The engine advertises the right default per family.
    default_steps = getattr(img_engine, "default_steps", 4)
    return img_engine.generate(
        prompt=request.prompt,
        width=width,
        height=height,
        num_inference_steps=request.steps
        if request.steps is not None
        else default_steps,
        seed=seed,
        # None → each model uses its own trained default guidance (Klein is
        # guidance-distilled; forcing 4.0 washes it out).
        guidance=request.guidance,
        negative_prompt=request.negative_prompt,
    )


@router.post("/v1/images/generations")
async def create_image(request: ImageGenerationRequest = Body(...)):
    """Generate one or more images from a text prompt.

    Returns the OpenAI ``{created, data:[{b64_json}]}`` envelope. ``url``
    responses are not offered by the local lane (there is no object store to
    host the bytes) — callers must request ``b64_json``.
    """
    from ..image.engine import ImageRuntimeError

    img_engine = _image_engine(request.model)

    # When the selected resident engine is an instruction-edit model,
    # text-to-image generation is the wrong endpoint. Point the caller to
    # /v1/images/edits instead of silently ignoring the mismatch.
    if not _supports_generation(img_engine):
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "message": (
                        "This server is running an image-edit model; use "
                        "/v1/images/edits with an input image."
                    ),
                    "type": "invalid_request_error",
                    "code": "wrong_image_endpoint",
                    "param": "model",
                }
            },
        )

    if request.response_format == "url":
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "The local image lane only returns base64 data; request "
                        "response_format='b64_json'."
                    ),
                    "type": "invalid_request_error",
                    "code": "unsupported_response_format",
                    "param": "response_format",
                }
            },
        )

    # When no seed is pinned, draw a random one so successive calls vary — even
    # two unseeded requests within the same wall-clock second. Multi-image
    # (``n``) requests offset per index off that base.
    base_seed = request.seed if request.seed is not None else secrets.randbelow(2**31)

    from ..image.engine import ImageGenerationCancelled

    data = []
    cancelled = False
    for index in range(request.n):
        try:
            png_bytes = await run_to_completion(
                _generate_one, img_engine, request, (base_seed + index) & 0x7FFFFFFF
            )
        except ImageGenerationCancelled:
            # User stopped mid-render: return whatever finished rather than an
            # error, so a cancelled multi-image batch keeps its earlier images.
            cancelled = True
            break
        except ImageRuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": str(exc),
                        "type": "image_generation_error",
                        "code": "image_generation_failed",
                    }
                },
            ) from exc
        data.append({"b64_json": base64.b64encode(png_bytes).decode("ascii")})

    return {"created": int(time.time()), "data": data, "cancelled": cancelled}


@router.get("/v1/images/progress")
async def image_progress(model: str = ""):
    """Live denoise progress for the single in-flight render.

    Diffusion has a fixed step count, so this is a *true* ``step / total``
    signal the client polls to drive a determinate progress bar and ETA — no
    streaming parser, and honest on slow hardware (the bar can't outrun the
    real steps). Single-flight: the server renders one image at a time.
    """
    img_engine = _image_engine(model)
    return img_engine.progress_snapshot()


@router.post("/v1/images/cancel")
async def image_cancel(model: str = ""):
    """Ask the in-flight render to stop at the next denoise step.

    By design the image lane is **single-flight**: the engine's process lock
    serialises one render at a time, so progress and cancel are intentionally
    process-global — they refer to the one active render. This is a local,
    single-user server (the desktop app issues one generation at a time), so
    there is deliberately no per-request generation id to thread through.
    """
    img_engine = _image_engine(model)
    img_engine.request_cancel()
    return {"ok": True}


def _reject(condition: bool, message: str, param: str) -> None:
    """Raise a 400 invalid_request_error when ``condition`` holds."""
    if condition:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": param,
                }
            },
        )


def _validate_edit_image(raw: bytes) -> None:
    """Reject unsupported or excessive decoded dimensions without rasterizing."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(raw)) as source:
            width, height = source.size
            image_format = (source.format or "").upper()
    except Image.DecompressionBombError as exc:
        raise HTTPException(
            status_code=413,
            detail={
                "error": {
                    "message": "image dimensions exceed the 8192 px / 40 megapixel limit",
                    "type": "invalid_request_error",
                    "param": "image",
                }
            },
        ) from exc
    except (OSError, UnidentifiedImageError) as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "image must be a readable PNG or JPEG",
                    "type": "invalid_request_error",
                    "param": "image",
                }
            },
        ) from exc
    if image_format not in {"PNG", "JPEG"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "image must be a PNG or JPEG",
                    "type": "invalid_request_error",
                    "param": "image",
                }
            },
        )
    if (
        width <= 0
        or height <= 0
        or width > _MAX_EDIT_IMAGE_DIMENSION
        or height > _MAX_EDIT_IMAGE_DIMENSION
        or width * height > _MAX_EDIT_IMAGE_PIXELS
    ):
        raise HTTPException(
            status_code=413,
            detail={
                "error": {
                    "message": "image dimensions exceed the 8192 px / 40 megapixel limit",
                    "type": "invalid_request_error",
                    "param": "image",
                }
            },
        )


def _generate_edit_one(
    img_engine, prompt, steps, seed, guidance, negative_prompt, image_path
) -> bytes:
    """Blocking single instruction-edit render — runs off the event loop.

    No width/height is threaded through the API. Each edit engine chooses its
    compatible default: FLUX.2 uses 1024×1024, while Qwen derives a canvas from
    the input image. The request ``size`` is accepted for OpenAI compatibility
    but deliberately not honored.
    """
    return img_engine.generate(
        prompt=prompt,
        num_inference_steps=steps
        if steps is not None
        else getattr(img_engine, "default_edit_steps", _DEFAULT_EDIT_STEPS),
        seed=seed,
        guidance=guidance
        if guidance is not None
        else getattr(img_engine, "default_edit_guidance", 4.0),
        negative_prompt=negative_prompt,
        image_paths=[image_path],
    )


@router.post("/v1/images/edits")
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form(""),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("b64_json"),
    seed: int | None = Form(None),
    steps: int | None = Form(None),
    guidance: float | None = Form(None),
    negative_prompt: str | None = Form(None),
):
    """Instruction-edit an input image (OpenAI ``/v1/images/edits`` compatible).

    Requires a server running an edit-capable image model (e.g.
    ``rapid-mlx serve flux2-klein-4b``); the uploaded image plus the prompt
    drive a global instruction edit (no mask). Returns the same
    ``{created, data:[{b64_json}]}`` envelope as generations.
    """
    from ..image.engine import ImageGenerationCancelled, ImageRuntimeError

    img_engine = _image_engine(model)

    # /v1/images/edits requires the edit family; a txt2img server points the
    # caller at /v1/images/generations instead of silently ignoring the image.
    if not _supports_editing(img_engine):
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "message": (
                        "This server is running a text-to-image model; use "
                        "/v1/images/generations, or start an image-edit model "
                        "(e.g. `rapid-mlx serve flux2-klein-4b`)."
                    ),
                    "type": "invalid_request_error",
                    "code": "wrong_image_endpoint",
                    "param": "model",
                }
            },
        )

    if not prompt or not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "prompt must not be empty",
                    "type": "invalid_request_error",
                    "param": "prompt",
                }
            },
        )
    if response_format != "b64_json":
        # Reject any non-b64_json value (not just "url"), matching the validated
        # generations contract — the local lane has no object store for URLs.
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "The local image lane only returns base64 "
                    "data; request response_format='b64_json'.",
                    "type": "invalid_request_error",
                    "code": "unsupported_response_format",
                    "param": "response_format",
                }
            },
        )
    if not 1 <= n <= 4:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "n must be between 1 and 4",
                    "type": "invalid_request_error",
                    "param": "n",
                }
            },
        )
    # ``size`` is accepted for OpenAI-API compatibility but edit backends own
    # their compatible canvas sizing. Validate malformed input, then discard it.
    try:
        parse_image_size(size)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "param": "size",
                }
            },
        ) from exc

    # A raw multipart form bypasses the validated bounds ``ImageGenerationRequest``
    # enforces on the JSON path, so a negative seed, non-finite guidance, or an
    # enormous step count could otherwise reach — and monopolize/crash — the
    # inference server. Validate the numeric knobs to the same bounds here.
    # 1..50 matches the JSON generations contract (ImageGenerationRequest:
    # ge=1, le=50) so an edit can't run for twice the documented maximum.
    _reject(
        steps is not None and not (1 <= steps <= 50),
        "steps must be between 1 and 50",
        "steps",
    )
    _reject(
        guidance is not None
        and not (math.isfinite(guidance) and 0.0 <= guidance <= 20.0),
        "guidance must be a finite number between 0 and 20",
        "guidance",
    )
    _reject(
        seed is not None and not (0 <= seed <= 0x7FFFFFFF),
        "seed must be between 0 and 2147483647",
        "seed",
    )

    # Bounded, streaming read: abort as soon as the cumulative size exceeds the
    # cap rather than materializing the whole (possibly huge) upload first —
    # otherwise an unauthenticated oversized multipart request exhausts memory.
    raw = bytearray()
    while True:
        chunk = await image.read(1024 * 1024)
        if not chunk:
            break
        raw.extend(chunk)
        if len(raw) > _MAX_EDIT_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": {
                        "message": f"image exceeds "
                        f"{_MAX_EDIT_IMAGE_BYTES // (1024 * 1024)} MB limit",
                        "type": "invalid_request_error",
                        "param": "image",
                    }
                },
            )
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "image file is empty",
                    "type": "invalid_request_error",
                    "param": "image",
                }
            },
        )
    raw = bytes(raw)
    _validate_edit_image(raw)

    # A fixed suffix — never the attacker-controlled upload filename, whose
    # length/bytes could otherwise raise an uncaught OSError from the temp-file
    # layer. mflux/PIL sniff the real format from content, not the extension.
    base_seed = seed if seed is not None else secrets.randbelow(2**31)
    data = []
    cancelled = False
    # One temp file for the whole request; the process lock in the img_engine keeps
    # generations serial, so a shared init image is safe across the n renders.
    # Creation + write live inside the try so a partial-write / close failure
    # can't leak the file — ``tmp_path`` is captured before the write, and the
    # finally unlinks whatever path was created on every exit.
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(raw)
        for index in range(n):
            try:
                png_bytes = await run_to_completion(
                    _generate_edit_one,
                    img_engine,
                    prompt,
                    steps,
                    (base_seed + index) & 0x7FFFFFFF,
                    guidance,
                    negative_prompt,
                    tmp_path,
                )
            except ImageGenerationCancelled:
                # A mid-render cancel is a success, not a 500: return whatever
                # finished (matching the generations envelope), never an error.
                cancelled = True
                break
            except ImageRuntimeError as exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": {
                            "message": str(exc),
                            "type": "image_generation_error",
                            "code": "image_generation_failed",
                        }
                    },
                ) from exc
            data.append({"b64_json": base64.b64encode(png_bytes).decode("ascii")})
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                # Don't fail the response on cleanup, but don't swallow it
                # silently either — a leaked upload in the temp dir is worth a
                # log line (the basename only, not the full path).
                logger.warning(
                    "failed to remove temp edit image %s: %s",
                    os.path.basename(tmp_path),
                    exc,
                )

    return {"created": int(time.time()), "data": data, "cancelled": cancelled}
