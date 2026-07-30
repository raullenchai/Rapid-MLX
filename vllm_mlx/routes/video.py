# SPDX-License-Identifier: Apache-2.0
"""Video-generation endpoint (content-farm lane).

``POST /v1/video/generations`` — text→video and image→video, served
through whatever :class:`vllm_mlx.video.engine.VideoEngine` backend is
configured. Wan 2.1 / 2.2 is the built-in backend (see
:mod:`vllm_mlx.video.wan`); the route itself depends only on the
Protocol, so it never changes when a backend is added or swapped.

With no backend configured the route still validates the full request and
answers HTTP 501, which keeps the wire contract discoverable and
developable-against on a text-only server.
"""

import asyncio
import base64
import logging
import os
import shutil
import tempfile
import time

from fastapi import APIRouter, Body, Depends, HTTPException

from ..api.models import (
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationResult,
)
from ..middleware.auth import verify_api_key
from ..video.engine import InvalidVideoRequestError, VideoBackendUnavailableError
from ._async_utils import run_to_completion

logger = logging.getLogger(__name__)

router = APIRouter()

#: Largest clip the wired handler will inline as base64. Encoding costs
#: 4/3 in size and holds the raw bytes plus the encoded string at once, so
#: this bounds peak memory per request. A backend that renders longer clips
#: should upload to object storage and populate ``url`` instead of
#: ``b64_video``.
MAX_INLINE_VIDEO_BYTES: int = 256 * 1024 * 1024

#: Sentinel distinguishing "the backend does not declare this optional
#: attribute" from "it declares it as None". Both are meaningful and they
#: mean different things for ``frame_rate`` reporting, so a plain
#: ``getattr(..., None)`` would conflate them.
_NOT_DECLARED = object()

#: Serialises rendering, for the same reason the audio lanes do it: a
#: video model is the heaviest thing this server can be asked to run, and
#: concurrent renders would multiply peak unified memory. On the event
#: loop (not a ``threading.Lock`` in the worker) so queued requests cost a
#: coroutine rather than pinning shared-executor threads — see
#: :mod:`vllm_mlx.routes._async_utils`.
#:
#: Lazily created: the module imports before any event loop exists, and an
#: ``asyncio.Lock`` must bind to the running loop.
_render_lock: asyncio.Lock | None = None


def _get_render_lock() -> asyncio.Lock:
    """Return the process-wide video render lock, creating it on first use."""
    global _render_lock
    if _render_lock is None:
        _render_lock = asyncio.Lock()
    return _render_lock


def _resolve_inside(out_dir: str, candidate: str) -> str:
    """Resolve ``candidate`` under ``out_dir``, refusing anything outside.

    A backend returns a path; we must not treat that path as trusted. A
    relative return value is resolved against the directory we handed the
    backend (not the server's cwd), and the result must still live beneath
    that directory. ``..`` traversal therefore can't get an arbitrary file
    read, base64'd into a response — containment is checked BEFORE we open
    the file, not only before we delete it.
    """
    resolved = (
        candidate if os.path.isabs(candidate) else os.path.join(out_dir, candidate)
    )
    real_dir = os.path.realpath(out_dir)
    real_path = os.path.realpath(resolved)
    if os.path.commonpath([real_dir, real_path]) != real_dir:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": (
                        "Video backend returned an output path outside its "
                        "working directory"
                    ),
                    "type": "api_error",
                    "code": "video_generation_failed",
                    "param": None,
                }
            },
        )
    return real_path


@router.post("/v1/video/generations", dependencies=[Depends(verify_api_key)])
async def create_video(request: VideoGenerationRequest = Body(...)):
    """Generate a video from text (t2v) or an image + text (i2v).

    Status codes worth knowing apart:

    * **501** ``video_backend_not_implemented`` — no backend configured.
      The request was still fully validated, so this is also the contract
      check clients develop against.
    * **503** ``video_backend_unavailable`` — a backend IS configured but
      its runtime dependency is missing (or is the wrong package). The
      message carries the install command.
    * **400** ``invalid_video_request`` — a model-specific constraint the
      generic schema can't express, e.g. Wan's ``num_frames == 4n+1`` or a
      per-checkpoint pixel-area ceiling.
    """
    # Lazy import mirrors the audio routes — keeps the backend's optional
    # dependency off the module-import hot path, which matters because
    # server.py mounts this router unconditionally.
    from ..video.engine import resolve_video_engine

    try:
        engine = resolve_video_engine(request.model)
    except NotImplementedError as e:
        # No backend configured. Surface a clean 501 with the OpenAI-shape
        # envelope so callers get the exact message + how to enable the
        # lane, not a stack trace. The request was still fully validated,
        # so this doubles as a contract check for client development.
        raise HTTPException(
            status_code=501,
            detail={
                "error": {
                    "message": str(e),
                    "type": "not_implemented_error",
                    "code": "video_backend_not_implemented",
                    "param": None,
                }
            },
        )
    except (ImportError, VideoBackendUnavailableError) as e:
        # A backend IS configured but its runtime dependency is missing or
        # is the WRONG package (the `mlx-video` PyPI name belongs to an
        # unrelated project — see vllm_mlx/video/wan.py). That is an
        # operator-fixable install problem, not "no video support", so it
        # gets a 503 carrying the actual install command rather than the
        # 501 above.
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "video_backend_unavailable",
                    "param": None,
                }
            },
        )

    return await _render_and_serialize(engine, request)


async def _render_and_serialize(
    engine, request: VideoGenerationRequest
) -> VideoGenerationResponse:
    """Render via ``engine`` and serialize to the wire response.

    Two things this deliberately does NOT do the naive way:

    * ``out_path`` is a real temp mp4 path. :meth:`VideoEngine.generate`
      declares ``out_path: str | Path`` and writes the clip there —
      passing ``None`` would break every conforming backend.
    * the clip comes back as base64 in ``b64_video``, not as a ``url``
      holding a server-side filesystem path. A local path is not a URL
      the client can fetch, and echoing one leaks the server's
      filesystem layout. A backend that uploads to real object storage
      can populate ``url`` instead.
    """
    out_dir = tempfile.mkdtemp(prefix="rapidmlx-video-")
    out_path = os.path.join(out_dir, "out.mp4")
    try:
        # Rendering is heavy blocking compute — keep it off the event loop
        # so concurrent requests and health probes stay responsive.
        # ``run_to_completion`` (not bare ``to_thread``) so a client
        # disconnect can't send us into ``finally`` and delete the output
        # directory while the worker is still rendering into it.
        # Serialised on the loop before offloading — one render at a time.
        try:
            async with _get_render_lock():
                written = await run_to_completion(
                    lambda: engine.generate(
                        request.prompt,
                        out_path,
                        image=request.image,
                        height=request.height,
                        width=request.width,
                        num_frames=request.num_frames,
                        frame_rate=request.frame_rate,
                        steps=request.steps,
                        negative_prompt=request.negative_prompt,
                        seed=request.seed,
                    )
                )
        except InvalidVideoRequestError as e:
            # ONLY the dedicated type, never a bare ValueError. Backends
            # raise this for caller-fixable requests the generic schema
            # can't catch because the constraint is model-specific (Wan
            # needs num_frames == 4n+1, and enforces a per-checkpoint
            # pixel-area ceiling). Catching plain ValueError here would
            # also swallow corrupt weights, a bad LoRA and scheduler
            # faults, reporting all of them as "your request is invalid".
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "code": "invalid_video_request",
                        "param": None,
                    }
                },
            )
        except VideoBackendUnavailableError as e:
            # The engine re-probes its dependency at call time and raises
            # this on failure, so the 503 reasoning from create_video
            # applies. Deliberately NOT catching bare ImportError here:
            # mlx-video does lazy imports mid-render, and one of those
            # failing is an internal fault whose raw message must not reach
            # the client — that falls through to the sanitized 500 below.
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": str(e),
                        "type": "api_error",
                        "code": "video_backend_unavailable",
                        "param": None,
                    }
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            # Everything else is OUR fault, not the caller's: corrupt
            # weights, an incompatible LoRA, a scheduler fault. Own the
            # envelope here rather than letting it escape to the global
            # handler, so this lane answers in the same shape as the audio
            # lanes — full traceback to the operator log, generic message to
            # the client so we don't leak filesystem or subprocess detail.
            logger.exception("Video generation failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": "Video generation failed",
                        "type": "api_error",
                        "code": "video_generation_failed",
                        "param": None,
                    }
                },
            )
        # Resolve relative against out_dir (not the server's cwd — a
        # backend returning ``Path("out.mp4")`` means the file it wrote
        # into our directory) and refuse anything that escapes it.
        video_path = _resolve_inside(out_dir, str(written or out_path))

        size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        if not size:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": "Video generation produced no output",
                        "type": "api_error",
                        "code": "video_generation_failed",
                        "param": None,
                    }
                },
            )
        # Refuse to inline a clip too large to base64 safely. Encoding
        # inflates by 4/3 and both the bytes and the encoded string are
        # resident at once, so an unbounded render could push the server
        # into swap. A backend producing clips this size should upload to
        # object storage and populate ``url`` instead.
        if size > MAX_INLINE_VIDEO_BYTES:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": (
                            f"Generated video is {size} bytes, over the "
                            f"{MAX_INLINE_VIDEO_BYTES}-byte limit for inline "
                            "base64 delivery"
                        ),
                        "type": "api_error",
                        "code": "video_too_large_to_inline",
                        "param": None,
                    }
                },
            )

        # Read + base64 off the event loop: a multi-MB read and encode on
        # the loop thread stalls every other request for its duration.
        # ``run_to_completion`` again, not bare ``to_thread`` — cancelling
        # here would otherwise drop into ``finally`` and rmtree the file
        # the reader is holding open.
        def _read_and_encode() -> str:
            with open(video_path, "rb") as fh:
                return base64.b64encode(fh.read()).decode("ascii")

        b64 = await run_to_completion(_read_and_encode)

        # Report the clip's REAL playback rate, not the requested one. Some
        # model families (Wan) emit frames at a fixed trained rate and
        # cannot vary fps — echoing back the request would tell the client
        # the clip is something it isn't. A backend that honours arbitrary
        # fps simply doesn't set ``native_frame_rate`` and the request
        # value stands.
        # Three distinct states, and the wire has a representation for each:
        #
        #   * backend reports a rate      -> report it (it's the real one)
        #   * backend has no such concept -> the request value stands, since
        #     that backend honours what it was asked for
        #   * backend HAS the concept but doesn't know for this checkpoint
        #     (a Wan checkpoint with no config.json: 2.1 is 16 fps, 2.2 is
        #     24, and weights don't distinguish them) -> ``null``
        #
        # That last case must not echo ``request.frame_rate``: Wan never
        # forwards it, so reporting 30 for a clip that is actually 16 or 24
        # is a fabricated number, which is the exact failure this reporting
        # exists to prevent.
        native = getattr(engine, "native_frame_rate", _NOT_DECLARED)
        if native is _NOT_DECLARED:
            actual_fps = float(request.frame_rate)
        else:
            actual_fps = float(native) if native is not None else None
        # Same principle for the model echo: report what RAN. The request's
        # ``model`` is a schema default (``ltx-2.3``) that selects nothing —
        # echoing it on a Wan-rendered clip actively misattributes the
        # result. Backends that don't identify themselves fall back to the
        # request value.
        actual_model = str(getattr(engine, "served_model", None) or request.model)
        return VideoGenerationResponse(
            created=int(time.time()),
            model=actual_model,
            data=[
                VideoGenerationResult(
                    b64_video=b64,
                    format=request.response_format,
                    width=request.width,
                    height=request.height,
                    num_frames=request.num_frames,
                    frame_rate=actual_fps,
                )
            ],
        )
    finally:
        # ``rmtree``, not ``rmdir``: a backend may drop sidecars (a probe
        # log, a separate audio track) beside the clip, and a non-empty dir
        # would make ``rmdir`` raise and leak the whole directory.
        #
        # ``written_path`` is already guaranteed to be inside out_dir by
        # ``_resolve_inside``, so the sweep below covers it — no separate
        # unlink needed, and no risk of deleting a backend-owned artifact
        # we never created.
        shutil.rmtree(out_dir, ignore_errors=True)
