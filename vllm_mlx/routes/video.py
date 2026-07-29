# SPDX-License-Identifier: Apache-2.0
"""Video-generation endpoint (content-farm lane — CONTRACT-ONLY).

``POST /v1/video/generations`` ships the full text→video / image→video
request+response CONTRACT and wires to the
:class:`vllm_mlx.video.engine.VideoEngine` interface. No concrete
backend exists yet, so :func:`vllm_mlx.video.engine.resolve_video_engine`
raises :class:`NotImplementedError` and the route returns a clean HTTP
501 with an OpenAI-shape envelope. The day an LTX-2.3 backend registers
itself the route goes live unchanged.
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

logger = logging.getLogger(__name__)

router = APIRouter()

#: Largest clip the wired handler will inline as base64. Encoding costs
#: 4/3 in size and holds the raw bytes plus the encoded string at once, so
#: this bounds peak memory per request. A backend that renders longer clips
#: should upload to object storage and populate ``url`` instead of
#: ``b64_video``.
MAX_INLINE_VIDEO_BYTES: int = 256 * 1024 * 1024


@router.post("/v1/video/generations", dependencies=[Depends(verify_api_key)])
async def create_video(request: VideoGenerationRequest = Body(...)):
    """Generate a video from text (t2v) or an image + text (i2v).

    CONTRACT-ONLY: the request/response schema and the ``VideoEngine``
    interface are fixed, but no backend is integrated yet. The route
    validates the request (so callers can develop against the real wire
    contract) and then returns HTTP 501 ``not_implemented`` because
    :func:`vllm_mlx.video.engine.resolve_video_engine` has no backend to
    hand back.

    A future LTX-2.3 backend implementing
    :class:`vllm_mlx.video.engine.VideoEngine` makes this route go live
    with no change here — the resolver stops raising and the engine call
    below runs.
    """
    # Lazy import mirrors the audio routes — keeps the video lane's deps
    # (none yet) off the module-import hot path.
    from ..video.engine import resolve_video_engine

    try:
        engine = resolve_video_engine(request.model)
    except NotImplementedError as e:
        # CONTRACT-ONLY state: surface a clean 501 with the OpenAI-shape
        # envelope so colleagues get the exact message + the pointer to
        # the backend-integration task, not a stack trace.
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

    # Reached only once a backend is registered. Kept wired so the route
    # is live the moment resolve_video_engine stops raising — no handler
    # edit required.
    return await _render_and_serialize(engine, request)  # pragma: no cover


async def _render_and_serialize(  # pragma: no cover — no backend yet
    engine, request: VideoGenerationRequest
) -> VideoGenerationResponse:
    """Render via ``engine`` and serialize to the wire response.

    Unreachable while the lane is contract-only (``resolve_video_engine``
    raises first), but kept correct and wired so registering a backend is
    the ONLY change needed to make ``/v1/video/generations`` live.

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
    # Whatever the backend actually wrote — it may return a path other
    # than the one we handed it. Tracked so cleanup deletes the real
    # artifact instead of leaking a multi-MB mp4 on every request.
    written_path: str | None = None
    try:
        # Rendering is heavy blocking compute — keep it off the event loop
        # so concurrent requests and health probes stay responsive.
        written = await asyncio.to_thread(
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
        video_path = str(written or out_path)
        written_path = video_path

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
        def _read_and_encode() -> str:
            with open(video_path, "rb") as fh:
                return base64.b64encode(fh.read()).decode("ascii")

        b64 = await asyncio.to_thread(_read_and_encode)

        return VideoGenerationResponse(
            created=int(time.time()),
            model=request.model,
            data=[
                VideoGenerationResult(
                    b64_video=b64,
                    format=request.response_format,
                    width=request.width,
                    height=request.height,
                    num_frames=request.num_frames,
                    frame_rate=request.frame_rate,
                )
            ],
        )
    finally:
        # Delete the artifact the backend actually produced, wherever it
        # put it, then the whole temp dir. ``rmtree`` rather than
        # ``rmdir`` because a backend may drop sidecars (a probe log, a
        # separate audio track) next to the clip, and a non-empty dir
        # would make ``rmdir`` raise and leak the entire directory.
        try:
            if written_path and os.path.exists(written_path):
                os.unlink(written_path)
        except OSError as cleanup_err:
            logger.warning(
                "Failed to unlink video output %s: %s", written_path, cleanup_err
            )
        shutil.rmtree(out_dir, ignore_errors=True)
