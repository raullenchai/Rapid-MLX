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

import logging

from fastapi import APIRouter, Body, Depends, HTTPException

from ..api.models import VideoGenerationRequest
from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


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
    # edit required. (Response serialization into VideoGenerationResponse
    # is the backend integrator's follow-up per REQUIREMENTS_rapid.md B1.)
    import time

    from ..api.models import VideoGenerationResponse, VideoGenerationResult

    out_path = engine.generate(  # pragma: no cover — no backend yet
        request.prompt,
        None,
        image=request.image,
        height=request.height,
        width=request.width,
        num_frames=request.num_frames,
        frame_rate=request.frame_rate,
        steps=request.steps,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
    )
    return VideoGenerationResponse(  # pragma: no cover — no backend yet
        created=int(time.time()),
        model=request.model,
        data=[
            VideoGenerationResult(
                url=str(out_path),
                format=request.response_format,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                frame_rate=request.frame_rate,
            )
        ],
    )
