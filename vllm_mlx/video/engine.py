# SPDX-License-Identifier: Apache-2.0
"""Text→video / image→video engine interface (content-farm lane).

This is the fourth generation lane alongside ``TTSEngine`` (text→speech),
``STTEngine`` (speech→text) and ``MusicEngine`` (text→music/SFX):
``VideoEngine`` does text→video AND image→video on Apple Silicon.

This module owns only the INTERFACE and the factory. Concrete backends
live beside it — currently :mod:`vllm_mlx.video.wan` (Wan 2.1 / 2.2 via
mlx-video). The ``/v1/video/generations`` route depends on the Protocol
alone, so adding or swapping a backend never touches the route.

The lane stays contract-only (HTTP 501, full request validation) until an
operator configures a backend, which keeps ``/v1/video/generations``
discoverable and developable-against on a text-only server.

See ``docs/content_farm_api.md`` for the wire contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class VideoEngine(Protocol):
    """The surface a text→/image→video backend must implement.

    A backend (e.g. an MLX-native LTX-2.3 port) implements ``generate``
    and is returned by :func:`resolve_video_engine`. The route layer
    depends ONLY on this Protocol, so swapping backends never touches
    ``vllm_mlx.routes.video``.
    """

    def generate(
        self,
        prompt: str,
        out_path: str | Path,
        *,
        image: str | None = None,
        height: int = 704,
        width: int = 1216,
        num_frames: int = 97,
        frame_rate: float = 25.0,
        steps: int | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> Path:
        """Render a clip for ``prompt`` → ``out_path`` (mp4). Returns the path.

        Args:
            prompt: Natural-language description of the video.
            out_path: Output mp4 path (absolute recommended).
            image: Conditioning first frame for image-to-video — a base64
                (optionally ``data:`` URI) string or an ``http(s)`` URL.
                ``None`` selects text-to-video.
            height: Output height in pixels.
            width: Output width in pixels.
            num_frames: Number of frames to render.
            frame_rate: Playback frame rate (fps).
            steps: Denoising steps (``None`` = backend default).
            negative_prompt: CFG negative branch.
            seed: Fixed seed for reproducibility (``None`` = random).

        Returns:
            The output ``Path`` (an mp4, with a native-audio track when
            the backend emits one).
        """
        ...


# Registry hook: a concrete backend assigns this to a callable taking the
# requested ``model`` id and returning a :class:`VideoEngine` (that is the
# signature :func:`resolve_video_engine` invokes it with). Left ``None``
# until a backend claims the lane so the resolver fails loudly.
_VIDEO_ENGINE_FACTORY = None

#: Guards :func:`_autoregister` so a lane with no configured backend
#: doesn't re-probe the environment on every request.
_AUTOREGISTER_DONE = False


def _autoregister() -> None:
    """Give known backends a chance to claim the lane, once.

    Called from :func:`resolve_video_engine` rather than at import time so
    that merely importing ``vllm_mlx.video`` never touches the
    environment or pulls a heavy optional dependency — the route module is
    mounted unconditionally in ``server.py``, including on text-only
    servers that will never serve a video request.

    A backend that isn't configured returns ``False`` and leaves the lane
    unclaimed, which keeps the contract-only 501 as the default answer.
    """
    global _AUTOREGISTER_DONE
    if _AUTOREGISTER_DONE:
        return
    _AUTOREGISTER_DONE = True
    try:
        from .wan import register as _register_wan

        if _register_wan():
            logger.info("Video lane: Wan backend registered")
    except Exception:  # noqa: BLE001 — never let a backend break the route
        logger.exception("Video backend auto-registration failed")


def resolve_video_engine(model: str) -> VideoEngine:
    """Return a :class:`VideoEngine` for ``model`` — or raise if none exists.

    Resolution order:

    1. an explicitly-installed :data:`_VIDEO_ENGINE_FACTORY` (tests, or a
       future backend that registers itself at startup), else
    2. auto-registration of the built-in backends (currently Wan 2.1/2.2
       via mlx-video, active only when the operator has pointed
       ``$RAPID_MLX_WAN_MODEL_DIR`` at a converted checkpoint).

    Raises:
        NotImplementedError: no backend is configured. The route turns
            this into HTTP 501 — the request/response contract is still
            fully validated, so clients can develop against it.
        ImportError: a backend IS configured but its runtime dependency
            is missing or is the wrong package. The route turns this into
            a 503 carrying the install instruction, which is a materially
            different problem from "no backend exists".
    """
    if _VIDEO_ENGINE_FACTORY is None:
        _autoregister()
    if _VIDEO_ENGINE_FACTORY is None:
        raise NotImplementedError(
            "no video backend configured. Set $RAPID_MLX_WAN_MODEL_DIR to a "
            "converted MLX Wan 2.1/2.2 checkpoint to serve this route; see "
            "docs/content_farm_api.md"
        )
    return _VIDEO_ENGINE_FACTORY(model)
