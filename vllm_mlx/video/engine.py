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


class InvalidVideoRequestError(ValueError):
    """A request a BACKEND rejects for a model-specific, caller-fixable reason.

    Exists so the route can tell "your request violates this model's
    constraints" apart from "something broke inside the backend". A bare
    ``except ValueError`` around a generate call cannot: corrupt weights, an
    incompatible LoRA and a scheduler fault all raise ValueError too, and
    reporting those as ``400 invalid_request`` sends the caller off to fix a
    request that was fine.

    Subclasses ``ValueError`` so a backend that raises it is still correct
    under the Protocol's documented contract.
    """


class VideoBackendUnavailableError(RuntimeError):
    """A backend is configured but cannot run — an OPERATOR-fixable fault.

    Missing runtime dependency, a model directory that doesn't exist, a
    checkpoint that won't load. Distinct from :class:`NotImplementedError`
    ("no backend configured", a 501) because "your install or config is
    wrong" is a different instruction than "this server has no video
    support", and distinct from :class:`InvalidVideoRequestError` because
    the caller can do nothing about it. The route maps this to
    ``503 video_backend_unavailable``.
    """


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
#: doesn't re-probe the environment on every request. Only set after an
#: attempt that did NOT raise, so a failure retries rather than latching.
_AUTOREGISTER_DONE = False

#: Why the last auto-registration attempt failed, if it did. Surfaced as a
#: 503 rather than being swallowed into a misleading 501.
_AUTOREGISTER_ERROR: BaseException | None = None


def _autoregister() -> None:
    """Give known backends a chance to claim the lane, once.

    Called from :func:`resolve_video_engine` rather than at import time so
    that merely importing ``vllm_mlx.video`` never touches the
    environment or pulls a heavy optional dependency — the route module is
    mounted unconditionally in ``server.py``, including on text-only
    servers that will never serve a video request.

    A backend that isn't configured returns ``False`` and leaves the lane
    unclaimed, which keeps the contract-only 501 as the default answer.

    On failure the attempt is NOT marked done and the reason is retained:
    latching "done" before success would turn a transient import error into
    a permanent, misleading 501 for the life of the process, and swallowing
    the reason would hide a configured-but-broken backend behind "no video
    support". The retained error makes the next request retry and, if it
    fails again, report it as unavailable (503) instead.
    """
    global _AUTOREGISTER_DONE, _AUTOREGISTER_ERROR
    if _AUTOREGISTER_DONE:
        return
    try:
        from .wan import register as _register_wan

        claimed = _register_wan()
    except Exception as e:  # noqa: BLE001 — never let a backend break the route
        logger.exception("Video backend auto-registration failed")
        _AUTOREGISTER_ERROR = e
        return
    _AUTOREGISTER_ERROR = None
    _AUTOREGISTER_DONE = True
    if claimed:
        logger.info("Video lane: Wan backend registered")


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
    if _VIDEO_ENGINE_FACTORY is None and _AUTOREGISTER_ERROR is not None:
        # A backend tried to claim the lane and blew up. That is an
        # operator-fixable install/config fault, not "no video support",
        # so it must not masquerade as the contract-only 501.
        raise ImportError(
            f"video backend failed to initialise: {_AUTOREGISTER_ERROR}"
        ) from _AUTOREGISTER_ERROR
    if _VIDEO_ENGINE_FACTORY is None:
        raise NotImplementedError(
            "no video backend configured. Set $RAPID_MLX_WAN_MODEL_DIR to a "
            "converted MLX Wan 2.1/2.2 checkpoint to serve this route; see "
            "docs/content_farm_api.md"
        )
    return _VIDEO_ENGINE_FACTORY(model)
