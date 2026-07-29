# SPDX-License-Identifier: Apache-2.0
"""Text→video / image→video engine interface (content-farm lane).

This is the fourth generation lane alongside ``TTSEngine`` (text→speech),
``STTEngine`` (speech→text) and ``MusicEngine`` (text→music/SFX):
``VideoEngine`` does text→video AND image→video, targeted at an
MLX-native LTX-2.3 backend on Apple Silicon.

CONTRACT-ONLY: there is **no** concrete backend wired yet. This module
defines the :class:`VideoEngine` interface a future LTX-2.3
implementation must satisfy and a :func:`resolve_video_engine` factory
that raises :class:`NotImplementedError` until one is registered. The
``/v1/video/generations`` route calls the factory so the day a backend
lands the route goes live with zero handler changes.

See ``docs/content_farm_api.md`` for the wire contract and the
backend-integration task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


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
# while the lane is contract-only so the resolver fails loudly.
_VIDEO_ENGINE_FACTORY = None


def resolve_video_engine(model: str) -> VideoEngine:
    """Return a :class:`VideoEngine` for ``model`` — or raise if none exists.

    CONTRACT-ONLY: no backend is registered yet, so this always raises
    :class:`NotImplementedError`. The route surfaces that as a clean
    HTTP 501 so colleagues see the exact request/response contract and
    the interface to implement LTX-2.3 behind.

    A future backend registers itself by setting
    :data:`_VIDEO_ENGINE_FACTORY`; this resolver then hands the route a
    ready engine with no other route change.
    """
    if _VIDEO_ENGINE_FACTORY is None:
        raise NotImplementedError(
            "video backend not yet integrated; see docs/content_farm_api.md"
        )
    return _VIDEO_ENGINE_FACTORY(model)
