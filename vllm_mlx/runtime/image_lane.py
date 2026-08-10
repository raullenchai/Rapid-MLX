# SPDX-License-Identifier: Apache-2.0
"""MLX-native text-to-image / image-edit generation lane (mflux backend).

Mirrors the video lane's split: this module is the thin, duck-typed engine
adapter that ``server.load_model`` dispatches to for ``modality=image-gen``
aliases; ``vllm_mlx/image/engine.py`` owns the mflux pipeline and the
``vllm_mlx/routes/images.py`` router owns the OpenAI-compatible transport.
"""

from __future__ import annotations

import importlib.util
import sys

from ..image.engine import (
    ImageGenerationCancelled,
    ImageGenerationEngine,
    ImageRuntimeError,
)

__all__ = [
    "ImageEngine",
    "ImageGenerationCancelled",
    "ImageRuntimeError",
    "require_image_runtime_or_exit",
]


def require_image_runtime_or_exit(model_name: str | None = None) -> None:
    """Fail before model download when the optional image stack is absent."""
    if sys.version_info < (3, 11):
        print(
            "\n  Error: image generation requires Python 3.11 or newer "
            f"(current: {sys.version_info.major}.{sys.version_info.minor}). "
            "Rapid-MLX core still supports Python 3.10, but the mflux runtime "
            "does not.\n",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if importlib.util.find_spec("mflux") is None:
        print(
            "\n  Error: image generation requires the `rapid-mlx[image]` "
            "Python extra (`pip install 'rapid-mlx[image]'`).\n",
            file=sys.stderr,
        )
        raise SystemExit(2)


class ImageEngine:
    """Thin adapter over the mflux image backend.

    Duck-typed like ``VideoEngine`` (``is_image_gen`` / ``_loaded``) so the
    router and ``/v1/models`` probes can recognise the lane. The underlying
    mflux model loads lazily on the first ``generate`` call.
    """

    is_image_gen = True
    is_mllm = False
    _loaded = True

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._engine = ImageGenerationEngine(model_name)

    def get_stats(self) -> dict:
        """Route-facing engine surface (mirrors ``BaseEngine.get_stats``).

        ``/health`` and other probes call ``engine.get_stats()``
        unconditionally; without this the image lane raised ``AttributeError``
        and answered 500 for its whole lifetime (issue #1776). The route-engine
        contract bans hasattr-guarding the call, so the method lives here.
        """
        return {"engine_type": "image"}

    @property
    def is_edit(self) -> bool:
        return self._engine.is_edit

    @property
    def family(self) -> str:
        return self._engine.family

    @property
    def default_steps(self) -> int:
        """Per-family default denoise steps when the request pins none."""
        return self._engine.default_steps

    def request_cancel(self) -> None:
        """Ask the in-flight generation to stop at the next denoise step."""
        self._engine.request_cancel()

    def progress_snapshot(self) -> dict:
        """Live denoise progress for the single in-flight render."""
        return self._engine.progress_snapshot()

    def generate(
        self,
        *,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        seed: int = 0,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        image_paths: list[str] | None = None,
    ) -> bytes:
        """Generate one image; returns PNG bytes. Raises ``ImageRuntimeError``."""
        return self._engine.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
            guidance=guidance,
            negative_prompt=negative_prompt,
            image_paths=image_paths,
        )

    def generate_warmup(self) -> None:
        """Image weights load lazily; startup must not trigger a multi-GB pull."""

    async def stop(self) -> None:
        """Release the backing model reference (mflux holds no async resources)."""
        self._engine._model = None  # noqa: SLF001 — internal drop for restart hygiene
