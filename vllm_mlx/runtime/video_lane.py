# SPDX-License-Identifier: Apache-2.0
"""MLX-native LTX-2.3 video-generation lane."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import threading
from pathlib import Path


class VideoRuntimeError(RuntimeError):
    """Safe, actionable generation error suitable for the public API."""


_PROCESS_GENERATION_LOCK = threading.Lock()


def require_video_runtime_or_exit() -> None:
    """Fail before model download when the optional video stack is absent."""
    missing = []
    if importlib.util.find_spec("mlx_video") is None:
        missing.append("the `rapid-mlx[video]` Python extra")
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg (`brew install ffmpeg`)")
    if missing:
        print(
            "\n  Error: LTX-2.3 video generation requires "
            + " and ".join(missing)
            + ".\n",
            file=sys.stderr,
        )
        raise SystemExit(2)


class VideoEngine:
    """Thin adapter over ``mlx-video-with-audio``'s LTX-2.3 pipeline.

    The upstream function owns model loading and generation. Rapid-MLX owns
    request validation, job lifecycle, concurrency and the OpenAI-compatible
    transport surface.
    """

    is_video_gen = True
    _loaded = True

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # Shared across engine instances and app lifespans. A bounded shutdown
        # may detach an old daemon worker; an in-process restart must not run a
        # second Metal graph concurrently with that still-draining worker.
        self._generation_lock = _PROCESS_GENERATION_LOCK

    def generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
        image: Path | None,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise VideoRuntimeError(
                "LTX-2.3 video generation requires ffmpeg. "
                "Install it with `brew install ffmpeg`."
            )
        try:
            from mlx_video import generate_video_with_audio
        except ImportError as exc:
            raise VideoRuntimeError(
                "LTX-2.3 support is not installed. "
                "Run `pip install 'rapid-mlx[video]'`."
            ) from exc

        # The 22B pipeline is not re-entrant and a second concurrent graph can
        # exhaust unified memory. Serialize jobs per served model.
        with self._generation_lock:
            generate_video_with_audio(
                model_repo=self.model_name,
                text_encoder_repo=None,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                fps=fps,
                output_path=str(output_path),
                image=str(image) if image is not None else None,
                verbose=False,
                enhance_prompt=False,
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise VideoRuntimeError(
                "LTX-2.3 generation completed without an MP4 output."
            )
        requested_width = output_width or width
        requested_height = output_height or height
        if (requested_width, requested_height) != (width, height):
            cropped = output_path.with_name(f"{output_path.stem}.cropped.mp4")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(output_path),
                        "-vf",
                        (
                            f"crop={requested_width}:{requested_height}:"
                            "(iw-ow)/2:(ih-oh)/2"
                        ),
                        "-c:v",
                        "libx264",
                        "-c:a",
                        "copy",
                        str(cropped),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                )
                cropped.replace(output_path)
            except (
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as exc:
                raise VideoRuntimeError(
                    "LTX-2.3 generated video but could not crop it to the "
                    "requested OpenAI-compatible size."
                ) from exc
            finally:
                cropped.unlink(missing_ok=True)

    def generate_warmup(self) -> None:
        """Video weights load lazily; startup must not trigger a 40+ GB pull."""

    async def stop(self) -> None:
        """No persistent worker is owned by this thin adapter."""
