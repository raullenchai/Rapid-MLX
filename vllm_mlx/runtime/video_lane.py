# SPDX-License-Identifier: Apache-2.0
"""MLX-native LTX-2.3, CogVideoX-Fun and Wan video-generation lane."""

from __future__ import annotations

import importlib.util
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


class VideoRuntimeError(RuntimeError):
    """Safe, actionable generation error suitable for the public API."""


_PROCESS_GENERATION_LOCK = threading.Lock()


def validate_video_request(
    engine,
    *,
    width: int,
    height: int,
    num_frames: int,
    image: Path | None,
) -> None:
    """Run family-specific validation before a video job is queued."""
    wan_engine = getattr(engine, "_wan_engine", None)
    if wan_engine is not None:
        wan_engine.validate_request(
            width=width,
            height=height,
            num_frames=num_frames,
            image=image,
        )


def _is_cogvideox_name(model_name: str | None) -> bool:
    return bool(model_name and "cogvideox" in model_name.casefold())


def _is_wan_name(model_name: str | None) -> bool:
    if not model_name:
        return False
    from ..video.wan import is_wan_model

    return is_wan_model(model_name)


def require_video_runtime_or_exit(model_name: str | None = None) -> None:
    """Fail before model download when the optional video stack is absent."""
    if sys.version_info < (3, 11):
        print(
            "\n  Error: video generation requires Python 3.11 or newer "
            f"(current: {sys.version_info.major}.{sys.version_info.minor}). "
            "Rapid-MLX core still supports Python 3.10, but the upstream "
            "mlx-video runtime does not.\n",
            file=sys.stderr,
        )
        raise SystemExit(2)

    missing = []
    if _is_cogvideox_name(model_name):
        cogvideox_modules = {
            "videox_fun_mlx": "the bundled VideoX-Fun-mlx runtime",
            "mlx_arsenal": "mlx-arsenal",
            "imageio": "imageio",
            "PIL": "Pillow",
            "numpy": "numpy",
            "huggingface_hub": "huggingface-hub",
        }
        missing.extend(
            label
            for module, label in cogvideox_modules.items()
            if importlib.util.find_spec(module) is None
        )
    else:
        mlx_video_available = importlib.util.find_spec("mlx_video") is not None
        wan_available = not _is_wan_name(model_name) or (
            mlx_video_available
            and importlib.util.find_spec("mlx_video.generate_wan") is not None
        )
        if not mlx_video_available or not wan_available:
            missing.append("the `rapid-mlx[video]` Python extra")
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg (`brew install ffmpeg`)")
    if missing:
        print(
            "\n  Error: video generation requires " + " and ".join(missing) + ".\n",
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
        self.video_family = (
            "cogvideox-fun"
            if _is_cogvideox_name(model_name)
            else ("wan" if _is_wan_name(model_name) else "ltx-2.3")
        )
        self._cog_engine = None
        self._wan_engine = None
        if self.video_family == "cogvideox-fun":
            from ..video.engine import VideoGenerationEngine

            self._cog_engine = VideoGenerationEngine(model_name)
        elif self.video_family == "wan":
            from ..video.wan import WanVideoEngine

            self._wan_engine = WanVideoEngine(model_name)
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
        negative_prompt: str | None = None,
        guidance_scale: float | None = None,
        conditioning_strength: float | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> None:
        if self._wan_engine is not None:
            from ..video.wan import WanBackendError

            try:
                with self._generation_lock:
                    self._wan_engine.generate(
                        prompt=prompt,
                        output_path=output_path,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        seed=seed,
                        image=image,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                    )
            except WanBackendError as exc:
                raise VideoRuntimeError(str(exc)) from exc
            self._crop_generated_output(
                output_path=output_path,
                width=width,
                height=height,
                output_width=output_width,
                output_height=output_height,
                family="Wan",
            )
            return
        if self._cog_engine is not None:
            if image is not None:
                raise VideoRuntimeError(
                    "CogVideoX-Fun MVP currently supports text-to-video only."
                )
            with self._generation_lock:
                self._cog_engine.generate_sync(
                    output_path=output_path,
                    prompt=prompt,
                    width=width,
                    height=height,
                    frames=num_frames,
                    fps=fps,
                    seed=seed,
                    negative_prompt=negative_prompt or "",
                    guidance_scale=(6.0 if guidance_scale is None else guidance_scale),
                )
            return
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
            generation_kwargs = {
                "model_repo": self.model_name,
                "text_encoder_repo": None,
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "seed": seed,
                "fps": fps,
                "output_path": str(output_path),
                "image": str(image) if image is not None else None,
                "verbose": False,
                "enhance_prompt": False,
            }
            if negative_prompt is not None:
                generation_kwargs["negative_prompt"] = negative_prompt
            if guidance_scale is not None:
                generation_kwargs["cfg_scale"] = guidance_scale
            if conditioning_strength is not None:
                generation_kwargs["image_strength"] = conditioning_strength
            generate_video_with_audio(**generation_kwargs)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise VideoRuntimeError(
                "LTX-2.3 generation completed without an MP4 output."
            )
        self._crop_generated_output(
            output_path=output_path,
            width=width,
            height=height,
            output_width=output_width,
            output_height=output_height,
            family="LTX-2.3",
        )
        self._remove_audio_track(output_path)

    @staticmethod
    def _remove_audio_track(output_path: Path) -> None:
        """Remux an audio-less LTX generation as a video-only MP4."""
        video_only: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.stem}.",
                suffix=".video-only.mp4",
                delete=False,
            ) as temporary:
                video_only = Path(temporary.name)
            if video_only is None:  # pragma: no cover - assigned by the context manager
                raise OSError("could not create a video-only temporary file")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(output_path),
                    "-map",
                    "0:v:0",
                    "-c:v",
                    "copy",
                    "-an",
                    str(video_only),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            if not video_only.is_file() or video_only.stat().st_size == 0:
                raise OSError("ffmpeg completed without a video-only MP4")
            video_only.chmod(stat.S_IMODE(output_path.stat().st_mode))
            video_only.replace(output_path)
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise VideoRuntimeError(
                "LTX-2.3 generated video but its silent audio track could not "
                "be removed."
            ) from exc
        finally:
            if video_only is not None:
                try:
                    video_only.unlink(missing_ok=True)
                except OSError:
                    # Cleanup must not mask either a successful atomic replace
                    # or the actionable remux error raised above.
                    pass

    @staticmethod
    def _crop_generated_output(
        *,
        output_path: Path,
        width: int,
        height: int,
        output_width: int | None,
        output_height: int | None,
        family: str,
    ) -> None:
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
                    f"{family} generated video but could not crop it to the "
                    "requested OpenAI-compatible size."
                ) from exc
            finally:
                cropped.unlink(missing_ok=True)

    @property
    def native_fps(self) -> int:
        if self._wan_engine is not None:
            return self._wan_engine.native_fps
        return 5 if self._cog_engine is not None else 24

    def generate_warmup(self) -> None:
        """Video weights load lazily; startup must not trigger a 40+ GB pull."""

    async def stop(self) -> None:
        if self._cog_engine is not None:
            await self._cog_engine.close()
