# SPDX-License-Identifier: Apache-2.0
"""MLX-native LTX, CogVideoX-Fun and Wan video-generation lane."""

from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Literal


class VideoRuntimeError(RuntimeError):
    """Safe, actionable generation error suitable for the public API."""


_PROCESS_GENERATION_LOCK = threading.Lock()

_FFMPEG_FALLBACK_PATHS = (
    Path("/opt/homebrew/bin/ffmpeg"),
    Path("/usr/local/bin/ffmpeg"),
    Path("/usr/bin/ffmpeg"),
)


def _resolve_imageio_ffmpeg() -> str | None:
    """Return the executable bundled by the declared video extra, if usable."""
    try:
        import imageio_ffmpeg

        candidate = Path(imageio_ffmpeg.get_ffmpeg_exe()).expanduser()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.absolute())
    return None


def _resolve_ffmpeg() -> str | None:
    """Resolve the ffmpeg binary consistently across the video lane."""
    override = os.environ.get("FFMPEG_BINARY", "").strip()
    if override:
        has_path_separator = os.sep in override or bool(
            os.altsep and os.altsep in override
        )
        if not has_path_separator:
            resolved_override = shutil.which(override)
            return (
                str(Path(resolved_override).absolute()) if resolved_override else None
            )
        override_path = Path(override).expanduser()
        if override_path.is_file() and os.access(override_path, os.X_OK):
            # Keep symlinks valid while preventing a relative override such as
            # ``./ffmpeg`` from becoming the PATH-searched argv[0] ``ffmpeg``.
            return str(override_path.absolute())
        return None
    resolved = shutil.which("ffmpeg")
    if resolved:
        return str(Path(resolved).absolute())
    for candidate in _FFMPEG_FALLBACK_PATHS:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return _resolve_imageio_ffmpeg()


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


def _is_ltx25_name(model_name: str | None) -> bool:
    from ..video.ltx25 import is_ltx25_model

    return is_ltx25_model(model_name)


def _submodule_spec_exists_without_import(parent: str, child: str) -> bool:
    """Find a child module without executing the parent's ``__init__``."""

    parent_spec = importlib.util.find_spec(parent)
    if parent_spec is None or parent_spec.submodule_search_locations is None:
        return False
    return (
        PathFinder.find_spec(
            f"{parent}.{child}", parent_spec.submodule_search_locations
        )
        is not None
    )


# The mlx-video runtime ships in the base install (Python 3.11+ on macOS), so
# its absence means a damaged environment; the install hint repairs it.
_VIDEO_RUNTIME_LABEL = "the mlx-video runtime"


def _default_video_runtime_requirements(model_name: str | None) -> list[str]:
    """Pure dependency probe for the default/Wan mlx-video runtime."""

    mlx_video_available = importlib.util.find_spec("mlx_video") is not None
    wan_available = not _is_wan_name(model_name) or (
        mlx_video_available
        and _submodule_spec_exists_without_import("mlx_video", "generate_wan")
    )
    if not mlx_video_available or not wan_available:
        return [_VIDEO_RUNTIME_LABEL]
    return []


def registered_wan_runtime_issue(model_name: str) -> str | None:
    """Return an actionable, side-effect-free Wan benchmark runtime problem."""

    if sys.version_info < (3, 11):
        return (
            "video generation requires Python 3.11 or newer "
            f"(current: {sys.version_info.major}.{sys.version_info.minor}). "
            "Rapid-MLX core still supports Python 3.10, but the upstream "
            "mlx-video runtime does not."
        )
    missing = _default_video_runtime_requirements(model_name)
    if _resolve_ffmpeg() is None:
        missing.append("ffmpeg (`brew install ffmpeg`)")
    if missing:
        return "video generation requires " + " and ".join(missing) + "."
    return None


@dataclass(frozen=True)
class VideoRuntimeIssue:
    """Structured, side-effect-free result of the video runtime preflight."""

    detail: str
    install_hint: str
    status: Literal["absent", "broken", "incompatible"]
    marker_reason: str


_LTX25_FFMPEG_HINT = (
    "Install ffmpeg so LTX-2.5 can encode its MP4 output: "
    "`brew install ffmpeg`, then see docs/guides/video-generation.md."
)


def _ltx25_runtime_issue(*, provision: bool) -> VideoRuntimeIssue | None:
    """Shared LTX-2.5 preflight used by serve, pull and the model catalog."""
    from ..video import ltx25
    from .base_runtime import python_upgrade_hint
    from .optional_runtime import optional_extra_install_hint

    runtime = ltx25.ltx25_runtime()
    if runtime.state == "python_unsupported":
        return VideoRuntimeIssue(
            detail="\n  Error: " + ltx25.python_unsupported_detail() + "\n",
            install_hint=python_upgrade_hint(),
            status="incompatible",
            marker_reason="python_version_unsupported",
        )

    missing: list[str] = []
    install_hint: str | None = None
    if runtime.state in {"absent", "corrupt"}:
        if runtime.reason == "shadowed":
            missing.append(
                "the LTX-2.5 runtime shipped with rapid-mlx "
                "(a different copy shadows it)"
            )
            detail = (runtime.detail or "").rstrip(".")
            install_hint = (
                "Remove the conflicting LTX-2.5 copy from PYTHONPATH or run "
                "rapid-mlx from a different working directory"
                + (f" ({detail})" if detail else "")
                + "; see docs/guides/video-generation.md."
            )
        else:
            missing.append("the LTX-2.5 runtime shipped with rapid-mlx")
            install_hint = optional_extra_install_hint("video", status="broken")
    elif runtime.state == "override_invalid":
        missing.append("the pinned LTX-2.5 runtime selected by RAPID_MLX_LTX25_RUNTIME")
        install_hint = (
            "Fix or unset RAPID_MLX_LTX25_RUNTIME: it must point at an "
            "executable ltx-2-mlx checkout at the pinned revision "
            f"({ltx25.LTX25_RUNTIME_COMMIT}). See the LTX-2.5 setup in "
            "docs/guides/video-generation.md."
        )
    elif runtime.source == "override":
        assert runtime.executable is not None
        if shutil.which("uv") is None:
            missing.append("uv (`brew install uv`)")
            install_hint = (
                "Install uv to provision the pinned LTX-2.5 checkout: "
                "`brew install uv`, then see docs/guides/video-generation.md."
            )
        elif provision:
            try:
                ltx25.prepare_ltx25_runtime(runtime.executable)
            except ltx25.LTX25BackendError as exc:
                # The checkout already resolved — re-cloning is not the
                # fix. Surface the underlying provisioning failure.
                missing.append(
                    f"a provisioned pinned LTX-2.5 runtime ({exc}; "
                    "see docs/guides/video-generation.md)"
                )
                install_hint = (
                    "Fix the pinned LTX-2.5 checkout or unset "
                    "RAPID_MLX_LTX25_RUNTIME; see the LTX-2.5 setup in "
                    "docs/guides/video-generation.md."
                )
    if _resolve_ffmpeg() is None:
        missing.append("ffmpeg (`brew install ffmpeg`)")
        if install_hint is None:
            install_hint = _LTX25_FFMPEG_HINT
    if not missing:
        return None
    assert install_hint is not None
    detail = "\n  Error: video generation requires " + " and ".join(missing) + ".\n"
    return VideoRuntimeIssue(
        detail=detail,
        install_hint=install_hint,
        status="broken",
        marker_reason="runtime_dependency_missing",
    )


def video_runtime_issue(
    model_name: str | None, *, provision: bool = True
) -> VideoRuntimeIssue | None:
    """Return the actionable video runtime problem for *model_name*.

    Side-effect free when ``provision`` is false: the model catalog uses that
    form so listing aliases never installs anything. ``serve`` and ``pull``
    provision an explicit LTX-2.5 override before they touch model weights.
    """
    from .base_runtime import python_upgrade_hint
    from .optional_runtime import optional_extra_install_hint

    if _is_ltx25_name(model_name):
        return _ltx25_runtime_issue(provision=provision)

    if sys.version_info < (3, 11):
        return VideoRuntimeIssue(
            detail=(
                "\n  Error: video generation requires Python 3.11 or newer "
                f"(current: {sys.version_info.major}.{sys.version_info.minor}). "
                "Rapid-MLX core still supports Python 3.10, but the upstream "
                "mlx-video runtime does not.\n"
            ),
            install_hint=python_upgrade_hint(),
            status="incompatible",
            marker_reason="python_version_unsupported",
        )

    missing: list[str] = []
    if _is_cogvideox_name(model_name):
        cogvideox_modules = {
            "videox_fun_mlx": "the bundled VideoX-Fun-mlx runtime",
            "mlx_arsenal": "mlx-arsenal",
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
        missing.extend(_default_video_runtime_requirements(model_name))
    if _resolve_ffmpeg() is None:
        missing.append("ffmpeg (`brew install ffmpeg`)")
    if not missing:
        return None
    reason = (
        "runtime_extra_missing"
        if _VIDEO_RUNTIME_LABEL in missing
        else "runtime_dependency_missing"
    )
    status: Literal["absent", "broken"] = (
        "absent" if reason == "runtime_extra_missing" else "broken"
    )
    return VideoRuntimeIssue(
        detail="\n  Error: video generation requires " + " and ".join(missing) + ".\n",
        install_hint=optional_extra_install_hint("video", status=status),
        status=status,
        marker_reason=reason,
    )


def video_alias_runtime_ready(model_name: str | None) -> bool:
    """Whether this video alias can run here without installing anything."""
    return video_runtime_issue(model_name, provision=False) is None


def require_video_runtime_or_exit(model_name: str | None = None) -> None:
    """Fail before model download when the optional video stack is absent."""
    from .optional_runtime import OptionalRuntimeMissing

    issue = video_runtime_issue(model_name, provision=True)
    if issue is None:
        return
    raise OptionalRuntimeMissing(
        extra="video",
        install_hint=issue.install_hint,
        detail=issue.detail,
        status=issue.status,
        marker_reason=issue.marker_reason,
    )


class VideoEngine:
    """Adapter over the supported MLX video runtimes.

    The upstream function owns model loading and generation. Rapid-MLX owns
    request validation, job lifecycle, concurrency and the OpenAI-compatible
    transport surface.
    """

    is_video_gen = True
    is_mllm = False
    _loaded = True

    def get_stats(self) -> dict:
        """Route-facing engine surface (mirrors ``BaseEngine.get_stats``).

        ``/health`` calls ``engine.get_stats()`` unconditionally; without this
        the video lane raised ``AttributeError`` and answered 500 for its whole
        lifetime — the same shape as the image lane in issue #1776. The
        route-engine contract bans hasattr-guarding the call, so the method
        lives here.
        """
        return {"engine_type": "video"}

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.video_family = (
            "cogvideox-fun"
            if _is_cogvideox_name(model_name)
            else (
                "wan"
                if _is_wan_name(model_name)
                else ("ltx-2.5" if _is_ltx25_name(model_name) else "ltx-2.3")
            )
        )
        self._cog_engine = None
        self._wan_engine = None
        self._ltx25_engine = None
        if self.video_family == "cogvideox-fun":
            from ..video.engine import VideoGenerationEngine

            self._cog_engine = VideoGenerationEngine(model_name)
        elif self.video_family == "wan":
            from ..video.wan import WanVideoEngine

            self._wan_engine = WanVideoEngine(model_name)
        elif self.video_family == "ltx-2.5":
            from ..video.ltx25 import LTX25VideoEngine

            self._ltx25_engine = LTX25VideoEngine(model_name)
        # Shared across engine instances and app lifespans. A bounded shutdown
        # may detach an old daemon worker; an in-process restart must not run a
        # second Metal graph concurrently with that still-draining worker.
        self._generation_lock = _PROCESS_GENERATION_LOCK

    def _emit_model_served(self) -> None:
        from ..server import _emit_primary_model_served_once

        _emit_primary_model_served_once(self)

    def extend(
        self,
        *,
        prompt: str,
        source_video: Path,
        output_path: Path,
        extend_frames: int,
        seed: int,
    ) -> None:
        """Append video using the LTX-2.5 runtime's temporal conditioning."""
        if self._ltx25_engine is None:
            raise VideoRuntimeError("Video extension requires LTX-2.5.")
        from ..video.ltx25 import LTX25BackendError

        try:
            with self._generation_lock:
                self._ltx25_engine.extend(
                    prompt=prompt,
                    source_video=source_video,
                    output_path=output_path,
                    extend_frames=extend_frames,
                    seed=seed,
                    on_loaded=self._emit_model_served,
                )
        except LTX25BackendError as exc:
            raise VideoRuntimeError(str(exc)) from exc

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
        if getattr(self, "_ltx25_engine", None) is not None:
            from ..video.ltx25 import LTX25BackendError

            if negative_prompt is not None or guidance_scale is not None:
                raise VideoRuntimeError(
                    "LTX-2.5 distilled generation does not support negative_prompt "
                    "or guidance_scale."
                )
            if image is None and conditioning_strength is not None:
                raise VideoRuntimeError(
                    "LTX-2.5 conditioning_strength requires an input image."
                )
            try:
                with self._generation_lock:
                    self._ltx25_engine.generate(
                        prompt=prompt,
                        output_path=output_path,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        fps=fps,
                        seed=seed,
                        image=image,
                        conditioning_strength=conditioning_strength,
                        on_loaded=self._emit_model_served,
                    )
            except LTX25BackendError as exc:
                raise VideoRuntimeError(str(exc)) from exc
            self._crop_generated_output(
                output_path=output_path,
                width=width,
                height=height,
                output_width=output_width,
                output_height=output_height,
                family="LTX-2.5",
            )
            return
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
                        on_loaded=self._emit_model_served,
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
                    on_loaded=self._emit_model_served,
                )
            return
        if _resolve_ffmpeg() is None:
            raise VideoRuntimeError(
                "LTX-2.3 video generation requires ffmpeg. "
                "Install it with `brew install ffmpeg`."
            )
        try:
            from mlx_video import generate_video_with_audio
        except ImportError as exc:
            from .optional_runtime import optional_extra_install_hint

            raise VideoRuntimeError(
                "LTX-2.3 support is not installed. "
                + optional_extra_install_hint("video")
            ) from exc

        # The 22B pipeline is not re-entrant and a second concurrent graph can
        # exhaust unified memory. Serialize jobs per served model.
        with self._generation_lock:
            generation_kwargs: dict[str, object] = {
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
                # The public response is video-only. Avoid decoding and
                # muxing a silent audio track only to strip it again.
                "no_audio": True,
            }
            if negative_prompt is not None:
                generation_kwargs["negative_prompt"] = negative_prompt
            if guidance_scale is not None:
                generation_kwargs["cfg_scale"] = guidance_scale
            if conditioning_strength is not None:
                generation_kwargs["image_strength"] = conditioning_strength
            supports_load_callback = (
                "on_loaded" in inspect.signature(generate_video_with_audio).parameters
            )
            if supports_load_callback:
                generation_kwargs["on_loaded"] = self._emit_model_served
            generate_video_with_audio(**generation_kwargs)
            if not supports_load_callback:
                # The pinned mlx-video-with-audio 0.1.36 export is a
                # ``(*args, **kwargs)`` wrapper without an on_loaded hook, so
                # its honest compatibility signal is post-generation. Future
                # runtimes with an explicit hook signal before inference.
                self._emit_model_served()
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
            ffmpeg = _resolve_ffmpeg()
            if ffmpeg is None:
                raise OSError("ffmpeg not found")
            subprocess.run(
                [
                    ffmpeg,
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
                ffmpeg = _resolve_ffmpeg()
                if ffmpeg is None:
                    raise OSError("ffmpeg not found")
                subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-i",
                        str(output_path),
                        "-vf",
                        (
                            f"crop={requested_width}:{requested_height}:"
                            "(iw-ow)/2:(ih-oh)/2"
                        ),
                        "-c:v",
                        "h264_videotoolbox",
                        "-pix_fmt",
                        "yuv420p",
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
        if getattr(self, "_ltx25_engine", None) is not None:
            import asyncio

            await asyncio.to_thread(self._ltx25_engine.stop)
        if self._cog_engine is not None:
            await self._cog_engine.close()
