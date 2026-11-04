"""H3 reference video I/O through the same FFmpeg tools used for probing/audio."""

from __future__ import annotations

import importlib
import json
import math
import shutil
import subprocess
import tempfile
import threading
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

_original_decode_video = None
_DECODE_TIMEOUT = 300


@lru_cache(maxsize=16)
def _supports_videotoolbox(ffmpeg: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return any(
            line.strip() == "videotoolbox" for line in result.stdout.splitlines()
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return False


def _fps(stream):
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            value = float(Fraction(str(stream.get(key, "0/0"))))
            if math.isfinite(value) and value > 0:
                return value
        except (ValueError, ZeroDivisionError):
            pass
    return None


def _read_ppm_frames(stdout):
    """Read FFmpeg P6 images one at a time, including their display dimensions."""
    import mlx.core as mx
    import numpy as np

    while True:
        magic = stdout.readline()
        if not magic:
            return
        if magic.strip() != b"P6":
            raise ValueError("invalid PPM frame header")
        tokens = []
        while len(tokens) < 3:
            line = stdout.readline()
            if not line:
                raise ValueError("incomplete PPM frame header")
            tokens.extend(line.split(b"#", 1)[0].split())
        width, height, maximum = map(int, tokens)
        if width <= 0 or height <= 0 or maximum != 255:
            raise ValueError("invalid PPM frame dimensions or sample depth")
        pixels = stdout.read(width * height * 3)
        if len(pixels) != width * height * 3:
            raise ValueError("incomplete reference-video frame")
        yield mx.array(
            np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3),
            dtype=mx.uint8,
        )


def decode_video_with_ffmpeg(video) -> tuple[mx.array, float | None]:
    """Return all uint8 RGB display frames [T,H,W,3] and container FPS.

    Sampling and resizing remain the responsibility of the pinned processor.
    PPM carries each frame's dimensions after FFmpeg's default autorotation.
    """
    if not isinstance(video, (str, Path)):
        original = _original_decode_video
        if original is None:
            original = importlib.import_module(
                "mlx_vlm.models.minimax_h3.processing"
            ).decode_video
        decoded: tuple[mx.array, float | None] = original(video)
        return decoded

    import mlx.core as mx

    path = str(video).removeprefix("file://")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError(f"cannot open video: {video} (ffmpeg and ffprobe required)")
    try:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-select_streams",
                "V:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,codec_name",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if not streams:
            raise ValueError("no video stream")
        stream = streams[0]
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise ValueError(f"cannot open video: {video}: {detail[-2000:]}") from exc

    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-protocol_whitelist",
        "file",
        "-i",
        path,
        "-map",
        "0:V:0",
        "-fps_mode",
        "passthrough",
        "-f",
        "image2pipe",
        "-c:v",
        "ppm",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    use_hwaccel = _supports_videotoolbox(str(Path(ffmpeg).resolve()))
    while True:
        attempt = command.copy()
        if use_hwaccel:
            attempt[1:1] = ["-hwaccel", "videotoolbox"]
        timed_out = threading.Event()
        try:
            # A file prevents a full stderr pipe from blocking stdout frame reads.
            with (
                tempfile.TemporaryFile() as errors,
                subprocess.Popen(
                    attempt, stdout=subprocess.PIPE, stderr=errors
                ) as proc,
            ):

                def expire(proc=proc, timed_out=timed_out):
                    timed_out.set()
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass

                timer = threading.Timer(_DECODE_TIMEOUT, expire)
                timer.start()
                frame_error = None
                try:
                    frames = list(_read_ppm_frames(proc.stdout))
                    proc.wait()
                except ValueError as exc:
                    frame_error = exc
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    status = proc.wait()
                    timer.cancel()
                errors.seek(max(0, errors.tell() - 2000))
                detail = errors.read().decode(errors="replace").strip()
                if timed_out.is_set():
                    raise ValueError("reference video decode timed out")
                if use_hwaccel and (status or (frame_error is None and not frames)):
                    use_hwaccel = False
                    continue
                if frame_error is not None:
                    raise ValueError(f"{frame_error}: {detail}") from frame_error
                if status:
                    raise ValueError(detail or f"ffmpeg exited with status {status}")
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot decode video: {video}: {exc}") from exc
        if not frames:
            raise ValueError(f"video contains no decodable frames: {video}")
        return mx.stack(frames), _fps(stream)


def install_reference_decoders():
    """Patch both pinned import sites, preserving array input handling once."""
    global _original_decode_video
    processing = importlib.import_module("mlx_vlm.models.minimax_h3.processing")
    pipeline = importlib.import_module("mlx_vlm.models.minimax_h3.pipeline")
    if not all(
        callable(getattr(module, "decode_video", None))
        for module in (processing, pipeline)
    ):
        raise RuntimeError(
            "pinned H3 reference decoder contract changed: decode_video missing"
        )
    if processing.decode_video is not decode_video_with_ffmpeg:
        _original_decode_video = processing.decode_video
    processing.decode_video = decode_video_with_ffmpeg
    pipeline.decode_video = decode_video_with_ffmpeg
