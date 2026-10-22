"""FFmpeg discovery and video probing utilities."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass


def find_ffmpeg() -> str:
    """Find the ffmpeg binary path.

    Returns:
        Path to ffmpeg executable.

    Raises:
        RuntimeError: If ffmpeg is not found on PATH.
    """
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it with: brew install ffmpeg")
    return path


def find_ffprobe() -> str:
    """Find the ffprobe binary path.

    Returns:
        Path to ffprobe executable.

    Raises:
        RuntimeError: If ffprobe is not found on PATH.
    """
    path = shutil.which("ffprobe")
    if path is None:
        raise RuntimeError("ffprobe not found on PATH. Install it with: brew install ffmpeg")
    return path


@dataclass
class VideoInfo:
    """Probed video stream metadata."""

    width: int
    height: int
    num_frames: int
    fps: float
    duration: float
    has_audio: bool


def probe_video_info(video_path: str) -> VideoInfo:
    """Probe a video file for stream metadata.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoInfo with width, height, num_frames, fps, duration, has_audio.

    Raises:
        RuntimeError: If probing fails.
    """
    ffprobe = find_ffprobe()
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])

    video_stream: dict | None = None
    audio_found = False
    for s in streams:
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        elif s.get("codec_type") == "audio":
            audio_found = True

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    # Parse fps from r_frame_rate (e.g. "24/1")
    r_rate = video_stream.get("r_frame_rate", "24/1")
    num, den = r_rate.split("/")
    fps = float(num) / float(den)

    num_frames = int(video_stream.get("nb_frames", 0))
    duration = float(data.get("format", {}).get("duration", 0))

    # Fallback: estimate frames from duration
    if num_frames == 0 and duration > 0:
        num_frames = int(duration * fps)

    return VideoInfo(
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        num_frames=num_frames,
        fps=fps,
        duration=duration,
        has_audio=audio_found,
    )
