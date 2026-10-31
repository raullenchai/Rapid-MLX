"""Small, auditable RGB-to-MP4 bridge for Desktop video runtimes."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import threading
import wave
from contextlib import ExitStack
from pathlib import Path

_AFCONVERT = "/usr/bin/afconvert"
_AUDIO_PAD_SECONDS = 0.1


class VideoEncodingError(RuntimeError):
    """Raised when the local encoder cannot produce a usable MP4."""


def _ffmpeg_command(
    ffmpeg: str,
    *,
    width: int,
    height: int,
    fps: int,
    output_path: Path,
    audio_path: Path | None = None,
) -> list[str]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
    ]
    if audio_path is None:
        command += ["-an"]
    else:
        command += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0"]
    command += [
        "-c:v",
        "h264_videotoolbox",
        "-pix_fmt",
        "yuv420p",
    ]
    if audio_path is not None:
        command += ["-c:a", "copy", "-shortest"]
    command += [
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return command


def encode_rgb_video(
    frames, output_path: str | Path, fps: int, *, audio=None, sampling_rate=None
) -> None:
    """Atomically encode ``[frames, height, width, rgb]`` uint8 pixels.

    Frames are streamed one at a time so a long generation does not allocate
    a second full raw-video buffer. The destination is replaced only after a
    successful, non-empty MP4 has been produced.
    """
    import numpy as np

    pixels = np.asarray(frames)
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise VideoEncodingError("video frames must be uint8 [T, H, W, 3] RGB")
    frame_count, height, width, _ = pixels.shape
    if frame_count < 1 or height < 1 or width < 1 or fps < 1:
        raise VideoEncodingError(
            "video dimensions, frame count and fps must be positive"
        )

    from ..runtime.video_lane import _resolve_ffmpeg

    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        raise VideoEncodingError("ffmpeg is unavailable")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    process: subprocess.Popen | None = None
    with ExitStack() as stack:
        audio_path = None
        if audio is not None:
            if sampling_rate is None or not sampling_rate > 0:
                raise VideoEncodingError("audio sampling rate must be positive")
            try:
                samples = np.asarray(audio, dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as exc:
                raise VideoEncodingError("invalid audio samples") from exc
            if samples.ndim == 3 and samples.shape[0] == 1:
                samples = samples[0]
            if samples.ndim == 1:
                samples = samples[None, :]
            if (
                samples.ndim != 2
                or not 1 <= samples.shape[0] <= 8
                or samples.shape[1] == 0
                or not np.isfinite(samples).all()
            ):
                raise VideoEncodingError("invalid audio samples")
            # ``-shortest`` keeps a longer soundtrack from extending the clip.
            # Pad a shorter one with silence (plus margin for AAC framing) so
            # the audio never ends first: every frame is kept and ffmpeg never
            # stops reading frames early.
            minimum = math.ceil(frame_count * sampling_rate / fps)
            minimum += math.ceil(sampling_rate * _AUDIO_PAD_SECONDS)
            if samples.shape[1] < minimum:
                samples = np.pad(samples, ((0, 0), (0, minimum - samples.shape[1])))
            if not os.path.isfile(_AFCONVERT) or not os.access(_AFCONVERT, os.X_OK):
                raise VideoEncodingError("AAC audio encoder (afconvert) is unavailable")
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            wav_path, audio_path = directory / "audio.wav", directory / "audio.m4a"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(samples.shape[0])
                wav.setsampwidth(2)
                wav.setframerate(sampling_rate)
                wav.writeframes(
                    (np.clip(samples, -1, 1).T * 32767).astype("<i2").tobytes()
                )
            try:
                subprocess.run(
                    [
                        _AFCONVERT,
                        "-f",
                        "mp4f",
                        "-d",
                        "aac",
                        "-b",
                        "192000",
                        str(wav_path),
                        str(audio_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                OSError,
            ) as exc:
                stderr = getattr(exc, "stderr", None) or b""
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                detail = stderr.strip()[-500:]
                raise VideoEncodingError(
                    f"audio encoding failed: {type(exc).__name__}"
                    + (f": {detail}" if detail else "")
                ) from exc
        stderr_file = stack.enter_context(tempfile.TemporaryFile())
        try:
            handle = tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.stem}.",
                suffix=".encoding.mp4",
                delete=False,
            )
            temporary = Path(handle.name)
            handle.close()
            temporary.unlink(missing_ok=True)
            process = subprocess.Popen(
                _ffmpeg_command(
                    ffmpeg,
                    width=width,
                    height=height,
                    fps=fps,
                    output_path=temporary,
                    audio_path=audio_path,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
            stdin = process.stdin
            if stdin is None:  # pragma: no cover - guaranteed by PIPE
                raise VideoEncodingError("ffmpeg input pipe is unavailable")
            writer_errors: list[BaseException] = []

            def stream_frames() -> None:
                try:
                    for frame in pixels:
                        stdin.write(np.ascontiguousarray(frame).tobytes())
                except (OSError, ValueError) as exc:
                    writer_errors.append(exc)
                finally:
                    try:
                        stdin.close()
                    except OSError:
                        pass

            # Supervise pipe writes from this thread so the same deadline
            # covers a VideoToolbox/ffmpeg stall while stdin is still full.
            writer = threading.Thread(
                target=stream_frames,
                name="rapid-video-encoder-input",
                daemon=True,
            )
            writer.start()
            try:
                return_code = process.wait(timeout=120)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                writer.join(timeout=5)
                raise VideoEncodingError(
                    "ffmpeg timed out while encoding video"
                ) from exc
            writer.join(timeout=5)
            if writer.is_alive():
                raise VideoEncodingError("ffmpeg input writer did not stop")
            if return_code != 0:
                stderr_file.seek(0, os.SEEK_END)
                size = stderr_file.tell()
                stderr_file.seek(max(0, size - 65_536))
                detail = stderr_file.read().decode("utf-8", errors="replace").strip()
                raise VideoEncodingError(
                    f"ffmpeg exited with status {return_code}"
                    + (f": {detail}" if detail else "")
                )
            if writer_errors:
                raise VideoEncodingError(
                    f"ffmpeg input failed: {type(writer_errors[0]).__name__}"
                ) from writer_errors[0]
            if (
                temporary is None
                or not temporary.is_file()
                or temporary.stat().st_size == 0
            ):
                raise VideoEncodingError("ffmpeg completed without an MP4 output")
            os.replace(temporary, destination)
            temporary = None
        except (OSError, subprocess.SubprocessError) as exc:
            raise VideoEncodingError(
                f"video encoding failed: {type(exc).__name__}"
            ) from exc
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            if temporary is not None:
                temporary.unlink(missing_ok=True)
