from __future__ import annotations

import io
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from vllm_mlx.video.encoding import (
    VideoEncodingError,
    _ffmpeg_command,
    encode_rgb_video,
)


def test_ffmpeg_command_uses_bundled_videotoolbox_contract(tmp_path: Path) -> None:
    output = tmp_path / "video.mp4"
    command = _ffmpeg_command(
        "/bundle/bin/ffmpeg", width=64, height=32, fps=12, output_path=output
    )

    assert command[0] == "/bundle/bin/ffmpeg"
    assert command[command.index("-video_size") + 1] == "64x32"
    assert command[command.index("-c:v") + 1] == "h264_videotoolbox"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[-1] == str(output)


def test_encode_streams_frames_and_atomically_replaces_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = {}
    stdin_closed = threading.Event()

    class CaptureBytesIO(io.BytesIO):
        def close(self) -> None:
            stdin_closed.set()

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            created["command"] = command
            self._stdin = CaptureBytesIO()
            self.stdin = self._stdin
            self.returncode = None

        def wait(self, timeout=None):
            created["timeout"] = timeout
            assert stdin_closed.wait(timeout=1), "encoder input was not closed"
            created["bytes"] = self._stdin.getvalue()
            Path(created["command"][-1]).write_bytes(b"mp4")
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "vllm_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/bundle/bin/ffmpeg"
    )
    monkeypatch.setattr("vllm_mlx.video.encoding.subprocess.Popen", FakeProcess)
    output = tmp_path / "result.mp4"
    output.write_bytes(b"old")
    frames = np.zeros((2, 4, 8, 3), dtype=np.uint8)

    encode_rgb_video(frames, output, 8)

    assert output.read_bytes() == b"mp4"
    assert len(created["bytes"]) == frames.nbytes
    assert created["timeout"] == 120


def test_encode_timeout_kills_process_and_unblocks_a_stalled_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    killed = threading.Event()

    class BlockingInput:
        def write(self, _data):
            killed.wait()
            raise BrokenPipeError

        def close(self):
            pass

    class StalledProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdin = BlockingInput()

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("ffmpeg", timeout)

        def poll(self):
            return None

        def kill(self):
            killed.set()

    monkeypatch.setattr(
        "vllm_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/bundle/bin/ffmpeg"
    )
    monkeypatch.setattr("vllm_mlx.video.encoding.subprocess.Popen", StalledProcess)

    with pytest.raises(VideoEncodingError, match="timed out"):
        encode_rgb_video(
            np.zeros((2, 4, 4, 3), dtype=np.uint8), tmp_path / "result.mp4", 8
        )

    assert killed.is_set()
    assert not list(tmp_path.glob("*.encoding.mp4"))


def test_encode_rejects_missing_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vllm_mlx.runtime.video_lane._resolve_ffmpeg", lambda: None)

    with pytest.raises(VideoEncodingError, match="ffmpeg is unavailable"):
        encode_rgb_video(
            np.zeros((1, 4, 4, 3), dtype=np.uint8), tmp_path / "result.mp4", 8
        )


def test_encode_tolerates_stdin_close_error_after_all_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdin_closed = threading.Event()

    class CloseFailsInput(io.BytesIO):
        def close(self) -> None:
            stdin_closed.set()
            raise OSError("already closed")

    class SuccessfulProcess:
        def __init__(self, command, **_kwargs):
            self.command = command
            self.stdin = CloseFailsInput()
            self.returncode = None

        def wait(self, timeout=None):
            assert stdin_closed.wait(timeout=1)
            Path(self.command[-1]).write_bytes(b"mp4")
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "vllm_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/bundle/bin/ffmpeg"
    )
    monkeypatch.setattr("vllm_mlx.video.encoding.subprocess.Popen", SuccessfulProcess)
    output = tmp_path / "result.mp4"

    encode_rgb_video(np.zeros((1, 4, 4, 3), dtype=np.uint8), output, 8)

    assert output.read_bytes() == b"mp4"


def test_encode_rejects_writer_that_does_not_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CompletedProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdin = io.BytesIO()
            self.returncode = None

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    class StuckWriter:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

    monkeypatch.setattr(
        "vllm_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/bundle/bin/ffmpeg"
    )
    monkeypatch.setattr("vllm_mlx.video.encoding.subprocess.Popen", CompletedProcess)
    monkeypatch.setattr("vllm_mlx.video.encoding.threading.Thread", StuckWriter)

    with pytest.raises(VideoEncodingError, match="writer did not stop"):
        encode_rgb_video(
            np.zeros((1, 4, 4, 3), dtype=np.uint8), tmp_path / "result.mp4", 8
        )


def test_encode_surfaces_encoder_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdin_closed = threading.Event()

    class ClosingInput(io.BytesIO):
        def close(self) -> None:
            stdin_closed.set()

    class FailedProcess:
        def __init__(self, *_args, **kwargs):
            self.stdin = ClosingInput()
            self.stderr = kwargs["stderr"]
            self.returncode = None

        def wait(self, timeout=None):
            assert stdin_closed.wait(timeout=1)
            self.stderr.write(b"encoder failed")
            self.stderr.flush()
            self.returncode = 7
            return 7

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "vllm_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/bundle/bin/ffmpeg"
    )
    monkeypatch.setattr("vllm_mlx.video.encoding.subprocess.Popen", FailedProcess)

    with pytest.raises(VideoEncodingError, match="status 7: encoder failed"):
        encode_rgb_video(
            np.zeros((1, 4, 4, 3), dtype=np.uint8), tmp_path / "result.mp4", 8
        )


def test_encode_surfaces_pipe_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdin_closed = threading.Event()

    class BrokenInput:
        def write(self, _data):
            raise BrokenPipeError

        def close(self):
            stdin_closed.set()

    class Process:
        def __init__(self, *_args, **_kwargs):
            self.stdin = BrokenInput()
            self.returncode = None

        def wait(self, timeout=None):
            assert stdin_closed.wait(timeout=1)
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "vllm_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/bundle/bin/ffmpeg"
    )
    monkeypatch.setattr("vllm_mlx.video.encoding.subprocess.Popen", Process)

    with pytest.raises(VideoEncodingError, match="input failed: BrokenPipeError"):
        encode_rgb_video(
            np.zeros((1, 4, 4, 3), dtype=np.uint8), tmp_path / "result.mp4", 8
        )


def test_encode_rejects_empty_encoder_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdin_closed = threading.Event()

    class ClosingInput(io.BytesIO):
        def close(self) -> None:
            stdin_closed.set()

    class EmptyOutputProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdin = ClosingInput()
            self.returncode = None

        def wait(self, timeout=None):
            assert stdin_closed.wait(timeout=1)
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "vllm_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/bundle/bin/ffmpeg"
    )
    monkeypatch.setattr("vllm_mlx.video.encoding.subprocess.Popen", EmptyOutputProcess)

    with pytest.raises(VideoEncodingError, match="without an MP4 output"):
        encode_rgb_video(
            np.zeros((1, 4, 4, 3), dtype=np.uint8), tmp_path / "result.mp4", 8
        )


def test_encode_wraps_process_launch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_launch(*_args, **_kwargs):
        raise OSError("launch denied")

    monkeypatch.setattr(
        "vllm_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "/bundle/bin/ffmpeg"
    )
    monkeypatch.setattr("vllm_mlx.video.encoding.subprocess.Popen", fail_launch)

    with pytest.raises(VideoEncodingError, match="encoding failed: OSError"):
        encode_rgb_video(
            np.zeros((1, 4, 4, 3), dtype=np.uint8), tmp_path / "result.mp4", 8
        )


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((1, 4, 4, 4), dtype=np.uint8),
        np.zeros((1, 4, 4, 3), dtype=np.float32),
        np.zeros((0, 4, 4, 3), dtype=np.uint8),
    ],
)
def test_encode_rejects_invalid_frame_contract(frames: np.ndarray) -> None:
    with pytest.raises(VideoEncodingError):
        encode_rgb_video(frames, "/tmp/unused.mp4", 8)
