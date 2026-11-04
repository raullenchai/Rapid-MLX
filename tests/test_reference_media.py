"""Reference I/O contract without a model, native decoder wheel, or GPU."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rapid_mlx.video import reference_media as media


@pytest.fixture(autouse=True)
def fake_arrays(monkeypatch):
    core = ModuleType("mlx.core")
    core.array = np.array
    core.uint8 = np.uint8
    core.stack = np.stack
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(media, "_original_decode_video", None)


@pytest.fixture
def tools(tmp_path, monkeypatch):
    def create(
        *,
        avg="30000/1001",
        rate="0/0",
        frames=2,
        exit_code=0,
        probe_exit=0,
        codec="h264",
        partial=False,
        stall=False,
    ):
        probe = tmp_path / "ffprobe"
        probe.write_text(
            f"#!{sys.executable}\nimport sys\n"
            f"print({json.dumps({'streams': [{'avg_frame_rate': avg, 'r_frame_rate': rate, 'codec_name': codec}]})!r})\n"
            f"sys.exit({probe_exit})\n"
        )
        ffmpeg = tmp_path / "ffmpeg"
        ffmpeg.write_text(
            f"#!{sys.executable}\nimport sys, time\n"
            "assert sys.argv[sys.argv.index('-i') + 1] == 'clip.mp4'\n"
            "assert '-fps_mode' in sys.argv\n"
            "assert sys.argv[sys.argv.index('-pix_fmt') + 1] == 'rgb24'\n"
            f"sys.stdout.buffer.write((b'P6\\n2 1\\n255\\n' + bytes([255, 0, 0, 0, 255, 0])) * {frames})\n"
            f"sys.stdout.buffer.write(b'P6\\n2 1\\n255\\nx' if {partial} else b'')\n"
            "sys.stdout.flush()\n"
            f"time.sleep(10 if {stall} else 0)\n"
            "sys.stderr.write('decoder failed')\n"
            f"sys.exit({exit_code})\n"
        )
        probe.chmod(0o755)
        ffmpeg.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))

    return create


@pytest.mark.parametrize("codec", ["h264", "av1", "prores"])
@pytest.mark.parametrize("path", [Path("clip.mp4"), "clip.mp4", "file://clip.mp4"])
def test_rgb_shape_fps_and_file_prefix(tools, path, codec):
    tools(codec=codec)
    frames, fps = media.decode_video_with_ffmpeg(path)
    assert frames.shape == (2, 1, 2, 3)
    assert frames.dtype == np.uint8
    assert frames[0].tolist() == [[[255, 0, 0], [0, 255, 0]]]
    assert fps == pytest.approx(30000 / 1001)


@pytest.mark.parametrize(
    "avg,rate,expected",
    [
        ("0/0", "0/0", None),
        ("bad", "25/1", 25),
        ("-1", "0", None),
    ],
)
def test_fps_fallback(tools, avg, rate, expected):
    tools(avg=avg, rate=rate)
    assert media.decode_video_with_ffmpeg("clip.mp4")[1] == expected


@pytest.mark.parametrize(
    "options,message",
    [
        ({"frames": 0}, "video contains no decodable frames"),
        ({"exit_code": 1}, "decoder failed"),
        ({"probe_exit": 1}, "cannot open video"),
        ({"partial": True}, "incomplete reference-video frame"),
    ],
)
def test_failed_decode(tools, options, message):
    tools(**options)
    with pytest.raises(ValueError, match=message):
        media.decode_video_with_ffmpeg("clip.mp4")


def test_timeout_kills_decoder(tools, monkeypatch):
    tools(stall=True)
    monkeypatch.setattr(media, "_DECODE_TIMEOUT", 0.2)
    with pytest.raises(ValueError, match="timed out"):
        media.decode_video_with_ffmpeg("clip.mp4")


def test_install_and_array_delegation(monkeypatch):
    array = np.zeros((2, 1, 2, 3), dtype=np.uint8)
    calls = []

    def original(value):
        calls.append(value)
        return value, None

    processing = SimpleNamespace(decode_video=original)
    pipeline = SimpleNamespace(decode_video=original)
    monkeypatch.setattr(
        media.importlib,
        "import_module",
        lambda name: processing if name.endswith("processing") else pipeline,
    )
    # Delegation works before installation as well.
    assert media.decode_video_with_ffmpeg(array)[0] is array
    media.install_reference_decoders()
    media.install_reference_decoders()
    assert processing.decode_video is media.decode_video_with_ffmpeg
    assert pipeline.decode_video is media.decode_video_with_ffmpeg
    assert media.decode_video_with_ffmpeg(array)[0] is array
    assert len(calls) == 2


def test_install_fails_closed(monkeypatch):
    monkeypatch.setattr(
        media.importlib, "import_module", lambda name: SimpleNamespace()
    )
    with pytest.raises(RuntimeError, match="contract changed"):
        media.install_reference_decoders()


def test_real_ffmpeg_roundtrip(tmp_path):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg and ffprobe not on PATH")
    clip = tmp_path / "real.mov"
    # Raw RGB in MOV avoids needing a software encoder such as libx264.
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "8x6",
            "-r",
            "4",
            "-i",
            "pipe:0",
            "-c:v",
            "rawvideo",
            str(clip),
        ],
        input=bytes(3 * 8 * 6 * 3),
        check=True,
        timeout=30,
    )
    frames, fps = media.decode_video_with_ffmpeg(clip)
    assert frames.shape == (3, 6, 8, 3)
    assert fps == 4
