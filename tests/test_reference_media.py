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
    media._supports_videotoolbox.cache_clear()


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
        videotoolbox=False,
        hardware_failure=None,
    ):
        probe = tmp_path / "ffprobe"
        probe_calls = tmp_path / "probe-calls.jsonl"
        probe.write_text(
            f"#!{sys.executable}\nimport json, sys\n"
            f"with open({str(probe_calls)!r}, 'a') as log: log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            f"print({json.dumps({'streams': [{'avg_frame_rate': avg, 'r_frame_rate': rate, 'codec_name': codec}]})!r})\n"
            f"sys.exit({probe_exit})\n"
        )
        ffmpeg = tmp_path / "ffmpeg"
        calls = tmp_path / "calls.jsonl"
        ffmpeg.write_text(
            f"#!{sys.executable}\nimport json, sys, time\n"
            f"with open({str(calls)!r}, 'a') as log: log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if '-hwaccels' in sys.argv:\n"
            f"    print('Hardware acceleration methods:\\n' + ('videotoolbox' if {videotoolbox} else 'cuda'))\n"
            "    sys.exit(0)\n"
            "assert sys.argv[sys.argv.index('-i') + 1] == 'clip.mp4'\n"
            "assert '-fps_mode' in sys.argv\n"
            "assert sys.argv[sys.argv.index('-pix_fmt') + 1] == 'rgb24'\n"
            f"if '-hwaccel' in sys.argv and {hardware_failure!r} is not None:\n"
            f"    sys.exit({hardware_failure!r})\n"
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
        return calls

    return create


def test_video_commands_exclude_attached_pictures(tools):
    calls = tools()
    media.decode_video_with_ffmpeg("clip.mp4")
    probe = json.loads((calls.parent / "probe-calls.jsonl").read_text())
    assert probe[probe.index("-select_streams") + 1] == "V:0"
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    decode = next(command for command in commands if "-map" in command)
    assert decode[decode.index("-map") + 1] == "0:V:0"


@pytest.mark.parametrize("videotoolbox", [False, True])
def test_hwaccel_only_when_available(tools, videotoolbox):
    calls = tools(videotoolbox=videotoolbox)
    media.decode_video_with_ffmpeg("clip.mp4")
    media.decode_video_with_ffmpeg("clip.mp4")
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert commands[0] == ["-hide_banner", "-hwaccels"]
    assert len(commands) == 3  # Capability probe is cached by executable path.
    for command in commands[1:]:
        assert ("-hwaccel" in command) == videotoolbox
        if videotoolbox:
            assert command[command.index("-hwaccel") + 1] == "videotoolbox"


@pytest.mark.parametrize("hardware_failure", [0, 1])
def test_hwaccel_failure_retries_software(tools, hardware_failure):
    calls = tools(videotoolbox=True, hardware_failure=hardware_failure)
    frames, _ = media.decode_video_with_ffmpeg("clip.mp4")
    assert frames.shape == (2, 1, 2, 3)
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == 3
    assert "-hwaccel" in commands[1]
    assert "-hwaccel" not in commands[2]


@pytest.mark.parametrize(
    "failure",
    [
        OSError("missing"),
        subprocess.CalledProcessError(1, "ffmpeg"),
        subprocess.TimeoutExpired("ffmpeg", 5),
    ],
)
def test_hwaccel_probe_failure(monkeypatch, failure):
    def run(command, **kwargs):
        assert command == ["/ffmpeg", "-hide_banner", "-hwaccels"]
        assert kwargs["timeout"] == 5
        raise failure

    monkeypatch.setattr(media.subprocess, "run", run)
    assert media._supports_videotoolbox("/ffmpeg") is False


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
    calls = tools(stall=True, videotoolbox=True)
    monkeypatch.setattr(media, "_DECODE_TIMEOUT", 0.2)
    with pytest.raises(ValueError, match="timed out"):
        media.decode_video_with_ffmpeg("clip.mp4")
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == 2  # A decode timeout must not trigger a retry.


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
