"""Short-video extension contract for the LTX-2.5 runtime and Videos API."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rapid_mlx.routes import video
from rapid_mlx.video import ltx25


class Upload:
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def read(self, size: int) -> bytes:
        chunk, self.data = self.data[:size], self.data[size:]
        return chunk


def test_extension_runtime_uses_video_context_and_latent_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ltx25, "embedded_ltx25_interpreter", lambda: "/embedded/python")
    captured: dict = {}

    class Process:
        returncode = 0

        def __init__(self, command: list[str], **kwargs) -> None:
            captured["command"] = command

        def communicate(self, *, input: str, timeout: int) -> None:
            captured["prompt"] = input
            Path(
                captured["command"][captured["command"].index("--output") + 1]
            ).write_bytes(b"extended-mp4")

    monkeypatch.setattr(ltx25.subprocess, "Popen", Process)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "output.mp4"
    ltx25.LTX25VideoEngine("ltx-2.5-mlx-q8").extend(
        prompt="private motion prompt",
        source_video=source,
        output_path=output,
        extend_frames=16,
        seed=11,
    )

    command = captured["command"]
    assert command[3] == "extend"
    assert command[command.index("--video") + 1] == str(source)
    assert command[command.index("--extend-frames") + 1] == "2"
    assert command[command.index("--direction") + 1] == "after"
    assert "--distilled" not in command
    assert "--low-ram" not in command
    assert "private motion prompt" not in command
    assert captured["prompt"] == "private motion prompt"
    assert output.read_bytes() == b"extended-mp4"


def test_extension_probe_counts_decoded_frames(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=256x256:r=24",
            "-frames:v",
            "9",
            "-y",
            str(source),
        ],
        check=True,
    )
    assert video._probe_extension_video(source) == (256, 256, 9)
    short = tmp_path / "short.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(source),
            "-frames:v",
            "8",
            "-y",
            str(short),
        ],
        check=True,
    )
    with pytest.raises(HTTPException, match="8n\\+1 frames"):
        video._probe_extension_video(short)


def test_extension_probe_rejects_runtime_frame_rate_mismatch(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
    source = tmp_path / "variable-rate.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=256x256:r=24",
            "-frames:v",
            "9",
            "-vf",
            r"settb=1/48,setpts=floor(N/2)*4+mod(N\,2)",
            "-fps_mode",
            "vfr",
            "-enc_time_base",
            "1/48",
            "-video_track_timescale",
            "48",
            "-y",
            str(source),
        ],
        check=True,
    )
    details = json.loads(
        subprocess.check_output(
            [
                shutil.which("ffprobe"),
                "-v",
                "error",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(source),
            ]
        )
    )["streams"][0]
    assert details == {"avg_frame_rate": "24/1", "r_frame_rate": "48/1"}
    with pytest.raises(HTTPException, match="24 fps"):
        video._probe_extension_video(source)


@pytest.mark.parametrize("dimensions", [(4096, 4096), (1920, 1920)])
def test_extension_probe_rejects_dimensions_before_frame_decode(
    dimensions: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(video.shutil, "which", lambda _: "/ffprobe")
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        assert "-count_frames" not in command
        assert command[command.index("-protocol_whitelist") + 1] == "file"
        assert command[command.index("-format_whitelist") + 1] == "mov"
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "width": dimensions[0],
                            "height": dimensions[1],
                            "avg_frame_rate": "24/1",
                            "r_frame_rate": "24/1",
                        }
                    ],
                    "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
                }
            )
        )

    monkeypatch.setattr(video.subprocess, "run", probe)
    with pytest.raises(HTTPException) as exc:
        video._probe_extension_video(Path("source.mp4"))
    assert exc.value.status_code == 400
    assert len(calls) == 1


@pytest.mark.parametrize("frames", [89, 97, 1001])
def test_extension_probe_bounds_counting_with_audio(
    frames: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=256x256:r=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-frames:v",
            str(frames),
            "-t",
            str(frames / 24),
            "-c:a",
            "aac",
            "-y",
            str(source),
        ],
        check=True,
    )
    run = video.subprocess.run
    counted = []

    def probe(command, **kwargs):
        result = run(command, **kwargs)
        if "-count_frames" in command:
            counted.append(
                int(json.loads(result.stdout)["streams"][0]["nb_read_frames"])
            )
        return result

    monkeypatch.setattr(video.subprocess, "run", probe)
    if frames == 89:
        assert video._probe_extension_video(source) == (256, 256, 89)
    else:
        with pytest.raises(HTTPException, match="supported workload"):
            video._probe_extension_video(source)
    assert counted == ([89] if frames == 89 else [])


def test_extension_probe_rejects_corrupt_decoding(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required")
    source = tmp_path / "corrupt.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=256x256:r=24",
            "-frames:v",
            "9",
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-g",
            "1",
            "-y",
            str(source),
        ],
        check=True,
    )
    packets = json.loads(
        subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_packets",
                "-select_streams",
                "v:0",
                "-show_entries",
                "packet=pos,size",
                "-of",
                "json",
                str(source),
            ]
        )
    )["packets"]
    packet = packets[4]
    with source.open("r+b") as target:
        target.seek(int(packet["pos"]))
        target.write(b"\0" * int(packet["size"]))
    with pytest.raises(HTTPException, match="invalid input_video"):
        video._probe_extension_video(source)


@pytest.mark.parametrize("declared_count", [None, "0", "N/A"])
def test_extension_probe_runtime_metadata_compatibility(
    declared_count: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(video.shutil, "which", lambda _: "/ffprobe")
    stream = {
        "width": 256,
        "height": 256,
        "avg_frame_rate": "24/1",
        "r_frame_rate": "24/1",
    }
    if declared_count is not None:
        stream["nb_frames"] = declared_count

    def probe(command, **kwargs):
        details = (
            {"streams": [{"nb_read_frames": "9", "nb_read_packets": "9"}]}
            if "-count_frames" in command
            else {
                "streams": [stream],
                "format": {"format_name": "mov,mp4", "duration": "0.375"},
            }
        )
        return SimpleNamespace(stdout=json.dumps(details), stderr="")

    monkeypatch.setattr(video.subprocess, "run", probe)
    if declared_count == "N/A":
        # The pinned runtime int() conversion rejects this literal as well.
        with pytest.raises(HTTPException, match="invalid input_video"):
            video._probe_extension_video(Path("source.mp4"))
    else:
        assert video._probe_extension_video(Path("source.mp4")) == (256, 256, 9)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_extension_route_runs_job_and_removes_source(
    failure: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video.configure_video_jobs(tmp_path / "jobs")
    video.start_video_jobs()
    captured: dict = {}

    class Engine:
        model_name = "MrMofer/ltx-2.5-mlx-q8"
        video_family = "ltx-2.5"

        def extend(self, *, source_video: Path, output_path: Path, **kwargs) -> None:
            assert source_video.read_bytes() == b"uploaded-mp4"
            captured.update(kwargs)
            output_path.write_bytes(b"extended-mp4")
            if failure:
                raise RuntimeError("runtime failed after writing partial output")

    monkeypatch.setattr(video, "_video_engine", Engine)
    monkeypatch.setattr(video, "_probe_extension_video", lambda _: (256, 256, 9))
    try:
        created = await video.extend_video(
            prompt="follow the dragon",
            model="ltx-2.5-mlx-q8",
            extend_frames=16,
            seed=11,
            input_video=Upload(b"uploaded-mp4"),
        )
        for _ in range(200):
            current = await video.retrieve_video(created["id"])
            if current["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert current["status"] == ("failed" if failure else "completed")
        assert current["frames"] == 25
        assert current["fps"] == 24
        assert captured["extend_frames"] == 16
        assert captured["seed"] == 11
        job_dir = video._jobs_root / created["id"]
        assert not (job_dir / "source.mp4").exists()
        if failure:
            assert not job_dir.exists()
        else:
            assert (job_dir / "output.mp4").read_bytes() == b"extended-mp4"
    finally:
        video.configure_video_jobs(None)
        video.start_video_jobs()


@pytest.mark.asyncio
async def test_extension_rejects_invalid_frame_count_before_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        video,
        "_video_engine",
        lambda: SimpleNamespace(
            model_name="MrMofer/ltx-2.5-mlx-q8", video_family="ltx-2.5"
        ),
    )
    with pytest.raises(HTTPException, match="extend_frames") as exc:
        await video.extend_video(
            prompt="follow the dragon",
            model="ltx-2.5-mlx-q8",
            extend_frames=15,
            seed=11,
            input_video=Upload(b"uploaded-mp4"),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_extension_rejects_oversized_result_and_removes_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video.configure_video_jobs(tmp_path / "jobs")
    video.start_video_jobs()
    monkeypatch.setattr(
        video,
        "_video_engine",
        lambda: SimpleNamespace(
            model_name="MrMofer/ltx-2.5-mlx-q8", video_family="ltx-2.5"
        ),
    )
    monkeypatch.setattr(video, "_probe_extension_video", lambda _: (512, 512, 97))
    try:
        with pytest.raises(HTTPException, match="supported workload") as exc:
            await video.extend_video(
                prompt="follow the dragon",
                model="ltx-2.5-mlx-q8",
                extend_frames=8,
                seed=11,
                input_video=Upload(b"uploaded-mp4"),
            )
        assert exc.value.status_code == 400
        assert list(video._jobs_root.iterdir()) == []
    finally:
        video.configure_video_jobs(None)
        video.start_video_jobs()


def test_ltx25_capabilities_expose_extension_limits() -> None:
    capabilities = video._video_capabilities(
        SimpleNamespace(model_name="MrMofer/ltx-2.5-mlx-q8", video_family="ltx-2.5")
    )
    extension = capabilities["limits"]["video_extension"]
    assert extension["endpoint"] == "/v1/videos/extend"
    assert extension["added_frames"]["multiple_of"] == 8
    assert extension["maximum_output_frames"] == 97
