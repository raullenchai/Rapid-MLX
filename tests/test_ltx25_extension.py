"""Short-video extension contract for the LTX-2.5 runtime and Videos API."""

from __future__ import annotations

import asyncio
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


@pytest.mark.asyncio
async def test_extension_route_runs_job_and_removes_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        assert current["status"] == "completed"
        assert current["frames"] == 25
        assert current["fps"] == 24
        assert captured["extend_frames"] == 16
        assert captured["seed"] == 11
        job_dir = video._jobs_root / created["id"]
        assert not (job_dir / "source.mp4").exists()
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
        with pytest.raises(HTTPException, match="beta workload limit") as exc:
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
