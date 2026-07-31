"""LTX-2.3 MLX-native lane and OpenAI video API contract tests."""

from __future__ import annotations

import asyncio
import io
import stat
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.routes import video
from vllm_mlx.runtime import video_lane
from vllm_mlx.runtime.video_lane import (
    VideoEngine,
    VideoRuntimeError,
    require_video_runtime_or_exit,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def test_ltx23_alias_routes_to_video_lane() -> None:
    profile = resolve_profile("ltx-2.3-mlx-q4")
    assert profile is not None
    assert profile.hf_path == "notapalindrome/ltx23-mlx-av-q4"
    assert profile.modality == "video-gen"
    assert profile.min_memory_gb == 24
    assert profile.supports_spec_decode is False


def test_ltx23_model_discovery_is_video_shaped() -> None:
    from vllm_mlx.routes.models import _build_model_info

    info = _build_model_info("ltx-2.3-mlx-q4")
    assert info.modality == "video-gen"
    assert info.capabilities == ["video.generation"]


def test_video_runtime_preflight_fails_before_download(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
    monkeypatch.setattr("shutil.which", lambda _: None)

    with pytest.raises(SystemExit) as exc:
        require_video_runtime_or_exit()

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "rapid-mlx[video]" in error
    assert "brew install ffmpeg" in error


def test_video_runtime_preflight_reports_python_311_floor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Python310(tuple):
        major = 3
        minor = 10

    monkeypatch.setattr(video_lane.sys, "version_info", Python310((3, 10, 0)))

    with pytest.raises(SystemExit) as exc:
        require_video_runtime_or_exit()

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "requires Python 3.11 or newer" in error
    assert "current: 3.10" in error


def test_video_extra_marks_every_dependency_python_311_or_newer() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)

    video_specs = pyproject["project"]["optional-dependencies"]["video"]
    assert video_specs
    assert any(spec.startswith("mlx-video-with-audio==0.1.36;") for spec in video_specs)
    assert all("python_version >= '3.11'" in spec for spec in video_specs)


@pytest.mark.asyncio
async def test_video_multipart_gate_authenticates_before_reading_body() -> None:
    from vllm_mlx.config import get_config

    cfg = get_config()
    saved_key = cfg.api_key
    cfg.api_key = "secret"
    receive_calls = 0
    sent = []

    async def downstream(scope, receive, send) -> None:
        raise AssertionError("unauthenticated request reached FastAPI")

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": b"x", "more_body": False}

    async def send(message) -> None:
        sent.append(message)

    try:
        middleware = video.VideoBodyLimitMiddleware(downstream)
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/videos",
                "headers": [],
            },
            receive,
            send,
        )
    finally:
        cfg.api_key = saved_key

    assert receive_calls == 0
    assert sent[0]["status"] == 401


@pytest.mark.asyncio
async def test_video_multipart_gate_rejects_content_length_before_read() -> None:
    from vllm_mlx.config import get_config

    cfg = get_config()
    saved_key = cfg.api_key
    cfg.api_key = None
    receive_calls = 0
    sent = []

    async def downstream(scope, receive, send) -> None:
        raise AssertionError("oversized request reached FastAPI")

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": b"x", "more_body": False}

    async def send(message) -> None:
        sent.append(message)

    try:
        middleware = video.VideoBodyLimitMiddleware(downstream)
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/videos",
                "headers": [
                    (
                        b"content-length",
                        str(video._VIDEO_REQUEST_BYTES + 1).encode(),
                    )
                ],
            },
            receive,
            send,
        )
    finally:
        cfg.api_key = saved_key

    assert receive_calls == 0
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_video_multipart_gate_caps_chunked_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.config import get_config

    cfg = get_config()
    saved_key = cfg.api_key
    cfg.api_key = None
    monkeypatch.setattr(video, "_VIDEO_REQUEST_BYTES", 4)
    chunks = iter(
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]
    )
    sent = []

    async def downstream(scope, receive, send) -> None:
        while True:
            message = await receive()
            if not message.get("more_body", False):
                return

    async def receive():
        return next(chunks)

    async def send(message) -> None:
        sent.append(message)

    try:
        middleware = video.VideoBodyLimitMiddleware(downstream)
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/videos",
                "headers": [],
            },
            receive,
            send,
        )
    finally:
        cfg.api_key = saved_key

    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_video_multipart_gate_emits_one_413_through_starlette(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming cap must reject outside Starlette's 500 handler."""
    from vllm_mlx.config import get_config

    cfg = get_config()
    saved_key = cfg.api_key
    cfg.api_key = None
    monkeypatch.setattr(video, "_VIDEO_REQUEST_BYTES", 4)

    async def endpoint(request: Request) -> JSONResponse:
        await request.body()
        return JSONResponse({"ok": True})

    app = video.VideoBodyLimitMiddleware(
        Starlette(routes=[Route("/v1/videos", endpoint, methods=["POST"])])
    )
    chunks = iter(
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]
    )
    sent = []

    async def receive():
        return next(chunks)

    async def send(message) -> None:
        sent.append(message)

    try:
        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/videos",
                "raw_path": b"/v1/videos",
                "query_string": b"",
                "root_path": "",
                "headers": [],
                "client": ("test", 1),
                "server": ("test", 80),
            },
            receive,
            send,
        )
    finally:
        cfg.api_key = saved_key

    starts = [message for message in sent if message["type"] == "http.response.start"]
    assert [message["status"] for message in starts] == [413]


def test_serve_dispatches_video_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_mlx import cli
    from vllm_mlx.runtime import video_lane

    class PreflightReachedError(RuntimeError):
        pass

    def stop_at_preflight() -> None:
        raise PreflightReachedError

    monkeypatch.setattr(video_lane, "require_video_runtime_or_exit", stop_at_preflight)
    args = SimpleNamespace(model="ltx-2.3-mlx-q4", max_tokens=None, watchdog_ppid=None)
    with pytest.raises(PreflightReachedError):
        cli.serve_command(args)


def test_video_engine_calls_mlx_native_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}
    ffmpeg_calls = []
    fake = ModuleType("mlx_video")

    def generate_video_with_audio(
        *,
        model_repo,
        text_encoder_repo,
        prompt,
        height,
        width,
        num_frames,
        seed,
        fps,
        output_path,
        image,
        verbose,
        enhance_prompt,
        negative_prompt=None,
        cfg_scale=3.0,
        image_strength=1.0,
    ) -> None:
        captured.update(locals())
        generated = Path(output_path)
        generated.write_bytes(b"mp4")
        generated.chmod(0o640)

    fake.generate_video_with_audio = generate_video_with_audio
    monkeypatch.setitem(sys.modules, "mlx_video", fake)
    monkeypatch.setattr("shutil.which", lambda _: "/opt/homebrew/bin/ffmpeg")

    def remux_video_only(command, **kwargs) -> None:
        ffmpeg_calls.append(command)
        Path(command[-1]).write_bytes(b"video-only-mp4")

    monkeypatch.setattr("subprocess.run", remux_video_only)

    output = tmp_path / "result.mp4"
    reference = tmp_path / "reference.png"
    Image.new("RGB", (64, 64), "blue").save(reference)
    engine = VideoEngine("notapalindrome/ltx23-mlx-av-q4")
    engine.generate(
        prompt="A fox runs through snow",
        output_path=output,
        width=768,
        height=512,
        num_frames=97,
        fps=24,
        seed=7,
        image=reference,
        negative_prompt="static",
        guidance_scale=4.5,
        conditioning_strength=0.25,
    )

    assert output.read_bytes() == b"video-only-mp4"
    assert stat.S_IMODE(output.stat().st_mode) == 0o640
    assert captured["model_repo"] == "notapalindrome/ltx23-mlx-av-q4"
    assert captured["num_frames"] == 97
    assert captured["image"] == str(reference)
    assert captured["negative_prompt"] == "static"
    assert captured["cfg_scale"] == 4.5
    assert captured["image_strength"] == 0.25
    assert len(ffmpeg_calls) == 1
    assert ffmpeg_calls[0][ffmpeg_calls[0].index("-map") + 1] == "0:v:0"
    assert "-an" in ffmpeg_calls[0]
    assert ffmpeg_calls[0][ffmpeg_calls[0].index("-c:v") + 1] == "copy"


def test_video_only_remux_failure_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.mp4"
    output.write_bytes(b"mp4-with-silent-audio")
    sibling = tmp_path / "result.video-only.mp4"
    sibling.write_bytes(b"unrelated-artifact")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr("subprocess.run", fail)

    with pytest.raises(VideoRuntimeError, match="silent audio track"):
        VideoEngine._remove_audio_track(output)

    assert output.read_bytes() == b"mp4-with-silent-audio"
    assert sibling.read_bytes() == b"unrelated-artifact"
    assert not list(tmp_path.glob(".result.*.video-only.mp4"))


def test_video_only_cleanup_failure_does_not_mask_remux_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.mp4"
    output.write_bytes(b"mp4-with-silent-audio")
    original_unlink = Path.unlink
    temporary_paths = []

    def fail_remux(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    def fail_cleanup(path: Path, *args, **kwargs) -> None:
        temporary_paths.append(path)
        raise PermissionError("cleanup denied")

    monkeypatch.setattr("subprocess.run", fail_remux)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)

    with pytest.raises(VideoRuntimeError, match="silent audio track") as exc:
        VideoEngine._remove_audio_track(output)

    assert isinstance(exc.value.__cause__, subprocess.CalledProcessError)
    assert len(temporary_paths) == 1
    original_unlink(temporary_paths[0], missing_ok=True)


def test_video_engines_share_process_generation_lock() -> None:
    first = VideoEngine("one/model")
    second = VideoEngine("two/model")
    assert first._generation_lock is second._generation_lock


def test_video_crop_timeout_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ModuleType("mlx_video")

    def generate_video_with_audio(**kwargs) -> None:
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    fake.generate_video_with_audio = generate_video_with_audio
    monkeypatch.setitem(sys.modules, "mlx_video", fake)
    monkeypatch.setattr("shutil.which", lambda _: "/opt/homebrew/bin/ffmpeg")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 120)

    monkeypatch.setattr("subprocess.run", timeout)
    engine = VideoEngine("notapalindrome/ltx23-mlx-av-q4")
    with pytest.raises(VideoRuntimeError, match="could not crop"):
        engine.generate(
            prompt="portrait",
            output_path=tmp_path / "result.mp4",
            width=768,
            height=1280,
            num_frames=9,
            fps=24,
            seed=1,
            image=None,
            output_width=720,
            output_height=1280,
        )


def test_video_parameter_validation() -> None:
    assert video._parse_size("768x512") == (768, 512)
    assert video._parse_size("1280x720") == (1280, 720)
    assert video._parse_size("720x1280") == (720, 1280)
    assert video._frame_count(4) == 97
    assert video._frame_count(1, 26) == 25
    with pytest.raises(HTTPException, match="divisible by 64") as exc:
        video._parse_size("700x512")
    assert exc.value.status_code == 400


def test_generation_gate_is_owned_by_event_loop() -> None:
    async def get_gate() -> asyncio.Lock:
        return video._generation_gate_for_current_loop()

    assert asyncio.run(get_gate()) is not asyncio.run(get_gate())


@pytest.mark.asyncio
async def test_video_rejects_unsafe_pixel_time_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    with pytest.raises(HTTPException, match="safe LTX-2.3 Q4 workload") as exc:
        await video.create_video(
            prompt="too large",
            model="ltx-2.3-mlx-q4",
            seconds="20",
            size="1920x1920",
            seed=1,
            input_reference=None,
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_openai_720p_size_is_aligned_then_cropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            captured.update(kwargs)
            output_path.write_bytes(b"mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="landscape",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="1280x720",
        seed=1,
        input_reference=None,
    )
    for _ in range(200):
        if (await video.retrieve_video(created["id"]))["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert captured["width"] == 1280
    assert captured["height"] == 768
    assert captured["output_width"] == 1280
    assert captured["output_height"] == 720
    await video.delete_video(created["id"])


@pytest.mark.asyncio
async def test_video_route_threads_motion_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    image_bytes = io.BytesIO()
    Image.new("RGB", (64, 64), "blue").save(image_bytes, format="PNG")

    class ReferenceUpload:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def read(self, size: int) -> bytes:
            payload, self._payload = self._payload[:size], self._payload[size:]
            return payload

    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            captured.update(kwargs)
            output_path.write_bytes(b"mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="a fast camera move",
        model="ltx-2.3-mlx-q4",
        seconds="ignored when frames is explicit",
        size="512x512",
        seed=3,
        fps=12,
        frames=17,
        guidance_scale=4.25,
        conditioning_strength=0.35,
        negative_prompt="static camera",
        input_reference=ReferenceUpload(image_bytes.getvalue()),
    )
    for _ in range(100):
        current = await video.retrieve_video(created["id"])
        if current["status"] == "completed":
            break
        await asyncio.sleep(0.01)

    assert current["status"] == "completed"
    assert current["seconds"] == "2"
    assert captured["fps"] == 12
    assert captured["num_frames"] == 17
    assert captured["guidance_scale"] == 4.25
    assert captured["negative_prompt"] == "static camera"
    assert captured["conditioning_strength"] == 0.35
    assert captured["image"].is_file()
    await video.delete_video(created["id"])


@pytest.mark.asyncio
async def test_video_route_validates_motion_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    base = {
        "prompt": "test",
        "model": "ltx-2.3-mlx-q4",
        "seconds": "1",
        "size": "512x512",
        "seed": 1,
        "input_reference": None,
    }
    with pytest.raises(HTTPException, match="fps must be between"):
        await video.create_video(**base, fps=0)
    with pytest.raises(HTTPException, match="LTX frames must be 8n"):
        await video.create_video(**base, frames=16)
    with pytest.raises(HTTPException, match="at least 9"):
        await video.create_video(**base, frames=1)
    with pytest.raises(HTTPException, match="guidance_scale must be between"):
        await video.create_video(**base, guidance_scale=0.5)
    with pytest.raises(HTTPException, match="requires input_reference"):
        await video.create_video(**base, conditioning_strength=0.5)


@pytest.mark.asyncio
async def test_failed_reference_upload_is_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

    class BrokenUpload:
        async def read(self, size: int) -> bytes:
            raise OSError("upload interrupted")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    before = set(video._jobs_root.iterdir())
    with pytest.raises(OSError, match="upload interrupted"):
        await video.create_video(
            prompt="test",
            model="ltx-2.3-mlx-q4",
            seconds="1",
            size="512x512",
            seed=1,
            input_reference=BrokenUpload(),
        )
    assert set(video._jobs_root.iterdir()) == before


@pytest.mark.asyncio
async def test_video_job_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"generated-mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="Ocean waves at sunset",
        model="ltx-2.3-mlx-q4",
        seconds="2",
        size="768x512",
        seed=42,
        input_reference=None,
    )
    video_id = created["id"]
    for _ in range(100):
        current = await video.retrieve_video(video_id)
        if current["status"] != "queued":
            if current["status"] == "completed":
                break
        await asyncio.sleep(0.01)

    assert current["status"] == "completed"
    assert current["progress"] == 100
    response = await video.retrieve_video_content(video_id)
    chunks = [chunk async for chunk in response.body_iterator]
    assert b"".join(chunks) == b"generated-mp4"
    deleted = await video.delete_video(video_id)
    assert deleted["deleted"] is True


@pytest.mark.asyncio
async def test_video_jobs_stay_queued_until_worker_is_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0

    class BlockingEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                assert release.wait(timeout=5)
            output_path.write_bytes(b"mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: BlockingEngine())
    first = await video.create_video(
        prompt="first",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=1,
        input_reference=None,
    )
    assert await asyncio.to_thread(started.wait, 2)
    second = await video.create_video(
        prompt="second",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=2,
        input_reference=None,
    )
    await asyncio.sleep(0.05)

    assert (await video.retrieve_video(first["id"]))["status"] == "in_progress"
    assert (await video.retrieve_video(second["id"]))["status"] == "queued"
    release.set()
    second_status = None
    for _ in range(200):
        second_status = (await video.retrieve_video(second["id"]))["status"]
        if second_status == "completed":
            break
        await asyncio.sleep(0.01)
    assert calls == 2
    assert second_status == "completed"
    response = await video.retrieve_video_content(second["id"])
    assert b"".join([chunk async for chunk in response.body_iterator]) == b"mp4"
    await video.delete_video(first["id"])
    await video.delete_video(second["id"])


@pytest.mark.asyncio
async def test_delete_cancels_a_queued_job(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0

    class BlockingEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            nonlocal calls
            calls += 1
            started.set()
            assert release.wait(timeout=5)
            output_path.write_bytes(b"mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: BlockingEngine())
    first = await video.create_video(
        prompt="first",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=1,
        input_reference=None,
    )
    assert await asyncio.to_thread(started.wait, 2)
    queued = await video.create_video(
        prompt="queued",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=2,
        input_reference=None,
    )
    assert (await video.retrieve_video(queued["id"]))["status"] == "queued"
    deleted = await video.delete_video(queued["id"])
    assert deleted["deleted"] is True
    release.set()
    for _ in range(200):
        if (await video.retrieve_video(first["id"]))["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert calls == 1
    await video.delete_video(first["id"])


@pytest.mark.asyncio
async def test_cancelled_job_reaches_terminal_state_and_cleans_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            started.set()
            assert release.wait(timeout=5)
            output_path.write_bytes(b"late-output")

    monkeypatch.setattr(video, "_video_engine", lambda: BlockingEngine())
    created = await video.create_video(
        prompt="cancel me",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=3,
        input_reference=None,
    )
    assert await asyncio.to_thread(started.wait, 2)
    task = video._tasks[created["id"]]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    current = await video.retrieve_video(created["id"])
    assert current["status"] == "failed"
    assert current["error"]["code"] == "video_generation_cancelled"
    assert video._jobs[created["id"]].generation_finished is False
    release.set()
    for _ in range(200):
        if (
            video._jobs[created["id"]].generation_finished
            and not (video._jobs_root / created["id"]).exists()
        ):
            break
        await asyncio.sleep(0.01)
    assert video._jobs[created["id"]].generation_finished is True
    assert not (video._jobs_root / created["id"]).exists()
    await video.delete_video(created["id"])


@pytest.mark.asyncio
async def test_failed_generation_removes_partial_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"partial")
            raise RuntimeError("private /tmp/detail")

    monkeypatch.setattr(video, "_video_engine", lambda: FailingEngine())
    created = await video.create_video(
        prompt="fail",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=4,
        input_reference=None,
    )
    for _ in range(200):
        current = await video.retrieve_video(created["id"])
        if current["status"] == "failed":
            break
        await asyncio.sleep(0.01)

    assert current["status"] == "failed"
    assert current["error"]["message"] == (
        "Video generation failed; check the server logs for details."
    )
    assert not (video._jobs_root / created["id"]).exists()
    await video.delete_video(created["id"])


@pytest.mark.asyncio
async def test_shutdown_is_bounded_and_stops_video_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingEngine:
        model_name = "notapalindrome/ltx23-mlx-av-q4"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            started.set()
            assert release.wait(timeout=5)
            output_path.write_bytes(b"late-output")

    monkeypatch.setattr(video, "_video_engine", lambda: BlockingEngine())
    video.start_video_jobs()
    created = await video.create_video(
        prompt="shutdown",
        model="ltx-2.3-mlx-q4",
        seconds="1",
        size="512x512",
        seed=5,
        input_reference=None,
    )
    assert await asyncio.to_thread(started.wait, 2)

    loop = asyncio.get_running_loop()
    began = loop.time()
    await video.shutdown_video_jobs(timeout=0.02)
    assert loop.time() - began < 0.5
    assert video._jobs[created["id"]].generation_finished is False
    assert video._jobs[created["id"]].error["code"] == "video_server_shutdown"
    with pytest.raises(HTTPException, match="shutting down") as exc:
        await video.create_video(
            prompt="too late",
            model="ltx-2.3-mlx-q4",
            seconds="1",
            size="512x512",
            seed=6,
            input_reference=None,
        )
    assert exc.value.status_code == 503

    release.set()
    for _ in range(200):
        if (
            not video._generation_threads
            and video._jobs[created["id"]].generation_finished
        ):
            break
        await asyncio.sleep(0.01)
    assert video._jobs[created["id"]].generation_finished is True
    video.start_video_jobs()
    await video.delete_video(created["id"])
