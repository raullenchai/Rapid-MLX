from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import HTTPException

from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.routes import video
from vllm_mlx.runtime.video_lane import VideoEngine


def test_cogvideox_aliases_route_to_video_lane() -> None:
    for alias in (
        "cogvideox-fun-5b-q4",
        "cogvideox-fun-5b-q8",
        "cogvideox-fun-5b-bf16",
    ):
        profile = resolve_profile(alias)
        assert profile is not None
        assert profile.modality == "video-gen"
        assert "CogVideoX-Fun" in profile.hf_path


def test_cogvideox_engine_uses_persistent_worker(monkeypatch, tmp_path) -> None:
    from vllm_mlx.video.engine import VideoGenerationEngine

    engine = VideoGenerationEngine("test/model", output_dir=tmp_path)
    thread_ids: list[int] = []

    def fake_generate(**kwargs):
        thread_ids.append(threading.get_ident())
        target = tmp_path / f"{len(thread_ids)}.mp4"
        target.write_bytes(b"video")
        return target

    monkeypatch.setattr(engine, "_generate_sync", fake_generate)
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    engine.generate_sync(output_path=first, prompt="one")
    engine.generate_sync(output_path=second, prompt="two")
    asyncio.run(engine.close())

    assert first.read_bytes() == b"video"
    assert second.read_bytes() == b"video"
    assert len(set(thread_ids)) == 1


def test_cogvideox_tokenizer_falls_back_to_upstream(tmp_path) -> None:
    from vllm_mlx.video.engine import _resolve_tokenizer_path

    calls = []

    def fake_download(repo, **kwargs):
        calls.append((repo, kwargs))
        return "/cached/upstream"

    assert _resolve_tokenizer_path(str(tmp_path), fake_download) == "/cached/upstream"
    assert calls[0][0] == "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"
    assert "tokenizer/spiece.model" in calls[0][1]["allow_patterns"]


@pytest.mark.asyncio
async def test_cogvideox_job_maps_validated_mvp_shape(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeEngine:
        model_name = "dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4"
        video_family = "cogvideox-fun"

        def generate(self, *, output_path: Path, **kwargs) -> None:
            captured.update(kwargs)
            output_path.write_bytes(b"mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="sunset",
        model="cogvideox-fun-5b-q4",
        seconds="1",
        size="672x384",
        seed=7,
        input_reference=None,
    )
    for _ in range(100):
        current = await video.retrieve_video(created["id"])
        if current["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert current["status"] == "completed"
    assert captured["width"] == 672
    assert captured["height"] == 384
    assert captured["num_frames"] == 5
    assert captured["fps"] == 5
    await video.delete_video(created["id"])


@pytest.mark.asyncio
async def test_cogvideox_mvp_rejects_unsupported_shape(monkeypatch) -> None:
    engine = SimpleNamespace(
        model_name="dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4",
        video_family="cogvideox-fun",
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)
    with pytest.raises(HTTPException, match="size=672x384"):
        await video.create_video(
            prompt="test",
            model="cogvideox-fun-5b-q4",
            seconds="1",
            size="768x512",
            seed=1,
            input_reference=None,
        )


def test_video_engine_delegates_cogvideox(monkeypatch, tmp_path) -> None:
    fake_module = ModuleType("vllm_mlx.video.engine")
    captured = {}

    class FakeCogEngine:
        def __init__(self, model_name):
            captured["model_name"] = model_name

        def generate_sync(self, **kwargs):
            captured.update(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"mp4")

        async def close(self):
            pass

    fake_module.VideoGenerationEngine = FakeCogEngine
    monkeypatch.setitem(sys.modules, "vllm_mlx.video.engine", fake_module)
    engine = VideoEngine("dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4")
    output = tmp_path / "result.mp4"
    engine.generate(
        prompt="sunset",
        output_path=output,
        width=672,
        height=384,
        num_frames=5,
        fps=5,
        seed=42,
        image=None,
    )
    assert output.read_bytes() == b"mp4"
    assert captured["frames"] == 5
