# SPDX-License-Identifier: Apache-2.0
"""No-weight, no-MLX HunyuanVideo checkpoint and serving contracts."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from rapid_mlx.model_aliases import resolve_profile, resolve_revision
from rapid_mlx.routes import video
from rapid_mlx.runtime.video_lane import VideoEngine, validate_video_request
from rapid_mlx.video.hunyuan_video15 import (
    ALIAS,
    REPO,
    REVISION,
    FlowMatchScheduler,
    PipelineConfig,
    is_hunyuan_video15_model,
    validate_request,
)
from rapid_mlx.video.hunyuan_video15.backend import HunyuanVideo15Engine


@pytest.fixture
def checkpoint(tmp_path):
    configs = {
        "model_index.json": {"_class_name": "HunyuanVideo15Pipeline"},
        "transformer/config.json": {
            "task_type": "t2v",
            "target_size": 640,
            "use_meanflow": False,
            "in_channels": 65,
            "out_channels": 32,
            "rope_axes_dim": [16, 56, 56],
            "attention_head_dim": 128,
        },
        "vae/config.json": {
            "latent_channels": 32,
            "spatial_compression_ratio": 16,
            "temporal_compression_ratio": 4,
        },
        "scheduler/scheduler_config.json": {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "shift": 5.0,
        },
        "guider/guider_config.json": {
            "_class_name": "ClassifierFreeGuidance",
            "guidance_scale": 6.0,
        },
    }
    for name, data in configs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return tmp_path


def test_checkpoint_config_defaults_and_detection(checkpoint):
    c = PipelineConfig.from_path(checkpoint)
    assert (c.width, c.height, c.frames, c.fps, c.steps) == (848, 480, 121, 24, 50)
    assert c.scheduler["shift"] == 5 and c.guider["guidance_scale"] == 6
    assert is_hunyuan_video15_model(str(checkpoint))
    assert is_hunyuan_video15_model(ALIAS)
    assert is_hunyuan_video15_model(REPO)
    assert not is_hunyuan_video15_model("tencent/HunyuanVideo-1.5")


@pytest.mark.parametrize(
    "file,key,value",
    [
        ("transformer/config.json", "task_type", "i2v"),
        ("transformer/config.json", "target_size", 960),
        ("transformer/config.json", "use_meanflow", True),
        ("transformer/config.json", "in_channels", 64),
        ("transformer/config.json", "rope_axes_dim", [1, 2, 3]),
        ("scheduler/scheduler_config.json", "use_dynamic_shifting", True),
        ("scheduler/scheduler_config.json", "shift", float("nan")),
        ("guider/guider_config.json", "guidance_rescale", 0.5),
    ],
)
def test_config_rejects_unsupported_variants(checkpoint, file, key, value):
    path = checkpoint / file
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        PipelineConfig.from_path(checkpoint)


def test_alias_and_live_catalog_match():
    profile = resolve_profile(ALIAS)
    assert profile.hf_path == REPO
    assert profile.modality == "video-gen"
    assert profile.min_memory_gb == 64
    assert profile.video_modes == ("text-to-video",)
    assert resolve_revision(REPO) == REVISION
    body = video._video_capabilities(
        SimpleNamespace(
            model_name=REPO, video_family="hunyuan-video-1.5", native_fps=24
        )
    )
    assert body["modes"] == list(profile.video_modes)
    assert body["limits"]["frames"]["maximum"] == 121
    assert body["limits"]["fps"]["fixed"]
    assert body["controls"]["conditioning_strength"] is None
    assert body["limits"]["workload"]["dimension_rounding"] == "ceil_to_16"


@pytest.mark.parametrize(
    "width,height,frames,image",
    [
        (848, 480, 33, None),
        (480, 848, 121, None),
        (256, 256, 5, None),
    ],
)
def test_valid_shapes(width, height, frames, image):
    validate_request(width=width, height=height, num_frames=frames, image=image)


@pytest.mark.parametrize(
    "width,height,frames,image",
    [
        (848, 480, 32, None),
        (848, 480, 125, None),
        (848, 480, 1, None),
        (1280, 720, 33, None),
        (512, 512, 33, None),
        (850, 480, 33, None),
        (848, 480, 33, Path("image.png")),
    ],
)
def test_invalid_shapes(width, height, frames, image):
    with pytest.raises(ValueError):
        validate_request(width=width, height=height, num_frames=frames, image=image)


def test_scheduler_sequence_and_step_bounds():
    schedule = FlowMatchScheduler({"shift": 5}, 2)
    np.testing.assert_allclose(schedule.sigmas, [1, 5 / 6, 0])
    sample = np.array([2.0], dtype=np.float32)
    for index in range(2):
        sample = schedule.step(np.array([3.0], dtype=np.float32), index, sample)
    np.testing.assert_allclose(sample, [-1.0])
    with pytest.raises(ValueError):
        FlowMatchScheduler({"shift": 5}, 0)


def test_hub_direct_inventory_exclusion():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "scripts/mirror_unmirrored.json").read_text())
    assert any(
        x["hf_path"] == REPO and "license:" in x["reason"] for x in data["entries"]
    )
    notice = (root / "rapid_mlx/video/hunyuan_video15/NOTICE").read_text()
    assert "Copyright © 2025 Tencent" in notice


@pytest.mark.asyncio
async def test_shared_lane_dispatch_and_worker_ownership(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        HunyuanVideo15Engine, "generate_sync", lambda self, **kw: captured.update(kw)
    )
    engine = VideoEngine(REPO)
    try:
        assert engine.video_family == "hunyuan-video-1.5"
        assert engine.native_fps == 24
        validate_video_request(engine, width=848, height=480, num_frames=33, image=None)
        engine.generate(
            prompt="fox",
            output_path=tmp_path / "clip.mp4",
            width=848,
            height=480,
            num_frames=33,
            fps=24,
            seed=42,
            image=None,
            guidance_scale=5,
            negative_prompt="rain",
        )
        assert captured["frames"] == 33 and captured["guidance_scale"] == 5
        assert captured["negative_prompt"] == "rain"
        with pytest.raises(ValueError, match="text-to-video"):
            validate_video_request(
                engine, width=848, height=480, num_frames=33, image=Path("input.png")
            )
    finally:
        await engine.stop()
    assert not engine._hunyuan_engine._worker.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"frames": 32}, r"4n\+1"),
        ({"fps": 12}, "fixed at 24"),
        ({"size": "1280x720"}, "480p dimensions"),
        ({"seconds": "6"}, "between 1 and 5"),
        ({"input_reference": object()}, "text-to-video"),
        ({"conditioning_strength": 0.5}, "LTX only"),
    ],
)
async def test_route_validation_before_admission(monkeypatch, overrides, match):
    video.start_video_jobs()
    engine = SimpleNamespace(
        model_name=REPO, video_family="hunyuan-video-1.5", native_fps=24
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)
    args = dict(
        prompt="fox",
        model=ALIAS,
        seconds="2",
        size="848x480",
        seed=42,
        input_reference=None,
    )
    args.update(overrides)
    with pytest.raises(HTTPException, match=match) as exc:
        await video.create_video(**args)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("frames,expected", [(None, 121), (33, 33)])
async def test_route_generates_default_resolution_without_rounding(
    monkeypatch, frames, expected
):
    video.start_video_jobs()
    captured = {}

    class FakeEngine:
        model_name = REPO
        video_family = "hunyuan-video-1.5"
        native_fps = 24

        def generate(self, *, output_path, **kwargs):
            captured.update(kwargs)
            output_path.write_bytes(b"mp4")

    monkeypatch.setattr(video, "_video_engine", lambda: FakeEngine())
    created = await video.create_video(
        prompt="fox",
        model=ALIAS,
        size=None,
        seconds=None,
        seed=42,
        frames=frames,
        input_reference=None,
    )
    try:
        for _ in range(200):
            result = await video.retrieve_video(created["id"])
            if result["status"] in ("failed", "completed"):
                break
            await asyncio.sleep(0.01)
        assert result["status"] == "completed", result
        assert result["model"] == ALIAS
        assert (captured["width"], captured["height"], captured["num_frames"]) == (
            848,
            480,
            expected,
        )
    finally:
        await video.delete_video(created["id"])
