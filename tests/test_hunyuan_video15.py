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
        ("model_index.json", "_class_name", "OtherPipeline"),
        ("vae/config.json", "spatial_compression_ratio", 8),
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


def test_detection_rejects_unreadable_snapshot(tmp_path):
    assert not is_hunyuan_video15_model(None)
    assert not is_hunyuan_video15_model("")
    (tmp_path / "model_index.json").write_text("{broken json")
    assert not is_hunyuan_video15_model(str(tmp_path))


@pytest.mark.parametrize(
    "key,expected",
    [
        ("shared.weight", "wte.weight"),
        ("encoder.embed_tokens.weight", None),
        ("encoder.final_layer_norm.weight", "encoder.ln.weight"),
        (
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            "encoder.relative_attention_bias.embeddings.weight",
        ),
        ("encoder.block.1.layer.0.layer_norm.weight", "encoder.layers.1.ln1.weight"),
        ("encoder.block.1.layer.1.layer_norm.weight", "encoder.layers.1.ln2.weight"),
        (
            "encoder.block.1.layer.1.DenseReluDense.wi_0.weight",
            "encoder.layers.1.dense.wi_0.weight",
        ),
        *[
            (
                f"encoder.block.1.layer.0.SelfAttention.{source}.weight",
                f"encoder.layers.1.attention.{target}.weight",
            )
            for source, target in (
                ("q", "query_proj"),
                ("k", "key_proj"),
                ("v", "value_proj"),
                ("o", "out_proj"),
            )
        ],
    ],
)
def test_t5_checkpoint_key_mapping(key, expected):
    from rapid_mlx.video.hunyuan_video15.loading import t5_key

    assert t5_key(key) == expected


@pytest.mark.parametrize(
    "key,expected",
    [
        ("model.embed_tokens.weight", "embed_tokens.weight"),
        ("model.layers.0.self_attn.q_proj.weight", "layers.0.self_attn.q_proj.weight"),
        ("layers.1.mlp.up_proj.weight", "layers.1.mlp.up_proj.weight"),
        ("model.layers.2.mlp.up_proj.weight", None),
        ("model.norm.weight", None),
    ],
)
def test_qwen_checkpoint_key_mapping_truncates_before_final_norm(key, expected):
    from rapid_mlx.video.hunyuan_video15.loading import qwen_key

    assert qwen_key(key, 2) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["conditioning", "validation", "backend"])
async def test_lane_translates_request_and_backend_errors(
    monkeypatch, tmp_path, failure
):
    from rapid_mlx.runtime.video_lane import VideoRuntimeError
    from rapid_mlx.video.engine import VideoBackendUnavailableError

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        raise VideoBackendUnavailableError("encoder unavailable")

    engine = VideoEngine(REPO)
    monkeypatch.setattr(engine._hunyuan_engine, "generate_sync", generate)
    try:
        with pytest.raises(VideoRuntimeError) as exc:
            engine.generate(
                prompt="fox",
                output_path=tmp_path / "clip.mp4",
                width=256,
                height=256,
                num_frames=6 if failure == "validation" else 5,
                fps=24,
                seed=1,
                image=None,
                conditioning_strength=0.5 if failure == "conditioning" else None,
            )
        assert isinstance(
            exc.value.__cause__,
            VideoBackendUnavailableError if failure == "backend" else ValueError,
        )
        assert len(calls) == (1 if failure == "backend" else 0)
        assert not (tmp_path / "clip.mp4").exists()
    finally:
        await engine.stop()


def test_backend_rejects_unrecognized_model():
    with pytest.raises(ValueError, match="pinned HunyuanVideo"):
        HunyuanVideo15Engine("unrecognized/repository")


@pytest.mark.parametrize("missing", [None, "ffmpeg", "mlx_lm"])
def test_hunyuan_runtime_dependencies(monkeypatch, missing):
    from rapid_mlx.runtime import video_lane

    monkeypatch.setattr(
        video_lane.importlib.util,
        "find_spec",
        lambda name: None if name == missing else SimpleNamespace(),
    )
    monkeypatch.setattr(
        video_lane, "_resolve_ffmpeg", lambda: None if missing == "ffmpeg" else "ffmpeg"
    )
    issue = video_lane.video_runtime_issue(REPO, provision=False)
    if missing is None:
        assert issue is None
    else:
        assert issue.detail == f"HunyuanVideo 1.5 requires {missing}"
        assert issue.status == "broken"
        assert issue.marker_reason == "runtime_dependency_missing"
        assert issue.install_hint


def test_lane_constructs_hunyuan_backend(monkeypatch):
    from rapid_mlx.video.hunyuan_video15 import backend

    calls = []
    sentinel = object()

    def construct(model_name):
        calls.append(model_name)
        return sentinel

    monkeypatch.setattr(backend, "HunyuanVideo15Engine", construct)
    engine = VideoEngine(REPO)
    assert calls == [REPO]
    assert engine._hunyuan_engine is sentinel
    assert engine.video_family == "hunyuan-video-1.5"


@pytest.mark.parametrize("remote", [False, True])
def test_backend_loads_local_or_pinned_snapshot(monkeypatch, checkpoint, remote):
    from rapid_mlx import model_downloads

    events, checks, downloads = [], [], []

    def download(repo, *, revision, tqdm_class):
        downloads.append((repo, revision))
        with tqdm_class(total=2, disable=True) as progress:
            progress.update(2)
        return str(checkpoint)

    monkeypatch.setattr(model_downloads, "snapshot_download", download)
    engine = HunyuanVideo15Engine(REPO if remote else str(checkpoint))
    try:
        assert (
            engine._load_sync(lambda: checks.append(True), lambda *a: events.append(a))
            == checkpoint
        )
        assert engine.config == PipelineConfig.from_path(checkpoint)
        assert engine._load_sync() == checkpoint
        assert downloads == ([(REPO, REVISION)] if remote else [])
        if remote:
            assert checks and events and events[-1][0] == "downloading"
    finally:
        asyncio.run(engine.close())


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
