# SPDX-License-Identifier: Apache-2.0
"""Wan 2.1 / 2.2 integration with the unified /v1/videos job API."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import HTTPException

from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.routes import video
from vllm_mlx.runtime.video_lane import VideoEngine, require_video_runtime_or_exit
from vllm_mlx.video.wan import WanBackendError, WanRequestError, WanVideoEngine


def _checkpoint(tmp_path: Path, **overrides) -> Path:
    config = {
        "model_type": "ti2v",
        "model_version": "2.2",
        "sample_fps": 24,
        "max_area": 901120,
    }
    config.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def _fake_generate_module(monkeypatch, captured: dict) -> None:
    module = ModuleType("mlx_video.generate_wan")

    def generate_video(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"\x00\x00\x00\x18ftypmp42")

    module.generate_video = generate_video
    monkeypatch.setitem(sys.modules, "mlx_video.generate_wan", module)


@pytest.fixture(autouse=True)
def _reset_video_job_lifecycle():
    video.start_video_jobs()
    yield


def test_wan_aliases_route_to_video_lane() -> None:
    expected = {
        "wan2.2-ti2v-5b-q8": "Anes1032/Wan2.2-TI2V-5B-mlx-q8",
        "wan2.2-ti2v-5b-bf16": "rickylin20260522/Wan2.2-TI2V-5B-mlx",
        "wan2.2-i2v-a14b-q8": "Anes1032/Wan2.2-I2V-A14B-mlx-q8",
        "wan2.2-t2v-a14b-bf16": "rickylin20260522/Wan2.2-T2V-A14B-mlx",
    }
    for alias, repo in expected.items():
        profile = resolve_profile(alias)
        assert profile is not None
        assert profile.hf_path == repo
        assert profile.modality == "video-gen"


def test_wan_engine_resolves_public_alias_at_pinned_revision(
    monkeypatch, tmp_path
) -> None:
    _checkpoint(tmp_path)
    captured = {}

    def fake_snapshot_download(repository, *, revision):
        captured.update(repository=repository, revision=revision)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    WanVideoEngine("wan2.2-ti2v-5b-q8")
    assert captured == {
        "repository": "Anes1032/Wan2.2-TI2V-5B-mlx-q8",
        "revision": "9624723c94ddf509832555c45e223a035baa7d1c",
    }


def test_wan_rejects_unpinned_remote_repository() -> None:
    with pytest.raises(WanBackendError, match="not registered at a pinned revision"):
        WanVideoEngine("someone/Wan2.2-custom-mlx")


def test_wan_wraps_non_utf8_config(monkeypatch, tmp_path) -> None:
    (tmp_path / "config.json").write_bytes(b"\xff\xfe")
    monkeypatch.setenv("RAPID_MLX_WAN_MODEL_DIR", str(tmp_path))
    with pytest.raises(WanBackendError, match="unreadable config.json"):
        WanVideoEngine("wan2.2-ti2v-5b-q8")


def test_wan_runtime_guard_checks_wan_module(monkeypatch, capsys) -> None:
    checked = []

    def fake_find_spec(module):
        checked.append(module)
        return None if module == "mlx_video.generate_wan" else object()

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)
    monkeypatch.setattr("shutil.which", lambda _: "/opt/homebrew/bin/ffmpeg")
    with pytest.raises(SystemExit):
        require_video_runtime_or_exit("Anes1032/Wan2.2-TI2V-5B-mlx-q8")
    assert "mlx_video.generate_wan" in checked
    assert "rapid-mlx[video]" in capsys.readouterr().err


def test_wan_runtime_guard_handles_missing_parent_package(monkeypatch, capsys) -> None:
    def fake_find_spec(module):
        if module == "mlx_video":
            return None
        raise AssertionError(f"must not probe child module without parent: {module}")

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)
    monkeypatch.setattr("shutil.which", lambda _: "/opt/homebrew/bin/ffmpeg")
    with pytest.raises(SystemExit):
        require_video_runtime_or_exit("Anes1032/Wan2.2-TI2V-5B-mlx-q8")
    assert "rapid-mlx[video]" in capsys.readouterr().err


def test_wan_engine_maps_current_mlx_video_api(monkeypatch, tmp_path) -> None:
    captured: dict = {}
    _fake_generate_module(monkeypatch, captured)
    engine = WanVideoEngine(_checkpoint(tmp_path))
    output = tmp_path / "result.mp4"

    engine.generate(
        prompt="a fox in snow",
        output_path=output,
        width=1280,
        height=704,
        num_frames=49,
        seed=7,
        image=None,
    )

    assert output.is_file()
    assert captured["model_dir"] == str(tmp_path)
    assert captured["prompt"] == "a fox in snow"
    assert captured["num_frames"] == 49
    assert captured["scheduler"] == "unipc"
    assert captured["tiling"] == "auto"
    assert "fps" not in captured


def test_wan_lane_crops_aligned_generation_to_requested_size(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "result.mp4"

    class FakeWanEngine:
        native_fps = 24

        def generate(self, **kwargs):
            Path(kwargs["output_path"]).write_bytes(b"aligned")

    lane = VideoEngine.__new__(VideoEngine)
    lane._wan_engine = FakeWanEngine()
    lane._cog_engine = None
    lane._generation_lock = threading.Lock()
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        Path(command[-1]).write_bytes(b"cropped")

    monkeypatch.setattr(subprocess, "run", fake_run)
    lane.generate(
        prompt="test",
        output_path=output,
        width=1280,
        height=768,
        num_frames=25,
        fps=24,
        seed=1,
        image=None,
        output_width=1280,
        output_height=720,
    )

    assert output.read_bytes() == b"cropped"
    assert "crop=1280:720:(iw-ow)/2:(ih-oh)/2" in captured["command"]


@pytest.mark.parametrize("frames", [2, 24, 50, 96])
def test_wan_rejects_non_4n_plus_1_frames(tmp_path, frames) -> None:
    engine = WanVideoEngine(_checkpoint(tmp_path))
    with pytest.raises(WanRequestError, match=r"4n\+1"):
        engine.validate_request(width=832, height=480, num_frames=frames, image=None)


def test_wan_enforces_checkpoint_area_ceiling(tmp_path) -> None:
    engine = WanVideoEngine(_checkpoint(tmp_path))
    with pytest.raises(WanRequestError, match="ceiling"):
        engine.validate_request(width=1920, height=1080, num_frames=49, image=None)


@pytest.mark.parametrize("max_area", [None, 0, "invalid"])
def test_wan_uses_safe_area_ceiling_when_config_has_no_limit(
    tmp_path, max_area
) -> None:
    engine = WanVideoEngine(_checkpoint(tmp_path, max_area=max_area))
    assert engine.max_area == 704 * 1280


@pytest.mark.parametrize("model_type", [None, "", "unsupported"])
def test_wan_rejects_unknown_model_type(tmp_path, model_type) -> None:
    with pytest.raises(WanBackendError, match="unsupported model_type"):
        WanVideoEngine(_checkpoint(tmp_path, model_type=model_type))


def test_wan_rejects_malformed_lora_strength(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAPID_MLX_WAN_LORA", "adapter.safetensors:abc")
    with pytest.raises(WanBackendError, match="invalid Wan LoRA strength"):
        WanVideoEngine(_checkpoint(tmp_path))


def test_wan_model_type_controls_reference_image(tmp_path) -> None:
    image = tmp_path / "reference.png"
    image.write_bytes(b"png")
    t2v_dir = tmp_path / "t2v"
    t2v_dir.mkdir()
    i2v_dir = tmp_path / "i2v"
    i2v_dir.mkdir()
    t2v = WanVideoEngine(_checkpoint(t2v_dir, model_type="t2v"))
    i2v = WanVideoEngine(_checkpoint(i2v_dir, model_type="i2v"))

    with pytest.raises(WanRequestError, match="text-to-video only"):
        t2v.validate_request(width=832, height=480, num_frames=49, image=image)
    with pytest.raises(WanRequestError, match="requires input_reference"):
        i2v.validate_request(width=832, height=480, num_frames=49, image=None)


def test_wan21_native_fps_falls_back_to_16(tmp_path) -> None:
    engine = WanVideoEngine(_checkpoint(tmp_path, model_version="2.1", sample_fps=None))
    assert engine.native_fps == 16


def test_wan_native_fps_rejects_config_value_that_rounds_to_zero(tmp_path) -> None:
    engine = WanVideoEngine(_checkpoint(tmp_path, sample_fps=0.4))
    assert engine.native_fps == 24


@pytest.mark.asyncio
async def test_wan_job_uses_native_fps_and_async_video_contract(
    monkeypatch, tmp_path
) -> None:
    captured: dict = {}

    class FakeWanEngine:
        model_name = "Anes1032/Wan2.2-TI2V-5B-mlx-q8"
        video_family = "wan"
        native_fps = 24

        _wan_engine = None

        def validate_request(self, **kwargs):
            captured["validated"] = kwargs

        def generate(self, *, output_path: Path, **kwargs):
            captured.update(kwargs)
            output_path.write_bytes(b"mp4")

    engine = FakeWanEngine()
    engine._wan_engine = engine
    monkeypatch.setattr(video, "_video_engine", lambda: engine)
    created = await video.create_video(
        prompt="ocean sunrise",
        model="wan2.2-ti2v-5b-q8",
        seconds="1",
        size="832x512",
        seed=9,
        input_reference=None,
    )
    for _ in range(100):
        current = await video.retrieve_video(created["id"])
        if current["status"] == "completed":
            break
        await asyncio.sleep(0.01)

    assert current["status"] == "completed"
    assert current["model"] == "wan2.2-ti2v-5b-q8"
    assert captured["fps"] == 24
    assert captured["num_frames"] == 25
    assert captured["validated"]["num_frames"] == 25
    await video.delete_video(created["id"])


@pytest.mark.asyncio
async def test_wan_route_returns_model_validation_as_400(monkeypatch) -> None:
    class FakeWanEngine:
        model_name = "Anes1032/Wan2.2-TI2V-5B-mlx-q8"
        video_family = "wan"
        native_fps = 24

        _wan_engine = None

        def validate_request(self, **kwargs):
            raise WanRequestError("checkpoint is image-to-video only")

    engine = FakeWanEngine()
    engine._wan_engine = engine
    monkeypatch.setattr(video, "_video_engine", lambda: engine)
    with pytest.raises(HTTPException) as error:
        await video.create_video(
            prompt="test",
            model="wan2.2-ti2v-5b-q8",
            seconds="1",
            size="832x512",
            seed=1,
            input_reference=None,
        )
    assert error.value.status_code == 400
    assert "image-to-video only" in str(error.value.detail)


@pytest.mark.asyncio
async def test_wan_route_rejects_oversized_pixel_frame_workload(monkeypatch) -> None:
    class FakeWanEngine:
        model_name = "Anes1032/Wan2.2-TI2V-5B-mlx-q8"
        video_family = "wan"
        native_fps = 24
        _wan_engine = object()

    monkeypatch.setattr(video, "_video_engine", lambda: FakeWanEngine())
    with pytest.raises(HTTPException) as error:
        await video.create_video(
            prompt="test",
            model="wan2.2-ti2v-5b-q8",
            seconds="20",
            size="1280x704",
            seed=1,
            input_reference=None,
        )
    assert error.value.status_code == 400
    assert "safe Wan workload limit" in str(error.value.detail)
