from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_mlx.routes import video


@pytest.mark.asyncio
async def test_ltx_capabilities_expose_combined_workload_budget(monkeypatch) -> None:
    engine = SimpleNamespace(
        model_name="notapalindrome/ltx23-mlx-av-q4",
        video_family="ltx-2.3",
        native_fps=24,
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)

    body = await video.video_capabilities()

    assert body["object"] == "video.capabilities"
    assert body["model"] == engine.model_name
    assert body["modality"] == "video-gen"
    assert body["modes"] == ["text-to-video", "image-to-video"]
    assert body["limits"]["seconds"]["maximum"] == 20
    assert body["limits"]["fps"] == {
        "minimum": 1,
        "maximum": 60,
        "default": 24,
        "fixed": False,
    }
    assert body["limits"]["frames"]["step"] == 8
    assert body["limits"]["workload"]["maximum"] == 768 * 512 * 97
    assert body["controls"]["conditioning_strength"]["maximum"] == 1.0


@pytest.mark.asyncio
async def test_cogvideox_capabilities_report_fixed_mvp_shape(monkeypatch) -> None:
    engine = SimpleNamespace(
        model_name="dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4",
        video_family="cogvideox-fun",
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)

    body = await video.video_capabilities()

    assert body["modes"] == ["text-to-video"]
    assert body["limits"]["size"] == {"type": "fixed", "values": ["672x384"]}
    assert body["limits"]["seconds"]["maximum"] == 1
    assert body["limits"]["fps"]["default"] == 5
    assert body["limits"]["workload"]["dimension_rounding"] == "none"
    assert body["controls"]["conditioning_strength"] is None


@pytest.mark.asyncio
async def test_wan_capabilities_use_checkpoint_limits(monkeypatch) -> None:
    engine = SimpleNamespace(
        model_name="Anes1032/Wan2.2-TI2V-5B-mlx-q8",
        video_family="wan",
        native_fps=24,
        _wan_engine=SimpleNamespace(model_type="i2v", max_area=901_120),
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)

    body = await video.video_capabilities()

    assert body["modes"] == ["image-to-video"]
    assert body["limits"]["size"]["maximum_area"] == 901_120
    assert body["limits"]["size"]["also_supported"] == []
    assert body["limits"]["fps"] == {
        "minimum": 24,
        "maximum": 24,
        "default": 24,
        "fixed": True,
    }
    assert body["limits"]["frames"]["step"] == 4


@pytest.mark.asyncio
async def test_wan_only_lists_openai_sizes_accepted_after_alignment(
    monkeypatch,
) -> None:
    engine = SimpleNamespace(
        model_name="custom/wan",
        video_family="wan",
        native_fps=16,
        _wan_engine=SimpleNamespace(model_type="t2v", max_area=1280 * 768),
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)

    body = await video.video_capabilities()

    assert body["limits"]["size"]["also_supported"] == [
        "1280x720",
        "720x1280",
    ]


def test_capabilities_route_precedes_dynamic_video_id_route() -> None:
    routes = [route.path for route in video.router.routes]
    assert routes.index("/v1/videos/capabilities") < routes.index(
        "/v1/videos/{video_id}"
    )
