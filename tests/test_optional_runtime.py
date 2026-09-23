# SPDX-License-Identifier: Apache-2.0
"""Optional serving lanes converge on one actionable telemetry failure."""

from __future__ import annotations

import sys
from collections import namedtuple
from types import SimpleNamespace

import pytest

from rapid_mlx import cli, server
from rapid_mlx.runtime.optional_runtime import OptionalRuntimeMissing
from rapid_mlx.telemetry import model_events, registry, server_start


@pytest.fixture(autouse=True)
def _reset_one_shot_state():
    server_start._reset_for_tests()
    model_events._reset_for_tests()
    yield
    server_start._reset_for_tests()
    model_events._reset_for_tests()


def _capture(monkeypatch):
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr("rapid_mlx.telemetry.track._upload_allowed", lambda: True)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.posthog_sender.install_atexit", lambda: None
    )
    monkeypatch.setattr(
        "rapid_mlx.telemetry.track.track",
        lambda event, props, **_kwargs: events.append((event, dict(props))),
    )
    return events


def _lane_failure(monkeypatch, lane: str) -> tuple[OptionalRuntimeMissing, str]:
    if lane == "vision":
        from rapid_mlx.models import mllm

        monkeypatch.setattr(
            mllm,
            "vision_runtime_status",
            lambda: (mllm.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
        )
        call = lambda: mllm.require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit")
        model = "ui-tars-1.5-7b-4bit"
    elif lane == "video":
        from rapid_mlx.runtime import video_lane

        version = namedtuple("Version", "major minor")
        monkeypatch.setattr(video_lane.sys, "version_info", version(3, 11))
        monkeypatch.setattr(video_lane, "_is_ltx25_name", lambda _name: False)
        monkeypatch.setattr(video_lane, "_is_cogvideox_name", lambda _name: False)
        monkeypatch.setattr(
            video_lane,
            "_default_video_runtime_requirements",
            lambda _name: ["the `rapid-mlx[video]` Python extra"],
        )
        monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "ffmpeg")
        call = lambda: video_lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
        model = "wan2.2-ti2v-5b-q8"
    elif lane == "image":
        from rapid_mlx.runtime import image_lane

        monkeypatch.setattr(
            image_lane,
            "image_runtime_issue",
            lambda _name: (
                "image generation requires the `rapid-mlx[image]` Python extra "
                "(`pip install 'rapid-mlx[image]'`)."
            ),
        )
        call = lambda: image_lane.require_image_runtime_or_exit("flux-schnell")
        model = "flux-schnell"
    else:
        import importlib.util

        from rapid_mlx.audio import probe

        monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
        call = lambda: probe.require_audio_or_exit("kokoro")
        model = "kokoro"

    with pytest.raises(OptionalRuntimeMissing) as caught:
        call()
    return caught.value, model


@pytest.mark.parametrize("lane", ["vision", "video", "audio", "image"])
def test_missing_lane_is_one_actionable_telemetered_failure(
    monkeypatch, capsys, lane: str
) -> None:
    failure, model = _lane_failure(monkeypatch, lane)
    events = _capture(monkeypatch)
    server_start.attempted(model, load_policy="eager")

    with pytest.raises(SystemExit) as caught:
        cli._handle_optional_runtime_missing(failure, alias_or_path=model)

    assert caught.value.code == 2
    stderr = capsys.readouterr().err
    assert failure.install_hint in stderr
    assert f"RAPID-MLX-STARTUP-FAILURE: {failure.marker_reason} extra={lane}" in stderr
    terminal = [
        props
        for name, props in events
        if name == "server_start_state" and props["state"] != "attempted"
    ]
    failures = [props for name, props in events if name == "model_serve_failed"]
    assert terminal == [
        {
            "state": "failed",
            "model_type": model_events.model_type(model),
            "load_policy": "eager",
            "failure_stage": "preflight",
        }
    ]
    assert len(failures) == 1
    assert failures[0]["error_class"] == "missing_extra"
    assert failures[0]["extra"] == lane
    assert "detail" not in failures[0]
    assert registry.validate("model_serve_failed", failures[0]) == failures[0]


def test_bonsai_engine_preflight_is_missing_vision_failure(monkeypatch, capsys) -> None:
    from rapid_mlx.models import mllm
    from rapid_mlx import model_aliases, model_metadata

    real_resolve_profile = model_aliases.resolve_profile
    monkeypatch.setattr(model_aliases, "resolve_model", lambda model: model)
    monkeypatch.setattr(model_aliases, "resolve_profile", lambda _model: None)
    monkeypatch.setattr(
        server, "_prefetch_routing_metadata", lambda _model: "/cached/bonsai"
    )
    monkeypatch.setattr(
        model_metadata,
        "read_model_metadata",
        lambda _path: SimpleNamespace(
            snapshot_dir=None,
            config={"model_type": "prism_hadamard_qwen35"},
        ),
    )
    monkeypatch.setattr(
        model_metadata, "checkpoint_has_multimodal_weights", lambda *_a: False
    )
    monkeypatch.setattr(model_metadata, "config_indicates_multimodal", lambda _c: False)
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )

    with pytest.raises(OptionalRuntimeMissing) as caught:
        server._preflight_vision_runtime("bonsai2-27b-2bit")
    monkeypatch.setattr(model_aliases, "resolve_profile", real_resolve_profile)

    events = _capture(monkeypatch)
    server_start.attempted("bonsai2-27b-2bit", load_policy="eager")
    with pytest.raises(SystemExit) as exited:
        cli._handle_optional_runtime_missing(
            caught.value, alias_or_path="bonsai2-27b-2bit"
        )

    assert exited.value.code == 2
    assert "rapid-mlx[vision]" in capsys.readouterr().err
    assert [props for name, props in events if name == "model_serve_failed"] == [
        {
            "error_class": "missing_extra",
            "extra": "vision",
            "model": "bonsai2-27b-2bit",
            "model_type": "vlm",
            "auto_selected": False,
            "quant": "2bit",
        }
    ]
    assert [
        props
        for name, props in events
        if name == "server_start_state" and props["state"] == "failed"
    ][0]["failure_stage"] == "preflight"


def test_present_extras_emit_no_failure(monkeypatch) -> None:
    from rapid_mlx.audio import probe
    from rapid_mlx.models import mllm
    from rapid_mlx.runtime import image_lane, video_lane

    events = _capture(monkeypatch)
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.OK, None),
    )
    monkeypatch.setattr(image_lane, "image_runtime_issue", lambda _name: None)
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())

    mllm.require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit")
    monkeypatch.setattr(video_lane.sys, "version_info", sys.version_info)
    monkeypatch.setattr(video_lane, "_is_ltx25_name", lambda _name: False)
    monkeypatch.setattr(video_lane, "_is_cogvideox_name", lambda _name: False)
    monkeypatch.setattr(video_lane, "_default_video_runtime_requirements", lambda _: [])
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "ffmpeg")
    video_lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
    image_lane.require_image_runtime_or_exit("flux-schnell")
    probe.require_audio_or_exit("kokoro")

    assert events == []
