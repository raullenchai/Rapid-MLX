# SPDX-License-Identifier: Apache-2.0
"""Live text concurrency is scheduler truth, independent of the lane name."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx.api.models import ModelInfo
from rapid_mlx.engine.batched import BatchedEngine
from rapid_mlx.routes import models as models_route


@pytest.fixture
def scheduler_config_stub(monkeypatch):
    """Keep the default-config branch independent of the Apple-only scheduler."""
    scheduler = ModuleType("rapid_mlx.scheduler")
    scheduler.SchedulerConfig = Mock(return_value=SimpleNamespace(max_num_seqs=256))
    monkeypatch.setitem(sys.modules, "rapid_mlx.scheduler", scheduler)
    return scheduler.SchedulerConfig


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ({}, None),
        ({"_loaded": False}, None),
        ({"_scheduler_config": SimpleNamespace(max_num_seqs=256)}, True),
        ({"_scheduler_config": SimpleNamespace(max_num_seqs=1)}, False),
        ({"_scheduler_config": None}, True),
        ({"_scheduler_config": SimpleNamespace()}, False),
        (
            {
                "_is_mllm": True,
                "_mllm_native_text_engine": True,
                "_scheduler_config": SimpleNamespace(max_num_seqs=256),
                "_mllm_scheduler": SimpleNamespace(
                    config=SimpleNamespace(max_num_seqs=1)
                ),
            },
            True,
        ),
        (
            {
                "_is_mllm": True,
                "_mllm_native_text_engine": True,
                "_scheduler_config": SimpleNamespace(max_num_seqs=1),
            },
            False,
        ),
        (
            {
                "_is_mllm": True,
                "_mllm_scheduler": SimpleNamespace(
                    config=SimpleNamespace(max_num_seqs=1)
                ),
            },
            False,
        ),
        (
            {
                "_is_mllm": True,
                "_mllm_scheduler": SimpleNamespace(
                    config=SimpleNamespace(max_num_seqs=16)
                ),
            },
            True,
        ),
        ({"_is_mllm": True, "_mllm_scheduler": None}, None),
        ({"_is_mllm": True, "_mllm_scheduler": SimpleNamespace()}, False),
    ],
)
def test_engine_concurrent_text_requests(attributes, expected, scheduler_config_stub):
    engine = BatchedEngine.__new__(BatchedEngine)
    if attributes:
        engine._loaded = True
    engine.__dict__.update(attributes)
    assert engine.concurrent_text_requests is expected


@pytest.mark.parametrize(
    ("engine", "expected"),
    [
        (SimpleNamespace(concurrent_text_requests=True), True),
        (SimpleNamespace(concurrent_text_requests=False), False),
        (SimpleNamespace(concurrent_text_requests=1), None),
        (SimpleNamespace(concurrent_text_requests="true"), None),
        (SimpleNamespace(concurrent_text_requests=None), None),
        (SimpleNamespace(), None),
        (None, None),
    ],
)
def test_served_text_concurrency(monkeypatch, engine, expected):
    lookup = Mock(return_value=engine)
    monkeypatch.setattr(models_route, "_engine_for", lookup)
    assert models_route._served_text_concurrency("m") is expected
    lookup.assert_called_once_with("m")


def test_served_text_concurrency_never_raises(monkeypatch):
    monkeypatch.setattr(models_route, "_engine_for", Mock(side_effect=RuntimeError))
    assert models_route._served_text_concurrency("m") is None

    class BrokenEngine:
        @property
        def concurrent_text_requests(self):
            raise RuntimeError("engine unavailable")

    monkeypatch.setattr(models_route, "_engine_for", Mock(return_value=BrokenEngine()))
    assert models_route._served_text_concurrency("m") is None


@pytest.mark.parametrize("concurrent", [True, False, None])
def test_model_info_serializes_concurrent_text_requests(concurrent):
    info = ModelInfo(id="m", concurrent_text_requests=concurrent)
    assert json.loads(info.model_dump_json())["concurrent_text_requests"] is concurrent
    assert ModelInfo(id="discovery").concurrent_text_requests is None


@pytest.mark.parametrize("profile_kind", ["registered", "raw_text", "raw_vision"])
@pytest.mark.parametrize("concurrent", [True, False, None])
def test_model_info_paths_propagate_concurrency(monkeypatch, profile_kind, concurrent):
    profile = models_route.resolve_profile("qwen3.5-4b-4bit")
    assert profile is not None
    monkeypatch.setattr(
        models_route,
        "resolve_profile",
        Mock(return_value=profile if profile_kind == "registered" else None),
    )
    monkeypatch.setattr(
        models_route,
        "_engine_for",
        Mock(
            return_value=SimpleNamespace(
                concurrent_text_requests=concurrent,
                serving_lane="vision",
                serving_lane_reason="vision_supported",
            )
        ),
    )
    monkeypatch.setattr(models_route, "_resolve_audio_entry", Mock(return_value=None))
    monkeypatch.setattr(models_route, "_locked_embedding_id", Mock(return_value=None))
    monkeypatch.setattr(models_route, "_locked_rerank_id", Mock(return_value=None))
    monkeypatch.setattr(
        models_route,
        "_reported_modality",
        Mock(return_value="image" if profile_kind == "raw_vision" else "text"),
    )
    info = models_route._build_model_info("m")
    assert info.concurrent_text_requests is concurrent
    assert info.serving_lane == "vision"
    assert json.loads(info.model_dump_json())["concurrent_text_requests"] is concurrent
