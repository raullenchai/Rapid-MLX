# SPDX-License-Identifier: Apache-2.0
"""Cache diagnostics follow the serving backend without optional global APIs."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.requires_mlx
@pytest.mark.parametrize("is_mllm", [False, True])
def test_installed_backend_diagnostics_do_not_claim_request_readiness(
    monkeypatch, is_mllm
):
    from rapid_mlx.engine.batched import BatchedEngine
    from rapid_mlx.routes import health

    engine = BatchedEngine("not-started")
    engine._is_mllm = is_mllm
    assert engine._engine is None
    assert engine._mllm_scheduler is None
    assert engine.get_cache_stats() is None
    monkeypatch.setitem(
        sys.modules, "mlx_vlm", SimpleNamespace(utils=SimpleNamespace())
    )
    monkeypatch.setattr(
        health, "get_config", lambda: SimpleNamespace(engine=engine, ready=False)
    )
    result = asyncio.run(health.cache_stats())
    assert result["model_type"] == ("mllm" if is_mllm else "llm")
    assert result["multimodal_kv_cache"] is None
    assert result["pixel_values_cache"] is None
    assert result["pil_image_cache"] is None


@pytest.mark.parametrize("stats", [None, {"entry_count": 2, "hits": 3}])
def test_vision_cache_stats_use_active_engine(monkeypatch, stats):
    from rapid_mlx.routes import health

    monkeypatch.setitem(
        sys.modules, "mlx_vlm", SimpleNamespace(utils=SimpleNamespace())
    )
    engine = SimpleNamespace(is_mllm=True, get_cache_stats=Mock(return_value=stats))
    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))
    assert asyncio.run(health.cache_stats()) == {
        "model_type": "mllm",
        "multimodal_kv_cache": stats,
        "pixel_values_cache": None,
        "pil_image_cache": None,
    }
    engine.get_cache_stats.assert_called_once_with()


def test_vision_cache_stats_isolate_engine_counter_failure(monkeypatch):
    from rapid_mlx.routes import health

    utils = SimpleNamespace(
        get_pixel_values_cache_stats=Mock(return_value={"hits": 7}),
        get_pil_cache_stats=Mock(return_value={"hits": 11}),
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(utils=utils))
    engine = SimpleNamespace(
        is_mllm=True,
        get_cache_stats=Mock(side_effect=RuntimeError("counter unavailable")),
    )
    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))

    assert asyncio.run(health.cache_stats()) == {
        "model_type": "mllm",
        "multimodal_kv_cache": None,
        "pixel_values_cache": {"hits": 7},
        "pil_image_cache": {"hits": 11},
    }
    engine.get_cache_stats.assert_called_once_with()


@pytest.mark.parametrize(
    "engine,model_type", [(None, None), (SimpleNamespace(is_mllm=False), "llm")]
)
def test_cache_stats_do_not_infer_model_type_from_installed_packages(
    monkeypatch, engine, model_type
):
    from rapid_mlx.routes import health

    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))
    result = asyncio.run(health.cache_stats())
    assert result["model_type"] == model_type
    assert result["message"]
    assert result == {
        "message": result["message"],
        "model_type": model_type,
        "multimodal_kv_cache": None,
        "pixel_values_cache": None,
        "pil_image_cache": None,
    }


@pytest.mark.parametrize(
    "pixel_available,pil_available", [(True, True), (True, False), (False, True)]
)
def test_vision_cache_stats_preserve_available_optional_image_stats(
    monkeypatch, pixel_available, pil_available
):
    from rapid_mlx.routes import health

    kv_stats = {"entry_count": 2}
    pixel_stats = {"hits": 7}
    pil_stats = {"hits": 11}
    utils = SimpleNamespace()
    if pixel_available:
        utils.get_pixel_values_cache_stats = Mock(return_value=pixel_stats)
    if pil_available:
        utils.get_pil_cache_stats = Mock(return_value=pil_stats)
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(utils=utils))
    engine = SimpleNamespace(is_mllm=True, get_cache_stats=lambda: kv_stats)
    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))

    assert asyncio.run(health.cache_stats()) == {
        "model_type": "mllm",
        "multimodal_kv_cache": kv_stats,
        "pixel_values_cache": pixel_stats if pixel_available else None,
        "pil_image_cache": pil_stats if pil_available else None,
    }


def test_vision_cache_stats_work_without_optional_image_apis(monkeypatch):
    from rapid_mlx.routes import health

    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    engine = SimpleNamespace(is_mllm=True, get_cache_stats=lambda: {"hits": 3})
    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))
    result = asyncio.run(health.cache_stats())
    assert result["model_type"] == "mllm"
    assert result["multimodal_kv_cache"] == {"hits": 3}
    assert result["pixel_values_cache"] is None
    assert result["pil_image_cache"] is None


@pytest.mark.parametrize(
    "failed_api", ["get_pixel_values_cache_stats", "get_pil_cache_stats"]
)
@pytest.mark.parametrize(
    "error", [ImportError, RuntimeError, AttributeError, ValueError]
)
def test_vision_cache_stats_isolate_optional_image_api_failures(
    monkeypatch, failed_api, error
):
    from rapid_mlx.routes import health

    utils = SimpleNamespace(
        get_pixel_values_cache_stats=Mock(return_value={"hits": 7}),
        get_pil_cache_stats=Mock(return_value={"hits": 11}),
    )
    getattr(utils, failed_api).side_effect = error("optional counter unavailable")
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(utils=utils))
    engine = SimpleNamespace(is_mllm=True, get_cache_stats=lambda: {"hits": 3})
    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))

    assert asyncio.run(health.cache_stats()) == {
        "model_type": "mllm",
        "multimodal_kv_cache": {"hits": 3},
        "pixel_values_cache": None
        if failed_api == "get_pixel_values_cache_stats"
        else {"hits": 7},
        "pil_image_cache": None
        if failed_api == "get_pil_cache_stats"
        else {"hits": 11},
    }
