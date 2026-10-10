# SPDX-License-Identifier: Apache-2.0
"""Cache diagnostics follow the serving backend without optional global APIs."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.requires_mlx


@pytest.mark.parametrize("stats", [None, {"entry_count": 2, "hits": 3}])
def test_vision_cache_stats_use_active_engine(monkeypatch, stats):
    from rapid_mlx.routes import health

    engine = SimpleNamespace(is_mllm=True, get_cache_stats=Mock(return_value=stats))
    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))
    assert asyncio.run(health.cache_stats()) == {
        "model_type": "mllm",
        "multimodal_kv_cache": stats,
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
