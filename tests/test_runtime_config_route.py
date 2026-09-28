# SPDX-License-Identifier: Apache-2.0
"""Read-only wire DTO tests for migration 002 stage 3."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rapid_mlx.routes import runtime_config as route
from rapid_mlx.runtime.config_adapter import (
    DEFAULT_RUNTIME_LAUNCH_VALUES,
    resolve_programmatic_runtime_config,
)


@pytest.mark.asyncio
async def test_runtime_config_route_returns_versioned_provenance(monkeypatch):
    config = resolve_programmatic_runtime_config(
        surface="server.load_model", legacy=DEFAULT_RUNTIME_LAUNCH_VALUES
    )
    monkeypatch.setattr(
        route,
        "get_config",
        lambda: SimpleNamespace(
            ready=True,
            effective_runtime_config=config,
            effective_runtime_model="qwen-test",
        ),
    )

    payload = await route.effective_runtime_config()

    assert payload["model"] == "qwen-test"
    assert payload["schema_version"] == 1
    fields = payload["fields"]
    assert isinstance(fields, list)
    assert [item["field"] for item in fields] == [
        "prefill_step_size",
        "max_num_seqs",
        "gpu_memory_utilization",
        "enable_prefix_cache",
        "kv_cache_dtype",
    ]
    assert fields[0]["source"] == "global_default"
    assert fields[0]["trace"][0]["action"] == "selected"


@pytest.mark.asyncio
async def test_runtime_config_route_is_unavailable_before_resolution(monkeypatch):
    monkeypatch.setattr(
        route,
        "get_config",
        lambda: SimpleNamespace(
            ready=False,
            effective_runtime_config=None,
            effective_runtime_model=None,
        ),
    )

    with pytest.raises(HTTPException) as caught:
        await route.effective_runtime_config()
    assert caught.value.status_code == 503


@pytest.mark.asyncio
async def test_runtime_config_route_rejects_unowned_snapshot(monkeypatch):
    config = resolve_programmatic_runtime_config(
        surface="server.load_model", legacy=DEFAULT_RUNTIME_LAUNCH_VALUES
    )
    monkeypatch.setattr(
        route,
        "get_config",
        lambda: SimpleNamespace(
            ready=True,
            effective_runtime_config=config,
            effective_runtime_model=None,
        ),
    )

    with pytest.raises(HTTPException) as caught:
        await route.effective_runtime_config()
    assert caught.value.status_code == 503
