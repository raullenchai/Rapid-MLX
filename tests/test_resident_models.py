from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
from vllm_mlx.runtime.resident_models import (
    ResidentModelBusyError,
    ResidentModelCapacityError,
    ResidentModelManager,
    ResidentPerformanceConfig,
    resident_scheduler_kwargs,
)

GIB = 1024**3


def test_resident_performance_maps_to_the_scheduler_contract():
    assert resident_scheduler_kwargs(
        ResidentPerformanceConfig(
            kv_cache_dtype="int4",
            prefix_cache_enabled=False,
            cache_memory_mb=4096,
        )
    ) == {
        "kv_cache_dtype": "int4",
        "kv_cache_quantization": True,
        "kv_cache_quantization_bits": 4,
        "enable_prefix_cache": False,
        "cache_memory_mb": 4096,
    }
    assert resident_scheduler_kwargs(
        ResidentPerformanceConfig(kv_cache_turboquant="k8v4")
    ) == {
        "kv_cache_turboquant": True,
        "kv_cache_turboquant_mode": "k8v4",
    }


class FakeEngine:
    is_mllm = False

    def __init__(self) -> None:
        self.stopped = False
        self.running = 0

    def get_stats(self):
        return {"num_running": self.running, "num_waiting": 0}

    async def stop(self) -> None:
        self.stopped = True


class FakeImageEngine(FakeEngine):
    is_image_gen = True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def entry(name: str, engine: FakeEngine | None = None) -> ModelEntry:
    return ModelEntry(
        engine=engine or FakeEngine(),
        model_name=name,
        model_path=f"repo/{name}",
    )


def manager_fixture(*, limit_gib=10, ttl=0):
    registry = ModelRegistry()
    primary = entry("chat")
    registry.add(primary, is_default=True)
    loaded: dict[str, FakeEngine] = {}

    async def loader(name: str, path: str | None, performance=None):
        engine = FakeEngine()
        loaded[name] = engine
        return ModelEntry(
            engine=engine,
            model_name=name,
            model_path=path or f"repo/{name}",
        )

    clock = Clock()
    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=limit_gib * GIB,
        idle_ttl_seconds=ttl,
        clock=clock,
        memory_reader=lambda: 0,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    return manager, registry, loaded, clock


@pytest.mark.asyncio
async def test_accounting_reserves_lazy_model_estimates_and_tracks_larger_actual_usage():
    registry = ModelRegistry()
    primary = entry("chat")
    registry.add(primary, is_default=True)
    process_usage = [1 * GIB]

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=20 * GIB,
        memory_reader=lambda: process_usage[0],
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    await manager.load("image", estimated_bytes=3 * GIB)

    # A lazy engine may add almost nothing to phys_footprint at load time;
    # admission still reserves its full estimate.
    assert manager.snapshot()["memory_used_bytes"] == 7 * GIB

    process_usage[0] = 9 * GIB
    assert manager.snapshot()["memory_used_bytes"] == 9 * GIB


@pytest.mark.asyncio
async def test_load_evicts_least_recently_used_unpinned_model():
    manager, registry, loaded, clock = manager_fixture()
    clock.now = 1
    await manager.load("image-a", estimated_bytes=3 * GIB)
    clock.now = 2
    await manager.load("image-b", estimated_bytes=3 * GIB)

    # A becomes most recently used, leaving B as the eviction candidate.
    clock.now = 3
    registry.get_engine("image-a")
    await manager.load("image-c", estimated_bytes=3 * GIB)

    assert "image-a" in registry
    assert "image-b" not in registry
    assert "image-c" in registry
    assert loaded["image-b"].stopped is True
    assert manager.snapshot()["evictions_total"] == 1


@pytest.mark.asyncio
async def test_pin_and_active_lease_are_never_evicted():
    manager, registry, loaded, clock = manager_fixture(limit_gib=9)
    clock.now = 1
    await manager.load("pinned", estimated_bytes=2 * GIB, pin=True)
    clock.now = 2
    await manager.load("working", estimated_bytes=2 * GIB)

    async with manager.lease("working"):
        with pytest.raises(ResidentModelCapacityError, match="no idle unpinned"):
            await manager.load("incoming", estimated_bytes=2 * GIB)

    assert "pinned" in registry
    assert "working" in registry
    assert not loaded["pinned"].stopped
    assert not loaded["working"].stopped


@pytest.mark.asyncio
async def test_idle_ttl_reclaims_only_unpinned_secondary_models():
    manager, registry, loaded, clock = manager_fixture(limit_gib=20, ttl=60)
    await manager.load("old", estimated_bytes=2 * GIB)
    await manager.load("kept", estimated_bytes=2 * GIB, pin=True)
    clock.now = 61

    assert await manager.evict_expired() == ["old"]
    assert "old" not in registry
    assert "kept" in registry
    assert loaded["old"].stopped
    assert not loaded["kept"].stopped


@pytest.mark.asyncio
async def test_replacing_assistant_promotes_target_and_keeps_image_resident():
    registry = ModelRegistry()
    primary = entry("chat-old")
    registry.add(primary, is_default=True)
    loaded: dict[str, FakeEngine] = {}
    promoted: list[str] = []

    async def loader(name: str, path: str | None, performance=None):
        engine = FakeImageEngine() if name == "image" else FakeEngine()
        loaded[name] = engine
        return ModelEntry(
            engine=engine,
            model_name=name,
            model_path=path or f"repo/{name}",
        )

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=20 * GIB,
        memory_reader=lambda: 0,
        on_primary_changed=lambda value: promoted.append(value.model_name),
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    await manager.load("image", estimated_bytes=3 * GIB)
    await manager.load("chat-new", estimated_bytes=4 * GIB)

    # The desktop may select a chat model that is already resident. Reissuing
    # the load with a replacement group must still perform the handoff.
    replacement = await manager.load("chat-new", replace_group="assistant")

    assert replacement.primary is True
    assert replacement.pinned is True
    assert registry.default_name == "chat-new"
    assert promoted == ["chat-new"]
    assert primary.engine.stopped is True
    assert loaded["chat-new"].stopped is False
    assert loaded["image"].stopped is False
    assert "chat-old" not in registry
    assert "chat-new" in registry
    assert "image" in registry
    assert {item["id"] for item in manager.snapshot()["models"]} == {
        "chat-new",
        "image",
    }


@pytest.mark.asyncio
async def test_failed_assistant_replacement_rolls_back_newly_loaded_model():
    manager, registry, loaded, _ = manager_fixture(limit_gib=20)
    registry.get_engine("chat").running = 1

    with pytest.raises(ResidentModelBusyError, match="active request"):
        await manager.load(
            "chat-new",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
        )

    assert "chat" in registry
    assert "chat-new" not in registry
    assert loaded["chat-new"].stopped is True
    assert {item["id"] for item in manager.snapshot()["models"]} == {"chat"}


@pytest.mark.asyncio
async def test_per_model_performance_reload_replaces_only_the_target_engine():
    manager, registry, loaded, _ = manager_fixture(limit_gib=20)
    await manager.load("image", estimated_bytes=3 * GIB)
    image_engine = loaded["image"]
    old_chat_engine = registry.get_engine("chat")
    config = ResidentPerformanceConfig(
        kv_cache_dtype="int8",
        prefix_cache_enabled=False,
        cache_memory_mb=2048,
    )

    reloaded = await manager.load(
        "chat",
        performance=config,
        reload_if_changed=True,
    )

    assert reloaded.performance == config
    assert old_chat_engine.stopped is True
    assert registry.get_engine("chat") is not old_chat_engine
    assert loaded["image"] is image_engine
    assert image_engine.stopped is False
    assert {item["id"] for item in manager.snapshot()["models"]} == {
        "chat",
        "image",
    }
    chat = next(item for item in manager.snapshot()["models"] if item["id"] == "chat")
    assert chat["performance"] == {
        "kv_cache_dtype": "int8",
        "prefix_cache_enabled": False,
        "cache_memory_mb": 2048,
    }


@pytest.mark.asyncio
async def test_failed_performance_reload_restores_the_last_known_good_engine():
    registry = ModelRegistry()
    primary = entry("chat")
    registry.add(primary, is_default=True)
    calls: list[ResidentPerformanceConfig | None] = []

    async def loader(name: str, path: str | None, performance=None):
        calls.append(performance)
        if performance == ResidentPerformanceConfig(kv_cache_dtype="int4"):
            raise RuntimeError("new config failed")
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(RuntimeError, match="new config failed"):
        await manager.load(
            "chat",
            performance=ResidentPerformanceConfig(kv_cache_dtype="int4"),
            reload_if_changed=True,
        )

    assert calls == [ResidentPerformanceConfig(kv_cache_dtype="int4"), None]
    assert "chat" in registry
    assert registry.get_engine("chat") is not primary.engine
    assert manager.snapshot()["models"][0]["performance"] is None


@pytest.mark.asyncio
async def test_failed_stop_keeps_the_existing_engine_registered():
    manager, registry, _, _ = manager_fixture(limit_gib=20)
    old_engine = registry.get_engine("chat")

    async def failed_stop():
        raise RuntimeError("stop failed")

    old_engine.stop = failed_stop
    with pytest.raises(RuntimeError, match="stop failed"):
        await manager.load(
            "chat",
            performance=ResidentPerformanceConfig(kv_cache_dtype="int8"),
            reload_if_changed=True,
        )

    assert registry.get_engine("chat") is old_engine
    assert manager.snapshot()["models"][0]["id"] == "chat"


@pytest.mark.asyncio
async def test_soak_load_evict_cycles_stay_inside_ceiling():
    manager, registry, loaded, clock = manager_fixture(limit_gib=8)

    for index in range(40):
        clock.now = float(index + 1)
        await manager.load(f"image-{index}", estimated_bytes=4 * GIB)
        snapshot = manager.snapshot()
        accounted = sum(model["estimated_bytes"] for model in snapshot["models"])
        assert accounted <= snapshot["memory_limit_bytes"]
        assert len(snapshot["models"]) == 2  # protected chat + latest image

    assert manager.snapshot()["evictions_total"] == 39
    assert sum(engine.stopped for engine in loaded.values()) == 39
    assert len(registry.list_entries()) == 2


def test_residency_control_plane_load_pin_status_and_unload(monkeypatch):
    from types import SimpleNamespace

    from vllm_mlx.routes.residency import router

    manager, registry, _, _ = manager_fixture(limit_gib=12)
    monkeypatch.setattr(
        "vllm_mlx.routes.residency.get_config",
        lambda: SimpleNamespace(residency_manager=manager),
    )
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        loaded = client.post(
            "/v1/models/load",
            json={"model": "image", "estimated_size_gb": 3},
        )
        assert loaded.status_code == 200
        assert loaded.json()["id"] == "image"

        status = client.get("/v1/models/residency")
        assert status.status_code == 200
        assert status.json()["memory_limit_bytes"] == 12 * GIB
        assert {item["id"] for item in status.json()["models"]} == {"chat", "image"}

        pinned = client.put("/v1/models/image/pin", json={"pinned": True})
        assert pinned.status_code == 200
        assert pinned.json()["pinned"] is True
        assert client.delete("/v1/models/image").status_code == 409

        assert (
            client.put("/v1/models/image/pin", json={"pinned": False}).status_code
            == 200
        )
        assert client.delete("/v1/models/image").status_code == 204
        assert "image" not in registry


def test_residency_control_plane_validates_and_forwards_performance(monkeypatch):
    from types import SimpleNamespace

    from vllm_mlx.routes.residency import router

    manager, registry, loaded, _ = manager_fixture(limit_gib=20)
    monkeypatch.setattr(
        "vllm_mlx.routes.residency.get_config",
        lambda: SimpleNamespace(residency_manager=manager),
    )
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/load",
            json={
                "model": "chat",
                "reload_if_changed": True,
                "performance": {
                    "kv_cache_turboquant": "k8v4",
                    "prefix_cache_enabled": True,
                    "cache_memory_mb": 4096,
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["performance"] == {
            "kv_cache_turboquant": "k8v4",
            "prefix_cache_enabled": True,
            "cache_memory_mb": 4096,
        }
        assert registry.get_engine("chat") is loaded["chat"]

        conflict = client.post(
            "/v1/models/load",
            json={
                "model": "chat",
                "performance": {
                    "kv_cache_dtype": "int4",
                    "kv_cache_turboquant": "v4",
                },
            },
        )
        assert conflict.status_code == 422

        monkeypatch.setattr(
            "vllm_mlx.routes.residency.resolve_profile",
            lambda _name: SimpleNamespace(modality="image-gen"),
        )
        image_override = client.post(
            "/v1/models/load",
            json={
                "model": "image",
                "performance": {"prefix_cache_enabled": True},
            },
        )
        assert image_override.status_code == 422
        assert "only supported for text" in image_override.json()["detail"]


def test_resident_performance_uses_cli_kv_safety_gate(monkeypatch):
    from vllm_mlx.runtime.resident_models import resolve_resident_performance

    monkeypatch.setattr(
        "vllm_mlx.cli._gather_kv_cache_dtype_inputs",
        lambda _name: ({"sliding_window": 4096}, None),
    )
    resolved = resolve_resident_performance(
        ResidentPerformanceConfig(kv_cache_dtype="int4", cache_memory_mb=2048),
        model_name="example/sliding-model",
        model_path=None,
    )

    assert resolved == ResidentPerformanceConfig(
        kv_cache_dtype="bf16",
        cache_memory_mb=2048,
    )
