from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
from vllm_mlx.runtime.resident_models import (
    ResidentModelBusyError,
    ResidentModelCapacityError,
    ResidentModelManager,
)

GIB = 1024**3


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

    async def loader(name: str, path: str | None):
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

    async def loader(name: str, path: str | None):
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

    async def loader(name: str, path: str | None):
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
