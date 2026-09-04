from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
from vllm_mlx.runtime.resident_models import (
    ResidencyRecord,
    ResidentModelBusyError,
    ResidentModelCapacityError,
    ResidentModelError,
    ResidentModelManager,
    ResidentPerformanceConfig,
    ResidentRole,
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


def test_legacy_capacity_error_keeps_stable_envelope():
    error = ResidentModelCapacityError("legacy replacement rejected")
    assert error.envelope() == {
        "error": {
            "message": "legacy replacement rejected",
            "type": "insufficient_capacity_error",
            "code": "insufficient_capacity_error",
            "param": "model",
        }
    }
    assert resident_scheduler_kwargs(
        ResidentPerformanceConfig(kv_cache_turboquant="k8v4")
    ) == {
        "kv_cache_turboquant": True,
        "kv_cache_turboquant_mode": "k8v4",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name", "model_path"),
    [
        ("publisher/opaque-hybrid", None),
        ("opaque-local-hybrid", "/private/cache/opaque-hybrid/snapshot"),
    ],
)
async def test_dynamic_resident_auto_detected_hybrid_gets_bounded_prefix_reuse(
    monkeypatch,
    scheduler_config_stub,
    residency_activity_contract,
    model_name,
    model_path,
):
    """Runtime residency must consume the same architecture truth as serve."""
    from vllm_mlx import server
    from vllm_mlx.model_profile import ModelProfile

    captured = {}

    class FakeEngine:
        is_mllm = False

        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            pass

        def generate_warmup(self):
            pass

    monkeypatch.setattr(server, "BatchedEngine", FakeEngine)
    monkeypatch.setattr(server, "_ensure_routing_config", lambda _name: None)
    monkeypatch.setattr(
        server, "resolve_serving_lane", lambda _name, **_kwargs: (False, True)
    )
    monkeypatch.setattr("vllm_mlx.model_aliases.resolve_profile", lambda _name: None)
    monkeypatch.setattr(
        "vllm_mlx.model_auto_config.detect_model_config",
        lambda _name: ModelProfile(
            is_hybrid=True,
            is_hybrid_explicit=True,
            supports_spec_decode=False,
        ),
    )

    await server._load_dynamic_resident_model(model_name, model_path)

    scheduler = captured["scheduler_config"]
    assert scheduler.enable_prefix_cache is True
    assert scheduler.hybrid_cache_entries == 8
    assert scheduler.non_trimmable_exact_prefix_reuse is True
    await residency_activity_contract()


@pytest.mark.asyncio
async def test_dynamic_resident_prefix_disable_keeps_hybrid_entries_zero(
    monkeypatch, scheduler_config_stub
):
    from vllm_mlx import server
    from vllm_mlx.model_profile import ModelProfile

    captured = {}

    class FakeEngine:
        is_mllm = False

        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            pass

        def generate_warmup(self):
            pass

    monkeypatch.setattr(server, "BatchedEngine", FakeEngine)
    monkeypatch.setattr(server, "_ensure_routing_config", lambda _name: None)
    monkeypatch.setattr(
        server, "resolve_serving_lane", lambda _name, **_kwargs: (False, True)
    )
    monkeypatch.setattr("vllm_mlx.model_aliases.resolve_profile", lambda _name: None)
    monkeypatch.setattr(
        "vllm_mlx.model_auto_config.detect_model_config",
        lambda _name: ModelProfile(
            is_hybrid=True,
            is_hybrid_explicit=True,
            supports_spec_decode=False,
        ),
    )

    await server._load_dynamic_resident_model(
        "publisher/opaque-hybrid",
        None,
        ResidentPerformanceConfig(prefix_cache_enabled=False),
    )

    scheduler = captured["scheduler_config"]
    assert scheduler.enable_prefix_cache is False
    assert scheduler.hybrid_cache_entries == 0
    assert scheduler.non_trimmable_exact_prefix_reuse is False


@pytest.mark.asyncio
async def test_dynamic_resident_full_attention_stays_unbounded(
    monkeypatch, scheduler_config_stub
):
    from vllm_mlx import server
    from vllm_mlx.model_profile import ModelProfile

    captured = {}

    class FakeEngine:
        is_mllm = False

        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            pass

        def generate_warmup(self):
            pass

    monkeypatch.setattr(server, "BatchedEngine", FakeEngine)
    monkeypatch.setattr(server, "_ensure_routing_config", lambda _name: None)
    monkeypatch.setattr(
        server, "resolve_serving_lane", lambda _name, **_kwargs: (False, False)
    )
    monkeypatch.setattr("vllm_mlx.model_aliases.resolve_profile", lambda _name: None)
    monkeypatch.setattr(
        "vllm_mlx.model_auto_config.detect_model_config",
        lambda _name: ModelProfile(is_hybrid=False),
    )

    await server._load_dynamic_resident_model("publisher/full-attention", None)

    scheduler = captured["scheduler_config"]
    assert scheduler.hybrid_cache_entries == 0
    assert scheduler.non_trimmable_exact_prefix_reuse is False


@pytest.mark.asyncio
async def test_dynamic_resident_loads_singleton_no_refs_snapshot_offline(
    monkeypatch, tmp_path, scheduler_config_stub
):
    """A commit-pinned catalog snapshot is the runtime load path without main."""
    import json

    import huggingface_hub

    from vllm_mlx import server

    repo = "mlx-community/Qwen3.5-2B-MLX-4bit"
    revision = "93760be4f1f69842a46bc13dbdc0f19e291392a3"
    snapshot = (
        tmp_path
        / "hub"
        / "models--mlx-community--Qwen3.5-2B-MLX-4bit"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "vision_config": {"model_type": "qwen3_5_vision"},
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "layer_types": ["linear_attention", "full_attention"],
                },
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"complete")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.layers.0.weight": "model.safetensors",
                    "vision_tower.blocks.0.weight": "model.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub")
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def fail_on_network(_name):
        raise AssertionError("singleton snapshot must never need the network")

    monkeypatch.setattr("vllm_mlx.cli._ensure_model_downloaded", fail_on_network)
    monkeypatch.setattr(server, "_ensure_routing_config", lambda _name: None)
    monkeypatch.setattr(
        server,
        "resolve_serving_lane_decision",
        lambda _name, **_kwargs: SimpleNamespace(
            is_mllm=False,
            reason="text_lane_forced",
            auto_text_fallback=True,
        ),
    )
    captured = {}

    class FakeEngine:
        is_mllm = False

        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            pass

        def generate_warmup(self):
            pass

    monkeypatch.setattr(server, "BatchedEngine", FakeEngine)

    entry = await server._load_dynamic_resident_model("qwen3.5-2b-4bit", repo)

    assert entry.model_path == repo
    assert captured["model_name"] == str(snapshot)
    assert captured["force_text"] is True


@pytest.mark.asyncio
async def test_dynamic_resident_preserves_explicit_subfolder_alias(
    monkeypatch, scheduler_config_stub
):
    """Residency and startup must apply the same alias-over-marker precedence."""
    from types import SimpleNamespace

    from vllm_mlx import server

    alias = "lfm2.5-2.6b-4bit"
    repo = "LiquidAI/LFM2.5-2.6B-MLX"
    seen: list[str] = []
    captured: dict[str, object] = {}

    class FakeEngine:
        is_mllm = False

        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            pass

        def generate_warmup(self):
            pass

    def fake_resolve_checkpoint(model_name, **kwargs):
        seen.append(model_name)
        return SimpleNamespace(
            model_path=repo,
            load_path="/cache/snapshots/revision/4bit",
            auto_text_fallback=False,
            lane_reason="text_checkpoint",
        )

    monkeypatch.setattr(server, "BatchedEngine", FakeEngine)
    monkeypatch.setattr(server, "_resolve_serving_checkpoint", fake_resolve_checkpoint)

    entry = await server._load_dynamic_resident_model(alias, repo)

    assert seen == [alias]
    assert captured["model_name"] == "/cache/snapshots/revision/4bit"
    assert entry.model_path == repo


@pytest.mark.asyncio
async def test_dynamic_switch_restores_hybrid_text_lane(
    monkeypatch, tmp_path, scheduler_config_stub
):
    """A large hybrid checkpoint keeps its lane after a small-model switch."""
    import json

    from vllm_mlx import server
    from vllm_mlx.api import utils as utils_mod

    large = tmp_path / "qwen35-9b"
    small = tmp_path / "qwen3-06b"
    for path in (large, small):
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"complete")
    (large / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "vision_config": {"model_type": "qwen3_5_vision"},
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "layer_types": ["linear_attention", "full_attention"],
                },
            }
        ),
        encoding="utf-8",
    )
    (large / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.layers.0.weight": "model.safetensors",
                    "vision_tower.blocks.0.weight": "model.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (small / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        encoding="utf-8",
    )
    constructed = []

    class FakeEngine:
        is_mllm = False

        def __init__(self, **kwargs):
            constructed.append(kwargs)

        async def start(self):
            pass

        def generate_warmup(self):
            pass

    monkeypatch.setattr(server, "BatchedEngine", FakeEngine)
    monkeypatch.setattr("vllm_mlx.model_aliases.resolve_profile", lambda _name: None)
    # Exercise the real resolver across both switches.  Only host capabilities
    # are fixed: the large checkpoint's own metadata must select the hybrid
    # text fallback, while the small text checkpoint must not inherit it.
    monkeypatch.setattr(utils_mod, "physical_ram_gb", lambda: 256.0)
    monkeypatch.setattr(utils_mod, "mllm_hybrid_runtime_supported", lambda: False)

    await server._load_dynamic_resident_model("large", str(large))
    await server._load_dynamic_resident_model("small", str(small))
    await server._load_dynamic_resident_model("large", str(large))

    assert [item["model_name"] for item in constructed] == [
        str(large),
        str(small),
        str(large),
    ]
    assert [item["force_text"] for item in constructed] == [True, False, True]
    assert [item["serving_lane_reason"] for item in constructed] == [
        "vision_hybrid_runtime_unsupported",
        "text_checkpoint",
        "vision_hybrid_runtime_unsupported",
    ]


class FakeEngine:
    is_mllm = False

    def __init__(self) -> None:
        self.stopped = False
        self.running = 0
        self.waiting = 0

    def get_stats(self):
        return {"num_running": self.running, "num_waiting": self.waiting}

    async def stop(self) -> None:
        self.stopped = True


class FakeImageEngine(FakeEngine):
    is_image_gen = True


class FakeLifecycleEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.pauses: list[tuple[str, float | None]] = []
        self.paused = False

    async def pause_generation(self, mode="wait", *, timeout=None):
        self.pauses.append((mode, timeout))
        self.paused = True
        if self.running and timeout == 0:
            raise TimeoutError
        self.running = 0
        return self.lifecycle_status()

    async def resume_generation(self):
        self.paused = False
        return self.lifecycle_status()

    def lifecycle_status(self):
        return {
            "paused": self.paused,
            "pause_mode": self.pauses[-1][0] if self.pauses else None,
            "admitted_requests": self.running,
            "running_requests": self.running,
            "queued_requests": 0,
        }


class FailingResumeLifecycleEngine(FakeLifecycleEngine):
    async def resume_generation(self):
        raise RuntimeError("resume failed")


class FailingStopLifecycleEngine(FakeLifecycleEngine):
    async def stop(self) -> None:
        self.stopped = True
        raise RuntimeError("stop failed")


class FailOnceStopLifecycleEngine(FakeLifecycleEngine):
    def __init__(self) -> None:
        super().__init__()
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1
        if self.stop_calls == 1:
            raise RuntimeError("transient stop failure")
        self.stopped = True


class BlockingStopLifecycleEngine(FakeLifecycleEngine):
    def __init__(self) -> None:
        super().__init__()
        self.stop_started = asyncio.Event()
        self.stop_release = asyncio.Event()
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1
        self.stopped = True
        self.stop_started.set()
        await self.stop_release.wait()


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


def test_residency_snapshot_exposes_live_serving_lane_truth():
    manager, registry, _, _ = manager_fixture()
    engine = registry.get_entry("chat").engine
    engine.serving_lane = "text"
    engine.serving_lane_reason = "vision_hybrid_runtime_unsupported"

    payload = manager.snapshot()["models"][0]

    assert payload["serving_lane"] == "text"
    assert payload["serving_lane_reason"] == "vision_hybrid_runtime_unsupported"


def test_live_engine_exposes_serving_lane_decision():
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    engine._serving_lane_reason = "vision_hybrid_runtime_supported"

    assert engine.serving_lane == "vision"
    assert engine.serving_lane_reason == "vision_hybrid_runtime_supported"


def test_models_lane_fields_use_matching_live_engine(monkeypatch):
    from types import SimpleNamespace

    from vllm_mlx.routes import models as models_route

    engine = SimpleNamespace(
        serving_lane="text",
        serving_lane_reason="vision_hybrid_runtime_unsupported",
    )
    monkeypatch.setattr(
        models_route,
        "_engine_for",
        lambda model_id: engine if model_id == "served-alias" else None,
    )

    assert models_route._served_lane_fields("served-alias") == (
        "text",
        "vision_hybrid_runtime_unsupported",
    )
    assert models_route._served_lane_fields("unknown") == (None, None)


@pytest.fixture
def residency_activity_contract(monkeypatch):
    """Exercise the activity SSOT from the MLX-free Linux fixed selector."""
    from vllm_mlx.runtime.resident_models import _engine_active_requests

    class ProgressEngine:
        def progress_snapshot(self):
            return {"running": True}

    assert _engine_active_requests(ProgressEngine()) == 1

    class BrokenProgressEngine:
        def progress_snapshot(self):
            raise RuntimeError("progress unavailable")

    assert _engine_active_requests(BrokenProgressEngine()) is None

    class BrokenStatsEngine:
        def get_stats(self):
            raise RuntimeError("stats unavailable")

    assert _engine_active_requests(BrokenStatsEngine()) is None

    manager, registry, _, _ = manager_fixture(limit_gib=12)
    primary = registry.get_engine("chat")
    primary.running = 1
    primary.waiting = 1
    chat = next(item for item in manager.snapshot()["models"] if item["id"] == "chat")
    assert chat["active_requests"] == 2

    async def exercise_reload_identity() -> None:
        reload_registry = ModelRegistry()
        reload_primary = entry("reload-chat")
        reload_primary.aliases.add("reload-alias")
        reload_registry.add(reload_primary, is_default=True)

        async def loader(name: str, path: str | None, performance=None):
            if performance == ResidentPerformanceConfig(kv_cache_dtype="int4"):
                raise RuntimeError("new config failed")
            return entry(name)

        reload_manager = ResidentModelManager(
            reload_registry,
            loader,
            memory_reader=lambda: 0,
        )
        reload_manager.register_primary(reload_primary, estimated_bytes=1 * GIB)
        replacement = await reload_manager.load(
            "reload-alias",
            performance=ResidentPerformanceConfig(kv_cache_dtype="int8"),
            reload_if_changed=True,
        )
        assert replacement.entry.aliases == {"reload-alias"}
        assert (await reload_manager.load("reload-alias", pin=True)).pinned is True

        try:
            await reload_manager.load(
                "reload-alias",
                performance=ResidentPerformanceConfig(kv_cache_dtype="int4"),
                reload_if_changed=True,
            )
        except RuntimeError as exc:
            assert str(exc) == "new config failed"
        else:
            raise AssertionError("failed reload must surface its loader error")
        assert reload_registry.get_entry("reload-alias").aliases == {"reload-alias"}

        with pytest.raises(KeyError):
            await reload_manager.unload("missing")

        async def image_loader(name: str, path: str | None, performance=None):
            return entry(name, FakeImageEngine())

        with monkeypatch.context() as scoped:
            scoped.setattr(reload_manager, "loader", image_loader)
            with pytest.raises(ResidentModelError, match="image-gen.*assistant"):
                await reload_manager.load("wrong-new-group", replace_group="assistant")
            assert "wrong-new-group" not in reload_registry

            image = await reload_manager.load("resident-image")
            with pytest.raises(ResidentModelError, match="image-gen.*assistant"):
                await reload_manager.load(image.model_id, replace_group="assistant")

        with pytest.raises(ResidentModelError, match="does not belong"):
            await reload_manager._quiesce_group_locked(replacement, "image", "reject")
        with pytest.raises(ResidentModelError, match="unsupported replacement mode"):
            reload_manager._replacement_candidates_locked(
                "assistant", replace_mode="invalid"
            )
        with pytest.raises(ResidentModelError, match="unsupported replacement mode"):
            await reload_manager._quiesce_records_locked([], "invalid")

        busy = await reload_manager.load("busy")
        busy.entry.engine.running = 1
        busy.active_requests = 1
        with pytest.raises(ResidentModelBusyError, match="active request"):
            await reload_manager._quiesce_replacement_group_locked(
                "assistant", "wait", exclude_model_id=replacement.model_id
            )
        with pytest.raises(ResidentModelBusyError, match="active request"):
            await reload_manager.unload("busy")
        busy.active_requests = 0
        busy.entry.engine.running = 0
        busy.state = "evicting"
        with pytest.raises(ResidentModelBusyError, match="being evicted"):
            async with reload_manager.lease("busy"):
                pass
        busy.state = "resident"

        pinned = await reload_manager.load("pinned-secondary", pin=True)
        pinned.primary = False
        with pytest.raises(ResidentModelError, match="pinned model"):
            reload_manager._replacement_candidates_locked("assistant")

        rollback_engine = FakeLifecycleEngine()
        resumed = []

        async def quiesce(*_args, **_kwargs):
            rollback_engine.paused = True
            return [], [rollback_engine]

        async def fail_commit(*_args, **_kwargs):
            raise RuntimeError("commit failed")

        original_resume = rollback_engine.resume_generation

        async def resume():
            resumed.append(True)
            return await original_resume()

        rollback_engine.resume_generation = resume
        with monkeypatch.context() as scoped:
            scoped.setattr(reload_manager, "_quiesce_group_locked", quiesce)
            scoped.setattr(
                reload_manager, "_commit_group_replacement_locked", fail_commit
            )
            with pytest.raises(RuntimeError, match="commit failed"):
                await reload_manager._replace_group_locked(
                    replacement, "assistant", "wait"
                )
        assert resumed == [True]

    return exercise_reload_identity


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
async def test_image_residency_mode_reaches_extended_loader():
    registry = ModelRegistry()
    modes = []

    async def loader(name, path, performance, image_mode):
        modes.append(image_mode)
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)

    await manager.load("image", image_mode="editing")

    assert modes == ["editing"]


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
async def test_second_image_model_evicts_the_first_without_an_explicit_group():
    """Generative-media lanes are single-slot server-side.

    The desktop loads image models with no ``replace_group`` (unlike the chat
    picker's explicit ``assistant``). Image engines therefore only ever
    accumulated — two multi-GB checkpoints resident at once (measured 9.1 GB).
    A second image load must now evict the first even with no group supplied,
    while the ``assistant`` group is left untouched — including a NON-primary
    resident chat model, which (unlike the eviction-protected primary) would
    actually disappear if media replacement leaked across groups.
    """
    registry = ModelRegistry()
    primary = entry("chat")
    registry.add(primary, is_default=True)
    loaded: dict[str, FakeEngine] = {}

    async def loader(name: str, path: str | None, performance=None):
        engine = FakeImageEngine() if name.startswith("image") else FakeEngine()
        loaded[name] = engine
        return ModelEntry(
            engine=engine,
            model_name=name,
            model_path=path or f"repo/{name}",
        )

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=50 * GIB,
        memory_reader=lambda: 0,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    # A second, NON-primary chat model in the assistant group. It has no
    # independent eviction protection, so it is the real probe for cross-group
    # isolation: if media replacement touched the assistant group it would be
    # evicted here.
    await manager.load("chat-2", estimated_bytes=4 * GIB)

    await manager.load("image-a", estimated_bytes=5 * GIB)
    await manager.load("image-b", estimated_bytes=5 * GIB)

    ids = {item["id"] for item in manager.snapshot()["models"]}
    # image-a evicted by image-b; both assistant-group models survive.
    assert ids == {"chat", "chat-2", "image-b"}
    assert loaded["image-a"].stopped is True
    assert loaded["image-b"].stopped is False
    assert loaded["chat-2"].stopped is False
    assert "image-a" not in registry
    assert "image-b" in registry
    assert "chat-2" in registry


@pytest.mark.asyncio
async def test_failed_assistant_replacement_rolls_back_newly_loaded_model():
    manager, registry, loaded, _ = manager_fixture(limit_gib=20)
    registry.get_engine("chat").running = 1
    chat = next(item for item in manager.snapshot()["models"] if item["id"] == "chat")
    assert chat["active_requests"] == 1

    with pytest.raises(ResidentModelBusyError, match="active request"):
        await manager.load(
            "chat-new",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
        )

    assert "chat" in registry
    assert "chat-new" not in registry
    assert "chat-new" not in loaded
    assert {item["id"] for item in manager.snapshot()["models"]} == {"chat"}


@pytest.mark.asyncio
async def test_replacement_keeps_rollback_path_when_both_models_fit():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)

    async def loader(name: str, path: str | None, performance=None):
        assert old_engine.stopped is False
        return entry(name)

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=10 * GIB,
        memory_reader=lambda: 0,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    replacement = await manager.load(
        "chat-new",
        estimated_bytes=4 * GIB,
        replace_group="assistant",
        memory_policy="evict_first_if_needed",
    )

    projection = replacement.replacement_projection
    assert projection is not None
    assert projection.strategy == "keep_then_commit"
    assert projection.reason == "keep_both_fits"
    assert projection.models_to_free == (("chat-old", 4 * GIB),)
    assert projection.projected_bytes == 4 * GIB
    assert old_engine.stopped is True
    assert registry.default_name == "chat-new"


@pytest.mark.asyncio
async def test_keep_then_commit_preserves_existing_idle_lru_admission():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    image_engine = FakeImageEngine()
    image = entry("image", image_engine)
    spare_engine = FakeImageEngine()
    spare = entry("spare-image", spare_engine)
    registry.add(primary, is_default=True)
    registry.add(image)
    registry.add(spare)
    load_observations: list[tuple[bool, bool]] = []

    async def loader(name: str, path: str | None, performance=None):
        load_observations.append((old_engine.stopped, image_engine.stopped))
        return entry(name)

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=13 * GIB,
        memory_reader=lambda: 0,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    manager._index_record(
        ResidencyRecord(
            entry=image,
            estimated_bytes=4 * GIB,
            loaded_at=0,
            last_used_at=0,
        )
    )
    manager._index_record(
        ResidencyRecord(
            entry=spare,
            estimated_bytes=1 * GIB,
            loaded_at=1,
            last_used_at=1,
        )
    )

    replacement = await manager.load(
        "chat-new",
        estimated_bytes=8 * GIB,
        replace_group="assistant",
    )

    assert load_observations == [(False, True)]
    assert old_engine.stopped is True
    assert spare_engine.stopped is False
    assert replacement.primary is True
    assert registry.default_name == "chat-new"
    assert replacement.replacement_projection is not None
    assert replacement.replacement_projection.strategy == "keep_then_commit"
    assert replacement.replacement_projection.models_to_free == (
        ("image", 4 * GIB),
        ("chat-old", 4 * GIB),
    )


@pytest.mark.asyncio
async def test_replacement_evicts_first_only_when_that_makes_target_fit():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    load_observations: list[tuple[bool, str | None]] = []
    primary_changes: list[str | None] = []
    handoff = Mock()

    async def loader(name: str, path: str | None, performance=None):
        load_observations.append((old_engine.stopped, registry.default_name))
        return entry(name)

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: 0,
        on_primary_handoff=lambda _entry: handoff,
        on_primary_changed=lambda replacement: primary_changes.append(
            replacement.model_name if replacement is not None else None
        ),
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    replacement = await manager.load(
        "chat-new",
        estimated_bytes=4 * GIB,
        replace_group="assistant",
        replace_mode="wait",
        memory_policy="evict_first_if_needed",
        resolved_group="assistant",
    )

    assert load_observations == [(True, None)]
    projection = replacement.replacement_projection
    assert projection is not None
    assert projection.payload() == {
        "strategy": "evict_first",
        "reason": "role_capacity_evict_first_required",
        "models_to_free": [{"id": "chat-old", "estimated_bytes": 4 * GIB}],
        "current_bytes": 4 * GIB,
        "requested_bytes": 4 * GIB,
        "projected_bytes": 4 * GIB,
        "limit_bytes": 6 * GIB,
    }
    assert replacement.primary is True
    assert replacement.pinned is True
    assert registry.default_name == "chat-new"
    assert old_engine.pauses == [("wait", None)]
    assert primary_changes == [None, "chat-new"]
    handoff.commit.assert_called_once_with(replacement.entry)
    handoff.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_evict_first_projection_reuses_idle_lru_for_remaining_capacity():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    image_engine = FakeImageEngine()
    image = entry("image", image_engine)
    spare_engine = FakeImageEngine()
    spare = entry("spare-image", spare_engine)
    registry.add(primary, is_default=True)
    registry.add(image)
    registry.add(spare)

    async def loader(name: str, path: str | None, performance=None):
        assert old_engine.stopped is True
        assert image_engine.stopped is True
        assert spare_engine.stopped is False
        return entry(name)

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=12 * GIB,
        memory_reader=lambda: 0,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    manager._index_record(
        ResidencyRecord(
            entry=image,
            estimated_bytes=6 * GIB,
            loaded_at=0,
            last_used_at=0,
        )
    )
    manager._index_record(
        ResidencyRecord(
            entry=spare,
            estimated_bytes=1 * GIB,
            loaded_at=1,
            last_used_at=1,
        )
    )

    replacement = await manager.load(
        "chat-new",
        estimated_bytes=10 * GIB,
        replace_group="assistant",
        memory_policy="evict_first_if_needed",
        resolved_group="assistant",
    )

    assert replacement.replacement_projection is not None
    assert replacement.replacement_projection.models_to_free == (
        ("chat-old", 4 * GIB),
        ("image", 6 * GIB),
    )
    assert replacement.replacement_projection.projected_bytes == 11 * GIB
    assert registry.default_name == "chat-new"


@pytest.mark.asyncio
async def test_evict_first_load_failure_reports_primary_absent_without_rollback():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    primary_changes: list[str | None] = []
    handoff_commits: list[str | None] = []

    async def loader(name: str, path: str | None, performance=None):
        raise ImportError("replacement checkpoint is unavailable")

    class Handoff:
        def __init__(self):
            self.rollback = Mock()

        def commit(self, replacement):
            handoff_commits.append(
                replacement.model_name if replacement is not None else None
            )

    handoff = Handoff()

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: 0,
        on_primary_handoff=lambda _entry: handoff,
        on_primary_changed=lambda replacement: primary_changes.append(
            replacement.model_name if replacement is not None else None
        ),
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(ImportError, match="unavailable"):
        await manager.load(
            "chat-invalid",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
            memory_policy="evict_first_if_needed",
            resolved_group="assistant",
        )

    assert old_engine.stopped is True
    assert registry.default_name is None
    assert manager.snapshot()["models"] == []
    assert primary_changes == [None]
    assert handoff_commits == [None]
    handoff.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_evict_first_stop_failure_finishes_handoff_as_primary_absent():
    registry = ModelRegistry()
    old_engine = FailingStopLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    loader = AsyncMock()
    handoff = Mock()
    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: 0,
        on_primary_handoff=lambda _entry: handoff,
        on_primary_changed=lambda _entry: None,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(RuntimeError, match="stop failed"):
        await manager.load(
            "chat-new",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
            memory_policy="evict_first_if_needed",
            resolved_group="assistant",
        )

    loader.assert_not_awaited()
    handoff.commit.assert_called_once_with(None)
    handoff.rollback.assert_not_called()
    assert registry.default_name is None
    # The evict-first failure propagates, but the engine is NOT silently freed:
    # it is durably tracked as cleanup-failed and its bytes stay charged.
    snapshot_models = manager.snapshot()["models"]
    assert [item["id"] for item in snapshot_models] == ["chat-old"]
    assert snapshot_models[0]["state"] == "failed"
    assert snapshot_models[0]["cleanup_failed"] == "stop failed"


@pytest.mark.asyncio
async def test_evict_first_sibling_stop_failure_preserves_primary_publication():
    registry = ModelRegistry()
    sibling_engine = FailingStopLifecycleEngine()
    sibling = entry("chat-sibling", sibling_engine)
    primary_engine = FakeLifecycleEngine()
    primary = entry("chat-primary", primary_engine)
    registry.add(sibling)
    registry.add(primary, is_default=True)
    loader = AsyncMock()
    handoff = Mock()
    primary_changes: list[ModelEntry | None] = []
    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: 0,
        on_primary_handoff=lambda _entry: handoff,
        on_primary_changed=primary_changes.append,
    )
    manager._index_record(
        ResidencyRecord(
            entry=sibling,
            estimated_bytes=1 * GIB,
            loaded_at=0,
            last_used_at=0,
        )
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(RuntimeError, match="stop failed"):
        await manager.load(
            "chat-new",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
            memory_policy="evict_first_if_needed",
            resolved_group="assistant",
        )

    loader.assert_not_awaited()
    handoff.rollback.assert_called_once_with()
    handoff.commit.assert_not_called()
    assert primary_changes == []
    assert primary_engine.stopped is False
    assert primary_engine.paused is False
    assert registry.default_name == "chat-primary"
    # The sibling's cleanup failed and is durably tracked (bytes charged), while
    # the healthy primary stays published and un-touched.
    ids = {item["id"] for item in manager.snapshot()["models"]}
    assert ids == {"chat-primary", "chat-sibling"}
    sibling_row = next(
        item for item in manager.snapshot()["models"] if item["id"] == "chat-sibling"
    )
    assert sibling_row["state"] == "failed"
    assert sibling_row["cleanup_failed"] == "stop failed"


@pytest.mark.asyncio
async def test_evict_first_primary_callback_failure_reopens_old_admission():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    loader = AsyncMock()
    handoff = Mock()

    def fail_clear(replacement):
        assert replacement is None
        raise RuntimeError("primary clear failed")

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: 0,
        on_primary_handoff=lambda _entry: handoff,
        on_primary_changed=fail_clear,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(RuntimeError, match="primary clear failed"):
        await manager.load(
            "chat-new",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
            memory_policy="evict_first_if_needed",
            resolved_group="assistant",
        )

    loader.assert_not_awaited()
    handoff.rollback.assert_called_once_with()
    handoff.commit.assert_not_called()
    assert old_engine.pauses == [("wait", 0)]
    assert old_engine.paused is False
    assert old_engine.stopped is False
    assert registry.default_name == "chat-old"
    assert [item["id"] for item in manager.snapshot()["models"]] == ["chat-old"]


@pytest.mark.asyncio
async def test_evict_first_partial_target_publication_is_cleared_on_failure():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    image_engine = FakeImageEngine()
    image = entry("image", image_engine)
    registry.add(primary, is_default=True)
    registry.add(image)
    published: list[ModelEntry | None] = [primary]
    loaded: list[FakeEngine] = []
    handoff = Mock()

    async def loader(name: str, path: str | None, performance=None):
        replacement = entry(name)
        loaded.append(replacement.engine)
        return replacement

    def publish_then_fail(replacement):
        published[0] = replacement
        if replacement is not None:
            raise RuntimeError("parser construction failed")

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: 0,
        on_primary_handoff=lambda _entry: handoff,
        on_primary_changed=publish_then_fail,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    manager._index_record(
        ResidencyRecord(
            entry=image,
            estimated_bytes=1 * GIB,
            loaded_at=0,
            last_used_at=0,
            pinned=True,
        )
    )

    with pytest.raises(RuntimeError, match="parser construction failed"):
        await manager.load(
            "chat-new",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
            memory_policy="evict_first_if_needed",
            resolved_group="assistant",
        )

    assert loaded[0].stopped is True
    assert published == [None]
    assert registry.default_name is None
    assert [item["id"] for item in manager.snapshot()["models"]] == ["image"]
    assert image_engine.stopped is False
    handoff.commit.assert_called_once_with(None)
    handoff.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_evict_first_cleanup_callback_failure_does_not_mask_publish_error(caplog):
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    none_calls = 0

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    def fail_publish_and_cleanup(replacement):
        nonlocal none_calls
        if replacement is None:
            none_calls += 1
            if none_calls == 2:
                raise RuntimeError("cleanup clear failed")
            return
        raise RuntimeError("original publish failed")

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: 0,
        on_primary_handoff=lambda _entry: Mock(),
        on_primary_changed=fail_publish_and_cleanup,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(RuntimeError, match="original publish failed"):
        await manager.load(
            "chat-new",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
            memory_policy="evict_first_if_needed",
            resolved_group="assistant",
        )

    assert "Failed to clear rejected replacement primary" in caplog.text
    assert manager.snapshot()["models"] == []


@pytest.mark.asyncio
async def test_replacement_rejects_before_eviction_when_target_still_does_not_fit():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    loader = AsyncMock()

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: 0,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(ResidentModelCapacityError) as exc_info:
        await manager.load(
            "chat-too-large",
            estimated_bytes=8 * GIB,
            replace_group="assistant",
            memory_policy="evict_first_if_needed",
            resolved_group="assistant",
        )

    projection = exc_info.value.replacement_projection
    assert projection is not None
    assert projection.strategy == "reject"
    assert projection.reason == "role_capacity_insufficient_after_eviction"
    assert projection.projected_bytes == 8 * GIB
    loader.assert_not_awaited()
    assert old_engine.pauses == [("wait", 0)]
    assert old_engine.paused is False
    assert old_engine.stopped is False
    assert registry.default_name == "chat-old"


@pytest.mark.asyncio
async def test_manager_rejects_unknown_memory_policy():
    manager, _, _, _ = manager_fixture()

    with pytest.raises(ResidentModelError, match="unsupported memory policy"):
        await manager.load("chat-new", memory_policy="invented")


@pytest.mark.asyncio
async def test_evict_first_rejects_known_wrong_group_before_retiring_primary():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    loader = AsyncMock()
    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: 0,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(ResidentModelError, match="image-gen.*not 'assistant'"):
        await manager.load(
            "image-model",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
            memory_policy="evict_first_if_needed",
            resolved_group="image-gen",
        )

    loader.assert_not_awaited()
    assert old_engine.pauses == []
    assert old_engine.stopped is False
    assert registry.default_name == "chat-old"


@pytest.mark.asyncio
async def test_evict_first_falls_back_to_safe_admission_without_group_metadata():
    manager, registry, loaded, _ = manager_fixture(limit_gib=6)

    with pytest.raises(ResidentModelCapacityError):
        await manager.load(
            "unknown-model",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
            memory_policy="evict_first_if_needed",
        )

    assert loaded == {}
    assert registry.default_name == "chat"
    assert registry.get_engine("chat").stopped is False


@pytest.mark.asyncio
async def test_evict_first_never_subtracts_unattributed_process_footprint():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    loader = AsyncMock()
    footprint = [4 * GIB]
    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: footprint[0],
    )
    footprint[0] = 6 * GIB
    # Only the 2 GiB growth after manager configuration is attributable to the
    # startup model; the 4 GiB server baseline cannot be projected as freed.
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(ResidentModelCapacityError) as exc_info:
        await manager.load(
            "chat-new",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
            memory_policy="evict_first_if_needed",
            resolved_group="assistant",
        )

    projection = exc_info.value.replacement_projection
    assert projection is not None
    assert projection.reason == "role_capacity_insufficient_after_eviction"
    assert projection.current_bytes == 6 * GIB
    assert projection.projected_bytes == 8 * GIB
    loader.assert_not_awaited()
    assert old_engine.stopped is False
    assert registry.default_name == "chat-old"


@pytest.mark.asyncio
async def test_evict_first_credits_only_attributed_startup_model_footprint():
    footprint = [2 * GIB]

    class FootprintEngine(FakeLifecycleEngine):
        async def stop(self):
            await super().stop()
            footprint[0] = 2 * GIB

    registry = ModelRegistry()
    old_engine = FootprintEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)

    async def loader(name: str, path: str | None, performance=None):
        assert old_engine.stopped is True
        assert footprint[0] == 2 * GIB
        return entry(name)

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: footprint[0],
    )
    footprint[0] = 6 * GIB
    primary_record = manager.register_primary(primary, estimated_bytes=4 * GIB)
    assert primary_record.measured_bytes == 4 * GIB

    replacement = await manager.load(
        "chat-new",
        estimated_bytes=4 * GIB,
        replace_group="assistant",
        memory_policy="evict_first_if_needed",
        resolved_group="assistant",
    )

    assert replacement.replacement_projection is not None
    assert replacement.replacement_projection.strategy == "evict_first"
    assert replacement.replacement_projection.current_bytes == 6 * GIB
    assert replacement.replacement_projection.projected_bytes == 6 * GIB
    assert registry.default_name == "chat-new"


@pytest.mark.asyncio
async def test_busy_reject_precedes_unrelated_capacity_eviction():
    registry = ModelRegistry()
    chat_engine = FakeLifecycleEngine()
    chat_engine.running = 1
    primary = entry("chat", chat_engine)
    image_engine = FakeImageEngine()
    image = entry("image", image_engine)
    registry.add(primary, is_default=True)
    registry.add(image)
    loaded = []

    async def loader(name: str, path: str | None, performance=None):
        loaded.append(name)
        return entry(name)

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=8 * GIB,
        memory_reader=lambda: 0,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    manager._index_record(
        ResidencyRecord(
            entry=image,
            estimated_bytes=3 * GIB,
            loaded_at=0,
            last_used_at=0,
        )
    )

    with pytest.raises(ResidentModelBusyError, match="active request"):
        await manager.load(
            "chat-new",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
            replace_mode="reject",
        )

    assert loaded == []
    assert registry.default_name == "chat"
    assert {item["id"] for item in manager.snapshot()["models"]} == {
        "chat",
        "image",
    }
    assert chat_engine.stopped is False
    assert image_engine.stopped is False


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_mode", ["wait", "abort"])
async def test_assistant_replacement_quiesces_before_primary_handoff(replace_mode):
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    events: list[str] = []

    async def loader(name: str, path: str | None, performance=None):
        assert old_engine.pauses == []
        events.append("loaded")
        return entry(name)

    class Handoff:
        def commit(self, _entry):
            events.append("handoff-commit")

        def rollback(self):
            events.append("handoff-rollback")

    def handoff(_entry):
        events.append("handoff-start")
        assert old_engine.paused is True
        return Handoff()

    manager = ResidentModelManager(
        registry,
        loader,
        memory_reader=lambda: 0,
        on_primary_handoff=handoff,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    replacement = await manager.load(
        "chat-new", replace_group="assistant", replace_mode=replace_mode
    )

    assert old_engine.pauses == [(replace_mode, None)]
    assert old_engine.stopped is True
    assert replacement.primary is True
    assert registry.default_name == "chat-new"
    assert events == ["loaded", "handoff-start", "handoff-commit"]


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_mode", ["wait", "abort"])
async def test_failed_replacement_load_does_not_quiesce_active_assistant(
    replace_mode,
):
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    old_engine.running = 1
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)

    async def loader(name: str, path: str | None, performance=None):
        raise ImportError("replacement checkpoint is unavailable")

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(ImportError, match="unavailable"):
        await manager.load(
            "chat-invalid",
            replace_group="assistant",
            replace_mode=replace_mode,
        )

    assert old_engine.pauses == []
    assert old_engine.running == 1
    assert old_engine.paused is False
    assert old_engine.stopped is False
    assert registry.default_name == "chat-old"
    assert [item.model_name for item in registry.list_entries()] == ["chat-old"]


@pytest.mark.asyncio
async def test_replacement_does_not_publish_target_until_old_engine_drains():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    old_engine.running = 1
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    allow_drain = asyncio.Event()

    async def pause_generation(mode="wait", *, timeout=None):
        old_engine.pauses.append((mode, timeout))
        old_engine.paused = True
        await allow_drain.wait()
        old_engine.running = 0
        return old_engine.lifecycle_status()

    old_engine.pause_generation = pause_generation

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    replacement = asyncio.create_task(
        manager.load("chat-new", replace_group="assistant", replace_mode="wait")
    )
    await asyncio.sleep(0)

    assert "chat-new" not in registry
    assert registry.default_name == "chat-old"

    allow_drain.set()
    await replacement
    assert registry.default_name == "chat-new"


@pytest.mark.asyncio
async def test_committed_replacement_does_not_rollback_to_stopped_sibling(caplog):
    registry = ModelRegistry()
    primary_engine = FakeLifecycleEngine()
    primary = entry("chat-primary", primary_engine)
    sibling_engine = FailingStopLifecycleEngine()
    sibling = entry("chat-sibling", sibling_engine)
    registry.add(primary, is_default=True)
    registry.add(sibling)

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    manager._index_record(
        ResidencyRecord(
            entry=sibling,
            estimated_bytes=4 * GIB,
            loaded_at=0,
            last_used_at=0,
        )
    )

    replacement = await manager.load(
        "chat-new", replace_group="assistant", replace_mode="wait"
    )

    assert primary_engine.stopped is True
    assert replacement.primary is True
    assert registry.default_name == "chat-new"
    assert [item.model_name for item in registry.list_entries()] == ["chat-new"]
    # The failed sibling is durably tracked as cleanup-failed (not resurrected
    # as a route, but never silently freed) with its bytes still charged.
    ids = {item["id"] for item in manager.snapshot()["models"]}
    assert ids == {"chat-new", "chat-sibling"}
    sibling_row = next(
        item for item in manager.snapshot()["models"] if item["id"] == "chat-sibling"
    )
    assert sibling_row["state"] == "failed"
    assert sibling_row["cleanup_failed"] == "stop failed"
    assert "Failed to clean up resident model 'chat-sibling'" in caplog.text


@pytest.mark.asyncio
async def test_primary_stop_failure_keeps_committed_replacement_routable(
    caplog,
):
    registry = ModelRegistry()
    old_engine = FailingStopLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    handoff_events: list[str] = []

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    class Handoff:
        def commit(self, committed):
            handoff_events.append(f"commit:{committed.model_name}")

        def rollback(self):
            handoff_events.append("rollback")

    manager = ResidentModelManager(
        registry,
        loader,
        memory_reader=lambda: 0,
        on_primary_handoff=lambda _entry: Handoff(),
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    replacement = await manager.load(
        "chat-new", replace_group="assistant", replace_mode="wait"
    )

    assert replacement.primary is True
    assert registry.default_name == "chat-new"
    assert [item.model_name for item in registry.list_entries()] == ["chat-new"]
    # The old primary's cleanup failed and is durably tracked (bytes charged),
    # never silently freed, while the committed new route stays authoritative.
    assert {item["id"] for item in manager.snapshot()["models"]} == {
        "chat-new",
        "chat-old",
    }
    old_row = next(
        item for item in manager.snapshot()["models"] if item["id"] == "chat-old"
    )
    assert old_row["state"] == "failed"
    assert old_row["cleanup_failed"] == "stop failed"
    assert handoff_events == ["commit:chat-new"]
    assert "Failed to clean up resident model 'chat-old'" in caplog.text


@pytest.mark.asyncio
async def test_task_cancel_after_primary_commit_preserves_new_route():
    registry = ModelRegistry()
    old_engine = BlockingStopLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    loaded: dict[str, FakeEngine] = {}

    async def loader(name: str, path: str | None, performance=None):
        loaded[name] = FakeEngine()
        return entry(name, loaded[name])

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    replacement = asyncio.create_task(
        manager.load("chat-new", replace_group="assistant", replace_mode="wait")
    )
    await old_engine.stop_started.wait()

    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement

    assert registry.default_name == "chat-new"
    assert registry.get_engine("chat-new") is loaded["chat-new"]
    assert [item.model_name for item in registry.list_entries()] == ["chat-new"]
    # The old engine's stop was suspended when the caller cancelled; it must stay
    # durably tracked as retiring (bytes charged) until a later shutdown rejoins
    # and drains its cleanup -- the old route is never resurrected.
    assert {item["id"] for item in manager.snapshot()["models"]} == {
        "chat-new",
        "chat-old",
    }
    old_row = next(
        item for item in manager.snapshot()["models"] if item["id"] == "chat-old"
    )
    assert old_row["state"] == "retiring"
    assert "cleanup_failed" in old_row and old_row["cleanup_failed"] is None


@pytest.mark.asyncio
async def test_task_cancel_during_sibling_cleanup_reopens_remaining_sibling():
    registry = ModelRegistry()
    primary_engine = FakeLifecycleEngine()
    primary = entry("chat-old", primary_engine)
    blocking_engine = BlockingStopLifecycleEngine()
    blocking = entry("chat-blocking", blocking_engine)
    remaining_engine = FakeLifecycleEngine()
    resume_started = asyncio.Event()
    resume_release = asyncio.Event()

    async def blocking_resume() -> dict:
        resume_started.set()
        await resume_release.wait()
        remaining_engine.paused = False
        return remaining_engine.lifecycle_status()

    remaining_engine.resume_generation = blocking_resume
    remaining = entry("chat-remaining", remaining_engine)
    registry.add(primary, is_default=True)
    registry.add(blocking)
    registry.add(remaining)

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    for sibling in (blocking, remaining):
        manager._index_record(
            ResidencyRecord(
                entry=sibling,
                estimated_bytes=4 * GIB,
                loaded_at=0,
                last_used_at=0,
            )
        )

    replacement = asyncio.create_task(
        manager.load("chat-new", replace_group="assistant", replace_mode="wait")
    )
    await blocking_engine.stop_started.wait()

    replacement.cancel()
    await resume_started.wait()
    # A second cancellation during rollback recovery must not strand the
    # unretired sibling paused. The original cancellation is delayed until the
    # manager-owned recovery finishes.
    replacement.cancel()
    await asyncio.sleep(0)
    assert replacement.done() is False
    resume_release.set()
    with pytest.raises(asyncio.CancelledError):
        await replacement

    assert registry.default_name == "chat-new"
    assert [item.model_name for item in registry.list_entries()] == [
        "chat-remaining",
        "chat-new",
    ]
    assert remaining_engine.paused is False


@pytest.mark.asyncio
async def test_existing_target_cancel_reopens_unretired_sibling(monkeypatch):
    # This contract is about cancellation ownership, not allocator latency.
    monkeypatch.setattr(
        "vllm_mlx.runtime.resident_models._release_allocator_cache", lambda: None
    )
    registry = ModelRegistry()
    primary_engine = BlockingStopLifecycleEngine()
    primary = entry("chat-primary", primary_engine)
    target_engine = FakeLifecycleEngine()
    target = entry("chat-target", target_engine)
    remaining_engine = FakeLifecycleEngine()
    remaining = entry("chat-remaining", remaining_engine)
    for model in (primary, target, remaining):
        registry.add(model, is_default=model is primary)

    manager = ResidentModelManager(registry, AsyncMock(), memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    for model in (target, remaining):
        manager._index_record(
            ResidencyRecord(
                entry=model,
                estimated_bytes=2 * GIB,
                loaded_at=0,
                last_used_at=0,
            )
        )

    replacement = asyncio.create_task(
        manager.load("chat-target", replace_group="assistant", replace_mode="wait")
    )
    await primary_engine.stop_started.wait()

    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement

    # The old primary is already owned by retirement, but the later sibling
    # stays routable and must be reopened rather than left paused indefinitely.
    assert registry.default_name == "chat-target"
    assert "chat-primary" not in {model.model_name for model in registry.list_entries()}
    assert "chat-remaining" in {model.model_name for model in registry.list_entries()}
    assert remaining_engine.paused is False

    primary_engine.stop_release.set()
    await asyncio.wait_for(manager.shutdown(), timeout=1)


@pytest.mark.asyncio
async def test_wait_replacement_retires_drained_engine_with_http_lease_finalizing():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    old_engine.running = 1
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    async with manager.lease("chat-old"):
        replacement = asyncio.create_task(
            manager.load("chat-new", replace_group="assistant", replace_mode="wait")
        )
        await asyncio.sleep(0)

        assert replacement.done() is False
        assert old_engine.pauses == [("wait", None)]
        assert old_engine.stopped is False
        assert registry.default_name == "chat-old"

    await asyncio.wait_for(replacement, timeout=1)

    assert old_engine.stopped is True
    assert registry.default_name == "chat-new"


# --- Issue #2383: post-commit retirement is durably tracked across ------------
# cancellation. These assert the engine stays memory-charged + snapshot-visible
# until its stop()/allocator cleanup finishes or fails truthfully, while the lock
# stays free during a suspended stop.


def _accounted_bytes(manager: ResidentModelManager, model_id: str) -> int:
    """Sum the retired/resident model's reserved bytes as snapshot would."""
    for retirement in manager._retiring.values():
        if retirement.record.model_id == model_id:
            return max(
                retirement.record.estimated_bytes, retirement.record.measured_bytes
            )
    record = manager._records.get(model_id)
    if record is not None:
        return max(record.estimated_bytes, record.measured_bytes)
    return 0


@pytest.mark.asyncio
async def test_cancel_primary_replace_keeps_old_bytes_charged_until_cleanup():
    """PRIMARY replacement: suspended old stop keeps bytes charged; cancel then
    release the stop proves eventual cleanup with no old-route resurrection."""
    registry = ModelRegistry()
    old_engine = BlockingStopLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    loaded: dict[str, FakeEngine] = {}

    async def loader(name: str, path: str | None, performance=None):
        loaded[name] = FakeEngine()
        return entry(name, loaded[name])

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    replacement = asyncio.create_task(
        manager.load("chat-new", replace_group="assistant", replace_mode="wait")
    )
    await old_engine.stop_started.wait()

    # The new route is already authoritative before cancellation.
    assert registry.default_name == "chat-new"
    old_row = next(
        item for item in manager.snapshot()["models"] if item["id"] == "chat-old"
    )
    assert old_row["state"] == "retiring"
    # During suspension the old engine's bytes are STILL charged.
    assert _accounted_bytes(manager, "chat-old") == 4 * GIB
    joined = _accounted_bytes(manager, "chat-old")
    assert joined > 0

    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement

    # Still retiring (suspended), bytes still charged, no resurrection.
    assert registry.default_name == "chat-new"
    assert [item.model_name for item in registry.list_entries()] == ["chat-new"]
    assert _accounted_bytes(manager, "chat-old") == 4 * GIB

    # Release the suspended stop and drain via shutdown -> cleanup completes,
    # bytes disappear, old engine no longer visible.
    old_engine.stop_release.set()
    await asyncio.wait_for(manager.shutdown(), timeout=1)
    assert {item["id"] for item in manager.snapshot()["models"]} == {"chat-new"}
    assert _accounted_bytes(manager, "chat-old") == 0


@pytest.mark.asyncio
async def test_cancel_secondary_replace_keeps_sibling_charged_and_cleaned():
    """SECONDARY replacement: a non-primary replacement candidate suspended in
    stop() keeps bytes charged, and completes cleanup on release."""
    registry = ModelRegistry()
    primary_engine = FakeLifecycleEngine()
    primary = entry("chat-primary", primary_engine)
    old_engine = BlockingStopLifecycleEngine()
    old_secondary = entry("chat-secondary", old_engine)
    registry.add(primary, is_default=True)
    registry.add(old_secondary)

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    manager._index_record(
        ResidencyRecord(
            entry=old_secondary,
            estimated_bytes=2 * GIB,
            loaded_at=0,
            last_used_at=0,
        )
    )

    replacement = asyncio.create_task(
        manager.load("chat-new", replace_group="assistant", replace_mode="wait")
    )
    # The old PRIMARY is retired first (fast), then the suspended SECONDARY.
    await old_engine.stop_started.wait()

    # The replacement group commits: chat-new becomes the new primary; the
    # suspended secondary candidate is retired but still reports as being
    # cleaned up, with its bytes charged.
    assert registry.default_name == "chat-new"
    assert _accounted_bytes(manager, "chat-secondary") == 2 * GIB
    secondary_row = next(
        item for item in manager.snapshot()["models"] if item["id"] == "chat-secondary"
    )
    assert secondary_row["state"] == "retiring"

    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement

    # Sibling still retiring + charged until its stop is released.
    assert _accounted_bytes(manager, "chat-secondary") == 2 * GIB

    old_engine.stop_release.set()
    await asyncio.wait_for(manager.shutdown(), timeout=1)
    assert "chat-secondary" not in {item["id"] for item in manager.snapshot()["models"]}
    assert _accounted_bytes(manager, "chat-secondary") == 0


@pytest.mark.asyncio
async def test_suspended_stop_does_not_block_lease_or_unrelated_op(monkeypatch):
    """LOCK SAFETY: suspended stop must not block unrelated manager ops."""
    monkeypatch.setattr(
        "vllm_mlx.runtime.resident_models._release_allocator_cache", lambda: None
    )
    registry = ModelRegistry()
    primary_engine = BlockingStopLifecycleEngine()
    primary = entry("chat-old", primary_engine)
    unrelated_engine = FakeLifecycleEngine()
    unrelated = entry("chat-unrelated", unrelated_engine)
    registry.add(primary, is_default=True)
    registry.add(unrelated)

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    manager._index_record(
        ResidencyRecord(
            entry=unrelated,
            estimated_bytes=2 * GIB,
            loaded_at=0,
            last_used_at=0,
        )
    )

    # Kick off a replacement that suspends the old primary's stop().
    replacement = asyncio.create_task(
        manager.load("chat-new", replace_group="assistant", replace_mode="wait")
    )
    await primary_engine.stop_started.wait()

    # The synchronous snapshot remains observable during suspended cleanup;
    # the async operations below are the assertions that the lock is free.
    manager.snapshot()

    # A lease on the UNRELATED resident model must not be blocked by the
    # suspended stop() -- it needs the manager lock, which must be free.
    async def use_unrelated_model() -> None:
        async with manager.lease("chat-unrelated"):
            pass

        # A non-conflicting set_pinned on the unrelated model must also proceed.
        await manager.set_pinned("chat-unrelated", True)

    await asyncio.wait_for(use_unrelated_model(), timeout=1)

    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement
    primary_engine.stop_release.set()
    await asyncio.wait_for(manager.shutdown(), timeout=1)


@pytest.mark.asyncio
async def test_failed_stop_exposes_cleanup_failed_and_keeps_bytes_charged():
    """FAILURE: stop() raises -> snapshot exposes cleanup_failed, bytes stay
    charged, route stays retired, and retry/shutdown behaves explicitly."""
    registry = ModelRegistry()
    old_engine = FailingStopLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    await manager.load("chat-new", replace_group="assistant", replace_mode="wait")

    # Route retired, failed cleanup exposed, bytes still charged.
    assert registry.default_name == "chat-new"
    assert [item.model_name for item in registry.list_entries()] == ["chat-new"]
    failed_row = next(
        item for item in manager.snapshot()["models"] if item["id"] == "chat-old"
    )
    assert failed_row["state"] == "failed"
    assert failed_row["cleanup_failed"] == "stop failed"
    assert _accounted_bytes(manager, "chat-old") == 4 * GIB

    # A later shutdown reports rather than silently claiming success or the
    # bytes free: the failed record remains observable.
    with pytest.raises(ResidentModelError, match="chat-old: stop failed"):
        await manager.shutdown()
    assert _accounted_bytes(manager, "chat-old") == 4 * GIB
    current_row = next(
        item for item in manager.snapshot()["models"] if item["id"] == "chat-old"
    )
    assert current_row["state"] == "failed"
    assert current_row["cleanup_failed"] == "stop failed"


@pytest.mark.asyncio
async def test_explicit_unload_propagates_stop_failure():
    registry = ModelRegistry()
    engine = FailingStopLifecycleEngine()
    target = entry("chat-target", engine)
    registry.add(target)
    manager = ResidentModelManager(registry, AsyncMock(), memory_reader=lambda: 0)
    manager._index_record(
        ResidencyRecord(
            entry=target,
            estimated_bytes=2 * GIB,
            loaded_at=0,
            last_used_at=0,
        )
    )

    with pytest.raises(RuntimeError, match="stop failed"):
        await manager.unload("chat-target")

    assert manager._retiring[id(engine)].state == "failed"
    assert _accounted_bytes(manager, "chat-target") == 2 * GIB


@pytest.mark.asyncio
async def test_explicit_unload_reports_cancelled_raw_cleanup():
    """A cancelled manager-owned cleanup remains charged and fails explicitly."""
    registry = ModelRegistry()
    engine = BlockingStopLifecycleEngine()
    target = entry("chat-target", engine)
    registry.add(target)
    manager = ResidentModelManager(registry, AsyncMock(), memory_reader=lambda: 0)
    manager._index_record(
        ResidencyRecord(
            entry=target,
            estimated_bytes=2 * GIB,
            loaded_at=0,
            last_used_at=0,
        )
    )

    unload = asyncio.create_task(manager.unload("chat-target"))
    await engine.stop_started.wait()
    retirement = manager._retiring[id(engine)]
    assert retirement.task is not None
    retirement.task.cancel()

    with pytest.raises(
        ResidentModelError, match="retirement cleanup cancelled before completion"
    ):
        await unload

    assert retirement.state == "failed"
    assert _accounted_bytes(manager, "chat-target") == 2 * GIB


@pytest.mark.asyncio
async def test_ttl_does_not_report_failed_cleanup_as_evicted():
    registry = ModelRegistry()
    engine = FailingStopLifecycleEngine()
    target = entry("chat-target", engine)
    registry.add(target)
    manager = ResidentModelManager(
        registry,
        AsyncMock(),
        idle_ttl_seconds=60,
        memory_reader=lambda: 0,
        clock=lambda: 61,
    )
    manager._index_record(
        ResidencyRecord(
            entry=target,
            estimated_bytes=2 * GIB,
            loaded_at=0,
            last_used_at=0,
        )
    )

    assert await manager.evict_expired() == []
    assert manager._retiring[id(engine)].state == "failed"
    assert _accounted_bytes(manager, "chat-target") == 2 * GIB


@pytest.mark.asyncio
async def test_shutdown_reports_every_failed_cleanup_after_draining_all():
    registry = ModelRegistry()
    manager = ResidentModelManager(registry, AsyncMock(), memory_reader=lambda: 0)
    engines = {}
    for model_id in ("chat-a", "chat-b"):
        engine = FailingStopLifecycleEngine()
        engines[model_id] = engine
        target = entry(model_id, engine)
        registry.add(target)
        manager._index_record(
            ResidencyRecord(
                entry=target,
                estimated_bytes=2 * GIB,
                loaded_at=0,
                last_used_at=0,
            )
        )

    with pytest.raises(ResidentModelError) as caught:
        await manager.shutdown()

    assert "chat-a: stop failed" in str(caught.value)
    assert "chat-b: stop failed" in str(caught.value)
    assert all(engine.stopped is True for engine in engines.values())
    assert all(
        manager._retiring[id(engine)].state == "failed" for engine in engines.values()
    )


@pytest.mark.asyncio
async def test_shutdown_reports_failure_without_a_preserved_cause(monkeypatch):
    """The aggregate error remains useful for cause-less cleanup failures."""
    manager = ResidentModelManager(
        ModelRegistry(), AsyncMock(), memory_reader=lambda: 0
    )
    monkeypatch.setattr(
        manager,
        "_cleanup_failures",
        lambda _identities: [("chat-target", "cleanup interrupted", None)],
    )

    with pytest.raises(
        ResidentModelError, match="chat-target: cleanup interrupted"
    ) as caught:
        await manager.shutdown()

    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_shutdown_retries_a_completed_failed_retirement():
    registry = ModelRegistry()
    engine = FailOnceStopLifecycleEngine()
    target = entry("chat-target", engine)
    registry.add(target)
    manager = ResidentModelManager(registry, AsyncMock(), memory_reader=lambda: 0)
    record = ResidencyRecord(
        entry=target,
        estimated_bytes=2 * GIB,
        loaded_at=0,
        last_used_at=0,
    )
    manager._index_record(record)

    async with manager._lock:
        first_attempt = manager._begin_evict_locked(record, reason="explicit")
    await first_attempt
    assert engine.stop_calls == 1
    assert manager._retiring[id(engine)].state == "failed"
    assert _accounted_bytes(manager, "chat-target") == 2 * GIB

    await manager.shutdown()
    assert engine.stop_calls == 2
    assert engine.stopped is True
    assert id(engine) not in manager._retiring
    assert _accounted_bytes(manager, "chat-target") == 0


@pytest.mark.asyncio
async def test_unload_shutdown_race_stops_engine_exactly_once():
    """IDEMPOTENCY: unload + shutdown for the same engine calls stop() exactly
    once (the second retirement joins the in-flight cleanup, never double-stops)."""
    registry = ModelRegistry()
    primary_engine = FakeLifecycleEngine()
    primary = entry("chat-primary", primary_engine)
    target_engine = BlockingStopLifecycleEngine()
    target = entry("chat-target", target_engine)
    registry.add(primary, is_default=True)
    registry.add(target)

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    manager._index_record(
        ResidencyRecord(
            entry=target,
            estimated_bytes=2 * GIB,
            loaded_at=0,
            last_used_at=0,
        )
    )

    # Start an unload that suspends the target's stop().
    unload_task = asyncio.create_task(manager.unload("chat-target"))
    await target_engine.stop_started.wait()

    # A concurrent shutdown must not call stop() again on the same engine; it
    # joins the in-flight cleanup.
    shutdown_task = asyncio.create_task(manager.shutdown())
    await asyncio.sleep(0)

    target_engine.stop_release.set()
    await asyncio.wait_for(asyncio.gather(unload_task, shutdown_task), timeout=1)
    assert target_engine.stop_calls == 1
    assert target_engine.stopped is True
    assert _accounted_bytes(manager, "chat-target") == 0


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_poison_later_retirement_join():
    registry = ModelRegistry()
    engine = BlockingStopLifecycleEngine()
    target = entry("chat-target", engine)
    registry.add(target)
    manager = ResidentModelManager(registry, AsyncMock(), memory_reader=lambda: 0)
    record = ResidencyRecord(
        entry=target,
        estimated_bytes=2 * GIB,
        loaded_at=0,
        last_used_at=0,
    )
    manager._index_record(record)

    async with manager._lock:
        first_waiter = manager._begin_evict_locked(record, reason="explicit")
    await engine.stop_started.wait()
    first_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_waiter

    async with manager._lock:
        second_waiter = manager._begin_evict_locked(record, reason="shutdown")
    assert second_waiter.cancelled() is False

    engine.stop_release.set()
    await asyncio.wait_for(second_waiter, timeout=1)
    assert engine.stop_calls == 1


@pytest.mark.asyncio
async def test_cancelled_shutdown_does_not_cancel_manager_owned_retirement():
    registry = ModelRegistry()
    engine = BlockingStopLifecycleEngine()
    target = entry("chat-target", engine)
    registry.add(target)
    manager = ResidentModelManager(registry, AsyncMock(), memory_reader=lambda: 0)
    record = ResidencyRecord(
        entry=target,
        estimated_bytes=2 * GIB,
        loaded_at=0,
        last_used_at=0,
    )
    manager._index_record(record)

    async with manager._lock:
        first_waiter = manager._begin_evict_locked(record, reason="explicit")
    await engine.stop_started.wait()
    first_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_waiter

    shutdown = asyncio.create_task(manager.shutdown())
    await asyncio.sleep(0)
    shutdown.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown

    retirement = manager._retiring[id(engine)]
    assert retirement.task is not None
    assert retirement.task.cancelled() is False

    engine.stop_release.set()
    await asyncio.wait_for(manager.shutdown(), timeout=1)
    assert engine.stop_calls == 1
    assert id(engine) not in manager._retiring


@pytest.mark.asyncio
async def test_retirement_does_not_remove_replacement_that_reuses_old_alias():
    registry = ModelRegistry()
    old_engine = BlockingStopLifecycleEngine()
    old = entry("chat-old", old_engine)
    registry.add(old, is_default=True)
    manager = ResidentModelManager(registry, AsyncMock(), memory_reader=lambda: 0)
    old_record = manager.register_primary(old, estimated_bytes=2 * GIB)

    replacement = entry("chat-new")
    replacement.aliases.add("chat-old")
    registry.add(replacement, is_default=True)
    replacement_record = ResidencyRecord(
        entry=replacement,
        estimated_bytes=2 * GIB,
        loaded_at=1,
        last_used_at=1,
        pinned=True,
        primary=True,
    )
    manager._index_record(replacement_record)

    async with manager._lock:
        cleanup = manager._begin_evict_locked(old_record, reason="replace_assistant")
    await old_engine.stop_started.wait()

    assert registry.get_entry("chat-old") is replacement
    assert manager._canonical("chat-old") == "chat-new"

    old_engine.stop_release.set()
    await asyncio.wait_for(cleanup, timeout=1)
    assert registry.get_entry("chat-old") is replacement
    assert registry.list_entries() == [replacement]
    assert manager._records["chat-new"] is replacement_record


@pytest.mark.asyncio
async def test_suspended_stop_cancellation_latency_stays_bounded():
    """CANCELLATION LATENCY: a cancelled caller returns promptly (not blocked on
    the suspended stop), while cleanup continues to completion when released."""
    registry = ModelRegistry()
    old_engine = BlockingStopLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    replacement = asyncio.create_task(
        manager.load("chat-new", replace_group="assistant", replace_mode="wait")
    )
    await old_engine.stop_started.wait()

    # Bounded wait proves the cancellation propagates promptly even though the
    # stop is still suspended (events + wait_for, never sleep-as-sync).
    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(replacement, timeout=1)

    # Cleanup is still in flight (suspended); bytes remain charged.
    assert _accounted_bytes(manager, "chat-old") == 4 * GIB
    # Release and drain.
    old_engine.stop_release.set()
    await asyncio.wait_for(manager.shutdown(), timeout=1)
    assert _accounted_bytes(manager, "chat-old") == 0


@pytest.mark.asyncio
async def test_cancelled_replacement_resumes_old_engine_and_discards_target():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    old_engine.running = 1
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    draining = asyncio.Event()

    async def pause_generation(mode="wait", *, timeout=None):
        old_engine.paused = True
        draining.set()
        await asyncio.Event().wait()

    old_engine.pause_generation = pause_generation
    loaded = None

    async def loader(name: str, path: str | None, performance=None):
        nonlocal loaded
        loaded = entry(name)
        return loaded

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    replacement = asyncio.create_task(
        manager.load("chat-new", replace_group="assistant", replace_mode="wait")
    )
    await draining.wait()
    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement

    assert old_engine.paused is False
    assert old_engine.stopped is False
    assert "chat-new" not in registry
    assert loaded is not None
    assert loaded.engine.stopped is True


@pytest.mark.asyncio
async def test_rejected_busy_replacement_reopens_engine_admission():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    old_engine.running = 1
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(ResidentModelBusyError, match="active request"):
        await manager.load("chat-new", replace_group="assistant")

    assert old_engine.pauses == [("wait", 0)]
    assert old_engine.paused is False
    assert old_engine.stopped is False
    assert registry.default_name == "chat-old"


@pytest.mark.asyncio
async def test_reject_replacement_keeps_old_engine_open_during_slow_load():
    """The healthy assistant serves while a reject-mode replacement loads."""
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)
    request_started = asyncio.Event()
    release_loader = asyncio.Event()

    async def loader(name: str, path: str | None, performance=None):
        old_engine.running = 1
        request_started.set()
        await asyncio.sleep(0)
        await release_loader.wait()
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    replacement = asyncio.create_task(
        manager.load("chat-new", replace_group="assistant")
    )
    await asyncio.wait_for(request_started.wait(), timeout=1)

    assert old_engine.paused is False
    assert old_engine.pauses == [("wait", 0)]
    assert old_engine.running == 1
    assert registry.default_name == "chat-old"

    release_loader.set()
    with pytest.raises(ResidentModelBusyError, match="active request"):
        await asyncio.wait_for(replacement, timeout=1)

    assert old_engine.pauses == [("wait", 0), ("wait", 0)]
    assert old_engine.paused is False
    assert old_engine.running == 1
    assert old_engine.stopped is False
    assert registry.default_name == "chat-old"
    assert [item.model_name for item in registry.list_entries()] == ["chat-old"]


def test_residency_status_uses_engine_owned_request_counts():
    registry = ModelRegistry()
    engine = FakeLifecycleEngine()
    engine.running = 2
    primary = entry("chat", engine)
    registry.add(primary, is_default=True)
    manager = ResidentModelManager(
        registry, lambda *_args: None, memory_reader=lambda: 0
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    model = manager.snapshot()["models"][0]

    assert model["active_requests"] == 2
    assert model["lifecycle"]["running_requests"] == 2


@pytest.mark.asyncio
async def test_resume_attempts_every_engine_after_one_resume_fails():
    manager, _, _, _ = manager_fixture()
    engines = [
        FakeLifecycleEngine(),
        FailingResumeLifecycleEngine(),
        FakeLifecycleEngine(),
    ]
    for engine in engines:
        engine.paused = True

    await manager._resume_engines(engines)  # noqa: SLF001

    assert engines[0].paused is False
    assert engines[2].paused is False


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


def test_performance_reload_preserves_alias_in_routing_and_models_list(monkeypatch):
    from vllm_mlx.config import get_config, reset_config
    from vllm_mlx.routes import models as models_route
    from vllm_mlx.routes import residency as residency_route

    repo = "mlx-community/Qwen3-0.6B-4bit"
    alias = "qwen3-0.6b-4bit"
    registry = ModelRegistry()
    primary = ModelEntry(
        engine=FakeEngine(),
        model_name=repo,
        model_path=repo,
        aliases={alias},
    )
    registry.add(primary, is_default=True)

    async def loader(name: str, path: str | None, performance=None):
        return ModelEntry(
            engine=FakeEngine(),
            model_name=name,
            model_path=path or name,
        )

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=1 * GIB)

    reset_config()
    cfg = get_config()
    cfg.model_registry = registry
    cfg.residency_manager = manager
    cfg.api_key = None
    cfg.embedding_model_locked = None
    app = FastAPI()
    app.include_router(residency_route.router)
    app.include_router(models_route.router)
    monkeypatch.setattr(
        residency_route,
        "resolve_resident_performance",
        lambda performance, **_kwargs: performance,
    )
    try:
        with TestClient(app) as client:
            reload_response = client.post(
                "/v1/models/load",
                json={
                    "model": alias,
                    "model_path": repo,
                    "reload_if_changed": True,
                    "performance": {"kv_cache_dtype": "int8"},
                },
            )
            assert reload_response.status_code == 200
            replacement = registry.get_entry(alias)
            assert replacement.model_name == repo
            assert replacement.model_path == repo
            assert replacement.aliases == {alias}
            registry.validate_model_name(alias)
            assert registry.get_engine(alias) is replacement.engine
            response = client.get("/v1/models")
        assert response.status_code == 200
        ids = {item["id"] for item in response.json()["data"]}
        assert {repo, alias}.issubset(ids)
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_failed_performance_reload_restores_the_last_known_good_engine():
    registry = ModelRegistry()
    primary = entry("chat")
    primary.aliases.add("chat-alias")
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
    assert registry.get_engine("chat-alias") is registry.get_engine("chat")
    assert registry.get_entry("chat").aliases == {"chat-alias"}
    assert registry.get_engine("chat") is not primary.engine
    assert manager.snapshot()["models"][0]["performance"] is None


@pytest.mark.asyncio
async def test_double_failed_primary_reload_clears_every_serving_owner():
    registry = ModelRegistry()
    primary = entry("chat")
    registry.add(primary, is_default=True)
    primary_changes: list[ModelEntry | None] = []

    async def loader(name: str, path: str | None, performance=None):
        del path, performance
        if name == "secondary":
            return entry(name)
        raise RuntimeError("loader unavailable")

    manager = ResidentModelManager(
        registry,
        loader,
        memory_reader=lambda: 0,
        on_primary_changed=primary_changes.append,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    await manager.load("secondary", estimated_bytes=1 * GIB)

    with pytest.raises(RuntimeError, match="loader unavailable"):
        await manager.load(
            "chat",
            performance=ResidentPerformanceConfig(kv_cache_dtype="int4"),
            reload_if_changed=True,
        )

    assert [item.model_name for item in registry.list_entries()] == ["secondary"]
    assert registry.default_name is None
    assert [item["id"] for item in manager.snapshot()["models"]] == ["secondary"]
    assert registry.get_engine("secondary") is not None
    with pytest.raises(KeyError, match="No default model set"):
        registry.get_engine(None)
    assert primary_changes == [None]


@pytest.mark.asyncio
async def test_primary_reload_clears_default_before_awaiting_replacement():
    registry = ModelRegistry()
    primary = entry("chat")
    registry.add(primary, is_default=True)
    replacement_started = asyncio.Event()
    finish_replacement = asyncio.Event()
    primary_changes: list[ModelEntry | None] = []

    async def loader(name: str, path: str | None, performance=None):
        del path, performance
        if name == "secondary":
            return entry(name)
        replacement_started.set()
        await finish_replacement.wait()
        return entry(name)

    manager = ResidentModelManager(
        registry,
        loader,
        memory_reader=lambda: 0,
        on_primary_changed=primary_changes.append,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    await manager.load("secondary", estimated_bytes=1 * GIB)

    reload_task = asyncio.create_task(
        manager.load(
            "chat",
            performance=ResidentPerformanceConfig(kv_cache_dtype="int4"),
            reload_if_changed=True,
        )
    )
    await replacement_started.wait()

    assert registry.default_name is None
    assert registry.get_engine("secondary") is not None
    with pytest.raises(KeyError, match="No default model set"):
        registry.get_engine(None)
    assert primary_changes == [None]

    finish_replacement.set()
    replacement = await reload_task
    assert registry.default_name == "chat"
    assert primary_changes == [None, replacement.entry]


@pytest.mark.asyncio
async def test_primary_reload_clear_callback_failure_restores_old_owner():
    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat", old_engine)
    registry.add(primary, is_default=True)
    changes: list[ModelEntry | None] = []
    handoff_events: list[str] = []

    async def loader(name: str, path: str | None, performance=None):
        del path, performance
        return entry(name)

    def publish(value: ModelEntry | None) -> None:
        changes.append(value)
        if value is None:
            raise RuntimeError("clear failed")

    class Handoff:
        def commit(self, committed):
            handoff_events.append(f"commit:{committed}")

        def rollback(self):
            handoff_events.append("rollback")

    manager = ResidentModelManager(
        registry,
        loader,
        memory_reader=lambda: 0,
        on_primary_handoff=lambda _entry: Handoff(),
        on_primary_changed=publish,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(RuntimeError, match="clear failed"):
        await manager.load(
            "chat",
            performance=ResidentPerformanceConfig(kv_cache_dtype="int4"),
            reload_if_changed=True,
        )

    assert changes == [None, primary]
    assert handoff_events == ["rollback"]
    assert registry.default_name == "chat"
    assert registry.get_engine(None) is old_engine
    assert old_engine.stopped is False
    assert manager.snapshot()["models"][0]["primary"] is True


@pytest.mark.asyncio
async def test_restore_publication_failure_clears_partially_published_primary():
    registry = ModelRegistry()
    primary = entry("chat")
    registry.add(primary, is_default=True)
    published: list[ModelEntry | None] = [primary]
    restored_engines: list[FakeEngine] = []

    async def loader(name: str, path: str | None, performance=None):
        del path
        if performance is not None:
            raise RuntimeError("replacement failed")
        restored = FakeEngine()
        restored_engines.append(restored)
        return entry(name, restored)

    def publish(value: ModelEntry | None) -> None:
        published[0] = value
        if value is not None and value is not primary:
            raise RuntimeError("restore publication failed")

    manager = ResidentModelManager(
        registry,
        loader,
        memory_reader=lambda: 0,
        on_primary_changed=publish,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(RuntimeError, match="replacement failed"):
        await manager.load(
            "chat",
            performance=ResidentPerformanceConfig(kv_cache_dtype="int4"),
            reload_if_changed=True,
        )

    assert published == [None]
    assert restored_engines[0].stopped is True
    assert registry.list_entries() == []
    assert registry.default_name is None
    assert manager.snapshot()["models"] == []


@pytest.mark.asyncio
async def test_restore_publication_failure_tolerates_cleanup_failures(caplog):
    """Cleanup failures cannot mask the original replacement error."""
    registry = ModelRegistry()
    primary = entry("chat")
    registry.add(primary, is_default=True)

    class FailingStopEngine(FakeEngine):
        async def stop(self):
            raise RuntimeError("cleanup stop failed")

    async def loader(name: str, path: str | None, performance=None):
        del path
        if performance is not None:
            raise RuntimeError("replacement failed")
        return entry(name, FailingStopEngine())

    publications: list[ModelEntry | None] = []

    def publish(value: ModelEntry | None) -> None:
        publications.append(value)
        if len(publications) == 1 and value is None:
            return
        if value is not None:
            raise RuntimeError("restore publication failed")
        raise RuntimeError("clear publication failed")

    manager = ResidentModelManager(
        registry,
        loader,
        memory_reader=lambda: 0,
        on_primary_changed=publish,
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    with pytest.raises(RuntimeError, match="replacement failed"):
        await manager.load(
            "chat",
            performance=ResidentPerformanceConfig(kv_cache_dtype="int4"),
            reload_if_changed=True,
        )

    assert registry.list_entries() == []
    assert registry.default_name is None
    assert "Failed to stop partially restored resident model" in caplog.text
    assert "Failed to clear serving-layer primary" in caplog.text


def test_clearing_resident_primary_disables_legacy_routing_and_readiness(monkeypatch):
    from types import SimpleNamespace

    from vllm_mlx import server

    cfg = SimpleNamespace()
    monkeypatch.setattr(server, "get_config", lambda: cfg)
    monkeypatch.setattr(server, "_engine", object())
    monkeypatch.setattr(server, "_model_name", "chat")

    server._set_resident_primary(None)

    assert server._engine is None
    assert server._model_name is None
    assert server._model_alias is None
    assert server._model_path is None
    assert server._enable_auto_tool_choice is False
    assert server._tool_call_parser is None
    assert server._tool_parser_instance is None
    assert server._reasoning_parser is None
    assert server._reasoning_parser_name is None
    assert cfg.engine is None
    assert cfg.model_name is None
    assert cfg.model_alias is None
    assert cfg.model_path is None
    assert cfg.ready is False

    replacement = entry("replacement")
    server._set_resident_primary(replacement)
    assert cfg.engine is replacement.engine
    assert cfg.ready is True


@pytest.mark.asyncio
async def test_failed_stop_rebuilds_the_existing_engine_before_rerouting():
    manager, registry, loaded, _ = manager_fixture(limit_gib=20)
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

    assert registry.get_engine("chat") is loaded["chat"]
    assert registry.get_engine("chat") is not old_engine
    assert manager.snapshot()["models"][0]["id"] == "chat"


@pytest.mark.asyncio
async def test_reload_waits_for_admitted_precommit_request_before_stop():
    admitted = asyncio.Event()
    drained = asyncio.Event()

    class PrecommitEngine(FakeLifecycleEngine):
        def lifecycle_status(self):
            status = super().lifecycle_status()
            status["active_requests"] = 1 if admitted.is_set() else 0
            status["admitted_requests"] = 1 if admitted.is_set() else 0
            return status

        async def pause_generation(self, mode="wait", *, timeout=None):
            self.pauses.append((mode, timeout))
            self.paused = True
            if admitted.is_set():
                await drained.wait()
                admitted.clear()
            return self.lifecycle_status()

    registry = ModelRegistry()
    old_engine = PrecommitEngine()
    primary = entry("chat", old_engine)
    registry.add(primary, is_default=True)
    loaded: list[FakeEngine] = []

    async def loader(name: str, path: str | None, performance=None):
        replacement = FakeEngine()
        loaded.append(replacement)
        return entry(name, replacement)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)
    admitted.set()

    reload_task = asyncio.create_task(
        manager.load(
            "chat",
            performance=ResidentPerformanceConfig(kv_cache_dtype="int8"),
            reload_if_changed=True,
            replace_mode="wait",
        )
    )
    await asyncio.sleep(0)

    assert reload_task.done() is False
    assert old_engine.stopped is False
    assert loaded == []

    drained.set()
    replacement = await asyncio.wait_for(reload_task, timeout=1)
    assert old_engine.stopped is True
    assert replacement.entry.engine is loaded[0]
    assert registry.default_name == "chat"


@pytest.mark.asyncio
async def test_reload_quiesces_and_retires_the_rest_of_explicit_group():
    manager, registry, loaded, _ = manager_fixture(limit_gib=20)
    sibling = await manager.load("chat-sibling")

    replacement = await manager.load(
        "chat",
        performance=ResidentPerformanceConfig(kv_cache_dtype="int8"),
        reload_if_changed=True,
        replace_group="assistant",
        replace_mode="wait",
    )

    assert replacement.primary is True
    assert registry.default_name == "chat"
    assert [item.model_name for item in registry.list_entries()] == ["chat"]
    assert sibling.entry.engine.stopped is True
    assert loaded["chat"].stopped is False


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
        assert isinstance(status.json()["audio_lanes"], list)

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


def test_models_load_requires_strict_json_booleans(monkeypatch):
    """``pin`` / ``pinned`` / ``reload_if_changed`` must be real JSON
    booleans (issue #2362). Pydantic v2's lax mode coerced ``"yes"`` /
    ``"on"`` / ``1`` / ``0`` onto ``bool``, so ``pin: "yes"`` silently
    became ``True`` and triggered a real resident-model reload. With
    ``StrictBool`` a non-boolean wire form is rejected by the schema with
    a 4xx naming the field, while ``true`` / ``false`` behave identically.
    """
    from types import SimpleNamespace

    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes.residency import router

    manager, registry, _, _ = manager_fixture(limit_gib=12)
    monkeypatch.setattr(
        "vllm_mlx.routes.residency.get_config",
        lambda: SimpleNamespace(residency_manager=manager),
    )
    app = FastAPI()
    app.include_router(router)
    install_exception_handlers(app)

    def _assert_400_field(client, field):
        for bad in ("yes", "on", "no", "off", 1, 0, 1.0):
            resp = client.post("/v1/models/load", json={"model": "image", field: bad})
            assert resp.status_code == 400, (
                f"{field}={bad!r} expected 400; got {resp.status_code} {resp.text[:120]}"
            )
            body = resp.json()
            assert body["error"]["type"] == "invalid_request_error"
            assert field in body["error"]["message"]

    def _pin_400(client, field, bad):
        resp = client.put("/v1/models/image/pin", json={field: bad})
        assert resp.status_code == 400, (
            f"{field}={bad!r} expected 400; got {resp.status_code} {resp.text[:120]}"
        )
        assert "pinned" in resp.json()["error"]["message"]

    with TestClient(app) as client:
        _assert_400_field(client, "pin")
        _assert_400_field(client, "reload_if_changed")
        for bad in ("yes", "on", "no", "off", 1, 0, 1.0):
            _pin_400(client, "pinned", bad)

        # Real JSON booleans keep working identically (pin/reload on the
        # load path; pinned on the pin path) and are not rejected.
        ok = client.post(
            "/v1/models/load",
            json={"model": "image", "estimated_size_gb": 1, "pin": True},
        )
        assert ok.status_code == 200
        ok_if = client.post(
            "/v1/models/load", json={"model": "image", "reload_if_changed": False}
        )
        assert ok_if.status_code == 200
        ok_pin = client.put("/v1/models/image/pin", json={"pinned": True})
        assert ok_pin.status_code == 200


def test_residency_snapshot_reports_primary_running_and_queued_requests(monkeypatch):
    """Primary requests bypass manager leases but still belong in residency."""
    from types import SimpleNamespace

    from vllm_mlx.routes.residency import router

    manager, registry, _, _ = manager_fixture(limit_gib=12)
    primary = registry.get_engine("chat")
    primary.running = 1
    primary.waiting = 1
    monkeypatch.setattr(
        "vllm_mlx.routes.residency.get_config",
        lambda: SimpleNamespace(residency_manager=manager),
    )
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.get("/v1/models/residency")

    assert response.status_code == 200
    chat = next(item for item in response.json()["models"] if item["id"] == "chat")
    assert chat["active_requests"] == 2


def test_residency_control_plane_forwards_abort_replacement_policy(monkeypatch):
    from types import SimpleNamespace

    from vllm_mlx.routes.residency import router

    registry = ModelRegistry()
    old_engine = FakeLifecycleEngine()
    primary = entry("chat-old", old_engine)
    registry.add(primary, is_default=True)

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(registry, loader, memory_reader=lambda: 0)
    manager.register_primary(primary, estimated_bytes=4 * GIB)
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
                "model": "chat-new",
                "replace_group": "assistant",
                "replace_mode": "abort",
            },
        )

    assert response.status_code == 200
    assert old_engine.pauses == [("abort", None)]
    assert registry.default_name == "chat-new"
    assert response.json()["replacement_projection"] == {
        "strategy": "keep_then_commit",
        "reason": "keep_both_fits",
        "models_to_free": [{"id": "chat-old", "estimated_bytes": 4 * GIB}],
        "current_bytes": 4 * GIB,
        "requested_bytes": response.json()["estimated_bytes"],
        "projected_bytes": response.json()["estimated_bytes"],
        "limit_bytes": 0,
    }


def test_residency_control_plane_returns_typed_replacement_capacity_projection(
    monkeypatch,
):
    from types import SimpleNamespace

    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes.residency import router

    manager, registry, loaded, _ = manager_fixture(limit_gib=6)
    monkeypatch.setattr(
        "vllm_mlx.routes.residency.get_config",
        lambda: SimpleNamespace(residency_manager=manager),
    )
    app = FastAPI()
    app.include_router(router)
    install_exception_handlers(app)

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/load",
            json={
                "model": "chat-too-large",
                "estimated_size_gb": 8,
                "replace_group": "assistant",
            },
        )

    assert response.status_code == 507
    assert response.json() == {
        "error": {
            "message": (
                "resident model memory ceiling exceeded after projected "
                "assistant replacement"
            ),
            "type": "insufficient_capacity_error",
            "code": "insufficient_capacity_error",
            "param": "estimated_size_gb",
        },
        "replacement_projection": {
            "strategy": "reject",
            "reason": "role_capacity_insufficient_after_eviction",
            "models_to_free": [{"id": "chat", "estimated_bytes": 4 * GIB}],
            "current_bytes": 4 * GIB,
            "requested_bytes": 8 * GIB,
            "projected_bytes": 8 * GIB,
            "limit_bytes": 6 * GIB,
        },
    }
    assert loaded == {}
    assert registry.default_name == "chat"


def test_residency_control_plane_uses_model_path_group_for_destructive_admission(
    monkeypatch,
):
    from types import SimpleNamespace

    from vllm_mlx.routes.residency import router

    manager, registry, loaded, _ = manager_fixture(limit_gib=6)
    profiles = {
        "chat-alias": SimpleNamespace(modality="text"),
        "repo/image": SimpleNamespace(modality="image-gen"),
    }
    monkeypatch.setattr("vllm_mlx.routes.residency.resolve_profile", profiles.get)
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
                "model": "chat-alias",
                "model_path": "repo/image",
                "estimated_size_gb": 4,
                "replace_group": "assistant",
            },
        )

    assert response.status_code == 409
    assert "image-gen" in response.json()["detail"]
    assert loaded == {}
    assert registry.default_name == "chat"
    assert registry.get_engine("chat").stopped is False


def test_residency_unknown_model_path_falls_back_to_safe_admission(monkeypatch):
    from types import SimpleNamespace

    from vllm_mlx.routes.residency import router

    manager, registry, loaded, _ = manager_fixture(limit_gib=6)
    monkeypatch.setattr(
        "vllm_mlx.routes.residency.resolve_profile",
        lambda name: SimpleNamespace(modality="text") if name == "chat-alias" else None,
    )
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
                "model": "chat-alias",
                "model_path": "/unknown/local/checkpoint",
                "estimated_size_gb": 4,
                "replace_group": "assistant",
            },
        )

    assert response.status_code == 507
    assert loaded == {}
    assert registry.default_name == "chat"
    assert registry.get_engine("chat").stopped is False


def test_residency_control_plane_preserves_generic_capacity_error(monkeypatch):
    from types import SimpleNamespace

    from vllm_mlx.routes.residency import router

    manager, _, _, _ = manager_fixture(limit_gib=6)
    monkeypatch.setattr(
        "vllm_mlx.routes.residency.get_config",
        lambda: SimpleNamespace(residency_manager=manager),
    )
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/load",
            json={"model": "image-too-large", "estimated_size_gb": 8},
        )

    assert response.status_code == 507
    assert "memory ceiling exceeded" in response.json()["detail"]


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
    """#78: a control-plane kv_cache_dtype is operator-explicit, so an
    unsupported family is now REJECTED with the shared typed error
    (mapped to 422 by the residency route) instead of being silently
    downgraded to bf16 as before."""
    from vllm_mlx.kv_cache_dtype import KVCacheQuantizationUnsupportedError
    from vllm_mlx.runtime.resident_models import resolve_resident_performance

    monkeypatch.setattr(
        "vllm_mlx.cli._gather_kv_cache_dtype_inputs",
        lambda _name: ({"sliding_window": 4096}, None),
    )
    with pytest.raises(KVCacheQuantizationUnsupportedError):
        resolve_resident_performance(
            ResidentPerformanceConfig(kv_cache_dtype="int4", cache_memory_mb=2048),
            model_name="example/sliding-model",
            model_path=None,
        )


def test_residency_route_maps_kv_unsupported_rejection_to_422(monkeypatch):
    """#78: the residency route must surface the shared typed rejection as an
    actionable 422 BEFORE load, not a 500 from the generic handler."""
    from types import SimpleNamespace

    from vllm_mlx.routes.residency import router

    manager, _, _, _ = manager_fixture(limit_gib=20)
    monkeypatch.setattr(
        "vllm_mlx.routes.residency.get_config",
        lambda: SimpleNamespace(residency_manager=manager),
    )
    monkeypatch.setattr(
        "vllm_mlx.cli._gather_kv_cache_dtype_inputs",
        # Sliding-window layout with an explicit quantized-KV perf request:
        # resolve_resident_performance raises KVCacheQuantizationUnsupportedError.
        lambda _name: ({"sliding_window": 4096}, None),
    )
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/load",
            json={
                "model": "sliding-model",
                "performance": {
                    "kv_cache_dtype": "int4",
                    "cache_memory_mb": 2048,
                },
            },
        )
    assert response.status_code == 422
    assert "#78" not in response.json()["detail"]
    assert "kv" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Protected-role admission (forced alignment). These exercise the shared
# residency ledger's ``admit_role``/``release_role`` transaction path with a
# bare manager (no registry primary) so the alignment role is tested in
# isolation — entirely MLX-free.
# ---------------------------------------------------------------------------


def role_manager_fixture(*, limit_gib=4.0):
    """A manager with no primary model so role admission is tested alone."""

    registry = ModelRegistry()
    clock = Clock()
    manager = ResidentModelManager(
        registry,
        lambda name, path=None, perf=None: entry(name),
        memory_limit_bytes=limit_gib * GIB,
        clock=clock,
        memory_reader=lambda: 0,
    )
    return manager, clock


def _aligner_bytes() -> int:
    # The real catalog footprint of qwen3-aligner /
    # mlx-community/Qwen3-ForcedAligner-0.6B-8bit (~1.19 GiB).
    return 1276473392


@pytest.mark.asyncio
async def test_alignment_role_admitted_under_budget_and_ledger_resident():
    manager, _ = role_manager_fixture(limit_gib=4.0)

    async with manager.admit_role(
        role="alignment",
        model_id="qwen3-aligner",
        requested_bytes=_aligner_bytes(),
        capacity_source="catalog",
    ):
        # During the load the reservation is charged in the "loading" state.
        roles = manager.snapshot()["roles"]
        entry = next(r for r in roles if r["role"] == "alignment")
        assert entry["state"] == "loading"
        assert entry["reserved_bytes"] == _aligner_bytes()

    # After the context succeeds the reservation is committed resident.
    roles = manager.snapshot()["roles"]
    entry = next(r for r in roles if r["role"] == "alignment")
    assert entry["state"] == "resident"
    assert entry["capacity_source"] == "catalog"
    # The reservation is charged to the shared accounted usage.
    assert manager._accounted_usage() >= _aligner_bytes()


@pytest.mark.asyncio
async def test_alignment_role_rejected_with_typed_507_when_over_ceiling():
    # Ceiling below the aligner footprint with no eligible eviction ->
    # the typed role_capacity envelope must be produced.
    manager, _ = role_manager_fixture(limit_gib=0.5)  # 0.5 GiB < aligner

    with pytest.raises(ResidentModelCapacityError) as exc_info:
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=_aligner_bytes(),
            capacity_source="catalog",
        ):
            pass

    exc = exc_info.value
    assert exc.reason == "role_capacity_alignment"
    assert exc.requested_bytes == _aligner_bytes()
    assert exc.limit_bytes == int(0.5 * GIB)
    assert exc.requested_role == "alignment"
    envelope = exc.envelope()
    assert envelope["error"]["type"] == "insufficient_capacity_error"
    assert envelope["error"]["code"] == "insufficient_capacity_error"
    assert envelope["error"]["reason"] == "role_capacity_alignment"
    assert envelope["error"]["requested_bytes"] == _aligner_bytes()
    assert envelope["error"]["limit_bytes"] == int(0.5 * GIB)
    # No leaked reservation on rejection.
    assert manager.snapshot()["roles"] == []


@pytest.mark.asyncio
async def test_alignment_role_rollback_on_load_failure_keeps_ledger_empty():
    manager, _ = role_manager_fixture(limit_gib=4.0)

    with pytest.raises(RuntimeError, match="weight load failed"):
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=_aligner_bytes(),
            capacity_source="catalog",
        ):
            raise RuntimeError("weight load failed")

    # A failed load leaves no leaked reservation.
    assert manager.snapshot()["roles"] == []
    assert manager._accounted_usage() == 0


@pytest.mark.asyncio
async def test_alignment_role_rollback_on_cancellation_keeps_ledger_empty():
    manager, _ = role_manager_fixture(limit_gib=4.0)

    with pytest.raises(asyncio.CancelledError):
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=_aligner_bytes(),
            capacity_source="catalog",
        ):
            raise asyncio.CancelledError()

    assert manager.snapshot()["roles"] == []
    assert manager._accounted_usage() == 0


@pytest.mark.asyncio
async def test_alignment_role_replace_retires_previous_on_failure():
    # A new aligner replaces a previous one; the load drops the old engine so
    # its reservation must not be restored on failure (retire_previous).
    manager, _ = role_manager_fixture(limit_gib=4.0)

    # First aligner loads and commits.
    async with manager.admit_role(
        role="alignment",
        model_id="qwen3-aligner",
        requested_bytes=_aligner_bytes(),
        capacity_source="catalog",
    ):
        pass
    assert manager.snapshot()["roles"][0]["model"] == "qwen3-aligner"

    # Replace with a second aligner whose load fails after retire_previous.
    with pytest.raises(RuntimeError, match="reload failed"):
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-forced-aligner",
            requested_bytes=_aligner_bytes(),
            capacity_source="catalog",
            replace_existing=True,
        ) as admission:
            admission.retire_previous()
            raise RuntimeError("reload failed")

    # The old engine is gone, so no reservation should remain.
    assert manager.snapshot()["roles"] == []


@pytest.mark.asyncio
async def test_alignment_role_replace_restores_previous_unless_retired():
    # Without retire_previous, a failed replacement restores the previous
    # reservation (the old engine still exists).
    manager, _ = role_manager_fixture(limit_gib=4.0)

    async with manager.admit_role(
        role="alignment",
        model_id="qwen3-aligner",
        requested_bytes=_aligner_bytes(),
        capacity_source="catalog",
    ):
        pass

    try:
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-forced-aligner",
            requested_bytes=_aligner_bytes(),
            capacity_source="catalog",
            replace_existing=True,
        ):
            raise RuntimeError("reload failed")
    except RuntimeError:
        pass

    # Previous aligner reservation restored (old engine survives).
    roles = manager.snapshot()["roles"]
    assert len(roles) == 1
    assert roles[0]["model"] == "qwen3-aligner"
    assert roles[0]["state"] == "resident"


@pytest.mark.asyncio
async def test_alignment_role_rejects_second_loading_admission():
    """A role must never host two concurrent in-flight LOADS.

    pr_validate codex BLOCKING #1: a second admission while the first is
    still loading would overwrite the ledger entry, and a later rollback
    could resurrect a stale ``\"loading\"`` record. The ledger layer rejects
    the second admission instead of corrupting the reservation.
    """
    manager, _ = role_manager_fixture(limit_gib=4.0)

    started = asyncio.Event()
    release = asyncio.Event()

    async def hold_open():
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=_aligner_bytes(),
            capacity_source="catalog",
        ):
            started.set()
            await release.wait()

    first = asyncio.create_task(hold_open())
    await started.wait()

    # While the first load is in flight ("loading"), a second admission for
    # the same role — even with replace_existing=True — must be rejected, not
    # overwrite the in-flight reservation.
    with pytest.raises(ResidentModelError, match="loading admission in flight"):
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-forced-aligner",
            requested_bytes=_aligner_bytes(),
            capacity_source="catalog",
            replace_existing=True,
        ):
            pass

    release.set()
    await first

    # The original (first) reservation committed resident, untouched by the
    # rejected second admission.
    roles = [r for r in manager.snapshot()["roles"] if r["role"] == "alignment"]
    assert len(roles) == 1
    assert roles[0]["model"] == "qwen3-aligner"
    assert roles[0]["state"] == "resident"


@pytest.mark.asyncio
async def test_alignment_role_unknown_capacity_fails_closed_under_ceiling():
    # Under a configured ceiling, a model with no catalog/local-cache size
    # must fail closed rather than admit blind.
    manager, _ = role_manager_fixture(limit_gib=4.0)

    with pytest.raises(ResidentModelCapacityError) as exc_info:
        async with manager.admit_role(
            role="alignment",
            model_id="some-unknown-aligner",
            requested_bytes=None,
            capacity_source="unknown",
        ):
            pass

    assert exc_info.value.reason == "role_capacity_unknown"
    assert exc_info.value.requested_bytes is None
    assert exc_info.value.envelope()["error"]["reason"] == "role_capacity_unknown"
    assert manager.snapshot()["roles"] == []


@pytest.mark.asyncio
async def test_alignment_role_no_ceiling_admits_without_typed_rejection():
    # With no configured ceiling (limit 0) admission bypasses capacity checks
    # entirely, preserving pre-existing behavior for unmanaged servers.
    manager, _ = role_manager_fixture(limit_gib=0.0)

    async with manager.admit_role(
        role="alignment",
        model_id="qwen3-aligner",
        requested_bytes=None,  # unknown, but no ceiling -> admitted
        capacity_source="unknown",
    ):
        pass

    roles = manager.snapshot()["roles"]
    assert len(roles) == 1
    assert roles[0]["state"] == "resident"


@pytest.mark.asyncio
async def test_alignment_role_release_removes_ledger_charge():
    manager, _ = role_manager_fixture(limit_gib=4.0)

    async with manager.admit_role(
        role="alignment",
        model_id="qwen3-aligner",
        requested_bytes=_aligner_bytes(),
        capacity_source="catalog",
    ):
        pass
    assert manager._accounted_usage() >= _aligner_bytes()

    await manager.release_role("alignment")
    assert manager.snapshot()["roles"] == []
    assert manager._accounted_usage() == 0


@pytest.mark.asyncio
async def test_alignment_role_release_exclusive_retires_sibling_on_success():
    """pr_validate codex BLOCKING (round-17): alignment and dictation are
    mutually exclusive, so a resident ``speech-input`` reservation must be
    retired atomically WITHIN the alignment admission transaction — read from
    the real auxiliary ``_roles`` ledger, never the ``snapshot()`` projection
    (which mixes in synthesized ordinary-model entries a route could wrongly
    release). On SUCCESS the sibling stays retired (the ASR engine was
    evicted) so the ledger charges only the alignment role."""
    manager, _ = role_manager_fixture(limit_gib=4.0)

    small = int(0.05 * 1024**3)
    async with manager.admit_role(
        role="speech-input",
        model_id="whisper-large",
        requested_bytes=small,
        capacity_source="catalog",
    ):
        pass
    assert any(r["role"] == "speech-input" for r in manager.snapshot()["roles"])

    async with manager.admit_role(
        role="alignment",
        model_id="qwen3-aligner",
        requested_bytes=_aligner_bytes(),
        capacity_source="catalog",
        release_exclusive_role="speech-input",
    ):
        pass

    roles = {r["role"] for r in manager.snapshot()["roles"]}
    assert "alignment" in roles
    assert "speech-input" not in roles  # retired with the successful admission


@pytest.mark.asyncio
async def test_alignment_role_release_exclusive_restores_sibling_on_rollback():
    """pr_validate codex BLOCKING (round-17): when the alignment admission
    transaction rolls back (no engine evicted), the retired ``speech-input``
    reservation MUST be restored atomically under the same lock — a
    still-resident ASR engine can never be left silently unaccounted. There is
    no window for a concurrent admission to claim the freed capacity, because
    the release and the restore are the SAME manager transaction."""
    manager, _ = role_manager_fixture(limit_gib=0.5)

    small = int(0.05 * 1024**3)
    async with manager.admit_role(
        role="speech-input",
        model_id="whisper-large",
        requested_bytes=small,
        capacity_source="catalog",
    ):
        pass
    assert any(r["role"] == "speech-input" for r in manager.snapshot()["roles"])

    # The aligner (1.19 GiB) cannot fit under 0.5 GiB even after the small
    # speech-input reservation is released -> the in-lock rollback runs and
    # must restore speech-input before the capacity error propagates.
    with pytest.raises(ResidentModelCapacityError):
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=_aligner_bytes(),
            capacity_source="catalog",
            release_exclusive_role="speech-input",
        ):
            pass

    roles = {r["role"] for r in manager.snapshot()["roles"]}
    assert "speech-input" in roles  # restored: the ASR engine is still resident
    assert "alignment" not in roles  # the rejected aligner left no reservation
    # The restored reservation keeps its original footprint (accounted once).
    (speech,) = [r for r in manager.snapshot()["roles"] if r["role"] == "speech-input"]
    assert speech["reserved_bytes"] == small


@pytest.mark.asyncio
async def test_alignment_role_exclusive_sibling_capacity_is_retained_during_load():
    """pr_validate codex BLOCKING (round-19/21): the manager lock is NOT held
    across the yielded aligner load, so the mutually-exclusive sibling's
    capacity could be STOLEN by a concurrent admission if it were released
    up front. Instead the sibling reservation is RETAINED (credited against the
    aligner, never freed), so a concurrent admission CANNOT claim its bytes:
    it is rejected with 507 while the aligner loads, and the rollback leaves
    the sibling's still-resident engine correctly accounted — no unaccounted
    engine and no over-ceiling ledger."""
    manager, _ = role_manager_fixture(limit_gib=1.0)
    GIB = 1024**3
    speech = int(0.4 * GIB)
    other = int(0.2 * GIB)
    aligner = int(0.3 * GIB)
    concurrent = int(0.5 * GIB)

    # Pre-existing roles: speech-input (0.4) + a speech-output sibling (0.2) = 0.6.
    async with manager.admit_role(
        role="speech-output",
        model_id="some-tts",
        requested_bytes=other,
        capacity_source="catalog",
    ):
        pass
    async with manager.admit_role(
        role="speech-input",
        model_id="whisper-large",
        requested_bytes=speech,
        capacity_source="catalog",
    ):
        pass
    assert manager._accounted_usage() == other + speech

    # Enter alignment (speech-input retained + credited, so no false 507 and
    # no freed capacity). During the yielded LOAD a concurrent admission wants
    # the sibling's bytes, but they are RETAINED -> it cannot fit -> 507.
    with pytest.raises(ResidentModelCapacityError):
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=aligner,
            capacity_source="catalog",
            release_exclusive_role="speech-input",
        ):
            async with manager.admit_role(
                role="image-generation",
                model_id="other-gen",
                requested_bytes=concurrent,
                capacity_source="catalog",
            ):
                pass
            raise RuntimeError("load failed")

    # The concurrent image-generation admission was REJECTED (sibling capacity
    # retained), so on rollback the ledger is exactly as before the aligner:
    # speech-input restored (its engine is still resident) + speech-output,
    # alignment gone.
    roles = {r["role"] for r in manager.snapshot()["roles"]}
    assert "alignment" not in roles
    assert "speech-output" in roles
    assert "image-generation" not in roles
    assert "speech-input" in roles  # retained + credited, its engine accounted
    assert manager._accounted_usage() == other + speech


@pytest.mark.asyncio
async def test_alignment_role_commit_finalize_is_cancellation_shielded():
    """pr_validate codex BLOCKING (round-22): after ``admission.commit()`` the
    engine is resident, so a cancellation arriving while the success-path
    finalization waits on the manager lock must STILL retire the evicted
    sibling (and keep the committed record resident) — never leak a phantom
    sibling charge. The success finalization runs under a cancel-drained
    shield, so whether the cancellation is handled by that shield or by the
    ``committed`` rollback branch, the invariant holds: sibling retired,
    alignment record resident."""
    from vllm_mlx.runtime.resident_models import ResidentRoleAdmission

    manager, _ = role_manager_fixture(limit_gib=1.0)
    GIB = 1024**3
    speech = int(0.4 * GIB)
    aligner = int(0.3 * GIB)

    async with manager.admit_role(
        role="speech-input",
        model_id="whisper-large",
        requested_bytes=speech,
        capacity_source="catalog",
    ):
        pass

    # align() commits the engine inside its body, then BLOCKS holding the
    # manager lock so cancellation lands deterministically AFTER publication.
    committed_in_body = asyncio.Event()
    release_body_lock = asyncio.Event()

    async def _align():
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=aligner,
            capacity_source="catalog",
            release_exclusive_role="speech-input",
        ) as admission:
            # Engine published -> mark resident; a cancellation anywhere after
            # this must not resurrect the evicted sibling.
            if isinstance(admission, ResidentRoleAdmission):
                admission.commit()
            # Hold the manager lock and park: the body owns the lock here, so
            # cancelling now exercises the post-commit cleanup under lock
            # contention (either the committed-rollback branch or the shielded
            # else-finalize) — both must retire the sibling.
            async with manager._lock:
                committed_in_body.set()
                await release_body_lock.wait()

    align_task = asyncio.create_task(_align())
    await committed_in_body.wait()  # committed and holding the manager lock
    align_task.cancel()  # cancel AFTER publication
    release_body_lock.set()  # let the post-commit cleanup run (drains)

    try:
        await align_task
    except asyncio.CancelledError:
        pass

    # The cancellation must not leak: sibling retired (ASR engine evicted) and
    # the committed alignment record resident.
    roles = {r["role"] for r in manager.snapshot()["roles"]}
    assert "speech-input" not in roles
    (aligned,) = [r for r in manager.snapshot()["roles"] if r["role"] == "alignment"]
    assert aligned["state"] == "resident"
    assert aligned["reserved_bytes"] == aligner


@pytest.mark.asyncio
async def test_alignment_role_success_finalize_drains_repeated_cancellation(
    monkeypatch,
):
    """Success finalization completes even if its waiter is cancelled twice."""
    manager, _ = role_manager_fixture(limit_gib=4.0)
    finalize_started = asyncio.Event()
    release_finalize = asyncio.Event()
    original = manager._finalize_role_commit

    async def delayed_finalize(**kwargs):
        finalize_started.set()
        await release_finalize.wait()
        await original(**kwargs)

    monkeypatch.setattr(manager, "_finalize_role_commit", delayed_finalize)

    async def admit():
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=_aligner_bytes(),
            capacity_source="catalog",
        ):
            pass

    task = asyncio.create_task(admit())
    await finalize_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    release_finalize.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    (role,) = manager.snapshot()["roles"]
    assert role["role"] == "alignment"
    assert role["state"] == "resident"


@pytest.mark.asyncio
async def test_alignment_role_rollback_preserves_concurrently_replaced_sibling():
    """pr_validate codex BLOCKING (round-24): rollback must only retire the
    sibling reservation THIS transaction retained — never erase a NEWER
    reservation a concurrent admission installed for that role while the
    aligner load ran. The identity-checked pop preserves the replacement."""
    manager, _ = role_manager_fixture(limit_gib=1.0)
    GIB = 1024**3
    s1 = int(0.4 * GIB)
    s2 = int(0.25 * GIB)
    aligner = int(0.3 * GIB)

    async with manager.admit_role(
        role="speech-input",
        model_id="whisper-large",
        requested_bytes=s1,
        capacity_source="catalog",
    ):
        pass

    with pytest.raises(RuntimeError, match="load failed"):
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=aligner,
            capacity_source="catalog",
            release_exclusive_role="speech-input",
        ) as admission:
            # The load evicted the ASR engine -> retire_exclusive fires.
            admission.retire_exclusive()
            # A concurrent admission REPLACES the speech-input reservation with
            # a newer (still-resident engine) one while the aligner loads.
            async with manager.admit_role(
                role="speech-input",
                model_id="whisper-large",
                requested_bytes=s2,
                capacity_source="catalog",
                replace_existing=True,
            ):
                pass
            raise RuntimeError("load failed")

    # The rollback must NOT erase the newer reservation (its engine is
    # resident); only alignment is rolled back.
    (speech,) = [r for r in manager.snapshot()["roles"] if r["role"] == "speech-input"]
    assert speech["reserved_bytes"] == s2
    assert "alignment" not in {r["role"] for r in manager.snapshot()["roles"]}


# ---------------------------------------------------------------------------
# Engine-layer contract for #2305: closed role enum + typed capacity envelope.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closed_role_enum_rejects_unknown_and_admits_all_six():
    # The six roles the shared residency lifecycle owns are all admissible.
    manager, _ = role_manager_fixture(limit_gib=16.0)
    for role in (
        "assistant",
        "speech-input",
        "speech-output",
        "alignment",
        "image-generation",
        "video-generation",
    ):
        async with manager.admit_role(
            role=role,
            model_id=f"model-{role}",
            requested_bytes=int(0.1 * GIB),
            capacity_source="catalog",
        ):
            pass
    resident = {r["role"] for r in manager.snapshot()["roles"]}
    assert resident == {
        "assistant",
        "speech-input",
        "speech-output",
        "alignment",
        "image-generation",
        "video-generation",
    }

    # An arbitrary/unknown role string must be rejected at the ledger gate with
    # a clear error — never silently admitted into the closed set.
    with pytest.raises(ResidentModelError, match="unknown resident role"):
        async with manager.admit_role(
            role="tts-extra",
            model_id="whatever",
            requested_bytes=int(0.1 * GIB),
            capacity_source="catalog",
        ):
            pass

    # release_role is gated by the same closed set.
    with pytest.raises(ResidentModelError, match="unknown resident role"):
        await manager.release_role("no-such-role")


def test_role_enum_accepts_canonical_underscore_aliases():
    # Callers may use the snake_case canonical names (speech_input,
    # speech_output, image_generation, video_generation) and the closed enum
    # coerces them to the authoritative dash-form wire values.
    for role, wire in (
        ("speech_input", "speech-input"),
        ("speech_output", "speech-output"),
        ("image_generation", "image-generation"),
        ("video_generation", "video-generation"),
    ):
        assert ResidentRole.coerce(role).value == wire

    with pytest.raises(ResidentModelError, match="invalid role"):
        ResidentRole.coerce(7)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_role_aliases_share_one_canonical_ledger_key():
    manager, _ = role_manager_fixture(limit_gib=4.0)

    async with manager.admit_role(
        role="speech_input",
        model_id="asr",
        requested_bytes=int(0.3 * GIB),
        capacity_source="catalog",
    ):
        pass

    roles = manager.snapshot()["roles"]
    assert [row["role"] for row in roles] == ["speech-input"]

    # The dash and underscore spellings are one logical role, not two ledger
    # slots that could be admitted and charged independently.
    with pytest.raises(ResidentModelError, match="already resident"):
        async with manager.admit_role(
            role="speech-input",
            model_id="other-asr",
            requested_bytes=int(0.3 * GIB),
            capacity_source="catalog",
        ):
            pass

    # Release accepts either spelling and removes the canonical reservation.
    await manager.release_role("speech_input")
    assert manager.snapshot()["roles"] == []


@pytest.mark.asyncio
async def test_capacity_507_envelope_carries_full_role_contract():
    # A capacity rejection must surface the stable machine fields #2306 will
    # parse: requested_role, the live resident_roles ledger, recovery_actions,
    # and the requested/used/limit bytes.
    manager, _ = role_manager_fixture(limit_gib=1.0)
    small = int(0.2 * GIB)

    # Seed a resident speech-output role so the ledger is non-empty.
    async with manager.admit_role(
        role="speech-output",
        model_id="some-tts",
        requested_bytes=small,
        capacity_source="catalog",
    ):
        pass

    with pytest.raises(ResidentModelCapacityError) as exc_info:
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=_aligner_bytes(),
            capacity_source="catalog",
        ):
            pass

    exc = exc_info.value
    assert exc.requested_role == "alignment"
    envelope = exc.envelope()["error"]
    assert envelope["type"] == "insufficient_capacity_error"
    assert envelope["requested_role"] == "alignment"
    assert envelope["requested_bytes"] == _aligner_bytes()
    assert envelope["limit_bytes"] == int(1.0 * GIB)
    assert envelope["used_bytes"] > 0
    # The live ledger is reflected in the envelope (speech-output resident).
    resident = envelope["resident_roles"]
    assert resident == [
        {
            "role": "speech-output",
            "model_id": "some-tts",
            "reserved_bytes": small,
            "state": "resident",
        }
    ]
    # Alignment's server-declared recovery actions.
    assert envelope["recovery_actions"] == ["unload_assistant"]


@pytest.mark.asyncio
async def test_capacity_507_envelope_includes_registered_assistant_role():
    registry = ModelRegistry()
    assistant = entry("assistant-model")
    registry.add(assistant, is_default=True)
    manager = ResidentModelManager(
        registry,
        lambda name, path=None, perf=None: entry(name),
        memory_limit_bytes=1 * GIB,
        memory_reader=lambda: 0,
    )
    manager.register_primary(assistant, estimated_bytes=int(0.8 * GIB))

    with pytest.raises(ResidentModelCapacityError) as exc_info:
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=int(0.3 * GIB),
            capacity_source="catalog",
        ):
            pass

    envelope = exc_info.value.envelope()["error"]
    assert envelope["used_bytes"] == int(0.8 * GIB)
    assert envelope["resident_roles"] == [
        {
            "role": "assistant",
            "model_id": "assistant-model",
            "reserved_bytes": int(0.8 * GIB),
            "state": "resident",
        }
    ]
    assert envelope["recovery_actions"] == ["unload_assistant"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_state", ["retiring", "failed"])
async def test_capacity_507_envelope_includes_charged_retirement(
    cleanup_state, monkeypatch
):
    monkeypatch.setattr(
        "vllm_mlx.runtime.resident_models._release_allocator_cache", lambda: None
    )
    registry = ModelRegistry()
    engine = BlockingStopLifecycleEngine()
    assistant = entry("retiring-assistant", engine)
    registry.add(assistant)
    manager = ResidentModelManager(
        registry,
        AsyncMock(),
        memory_limit_bytes=1 * GIB,
        memory_reader=lambda: 0,
    )
    record = ResidencyRecord(
        entry=assistant,
        estimated_bytes=int(0.8 * GIB),
        loaded_at=0,
        last_used_at=0,
    )
    manager._index_record(record)

    async with manager._lock:
        cleanup = manager._begin_evict_locked(record, reason="explicit")
    await engine.stop_started.wait()
    retirement = manager._retiring[id(engine)]
    if cleanup_state == "failed":
        assert retirement.task is not None
        retirement.task.cancel()
        await cleanup

    with pytest.raises(ResidentModelCapacityError) as exc_info:
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=int(0.3 * GIB),
            capacity_source="catalog",
        ):
            pass

    envelope = exc_info.value.envelope()["error"]
    assert envelope["used_bytes"] == int(0.8 * GIB)
    assert envelope["resident_roles"] == [
        {
            "role": "assistant",
            "model_id": "retiring-assistant",
            "reserved_bytes": int(0.8 * GIB),
            "state": cleanup_state,
        }
    ]

    if cleanup_state == "retiring":
        engine.stop_release.set()
        await cleanup


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resident_role", "requested_role", "admission_options"),
    [
        ("speech-output", "speech-output", {"replace_existing": True}),
        (
            "speech-input",
            "alignment",
            {"release_exclusive_role": "speech-input"},
        ),
    ],
)
async def test_capacity_507_used_bytes_remain_actual_when_role_is_credited(
    resident_role, requested_role, admission_options
):
    manager, _ = role_manager_fixture(limit_gib=1.0)
    resident_bytes = int(0.8 * GIB)
    async with manager.admit_role(
        role=resident_role,
        model_id="resident-model",
        requested_bytes=resident_bytes,
        capacity_source="catalog",
    ):
        pass

    with pytest.raises(ResidentModelCapacityError) as exc_info:
        async with manager.admit_role(
            role=requested_role,
            model_id="oversized-replacement",
            requested_bytes=int(1.1 * GIB),
            capacity_source="catalog",
            **admission_options,
        ):
            pass

    envelope = exc_info.value.envelope()["error"]
    assert envelope["used_bytes"] == resident_bytes
    assert envelope["resident_roles"] == [
        {
            "role": resident_role,
            "model_id": "resident-model",
            "reserved_bytes": resident_bytes,
            "state": "resident",
        }
    ]


@pytest.mark.asyncio
async def test_alignment_replace_conflict_reports_recovery_actions():
    # The capacity rejection for the alignment role must pair its
    # requested_role with the role-appropriate recovery actions.
    manager, _ = role_manager_fixture(limit_gib=0.5)

    with pytest.raises(ResidentModelCapacityError) as exc_info:
        async with manager.admit_role(
            role="alignment",
            model_id="qwen3-aligner",
            requested_bytes=_aligner_bytes(),
            capacity_source="catalog",
        ):
            pass

    envelope = exc_info.value.envelope()["error"]
    assert envelope["requested_role"] == "alignment"
    assert envelope["recovery_actions"] == ["unload_assistant"]


@pytest.mark.asyncio
async def test_recovery_actions_are_role_appropriate():
    # recovery_actions is data-driven per role: speech-input gets the three
    # actions; speech-output only stop_speech_output; assistant none.
    async def envelope_for_role(role):
        manager, _ = role_manager_fixture(limit_gib=0.1)
        with pytest.raises(ResidentModelCapacityError) as exc_info:
            async with manager.admit_role(
                role=role,
                model_id=f"model-{role}",
                requested_bytes=int(1.0 * GIB),  # far over the tiny ceiling
                capacity_source="catalog",
            ):
                pass
        return exc_info.value.envelope()["error"]

    assert (await envelope_for_role("speech-input"))["recovery_actions"] == [
        "select_smaller_speech_input",
        "stop_speech_output",
        "unload_assistant",
    ]
    assert (await envelope_for_role("speech-output"))["recovery_actions"] == [
        "stop_speech_output"
    ]
    assert (await envelope_for_role("alignment"))["recovery_actions"] == [
        "unload_assistant"
    ]
    # Assistant has no server-declared recovery actions.
    assert (await envelope_for_role("assistant"))["recovery_actions"] == []


@pytest.mark.asyncio
async def test_role_telemetry_consistent_across_success_rollback_cancel():
    # The role ledger stays consistent across the full lifecycle: an admitted
    # role commits resident; a failed load rolls back to empty; a cancellation
    # also leaves no leaked reservation (all deterministic — no real sleeps).
    manager, _ = role_manager_fixture(limit_gib=4.0)

    # Success: commits to resident.
    async with manager.admit_role(
        role="speech-input",
        model_id="whisper",
        requested_bytes=int(0.3 * GIB),
        capacity_source="catalog",
    ):
        pass
    (entry,) = [r for r in manager.snapshot()["roles"] if r["role"] == "speech-input"]
    assert entry["state"] == "resident"
    assert entry["reserved_bytes"] == int(0.3 * GIB)

    # Rollback on failure: the new role is not left in the ledger.
    with pytest.raises(RuntimeError, match="boom"):
        async with manager.admit_role(
            role="speech-output",
            model_id="tts",
            requested_bytes=int(0.3 * GIB),
            capacity_source="catalog",
        ):
            raise RuntimeError("boom")
    assert "speech-output" not in {r["role"] for r in manager.snapshot()["roles"]}

    # Cancellation: no leaked reservation.
    with pytest.raises(asyncio.CancelledError):
        async with manager.admit_role(
            role="speech-output",
            model_id="tts",
            requested_bytes=int(0.3 * GIB),
            capacity_source="catalog",
        ):
            raise asyncio.CancelledError()
    assert "speech-output" not in {r["role"] for r in manager.snapshot()["roles"]}

    # release_role drops a committed reservation so the snapshot is consistent.
    await manager.release_role("speech-input")
    assert "speech-input" not in {r["role"] for r in manager.snapshot()["roles"]}


@pytest.mark.asyncio
async def test_release_role_unknown_rejected_after_valid_release_works():
    manager, _ = role_manager_fixture(limit_gib=4.0)
    async with manager.admit_role(
        role="speech-output",
        model_id="tts",
        requested_bytes=int(0.3 * GIB),
        capacity_source="catalog",
    ):
        pass
    # Valid release succeeds.
    await manager.release_role("speech-output")
    assert "speech-output" not in {r["role"] for r in manager.snapshot()["roles"]}
    # Unknown release is gated by the closed enum.
    with pytest.raises(ResidentModelError, match="unknown resident role"):
        await manager.release_role("speech-output-extra")
