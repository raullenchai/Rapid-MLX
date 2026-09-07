import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from vllm_mlx.runtime.primary_lifecycle import PrimaryModelLifecycle


class FakeEngine:
    def __init__(self, *, loaded: bool = False) -> None:
        self._loaded = loaded
        self.start_calls = 0
        self.stop_calls = 0
        self.release_calls = 0
        self.active_requests = 0
        self.start_gate: asyncio.Event | None = None
        self.pause_gate: asyncio.Event | None = None
        self.fail_start = False
        self.fail_resume = False
        self.partial_start_failure = False
        self.paused = False

    async def start(self) -> None:
        self.start_calls += 1
        if self.start_gate is not None:
            await self.start_gate.wait()
        if self.fail_start:
            if self.partial_start_failure:
                self._loaded = True
            raise RuntimeError("load failed")
        self._loaded = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self._loaded = False

    def release_admission_reservation(self) -> None:
        self.release_calls += 1

    def lifecycle_status(self) -> dict[str, int]:
        return {"active_requests": self.active_requests}

    async def pause_generation(self, mode: str, *, timeout: float | None = None):
        assert mode == "wait"
        assert timeout == 0
        if self.active_requests:
            raise TimeoutError
        self.paused = True
        if self.pause_gate is not None:
            await self.pause_gate.wait()
        return self.lifecycle_status()

    async def resume_generation(self):
        if self.fail_resume:
            raise RuntimeError("resume failed")
        self.paused = False
        return self.lifecycle_status()


@pytest.mark.asyncio
async def test_concurrent_first_requests_share_load_and_cancellation_isolated():
    engine = FakeEngine()
    engine.start_gate = asyncio.Event()
    loaded_hooks = 0

    async def on_loaded() -> None:
        nonlocal loaded_hooks
        loaded_hooks += 1

    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True, on_loaded=on_loaded)
    first = asyncio.create_task(lifecycle.ensure_loaded())
    second = asyncio.create_task(lifecycle.ensure_loaded())
    await asyncio.sleep(0)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    engine.start_gate.set()
    await second

    assert engine.start_calls == 1
    assert loaded_hooks == 1
    assert lifecycle.snapshot()["state"] == "ready"


@pytest.mark.asyncio
async def test_idle_unload_keeps_engine_reloadable_and_runs_cache_hooks():
    now = [100.0]
    engine = FakeEngine(loaded=True)
    events: list[str] = []
    states: list[str] = []

    lifecycle = PrimaryModelLifecycle(
        engine,
        idle_unload_seconds=30,
        before_unload=lambda: events.append("save"),
        release_allocator_cache=lambda: events.append("release"),
        on_loaded=lambda: events.append("restore"),
        on_state_change=states.append,
        clock=lambda: now[0],
    )

    now[0] += 31
    assert await lifecycle.evict_if_idle() is True
    assert engine.stop_calls == 1
    assert events == ["save", "release"]
    assert lifecycle.snapshot()["state"] == "standby"
    assert engine.paused is False

    await lifecycle.ensure_loaded()
    assert engine.start_calls == 1
    assert events == ["save", "release", "restore"]
    assert states == ["unloading", "standby", "loading", "ready"]


@pytest.mark.asyncio
async def test_pre_unload_failure_still_honors_memory_policy():
    now = [100.0]
    engine = FakeEngine(loaded=True)

    async def fail_save() -> None:
        raise OSError("disk full")

    lifecycle = PrimaryModelLifecycle(
        engine,
        idle_unload_seconds=5,
        before_unload=fail_save,
        clock=lambda: now[0],
    )
    now[0] += 6

    assert await lifecycle.evict_if_idle() is True
    assert engine._loaded is False
    assert engine.stop_calls == 1
    assert engine.paused is False
    assert lifecycle.snapshot()["state"] == "standby"
    await lifecycle.ensure_loaded()
    assert engine.start_calls == 1


@pytest.mark.asyncio
async def test_resume_failure_never_publishes_false_standby():
    now = [100.0]
    engine = FakeEngine(loaded=True)
    engine.fail_resume = True
    lifecycle = PrimaryModelLifecycle(
        engine, idle_unload_seconds=5, clock=lambda: now[0]
    )
    now[0] += 6

    with pytest.raises(RuntimeError, match="resume failed"):
        await lifecycle.evict_if_idle()
    assert lifecycle.snapshot()["state"] == "error"
    assert lifecycle.snapshot()["error"] == "AdmissionResumeError"

    engine.fail_resume = False
    await lifecycle.ensure_loaded()
    assert lifecycle.snapshot()["state"] == "ready"
    assert engine.paused is False


@pytest.mark.asyncio
async def test_detach_refuses_in_flight_load_then_suppresses_callbacks():
    engine = FakeEngine()
    engine.start_gate = asyncio.Event()
    states: list[str] = []
    lifecycle = PrimaryModelLifecycle(
        engine, lazy_load=True, on_state_change=states.append
    )
    load = asyncio.create_task(lifecycle.ensure_loaded())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="transition is in progress"):
        lifecycle.detach()

    engine.start_gate.set()
    await load
    lifecycle.detach()
    lifecycle._set_state("standby")
    assert states == ["loading", "ready"]


@pytest.mark.asyncio
async def test_request_owner_closes_post_load_pre_admission_race():
    now = [10.0]
    engine = FakeEngine()
    lifecycle = PrimaryModelLifecycle(
        engine, lazy_load=True, idle_unload_seconds=1, clock=lambda: now[0]
    )

    lifecycle.acquire_request()
    await lifecycle.ensure_loaded()
    now[0] += 10
    assert await lifecycle.evict_if_idle() is False

    lifecycle.release_request()
    now[0] += 0.5
    assert await lifecycle.evict_if_idle() is False
    now[0] += 0.6
    assert await lifecycle.evict_if_idle() is True


@pytest.mark.asyncio
async def test_request_arriving_during_pause_aborts_idle_unload():
    now = [10.0]
    engine = FakeEngine(loaded=True)
    engine.pause_gate = asyncio.Event()
    lifecycle = PrimaryModelLifecycle(
        engine, idle_unload_seconds=1, clock=lambda: now[0]
    )
    now[0] += 2

    eviction = asyncio.create_task(lifecycle.evict_if_idle())
    await asyncio.sleep(0)
    assert lifecycle.snapshot()["state"] == "unloading"

    lifecycle.acquire_request()
    ready = asyncio.create_task(lifecycle.ensure_loaded())
    await asyncio.sleep(0)
    assert ready.done() is False

    engine.pause_gate.set()
    assert await eviction is False
    await ready
    assert engine.stop_calls == 0
    assert engine.paused is False
    assert lifecycle.snapshot()["state"] == "ready"
    lifecycle.release_request()


@pytest.mark.asyncio
async def test_shutdown_drains_cancellation_isolated_load():
    engine = FakeEngine()
    engine.start_gate = asyncio.Event()
    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)
    waiter = asyncio.create_task(lifecycle.ensure_loaded())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    shutdown = asyncio.create_task(lifecycle.shutdown())
    await asyncio.sleep(0)
    assert shutdown.done() is False
    engine.start_gate.set()
    await shutdown
    assert engine._loaded is True
    with pytest.raises(RuntimeError, match="closed"):
        await lifecycle.ensure_loaded()


@pytest.mark.asyncio
async def test_shutdown_load_drain_propagates_external_cancellation():
    engine = FakeEngine()
    engine.start_gate = asyncio.Event()
    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)
    waiter = asyncio.create_task(lifecycle.ensure_loaded())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    shutdown = asyncio.create_task(lifecycle.shutdown())
    await asyncio.sleep(0)
    shutdown.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    assert lifecycle._load_task is not None
    assert lifecycle._load_task.done() is False

    engine.start_gate.set()
    await lifecycle._load_task


@pytest.mark.asyncio
async def test_demand_warmup_runs_off_the_event_loop(monkeypatch):
    from vllm_mlx import server

    engine = FakeEngine(loaded=True)
    engine.generate_warmup = Mock()
    to_thread = AsyncMock()
    monkeypatch.setattr(server.asyncio, "to_thread", to_thread)

    await server._warmup_primary_engine(engine)
    to_thread.assert_awaited_once_with(engine.generate_warmup)


@pytest.mark.asyncio
async def test_active_request_restarts_idle_window():
    now = [10.0]
    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(
        engine, idle_unload_seconds=5, clock=lambda: now[0]
    )
    engine.active_requests = 1
    now[0] += 6

    assert await lifecycle.evict_if_idle() is False
    assert engine.stop_calls == 0

    engine.active_requests = 0
    now[0] += 4
    assert await lifecycle.evict_if_idle() is False
    now[0] += 2
    assert await lifecycle.evict_if_idle() is True


@pytest.mark.asyncio
async def test_route_release_starts_full_idle_window():
    from vllm_mlx.config import reset_config
    from vllm_mlx.service.helpers import _release_admission_unless_committed

    now = [10.0]
    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(
        engine, idle_unload_seconds=5, clock=lambda: now[0]
    )
    cfg = reset_config()
    cfg.primary_model_lifecycle = lifecycle

    now[0] += 20
    lifecycle.acquire_request()
    _release_admission_unless_committed(engine, False)
    assert engine.release_calls == 1
    now[0] += 4
    assert await lifecycle.evict_if_idle() is False
    now[0] += 2
    assert await lifecycle.evict_if_idle() is True
    reset_config()


@pytest.mark.asyncio
async def test_failed_load_is_retryable_and_reports_only_error_type():
    engine = FakeEngine()
    engine.fail_start = True
    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)

    with pytest.raises(RuntimeError, match="load failed"):
        await lifecycle.ensure_loaded()
    assert lifecycle.snapshot()["state"] == "error"
    assert lifecycle.snapshot()["error"] == "RuntimeError"

    engine.fail_start = False
    await lifecycle.ensure_loaded()
    assert engine.start_calls == 2
    assert lifecycle.snapshot()["state"] == "ready"


@pytest.mark.asyncio
async def test_partial_failed_load_is_cleaned_and_retryable():
    engine = FakeEngine()
    engine.fail_start = True
    engine.partial_start_failure = True
    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)

    with pytest.raises(RuntimeError, match="load failed"):
        await lifecycle.ensure_loaded()
    assert engine._loaded is False
    assert engine.stop_calls == 1

    engine.fail_start = False
    await lifecycle.ensure_loaded()
    assert engine.start_calls == 2
    assert lifecycle.snapshot()["state"] == "ready"


@pytest.mark.asyncio
async def test_streaming_release_starts_idle_window_after_final_chunk():
    from vllm_mlx.config import reset_config
    from vllm_mlx.service.helpers import _disconnect_guard

    class ConnectedRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def stream():
        yield "data: done\n\n"

    now = [10.0]
    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(
        engine, idle_unload_seconds=5, clock=lambda: now[0]
    )
    cfg = reset_config()
    cfg.primary_model_lifecycle = lifecycle
    now[0] += 20
    lifecycle.acquire_request()

    chunks = [
        chunk
        async for chunk in _disconnect_guard(
            stream(),
            ConnectedRequest(),
            poll_interval=0.01,
            engine=engine,
            keepalive_seconds=0,
        )
    ]
    assert chunks == ["data: done\n\n"]
    assert engine.release_calls == 1
    now[0] += 4
    assert await lifecycle.evict_if_idle() is False
    now[0] += 2
    assert await lifecycle.evict_if_idle() is True
    reset_config()


@pytest.mark.asyncio
async def test_non_generation_route_decorator_releases_primary_owner():
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.anthropic import _release_primary_request_after_route

    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(engine, idle_unload_seconds=5)
    cfg = reset_config()
    cfg.primary_model_lifecycle = lifecycle

    @_release_primary_request_after_route
    async def route():
        lifecycle.acquire_request()
        assert lifecycle.snapshot()["active_request_owners"] == 1
        return "done"

    assert await route() == "done"
    assert lifecycle.snapshot()["active_request_owners"] == 0
    reset_config()


@pytest.mark.asyncio
async def test_ready_engine_wakes_only_the_configured_primary():
    from vllm_mlx.config import reset_config
    from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from vllm_mlx.service.helpers import get_ready_engine

    primary = FakeEngine()
    secondary = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(primary, lazy_load=True)
    lifecycle.ensure_loaded = AsyncMock(wraps=lifecycle.ensure_loaded)
    registry = ModelRegistry()
    registry.add(
        ModelEntry(primary, "primary", "primary", aliases={"primary-alias"}),
        is_default=True,
    )
    registry.add(ModelEntry(secondary, "secondary", "secondary"))
    cfg = reset_config()
    cfg.engine = primary
    cfg.model_registry = registry
    cfg.primary_model_lifecycle = lifecycle

    assert await get_ready_engine("secondary") is secondary
    lifecycle.ensure_loaded.assert_not_awaited()
    assert await get_ready_engine("primary-alias") is primary
    lifecycle.ensure_loaded.assert_awaited_once()
    reset_config()


def test_residency_state_callback_is_bound_to_engine_identity():
    from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from vllm_mlx.runtime.resident_models import ResidentModelManager

    old_engine = FakeEngine(loaded=True)
    new_engine = FakeEngine(loaded=True)
    registry = ModelRegistry()
    registry.add(ModelEntry(old_engine, "old", "old"), is_default=True)
    manager = ResidentModelManager(registry, AsyncMock())
    old_record = manager.register_primary(registry.get_entry("old"))
    old_record.primary = False
    registry.add(ModelEntry(new_engine, "new", "new"), is_default=True)
    new_record = manager.register_primary(registry.get_entry("new"))

    manager.set_primary_lifecycle_state(old_engine, "standby")
    assert new_record.state == "resident"
    manager.set_primary_lifecycle_state(new_engine, "standby")
    assert new_record.state == "standby"


@pytest.mark.asyncio
async def test_ready_probe_reports_standby_as_available():
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.health import health_ready

    engine = FakeEngine()
    cfg = reset_config()
    cfg.ready = True
    cfg.engine = engine
    cfg.model_name = "configured-model"
    cfg.primary_model_lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)

    result = await health_ready()
    assert result == {
        "ready": True,
        "model": "configured-model",
        "state": "standby",
        "model_loaded": False,
    }
    reset_config()


def test_serve_parser_exposes_primary_standby_options():
    from vllm_mlx.cli import build_parser

    args = build_parser().parse_args(
        [
            "serve",
            "some/model",
            "--lazy-load",
            "--idle-unload-seconds",
            "300",
        ]
    )
    assert args.lazy_load is True
    assert args.idle_unload_seconds == 300
