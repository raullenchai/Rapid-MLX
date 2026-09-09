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
    assert lifecycle.snapshot()["unload_total"] == 1
    assert lifecycle.snapshot()["unload_total_by_reason"] == {"idle": 1}
    assert lifecycle.snapshot()["last_unload_reason"] == "idle"
    assert engine.paused is False

    await lifecycle.ensure_loaded()
    assert engine.start_calls == 1
    assert events == ["save", "release", "restore"]
    assert states == ["unloading", "standby", "loading", "ready"]
    assert lifecycle.snapshot()["load_total"] == 1
    assert lifecycle.snapshot()["load_failures_total"] == 0
    assert lifecycle.snapshot()["last_load_duration_seconds"] == 0.0


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
    assert lifecycle.snapshot()["load_total"] == 1
    assert lifecycle.snapshot()["load_failures_total"] == 1

    engine.fail_start = False
    await lifecycle.ensure_loaded()
    assert engine.start_calls == 2
    assert lifecycle.snapshot()["state"] == "ready"
    assert lifecycle.snapshot()["load_total"] == 2
    assert lifecycle.snapshot()["load_failures_total"] == 1


@pytest.mark.asyncio
async def test_load_duration_includes_post_load_hooks():
    now = [10.0]
    engine = FakeEngine()

    def finish_load() -> None:
        now[0] += 2.5

    lifecycle = PrimaryModelLifecycle(
        engine,
        lazy_load=True,
        on_loaded=finish_load,
        clock=lambda: now[0],
    )

    await lifecycle.ensure_loaded()

    assert lifecycle.snapshot()["last_load_duration_seconds"] == 2.5


@pytest.mark.asyncio
async def test_partial_failed_load_is_cleaned_and_retryable():
    now = [10.0]
    engine = FakeEngine()
    engine.fail_start = True
    engine.partial_start_failure = True

    async def release_allocator_cache():
        now[0] += 2.5

    lifecycle = PrimaryModelLifecycle(
        engine,
        lazy_load=True,
        release_allocator_cache=release_allocator_cache,
        clock=lambda: now[0],
    )

    with pytest.raises(RuntimeError, match="load failed"):
        await lifecycle.ensure_loaded()
    assert engine._loaded is False
    assert engine.stop_calls == 1
    assert lifecycle.snapshot()["last_load_duration_seconds"] == 2.5

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


@pytest.mark.asyncio
async def test_lifecycle_properties_and_request_token_edges():
    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(engine)

    assert lifecycle.enabled is False
    assert lifecycle.last_error is None
    assert lifecycle.model_loaded is True
    lifecycle.release_request()  # no request in this context

    lifecycle.acquire_request()
    lifecycle.acquire_request()  # acquisition is idempotent within one route
    assert lifecycle.snapshot()["active_request_owners"] == 1
    lifecycle.transfer_request_to_stream()
    assert list(lifecycle._request_tokens.values()) == [True]
    lifecycle.release_request()
    assert lifecycle.snapshot()["active_request_owners"] == 0


@pytest.mark.asyncio
async def test_start_shutdown_and_detach_monitor_paths(monkeypatch):
    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(engine, idle_unload_seconds=10)
    await lifecycle.start()
    monitor = lifecycle._monitor_task
    assert monitor is not None
    await lifecycle.start()  # does not install a duplicate monitor
    await lifecycle.shutdown()
    assert monitor.cancelled()

    detached = PrimaryModelLifecycle(engine, idle_unload_seconds=10)
    await detached.start()
    monitor = detached._monitor_task
    detached.detach()
    assert monitor is not None and monitor.cancelled() is False
    await asyncio.sleep(0)
    assert monitor.cancelled()
    await detached.start()  # detached coordinators stay retired
    assert detached._monitor_task is None


@pytest.mark.asyncio
async def test_shutdown_logs_failed_detached_load(caplog):
    engine = FakeEngine()
    engine.start_gate = asyncio.Event()
    engine.fail_start = True
    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)
    load = asyncio.create_task(lifecycle.ensure_loaded())
    await asyncio.sleep(0)
    load.cancel()
    with pytest.raises(asyncio.CancelledError):
        await load

    shutdown = asyncio.create_task(lifecycle.shutdown())
    await asyncio.sleep(0)
    engine.start_gate.set()
    await shutdown
    assert "load failed during shutdown drain" in caplog.text


@pytest.mark.asyncio
async def test_loaded_fast_paths_and_closed_acquisition():
    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(engine)
    lifecycle._last_error = "old"
    await lifecycle.ensure_loaded()
    assert lifecycle.last_error is None

    # Force the first fast-path check to miss, then become ready while waiting
    # for the transition lock so the in-lock fast path owns the decision.
    await lifecycle._transition_lock.acquire()
    lifecycle.state = "standby"
    waiter = asyncio.create_task(lifecycle.ensure_loaded())
    await asyncio.sleep(0)
    lifecycle.state = "ready"
    lifecycle._transition_lock.release()
    await waiter
    assert engine.start_calls == 0

    await lifecycle.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        lifecycle.acquire_request()


@pytest.mark.asyncio
async def test_error_state_loaded_retry_resets_before_loading():
    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(engine)
    lifecycle.state = "error"
    await lifecycle.ensure_loaded()
    assert engine.stop_calls == 1
    assert engine.start_calls == 1


@pytest.mark.asyncio
async def test_missing_or_ineffective_engine_lifecycle_methods():
    class NoStop:
        _loaded = True

    lifecycle = PrimaryModelLifecycle(NoStop())
    with pytest.raises(RuntimeError, match="cannot be reset"):
        await lifecycle._reset_partial_load()

    class IneffectiveStop:
        _loaded = True

        def stop(self):
            return None

    lifecycle = PrimaryModelLifecycle(IneffectiveStop())
    with pytest.raises(RuntimeError, match="did not stop"):
        await lifecycle._reset_partial_load()

    class NoResume:
        _loaded = False

    lifecycle = PrimaryModelLifecycle(NoResume())
    with pytest.raises(RuntimeError, match="resume_generation"):
        await lifecycle._resume_admission()

    class NoStart:
        _loaded = False

    lifecycle = PrimaryModelLifecycle(NoStart(), lazy_load=True)
    with pytest.raises(RuntimeError, match=r"support start\(\)"):
        await lifecycle.ensure_loaded()


@pytest.mark.asyncio
async def test_load_task_ready_shortcut_and_failed_cleanup_logging(caplog):
    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(engine)
    lifecycle.state = "standby"
    await lifecycle._load()
    assert lifecycle.state == "ready"

    engine = FakeEngine()
    engine.fail_start = True
    engine.partial_start_failure = True

    async def broken_stop():
        raise RuntimeError("cleanup failed")

    engine.stop = broken_stop
    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)
    with pytest.raises(RuntimeError, match="load failed"):
        await lifecycle.ensure_loaded()
    assert "Failed to reset partially loaded primary engine" in caplog.text


@pytest.mark.asyncio
async def test_idle_rechecks_state_and_clock_after_lock_wait():
    now = [20.0]
    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(
        engine, idle_unload_seconds=1, clock=lambda: now[0]
    )
    now[0] = 22.0
    await lifecycle._transition_lock.acquire()
    eviction = asyncio.create_task(lifecycle.evict_if_idle())
    await asyncio.sleep(0)
    lifecycle.state = "standby"
    lifecycle._transition_lock.release()
    assert await eviction is False

    lifecycle.state = "ready"
    lifecycle._last_activity = 20.0
    await lifecycle._transition_lock.acquire()
    eviction = asyncio.create_task(lifecycle.evict_if_idle())
    await asyncio.sleep(0)
    lifecycle.touch()
    lifecycle._transition_lock.release()
    assert await eviction is False


@pytest.mark.asyncio
async def test_pause_timeout_recovers_or_reports_resume_failure():
    now = [10.0]
    engine = FakeEngine(loaded=True)
    engine.active_requests = 1

    # Hide the activity from the pre-pause status check so pause_generation
    # remains the final authority and raises its TimeoutError.
    engine.lifecycle_status = lambda: {"active_requests": 0}
    lifecycle = PrimaryModelLifecycle(
        engine, idle_unload_seconds=1, clock=lambda: now[0]
    )
    now[0] = 12.0
    assert await lifecycle.evict_if_idle() is False
    assert lifecycle.state == "ready"

    engine.fail_resume = True
    now[0] = 14.0
    with pytest.raises(RuntimeError, match="resume failed"):
        await lifecycle.evict_if_idle()
    assert lifecycle.state == "error"


@pytest.mark.asyncio
async def test_pause_exception_recovers_or_reports_resume_failure():
    now = [10.0]
    engine = FakeEngine(loaded=True)

    async def broken_pause(*_args, **_kwargs):
        raise RuntimeError("pause failed")

    engine.pause_generation = broken_pause
    lifecycle = PrimaryModelLifecycle(
        engine, idle_unload_seconds=1, clock=lambda: now[0]
    )
    now[0] = 12.0
    with pytest.raises(RuntimeError, match="pause failed"):
        await lifecycle.evict_if_idle()
    assert lifecycle.state == "ready"

    engine.fail_resume = True
    now[0] = 14.0
    with pytest.raises(RuntimeError, match="resume failed"):
        await lifecycle.evict_if_idle()
    assert lifecycle.state == "error"


@pytest.mark.asyncio
async def test_request_during_pause_resume_failure_is_terminal():
    now = [10.0]
    engine = FakeEngine(loaded=True)
    engine.pause_gate = asyncio.Event()
    lifecycle = PrimaryModelLifecycle(
        engine, idle_unload_seconds=1, clock=lambda: now[0]
    )
    now[0] = 12.0
    eviction = asyncio.create_task(lifecycle.evict_if_idle())
    await asyncio.sleep(0)
    lifecycle.acquire_request()
    engine.fail_resume = True
    engine.pause_gate.set()
    with pytest.raises(RuntimeError, match="resume failed"):
        await eviction
    assert lifecycle.state == "error"
    lifecycle.release_request()


@pytest.mark.asyncio
async def test_unload_cancellation_and_stop_contract_failures():
    now = [10.0]
    engine = FakeEngine(loaded=True)

    async def cancelled_save():
        raise asyncio.CancelledError

    lifecycle = PrimaryModelLifecycle(
        engine,
        idle_unload_seconds=1,
        before_unload=cancelled_save,
        clock=lambda: now[0],
    )
    now[0] = 12.0
    with pytest.raises(asyncio.CancelledError):
        await lifecycle.evict_if_idle()
    assert lifecycle.state == "ready"

    class NoStop(FakeEngine):
        stop = None

    no_stop = NoStop(loaded=True)
    lifecycle = PrimaryModelLifecycle(
        no_stop, idle_unload_seconds=1, clock=lambda: now[0]
    )
    lifecycle._last_activity = 10.0
    with pytest.raises(RuntimeError, match=r"support stop\(\)"):
        await lifecycle.evict_if_idle()
    assert lifecycle.state == "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize("remains_loaded", [True, False])
async def test_stop_failure_publishes_truthful_state(remains_loaded):
    now = [10.0]
    engine = FakeEngine(loaded=True)

    async def broken_stop():
        engine._loaded = remains_loaded
        raise RuntimeError("stop failed")

    engine.stop = broken_stop
    lifecycle = PrimaryModelLifecycle(
        engine, idle_unload_seconds=1, clock=lambda: now[0]
    )
    now[0] = 12.0
    with pytest.raises(RuntimeError, match="stop failed"):
        await lifecycle.evict_if_idle()
    assert lifecycle.state == ("ready" if remains_loaded else "error")


@pytest.mark.asyncio
async def test_allocator_release_failure_is_best_effort(caplog):
    now = [10.0]
    engine = FakeEngine(loaded=True)

    async def broken_release():
        raise RuntimeError("allocator failed")

    lifecycle = PrimaryModelLifecycle(
        engine,
        idle_unload_seconds=1,
        release_allocator_cache=broken_release,
        clock=lambda: now[0],
    )
    now[0] = 12.0
    assert await lifecycle.evict_if_idle() is True
    assert lifecycle.state == "standby"
    assert "allocator cache release failed" in caplog.text


@pytest.mark.asyncio
async def test_monitor_logs_failures_and_propagates_cancellation(monkeypatch, caplog):
    from vllm_mlx.runtime import primary_lifecycle as lifecycle_module

    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(engine, idle_unload_seconds=4)
    lifecycle.evict_if_idle = AsyncMock(side_effect=RuntimeError("evict failed"))
    sleeps = 0

    async def fake_sleep(_interval):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(lifecycle_module.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await lifecycle._monitor_idle()
    assert "idle unload failed" in caplog.text

    lifecycle = PrimaryModelLifecycle(engine, idle_unload_seconds=4)
    lifecycle.evict_if_idle = AsyncMock(side_effect=asyncio.CancelledError)

    async def one_sleep(_interval):
        return None

    monkeypatch.setattr(lifecycle_module.asyncio, "sleep", one_sleep)
    with pytest.raises(asyncio.CancelledError):
        await lifecycle._monitor_idle()


@pytest.mark.asyncio
async def test_ensure_engine_ready_releases_on_cancel_and_failure():
    from fastapi import HTTPException

    from vllm_mlx.config import reset_config
    from vllm_mlx.service.helpers import ensure_engine_ready

    engine = FakeEngine()
    cfg = reset_config()
    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)
    cfg.primary_model_lifecycle = lifecycle
    lifecycle.ensure_loaded = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await ensure_engine_ready(engine)
    assert lifecycle.snapshot()["active_request_owners"] == 0

    lifecycle.ensure_loaded = AsyncMock(side_effect=RuntimeError("bad load"))
    with pytest.raises(HTTPException) as raised:
        await ensure_engine_ready(engine)
    assert raised.value.status_code == 503
    assert raised.value.headers == {"Retry-After": "5"}
    assert lifecycle.snapshot()["active_request_owners"] == 0

    other = FakeEngine(loaded=True)
    assert await ensure_engine_ready(other) is other
    reset_config()


@pytest.mark.asyncio
async def test_route_ownership_helpers_cover_all_lease_shapes():
    from vllm_mlx.config import reset_config
    from vllm_mlx.service.helpers import _release_route_ownership

    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(engine, idle_unload_seconds=5)
    cfg = reset_config()
    cfg.primary_model_lifecycle = lifecycle

    lifecycle.acquire_request()
    _release_route_ownership(engine, admission_acquired=True, committed=True)
    assert list(lifecycle._request_tokens.values()) == [True]
    lifecycle.release_request()

    lifecycle.acquire_request()
    _release_route_ownership(engine, admission_acquired=False, committed=True)
    assert list(lifecycle._request_tokens.values()) == [True]
    lifecycle.release_request()

    lifecycle.acquire_request()
    _release_route_ownership(engine, admission_acquired=False, committed=False)
    assert lifecycle.snapshot()["active_request_owners"] == 0

    lifecycle.acquire_request()
    _release_route_ownership(engine, admission_acquired=True, committed=False)
    assert lifecycle.snapshot()["active_request_owners"] == 0
    assert engine.release_calls == 1
    reset_config()


@pytest.mark.asyncio
async def test_stream_cleanup_releases_lifecycle_when_admission_release_raises():
    from vllm_mlx.config import reset_config
    from vllm_mlx.service import helpers

    class ConnectedRequest:
        async def is_disconnected(self) -> bool:
            return False

    class BrokenReleaseEngine(FakeEngine):
        def release_admission_reservation(self) -> None:
            raise RuntimeError("release failed")

    async def stream():
        yield "done"

    engine = BrokenReleaseEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(engine, idle_unload_seconds=5)
    cfg = reset_config()
    cfg.primary_model_lifecycle = lifecycle
    lifecycle.acquire_request()

    assert [
        chunk
        async for chunk in helpers._disconnect_guard(
            stream(),
            ConnectedRequest(),
            poll_interval=0.01,
            engine=engine,
            keepalive_seconds=0,
        )
    ] == ["done"]
    assert lifecycle.snapshot()["active_request_owners"] == 0

    aborted = Mock()
    original_abort = helpers._force_abort_request
    helpers._force_abort_request = aborted

    async def failed_stream():
        raise RuntimeError("stream failed")
        yield  # pragma: no cover - makes this an async generator

    try:
        chunks = [
            chunk
            async for chunk in helpers._disconnect_guard(
                failed_stream(),
                ConnectedRequest(),
                poll_interval=0.01,
                engine=None,
                keepalive_seconds=0,
            )
        ]
        assert chunks
        aborted.assert_called_once()
    finally:
        helpers._force_abort_request = original_abort

    cfg.engine = engine
    cfg.model_registry = None
    assert helpers.get_engine() is engine
    reset_config()


@pytest.mark.asyncio
async def test_health_probe_rejects_lifecycle_error():
    from fastapi import HTTPException

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.health import health_ready

    engine = FakeEngine()
    lifecycle = PrimaryModelLifecycle(engine, lazy_load=True)
    lifecycle.state = "error"
    cfg = reset_config()
    cfg.ready = True
    cfg.engine = engine
    cfg.primary_model_lifecycle = lifecycle
    with pytest.raises(HTTPException) as raised:
        await health_ready()
    assert raised.value.status_code == 503
    reset_config()


@pytest.mark.asyncio
async def test_primary_server_lifecycle_helpers(monkeypatch):
    from types import SimpleNamespace

    from vllm_mlx import server

    class HybridEngine(FakeEngine):
        async def stream_chat(self, **_kwargs):
            yield "token"

    hybrid = HybridEngine(loaded=True)
    monkeypatch.setattr(server, "_detect_hybrid_for_warmup", lambda _engine: True)
    await server._warmup_primary_engine(hybrid)

    async def broken_stream(**_kwargs):
        raise RuntimeError("warmup failed")
        yield  # pragma: no cover - makes this an async generator

    hybrid.stream_chat = broken_stream
    await server._warmup_primary_engine(hybrid)

    monkeypatch.setattr(
        server, "_detect_hybrid_for_warmup", Mock(side_effect=RuntimeError("detect"))
    )
    await server._warmup_primary_engine(hybrid)

    monkeypatch.setattr(server, "_engine", None)
    await server._finish_primary_demand_load()

    engine = FakeEngine(loaded=True)
    engine.load_cache_from_disk = Mock()
    monkeypatch.setattr(server, "_engine", engine)
    warmup = AsyncMock()
    grammar = AsyncMock(side_effect=RuntimeError("grammar"))
    cache_load = AsyncMock()
    monkeypatch.setattr(server, "_warmup_primary_engine", warmup)
    monkeypatch.setattr(server, "_warmup_tool_grammar", grammar)
    monkeypatch.setattr(server, "_deferred_load_prefix_cache", cache_load)
    await server._finish_primary_demand_load()
    assert server._prefix_cache_load_task is not None
    await server._prefix_cache_load_task

    drain = AsyncMock()
    save = AsyncMock()
    monkeypatch.setattr(server, "_drain_deferred_prefix_cache_load", drain)
    monkeypatch.setattr(server, "_shutdown_save_prefix_cache", save)
    await server._prepare_primary_idle_unload()
    drain.assert_awaited_once()
    save.assert_awaited_once()

    manager = SimpleNamespace(set_primary_lifecycle_state=Mock())
    monkeypatch.setattr(server, "_residency_manager", None)
    server._mirror_primary_lifecycle_state(engine, "standby")
    monkeypatch.setattr(server, "_residency_manager", manager)
    server._mirror_primary_lifecycle_state(engine, "ready")
    manager.set_primary_lifecycle_state.assert_called_once_with(engine, "ready")

    monkeypatch.setattr(server, "_primary_lazy_load", True)
    monkeypatch.setattr(server, "_primary_idle_unload_seconds", 9.0)
    lifecycle = server._build_primary_model_lifecycle(engine)
    assert lifecycle.lazy_load is True
    assert lifecycle.idle_unload_seconds == 9.0


def test_configure_primary_lifecycle_resets_existing_state(monkeypatch):
    from vllm_mlx import server
    from vllm_mlx.config import reset_config

    cfg = reset_config()
    cfg.primary_model_lifecycle = object()
    monkeypatch.setattr(server, "_primary_model_lifecycle", object())
    monkeypatch.setattr(server, "_primary_lazy_load", False)
    monkeypatch.setattr(server, "_primary_idle_unload_seconds", 0.0)
    server.configure_primary_model_lifecycle(lazy_load=True, idle_unload_seconds=12)
    assert server._primary_lazy_load is True
    assert server._primary_idle_unload_seconds == 12
    assert server._primary_model_lifecycle is None
    assert cfg.primary_model_lifecycle is None


@pytest.mark.asyncio
async def test_resident_primary_handoff_detaches_or_rejects(monkeypatch):
    from types import SimpleNamespace

    from vllm_mlx import server
    from vllm_mlx.config import reset_config
    from vllm_mlx.runtime.model_registry import ModelEntry
    from vllm_mlx.runtime.resident_models import ResidentModelBusyError

    cfg = reset_config()
    old = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(old, idle_unload_seconds=3)
    monkeypatch.setattr(server, "_primary_model_lifecycle", lifecycle)
    monkeypatch.setattr(server, "_primary_lazy_load", True)
    monkeypatch.setattr(server, "_primary_idle_unload_seconds", 3.0)
    replacement = ModelEntry(FakeEngine(loaded=True), "new", "new")
    server._set_resident_primary(replacement)
    assert lifecycle._detached is True
    assert server._primary_model_lifecycle is not None
    assert server._primary_model_lifecycle.engine is replacement.engine
    assert cfg.primary_model_lifecycle is server._primary_model_lifecycle
    await server._primary_model_lifecycle.shutdown()

    busy = SimpleNamespace(engine=old, detach=Mock(side_effect=RuntimeError("busy")))
    monkeypatch.setattr(server, "_primary_model_lifecycle", busy)
    with pytest.raises(ResidentModelBusyError, match="transition"):
        server._set_resident_primary(None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lazy_load", "idle_seconds", "expected_start_calls"),
    [(True, 0.0, 0), (False, 10.0, 1), (False, 0.0, 1)],
)
async def test_lifespan_primary_lifecycle_modes(
    monkeypatch, lazy_load, idle_seconds, expected_start_calls
):
    from types import SimpleNamespace

    from vllm_mlx import server
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes import audio, video
    from vllm_mlx.runtime import audio_worker

    engine = FakeEngine()
    manager = SimpleNamespace(
        start=AsyncMock(),
        shutdown=AsyncMock(),
        contains=Mock(return_value=False),
        register_primary=Mock(),
        set_primary_lifecycle_state=Mock(),
    )
    registry = SimpleNamespace(list_entries=Mock(return_value=[]))
    cfg = reset_config()
    cfg.bind_host = None
    cfg.bind_port = None
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_model_registry", registry)
    monkeypatch.setattr(server, "_residency_manager", manager)
    monkeypatch.setattr(server, "_primary_model_lifecycle", None)
    monkeypatch.setattr(server, "_primary_lazy_load", lazy_load)
    monkeypatch.setattr(server, "_primary_idle_unload_seconds", idle_seconds)
    monkeypatch.setattr(server, "_warmup_primary_engine", AsyncMock())
    monkeypatch.setattr(server, "_warmup_tool_grammar", AsyncMock())
    monkeypatch.setattr(server, "_drain_deferred_prefix_cache_load", AsyncMock())
    monkeypatch.setattr(server, "_shutdown_save_prefix_cache", AsyncMock())
    monkeypatch.setattr(audio, "audio_routes_should_register", Mock(return_value=False))
    monkeypatch.setattr(audio, "shutdown_audio_lanes", AsyncMock())
    monkeypatch.setattr(video, "start_video_jobs", Mock())
    monkeypatch.setattr(video, "shutdown_video_jobs", AsyncMock())
    monkeypatch.setattr(audio_worker, "bind_audio_worker", Mock())

    lifespan = server.lifespan(server.app)
    await lifespan.__anext__()
    assert engine.start_calls == expected_start_calls
    assert (server._primary_model_lifecycle is not None) is (
        lazy_load or idle_seconds > 0
    )
    with pytest.raises(StopAsyncIteration):
        await lifespan.__anext__()
    assert cfg.ready is False


@pytest.mark.asyncio
async def test_lifespan_rejects_unsupported_lazy_engine(monkeypatch):
    from vllm_mlx import server
    from vllm_mlx.config import reset_config

    reset_config()
    monkeypatch.setattr(server, "_engine", object())
    monkeypatch.setattr(server, "_primary_lazy_load", True)
    monkeypatch.setattr(server, "_primary_idle_unload_seconds", 0.0)
    monkeypatch.setattr(server, "_primary_model_lifecycle", None)
    lifespan = server.lifespan(server.app)
    with pytest.raises(RuntimeError, match="text and vision-language"):
        await lifespan.__anext__()


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
