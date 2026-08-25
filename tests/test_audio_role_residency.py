"""Auxiliary audio-role admission and lifecycle (#2305).

Covers the contract issue #2305 asks for: speech engines are budgeted against
the same ceiling as the conversation model, unsafe combinations are rejected
BEFORE any weights load, rejections name the conflicting roles, and transient
speech engines are released on an idle TTL without disturbing the conversation
model.
"""

from __future__ import annotations

import asyncio

import pytest

from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
from vllm_mlx.runtime.resident_models import (
    ResidentModelBusyError,
    ResidentModelManager,
    ResidentRoleConflictError,
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


def manager_fixture(*, limit_gib=10, primary_gib=4, audio_ttl=0.0):
    registry = ModelRegistry()
    primary = entry("chat")
    registry.add(primary, is_default=True)

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    clock = Clock()
    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=limit_gib * GIB,
        audio_role_idle_ttl_seconds=audio_ttl,
        clock=clock,
        memory_reader=lambda: 0,
    )
    manager.register_primary(primary, estimated_bytes=primary_gib * GIB)
    return manager, clock


async def admit(manager, *, role, lane, model, gib, source="manifest", loads=None):
    """Admit one role and record the load, mirroring the audio routes' shape."""

    async with manager.admitting_role(
        role=role,
        lane=lane,
        model_id=model,
        reserved_bytes=int(gib * GIB),
        capacity_source=source,
        weight_bytes=int(gib * GIB),
    ) as record:
        if loads is not None:
            loads.append(model)
        record.unload = lambda: None
    return record


@pytest.mark.asyncio
async def test_safe_combination_admits_speech_alongside_the_conversation_model():
    manager, _clock = manager_fixture(limit_gib=10, primary_gib=4)

    await admit(manager, role="speech-input", lane="stt", model="whisper", gib=3)
    await admit(manager, role="speech-output", lane="tts", model="kokoro", gib=1)

    snapshot = manager.snapshot()
    assert snapshot["memory_used_bytes"] == 8 * GIB
    assert [role["role"] for role in snapshot["roles"]] == [
        "speech-input",
        "speech-output",
    ]
    assert all(role["state"] == "resident" for role in snapshot["roles"])


@pytest.mark.asyncio
async def test_unsafe_combination_is_rejected_before_the_loader_runs():
    manager, _clock = manager_fixture(limit_gib=10, primary_gib=4)
    loads: list[str] = []

    await admit(
        manager, role="speech-output", lane="tts", model="kokoro", gib=1, loads=loads
    )

    with pytest.raises(ResidentRoleConflictError) as excinfo:
        await admit(
            manager,
            role="speech-input",
            lane="stt",
            model="whisper-large",
            gib=8,
            loads=loads,
        )

    # The whole point: the rejected role never reached its weight loader.
    assert loads == ["kokoro"]
    # ...and the speech-output role it could not displace is still resident,
    # because a busy-free idle role IS reclaimable but 1 GiB would not have
    # been enough to fit 8 GiB anyway.
    assert excinfo.value.requested.model == "whisper-large"
    assert excinfo.value.requested.bytes == 8 * GIB
    assert excinfo.value.limit_bytes == 10 * GIB


@pytest.mark.asyncio
async def test_conflict_names_the_conversation_model_and_marks_it_unevictable():
    manager, _clock = manager_fixture(limit_gib=6, primary_gib=4)

    with pytest.raises(ResidentRoleConflictError) as excinfo:
        await admit(manager, role="speech-input", lane="stt", model="whisper", gib=4)

    conflicts = {conflict.role: conflict for conflict in excinfo.value.conflicts}
    assert conflicts["conversation"].model == "chat"
    assert conflicts["conversation"].bytes == 4 * GIB
    # Pressing the microphone button is never consent to unload the model
    # answering the conversation (#2300 non-goal, restated by #2305).
    assert conflicts["conversation"].evictable is False
    assert conflicts["conversation"].reason == "active_conversation_model"


@pytest.mark.asyncio
async def test_conflict_envelope_carries_the_structured_detail_the_desktop_reads():
    manager, _clock = manager_fixture(limit_gib=6, primary_gib=4)

    with pytest.raises(ResidentRoleConflictError) as excinfo:
        await admit(manager, role="speech-input", lane="stt", model="whisper", gib=4)

    error = excinfo.value.envelope()["error"]
    assert error["type"] == "insufficient_capacity_error"
    assert error["code"] == "role_capacity_conflict"
    assert error["requested"] == {
        "role": "speech-input",
        "model": "whisper",
        "bytes": 4 * GIB,
    }
    assert error["limit_bytes"] == 6 * GIB
    assert error["conflicts"][0]["role"] == "conversation"
    assert isinstance(error["message"], str) and error["message"]


@pytest.mark.asyncio
async def test_idle_speech_role_is_reclaimed_to_admit_another_speech_role():
    manager, _clock = manager_fixture(limit_gib=10, primary_gib=4)

    await admit(manager, role="speech-output", lane="tts", model="kokoro", gib=3)
    # 4 (chat) + 3 (tts) + 4 (stt) > 10, but the idle TTS role can go.
    await admit(manager, role="speech-input", lane="stt", model="whisper", gib=4)

    roles = {role["role"]: role for role in manager.snapshot()["roles"]}
    assert set(roles) == {"speech-input"}
    assert manager.snapshot()["memory_used_bytes"] == 8 * GIB


@pytest.mark.asyncio
async def test_busy_speech_role_is_not_reclaimed_and_is_reported_as_such():
    manager, _clock = manager_fixture(limit_gib=10, primary_gib=4)
    await admit(manager, role="speech-output", lane="tts", model="kokoro", gib=3)

    async with manager.lease_role("speech-output"):
        with pytest.raises(ResidentRoleConflictError) as excinfo:
            await admit(
                manager, role="speech-input", lane="stt", model="whisper", gib=4
            )

    conflicts = {conflict.role: conflict for conflict in excinfo.value.conflicts}
    assert conflicts["speech-output"].evictable is False
    assert conflicts["speech-output"].reason == "serving_active_request"
    # The busy role kept its weights.
    assert [role["role"] for role in manager.snapshot()["roles"]] == ["speech-output"]


@pytest.mark.asyncio
async def test_failed_load_rolls_the_ledger_back_and_unloads():
    manager, _clock = manager_fixture(limit_gib=10, primary_gib=4)
    unloaded: list[str] = []

    with pytest.raises(RuntimeError, match="weights are corrupt"):
        async with manager.admitting_role(
            role="speech-input",
            lane="stt",
            model_id="whisper",
            reserved_bytes=3 * GIB,
            capacity_source="manifest",
        ) as record:
            record.unload = lambda: unloaded.append("whisper")
            raise RuntimeError("weights are corrupt")

    assert unloaded == ["whisper"]
    assert manager.snapshot()["roles"] == []
    assert manager.snapshot()["memory_used_bytes"] == 4 * GIB


@pytest.mark.asyncio
async def test_cancelled_load_rolls_the_ledger_back_and_unloads():
    manager, _clock = manager_fixture(limit_gib=10, primary_gib=4)
    unloaded: list[str] = []

    with pytest.raises(asyncio.CancelledError):
        async with manager.admitting_role(
            role="speech-input",
            lane="stt",
            model_id="whisper",
            reserved_bytes=3 * GIB,
            capacity_source="manifest",
        ) as record:
            record.unload = lambda: unloaded.append("whisper")
            raise asyncio.CancelledError

    assert unloaded == ["whisper"]
    assert manager.snapshot()["roles"] == []


@pytest.mark.asyncio
async def test_idle_ttl_releases_speech_roles_without_touching_the_conversation():
    manager, clock = manager_fixture(limit_gib=20, primary_gib=4, audio_ttl=300.0)
    unloaded: list[str] = []

    async with manager.admitting_role(
        role="speech-input",
        lane="stt",
        model_id="whisper",
        reserved_bytes=3 * GIB,
        capacity_source="manifest",
    ) as record:
        record.unload = lambda: unloaded.append("whisper")

    clock.now += 299.0
    assert await manager.evict_expired() == []

    clock.now += 2.0
    assert await manager.evict_expired() == ["speech-input"]
    assert unloaded == ["whisper"]
    # The conversation model is protected and unaffected.
    assert [model["id"] for model in manager.snapshot()["models"]] == ["chat"]


@pytest.mark.asyncio
async def test_idle_ttl_does_not_release_a_role_serving_a_request():
    manager, clock = manager_fixture(limit_gib=20, primary_gib=4, audio_ttl=300.0)
    await admit(manager, role="speech-input", lane="stt", model="whisper", gib=3)

    async with manager.lease_role("speech-input"):
        clock.now += 600.0
        assert await manager.evict_expired() == []

    # Releasing the lease marks the role as just-used, so it has to go idle
    # again from scratch rather than being swept the instant the request ends.
    assert await manager.evict_expired() == []
    clock.now += 301.0
    assert await manager.evict_expired() == ["speech-input"]


@pytest.mark.asyncio
async def test_ttl_sweeper_starts_for_an_audio_only_ttl():
    # A desktop server passes no model idle TTL in some configurations; the
    # audio TTL must still be swept.
    manager, _clock = manager_fixture(limit_gib=20, audio_ttl=300.0)
    assert manager.idle_ttl_seconds == 0
    await manager.start()
    try:
        assert manager._ttl_task is not None
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_replacing_a_role_releases_the_incumbent_before_charging_the_new_model():
    manager, _clock = manager_fixture(limit_gib=10, primary_gib=4)
    unloaded: list[str] = []

    async with manager.admitting_role(
        role="speech-input",
        lane="stt",
        model_id="whisper-small",
        reserved_bytes=3 * GIB,
        capacity_source="manifest",
    ) as record:
        record.unload = lambda: unloaded.append("whisper-small")

    # 3 + 5 would exceed the 6 GiB left, but this is a swap, not a co-load.
    await admit(manager, role="speech-input", lane="stt", model="whisper-large", gib=5)

    assert unloaded == ["whisper-small"]
    roles = manager.snapshot()["roles"]
    assert [role["model"] for role in roles] == ["whisper-large"]
    assert manager.snapshot()["memory_used_bytes"] == 9 * GIB


@pytest.mark.asyncio
async def test_concurrent_admission_of_one_role_is_rejected():
    manager, _clock = manager_fixture(limit_gib=20, primary_gib=4)

    async with manager.admitting_role(
        role="speech-input",
        lane="stt",
        model_id="whisper",
        reserved_bytes=3 * GIB,
        capacity_source="manifest",
    ) as record:
        record.unload = lambda: None
        with pytest.raises(ResidentModelBusyError):
            await admit(
                manager, role="speech-input", lane="stt", model="parakeet", gib=3
            )


@pytest.mark.asyncio
async def test_unknown_capacity_is_charged_by_measurement_after_the_load():
    # Tier 3 of the capacity resolver: nothing to reserve up front, but the
    # process-footprint delta must still be charged so the NEXT admission
    # sees the memory this role really took.
    registry = ModelRegistry()
    primary = entry("chat")
    registry.add(primary, is_default=True)
    # A live process always reports a non-zero footprint; zero means the
    # reader is unavailable, in which case there is nothing to measure.
    process_usage = [1 * GIB]

    async def loader(name: str, path: str | None, performance=None):
        return entry(name)

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=10 * GIB,
        clock=Clock(),
        memory_reader=lambda: process_usage[0],
    )
    manager.register_primary(primary, estimated_bytes=4 * GIB)

    async with manager.admitting_role(
        role="speech-input",
        lane="stt",
        model_id="someone/unlisted-asr",
        reserved_bytes=0,
        capacity_source="unknown",
    ) as record:
        process_usage[0] = 4 * GIB
        record.unload = lambda: None

    role = manager.snapshot()["roles"][0]
    assert role["capacity_source"] == "unknown"
    assert role["reserved_bytes"] == 0
    assert role["measured_bytes"] == 3 * GIB
    # 4 GiB primary + 3 GiB measured audio role, even once the live footprint
    # reading drops below the reservation total.
    process_usage[0] = 0
    assert manager.snapshot()["memory_used_bytes"] == 7 * GIB


@pytest.mark.asyncio
async def test_shutdown_releases_every_audio_role():
    manager, _clock = manager_fixture(limit_gib=20, primary_gib=4)
    unloaded: list[str] = []

    for role_name, lane, model in (
        ("speech-input", "stt", "whisper"),
        ("speech-output", "tts", "kokoro"),
    ):
        async with manager.admitting_role(
            role=role_name,
            lane=lane,
            model_id=model,
            reserved_bytes=1 * GIB,
            capacity_source="manifest",
        ) as record:
            record.unload = lambda name=model: unloaded.append(name)

    await manager.shutdown()

    assert sorted(unloaded) == ["kokoro", "whisper"]
    assert manager.snapshot()["roles"] == []


@pytest.mark.asyncio
async def test_a_model_load_reclaims_idle_speech_before_failing_admission():
    # The reverse direction of the budget: an explicit user model choice
    # outranks a transient speech engine.
    manager, _clock = manager_fixture(limit_gib=10, primary_gib=4)
    unloaded: list[str] = []

    async with manager.admitting_role(
        role="speech-output",
        lane="tts",
        model_id="kokoro",
        reserved_bytes=4 * GIB,
        capacity_source="manifest",
    ) as record:
        record.unload = lambda: unloaded.append("kokoro")

    await manager.load("vision", estimated_bytes=5 * GIB)

    assert unloaded == ["kokoro"]
    assert manager.snapshot()["roles"] == []
    assert {model["id"] for model in manager.snapshot()["models"]} == {
        "chat",
        "vision",
    }


@pytest.mark.asyncio
async def test_role_snapshot_reports_provenance_for_the_control_plane():
    manager, _clock = manager_fixture(limit_gib=20, primary_gib=4)
    await admit(
        manager,
        role="speech-input",
        lane="stt",
        model="whisper",
        gib=3,
        source="local_cache",
    )

    role = manager.snapshot()["roles"][0]
    assert role["lane"] == "stt"
    assert role["model"] == "whisper"
    assert role["capacity_source"] == "local_cache"
    assert role["weight_bytes"] == 3 * GIB
    assert role["active_requests"] == 0


def test_model_rows_report_their_lifecycle_role():
    manager, _clock = manager_fixture()
    assert manager.snapshot()["models"][0]["role"] == "conversation"


# ---------------------------------------------------------------------------
# Route wiring: the audio lanes must actually go through the ledger above.
# ---------------------------------------------------------------------------


@pytest.fixture
def route_manager(monkeypatch):
    """Install a real manager as the process residency manager for the routes."""

    from vllm_mlx.config import get_config

    manager, clock = manager_fixture(limit_gib=10, primary_gib=4, audio_ttl=300.0)
    config = get_config()
    previous = config.residency_manager
    config.residency_manager = manager
    try:
        yield manager, clock
    finally:
        config.residency_manager = previous


@pytest.mark.asyncio
async def test_tts_route_rejects_an_unsafe_model_before_loading_weights(
    monkeypatch, route_manager
):
    from fastapi import HTTPException

    from vllm_mlx.audio import tts as tts_module
    from vllm_mlx.routes import audio as audio_route
    from vllm_mlx.runtime import audio_capacity

    manager, _clock = route_manager
    loads: list[str] = []

    class _TTS:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            loads.append(self.model_name)

    monkeypatch.setattr(tts_module, "TTSEngine", _TTS)
    monkeypatch.setattr(audio_route, "_tts_engine", None)
    # 4 GiB primary + a 9 GiB speech-output model does not fit under 10 GiB.
    monkeypatch.setattr(
        audio_capacity,
        "resolve_audio_role_capacity",
        lambda model: audio_capacity.AudioRoleCapacity(
            reserved_bytes=9 * GIB,
            weight_bytes=9 * GIB,
            capacity_source="manifest",
            hf_id=model,
        ),
    )

    with pytest.raises(HTTPException) as excinfo:
        await audio_route._ensure_tts_engine("huge-tts")

    assert excinfo.value.status_code == 507
    error = excinfo.value.detail["error"]
    assert error["code"] == "role_capacity_conflict"
    assert error["requested"]["role"] == "speech-output"
    assert any(
        conflict["role"] == "conversation" and conflict["evictable"] is False
        for conflict in error["conflicts"]
    )
    # The rejection happened before TTSEngine.load ran.
    assert loads == []
    assert audio_route._tts_engine is None
    assert manager.snapshot()["roles"] == []


@pytest.mark.asyncio
async def test_tts_route_registers_its_role_in_the_shared_budget(
    monkeypatch, route_manager
):
    from vllm_mlx.audio import tts as tts_module
    from vllm_mlx.routes import audio as audio_route

    manager, clock = route_manager
    unloaded: list[str] = []

    class _TTS:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            pass

        def unload(self) -> None:
            unloaded.append(self.model_name)

    monkeypatch.setattr(tts_module, "TTSEngine", _TTS)
    monkeypatch.setattr(audio_route, "_tts_engine", None)

    await audio_route._ensure_tts_engine("kokoro")

    roles = manager.snapshot()["roles"]
    assert [role["role"] for role in roles] == ["speech-output"]
    assert roles[0]["capacity_source"] == "manifest"

    # The TTL sweeper must be able to unload through the registered callback
    # AND clear this module's cache, or the next request hands out an engine
    # whose weights are gone.
    clock.now += 301.0
    assert await manager.evict_expired() == ["speech-output"]
    assert unloaded == ["kokoro"]
    assert audio_route._tts_engine is None


@pytest.mark.asyncio
async def test_idle_sweep_cannot_unload_a_lane_a_request_already_owns(
    monkeypatch, route_manager
):
    """Regression: cache-hit requests hold the lane lock but no lease.

    A second request that finds the engine already cached never re-enters
    ``admitting_role``, so ``active_requests`` is still zero when it takes the
    lane lock. Before the ``can_release`` veto, a TTL sweep firing in that
    window unloaded the weights the request was about to use.
    """

    from vllm_mlx.audio import tts as tts_module
    from vllm_mlx.routes import audio as audio_route

    manager, clock = route_manager
    unloaded: list[str] = []

    class _TTS:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            pass

        def unload(self) -> None:
            unloaded.append(self.model_name)

    monkeypatch.setattr(tts_module, "TTSEngine", _TTS)
    monkeypatch.setattr(audio_route, "_tts_engine", None)

    await audio_route._ensure_tts_engine("kokoro")
    clock.now += 301.0

    async with audio_route._get_tts_lane_lock():
        # The request owns the lane; it has not reached its lease yet.
        assert await manager.evict_expired() == []
        assert unloaded == []
        assert audio_route._tts_engine is not None

    # Once the lane is free again the sweep proceeds normally.
    assert await manager.evict_expired() == ["speech-output"]
    assert unloaded == ["kokoro"]


@pytest.mark.asyncio
async def test_a_lane_owned_by_a_request_is_reported_as_a_busy_conflict(
    monkeypatch, route_manager
):
    from vllm_mlx.audio import tts as tts_module
    from vllm_mlx.routes import audio as audio_route

    manager, _clock = route_manager

    class _TTS:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            pass

        def unload(self) -> None:
            pass

    monkeypatch.setattr(tts_module, "TTSEngine", _TTS)
    monkeypatch.setattr(audio_route, "_tts_engine", None)
    await audio_route._ensure_tts_engine("kokoro")

    async with audio_route._get_tts_lane_lock():
        with pytest.raises(ResidentRoleConflictError) as excinfo:
            await admit(
                manager, role="speech-input", lane="stt", model="whisper", gib=6
            )

    conflicts = {conflict.role: conflict for conflict in excinfo.value.conflicts}
    assert conflicts["speech-output"].evictable is False
    assert conflicts["speech-output"].reason == "serving_active_request"
