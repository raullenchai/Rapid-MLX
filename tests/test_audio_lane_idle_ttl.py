"""Idle audio release uses the request barrier and preserves lazy loading."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rapid_mlx.runtime.audio_worker import AudioWorkerDispatcher


class Worker:
    async def execute_on_model_worker(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def execute_on_model_worker_sync(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class Cached:
    model_name = "whisper-small"

    def __init__(self, model_name="whisper-small", fail=False):
        self.model_name = model_name
        self.unloaded = False
        self.fail = fail

    def unload(self):
        if self.fail:
            raise RuntimeError("unload failed")
        self.unloaded = True


@pytest.fixture
def lanes(monkeypatch):
    from rapid_mlx.routes import audio
    from rapid_mlx.runtime import audio_worker as worker_module
    from rapid_mlx.runtime import resident_models

    dispatcher = AudioWorkerDispatcher()
    dispatcher.bind(Worker())
    monkeypatch.setattr(worker_module, "audio_worker", dispatcher)
    monkeypatch.setattr(audio, "_stt_engine", None)
    monkeypatch.setattr(audio, "_aligner_engine", None)
    monkeypatch.setattr(audio, "_tts_engine", None)
    monkeypatch.setattr(audio, "_release_alignment_role", AsyncMock())
    monkeypatch.setattr(resident_models, "_release_allocator_cache", lambda: None)
    return audio, dispatcher


def age(dispatcher, lane, seconds=100):
    # Only test clock state changes; all observable transitions use dispatch.
    dispatcher._lanes[lane].last_used_at -= seconds


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lane,attribute",
    [("stt", "_stt_engine"), ("alignment", "_aligner_engine"), ("tts", "_tts_engine")],
)
async def test_idle_release_reports_registered_and_clears_cache(lanes, lane, attribute):
    audio, dispatcher = lanes
    cached = Cached()
    setattr(audio, attribute, cached)
    await dispatcher.execute(lane, cached.model_name, "load", lambda: None)
    await audio.release_idle_audio_lanes(0)
    assert not cached.unloaded
    await audio.release_idle_audio_lanes(30)
    assert not cached.unloaded
    age(dispatcher, lane)
    await audio.release_idle_audio_lanes(30)
    assert cached.unloaded
    assert getattr(audio, attribute) is None
    assert dispatcher.snapshot()[0]["state"] == "registered"
    assert dispatcher.snapshot()[0]["loaded_at"] is None
    assert dispatcher.snapshot()[0]["model"] is None
    if lane == "alignment":
        audio._release_alignment_role.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lane,attribute,lock_name",
    [
        ("stt", "_stt_engine", "_get_stt_lane_lock"),
        ("tts", "_tts_engine", "_get_tts_lane_lock"),
    ],
)
async def test_queued_timer_rechecks_idle_age_after_request(
    lanes, lane, attribute, lock_name
):
    audio, dispatcher = lanes
    cached = Cached()
    setattr(audio, attribute, cached)
    await dispatcher.execute(lane, cached.model_name, "load", lambda: None)
    age(dispatcher, lane)
    async with getattr(audio, lock_name)():
        timer = asyncio.create_task(audio.release_idle_audio_lanes(30))
        await asyncio.sleep(0)
        assert not timer.done()
        assert not cached.unloaded
        # A request finishes while owning the same lock, refreshing activity.
        await dispatcher.execute(lane, cached.model_name, "infer", lambda: None)
    await timer
    assert not cached.unloaded
    assert getattr(audio, attribute) is cached


@pytest.mark.asyncio
async def test_worker_handoff_and_unload_failure_preserve_retry(lanes):
    audio, dispatcher = lanes
    cached = Cached(fail=True)
    audio._stt_engine = cached
    await dispatcher.execute("stt", cached.model_name, "load", lambda: None)
    age(dispatcher, "stt")
    handoff = dispatcher.begin_handoff()
    await audio.release_idle_audio_lanes(30)
    assert audio._stt_engine is cached
    assert dispatcher.snapshot()[0]["state"] == "resident"
    handoff.rollback()
    await audio.release_idle_audio_lanes(30)
    assert audio._stt_engine is cached
    assert dispatcher.snapshot()[0]["state"] == "failed"
    age(dispatcher, "stt")
    cached.fail = False
    await audio.release_idle_audio_lanes(30)
    assert audio._stt_engine is None
    assert dispatcher.snapshot()[0]["state"] == "registered"


@pytest.mark.asyncio
async def test_stt_request_loads_again_after_idle_release(lanes, monkeypatch):
    audio, dispatcher = lanes
    from rapid_mlx.audio import stt

    calls = []

    class STT(Cached):
        def load(self):
            calls.append("load")

        def transcribe(self, *_args, **_kwargs):
            calls.append("infer")
            return SimpleNamespace(text="hello", language="en", duration=1.0)

    class Upload:
        filename = "speech.wav"
        size = 4

        def __init__(self):
            self.read_once = False

        async def read(self, _size=-1):
            if self.read_once:
                return b""
            self.read_once = True
            return b"RIFF"

    monkeypatch.setattr(stt, "STTEngine", STT)
    for attempt in range(2):
        result = await audio._run_stt_request(
            Upload(), "whisper-small", "en", "json", "transcribe"
        )
        assert result["text"] == "hello"
        assert dispatcher.snapshot()[0]["state"] == "resident"
        if attempt == 0:
            age(dispatcher, "stt")
            await audio.release_idle_audio_lanes(30)
    assert calls == ["load", "infer", "load", "infer"]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_ttl_rejected_before_loading(value):
    from rapid_mlx.cli import _validate_primary_lifecycle_args

    with pytest.raises(SystemExit):
        _validate_primary_lifecycle_args(SimpleNamespace(audio_lane_idle_ttl=value))


@pytest.mark.asyncio
async def test_active_worker_operation_cannot_be_released(lanes):
    audio, dispatcher = lanes
    audio._stt_engine = Cached()
    await dispatcher.execute("stt", "whisper-small", "load", lambda: None)
    dispatcher._begin("stt", "whisper-small", "infer")
    age(dispatcher, "stt")
    try:
        await audio.release_idle_audio_lanes(1)
        assert not audio._stt_engine.unloaded
        assert dispatcher.snapshot()[0]["active_requests"] == 1
    finally:
        dispatcher._finish("stt", "whisper-small", "infer", None)


@pytest.mark.asyncio
async def test_timer_cancellation_drains_unload_and_bookkeeping(lanes):
    audio, dispatcher = lanes
    cached = Cached()
    audio._stt_engine = cached
    entered = asyncio.Event()
    finish = asyncio.Event()

    class SlowWorker(Worker):
        async def execute_on_model_worker(self, func, *args, **kwargs):
            entered.set()
            await finish.wait()
            return func(*args, **kwargs)

    await dispatcher.execute("stt", cached.model_name, "load", lambda: None)
    age(dispatcher, "stt")
    dispatcher.bind(SlowWorker())
    timer = asyncio.create_task(audio.audio_lane_idle_loop(0.01))
    await asyncio.wait_for(entered.wait(), timeout=3)
    timer.cancel()
    await asyncio.sleep(0)
    timer.cancel()
    await asyncio.sleep(0)
    assert not timer.done()
    assert audio._stt_engine is cached
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await timer
    assert audio._stt_engine is None
    assert dispatcher.snapshot()[0]["state"] == "registered"


@pytest.mark.asyncio
async def test_active_stt_does_not_delay_idle_tts_release(lanes):
    audio, dispatcher = lanes
    audio._stt_engine = Cached()
    tts = Cached("kokoro")
    audio._tts_engine = tts
    for lane, name in (("stt", "whisper-small"), ("tts", "kokoro")):
        await dispatcher.execute(lane, name, "load", lambda: None)
        age(dispatcher, lane)
    async with audio._get_stt_lane_lock():
        timer = asyncio.create_task(audio.release_idle_audio_lanes(30))
        for _ in range(100):
            if tts.unloaded:
                break
            await asyncio.sleep(0.001)
        assert tts.unloaded
        assert not audio._stt_engine.unloaded
        assert not timer.done()
    await timer
    assert audio._stt_engine is None
    assert audio._tts_engine is None


@pytest.mark.asyncio
async def test_busy_stt_does_not_stop_later_tts_idle_cycles(lanes):
    audio, dispatcher = lanes

    async def loaded_tts():
        async with audio._get_tts_lane_lock():
            cached = Cached("kokoro")
            audio._tts_engine = cached
            await dispatcher.execute("tts", "kokoro", "load", lambda: None)
            age(dispatcher, "tts")
            return cached

    async def wait_unloaded(cached):
        for _ in range(250):
            if cached.unloaded:
                return
            await asyncio.sleep(0.01)
        assert cached.unloaded, "busy STT stalled a later TTS idle cycle"

    timer = None
    try:
        async with audio._get_stt_lane_lock():
            first = await loaded_tts()
            timer = asyncio.create_task(audio.audio_lane_idle_loop(0.01))
            await wait_unloaded(first)
            second = await loaded_tts()
            await wait_unloaded(second)
    finally:
        if timer is not None:
            timer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await timer


@pytest.mark.asyncio
async def test_idle_backend_without_unload_keeps_truthful_residency(lanes):
    audio, dispatcher = lanes
    cached = SimpleNamespace(model_name="external-backend")
    audio._stt_engine = cached
    await dispatcher.execute("stt", cached.model_name, "load", lambda: None)
    age(dispatcher, "stt")
    await audio.release_idle_audio_lanes(30)
    assert audio._stt_engine is cached
    assert dispatcher.snapshot()[0]["state"] == "resident"
    assert dispatcher.snapshot()[0]["model"] == cached.model_name


@pytest.mark.asyncio
async def test_repeated_lane_cancellation_drains_worker_and_bookkeeping(lanes):
    audio, dispatcher = lanes
    cached = Cached()
    audio._stt_engine = cached
    entered = asyncio.Event()
    finish = asyncio.Event()

    class SlowWorker(Worker):
        async def execute_on_model_worker(self, func, *args, **kwargs):
            entered.set()
            await finish.wait()
            return func(*args, **kwargs)

    await dispatcher.execute("stt", cached.model_name, "load", lambda: None)
    age(dispatcher, "stt")
    dispatcher.bind(SlowWorker())
    timer = asyncio.create_task(audio.audio_lane_idle_loop(0.01))
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        lane = next(t for t in asyncio.all_tasks() if t.get_name() == "audio-idle-stt")
        lane.cancel()
        await asyncio.sleep(0)
        lane.cancel()
        await asyncio.sleep(0)
        assert not lane.done()
        assert audio._stt_engine is cached
        # Parent shutdown must also wait for this already-cancelling child.
        timer.cancel()
        await asyncio.sleep(0)
        assert not timer.done()
    finally:
        finish.set()
        timer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await timer
    assert audio._stt_engine is None
    assert dispatcher.snapshot()[0]["state"] == "registered"
