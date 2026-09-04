"""Contracts for server-owned auxiliary audio execution."""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.runtime.audio_worker import AudioWorkerDispatcher


@pytest.fixture(autouse=True)
def _stub_mlx_thread_init_on_non_mlx_hosts(monkeypatch):
    """Keep the worker contract testable in the Linux unit-test lane."""
    if importlib.util.find_spec("mlx") is not None:
        return

    engine_core = types.ModuleType("vllm_mlx.engine_core")
    engine_core._init_mlx_step_thread = lambda: None
    monkeypatch.setitem(sys.modules, "vllm_mlx.engine_core", engine_core)
    if importlib.util.find_spec("uvicorn") is None:
        monkeypatch.setitem(sys.modules, "uvicorn", types.ModuleType("uvicorn"))


class _RecordingWorker:
    def __init__(self) -> None:
        self.async_calls: list[tuple[object, tuple, dict]] = []
        self.sync_calls: list[tuple[object, tuple, dict]] = []

    async def execute_on_model_worker(self, func, *args, **kwargs):
        self.async_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    def execute_on_model_worker_sync(self, func, *args, **kwargs):
        self.sync_calls.append((func, args, kwargs))
        return func(*args, **kwargs)


class _ReplacementWorker:
    is_mllm = False

    def __init__(self, name: str, *, fail_stop: bool = False) -> None:
        self.name = name
        self.fail_stop = fail_stop
        self.stopped = False
        self.worker_active = False
        self.stopped_during_audio = False
        self.async_calls = 0
        self.stop_calls = 0

    def get_stats(self):
        return {"num_running": 0, "num_waiting": 0}

    async def execute_on_model_worker(self, func, *args, **kwargs):
        if self.stopped:
            raise RuntimeError("model worker is not running")
        self.async_calls += 1
        self.worker_active = True
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        finally:
            self.worker_active = False

    def execute_on_model_worker_sync(self, func, *args, **kwargs):
        if self.stopped:
            raise RuntimeError("model worker is not running")
        return func(*args, **kwargs)

    async def stop(self) -> None:
        self.stop_calls += 1
        self.stopped_during_audio = self.worker_active
        if self.fail_stop:
            raise RuntimeError("old stop failed")
        self.stopped = True


def _replacement_manager(server, old_worker):
    from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from vllm_mlx.runtime.resident_models import ResidentModelManager

    registry = ModelRegistry()
    primary = ModelEntry(
        engine=old_worker,
        model_name="chat-old",
        model_path="repo/chat-old",
    )
    registry.add(primary, is_default=True)
    loaded: dict[str, _ReplacementWorker] = {}

    async def loader(name: str, path: str | None, performance=None):
        worker = _ReplacementWorker(name)
        loaded[name] = worker
        return ModelEntry(
            engine=worker,
            model_name=name,
            model_path=path or f"repo/{name}",
        )

    manager = ResidentModelManager(
        registry,
        loader,
        memory_reader=lambda: 0,
        on_primary_handoff=server._handoff_resident_primary_audio_worker,
        on_primary_changed=server._set_resident_primary,
    )
    manager.register_primary(primary, estimated_bytes=1)
    return manager, registry, loaded


def _configure_server_primary(monkeypatch, server, worker) -> None:
    monkeypatch.setattr(server, "_engine", worker)
    monkeypatch.setattr(server, "_model_name", "chat-old")
    monkeypatch.setattr(server, "_model_alias", "chat-old")
    monkeypatch.setattr(server, "_model_path", "repo/chat-old")
    monkeypatch.setattr(server, "get_config", lambda: SimpleNamespace())


@pytest.mark.asyncio
async def test_audio_only_dispatch_uses_dedicated_worker_without_primary():
    dispatcher = AudioWorkerDispatcher()
    caller_thread = threading.get_ident()

    assert (
        await dispatcher.execute("stt", "whisper", "infer", threading.get_ident)
        != caller_thread
    )
    assert (
        dispatcher.execute_sync("tts", "kokoro", "infer", threading.get_ident)
        != caller_thread
    )
    dispatcher.bind(None)


@pytest.mark.asyncio
async def test_bound_dispatch_uses_public_worker_contract():
    dispatcher = AudioWorkerDispatcher()
    worker = _RecordingWorker()
    dispatcher.bind(worker)

    assert (
        await dispatcher.execute("stt", "whisper", "infer", lambda value: value + 1, 4)
        == 5
    )
    assert (
        dispatcher.execute_sync(
            "tts", "kokoro", "infer", lambda *, value: value + 1, value=6
        )
        == 7
    )
    assert len(worker.async_calls) == 1
    assert len(worker.sync_calls) == 1


def test_bind_rejects_engine_without_complete_worker_contract():
    dispatcher = AudioWorkerDispatcher()

    with pytest.raises(TypeError, match="model-worker contract"):
        dispatcher.bind(object())


def test_audio_worker_handoff_gates_requests_and_supports_rollback():
    from vllm_mlx.runtime.audio_worker import AudioWorkerBusyError

    dispatcher = AudioWorkerDispatcher()
    old_worker = _RecordingWorker()
    new_worker = _RecordingWorker()
    dispatcher.bind(old_worker)

    handoff = dispatcher.begin_handoff()
    with pytest.raises(AudioWorkerBusyError, match="handoff is in progress"):
        dispatcher.execute_sync("stt", "whisper", "infer", lambda: None)
    handoff.rollback()
    assert dispatcher.execute_sync("stt", "whisper", "infer", lambda: "old") == ("old")
    assert len(old_worker.sync_calls) == 1

    handoff = dispatcher.begin_handoff()
    handoff.commit(new_worker)
    assert dispatcher.execute_sync("stt", "whisper", "infer", lambda: "new") == ("new")
    assert len(new_worker.sync_calls) == 1


def test_audio_worker_handoff_defensive_contracts():
    from vllm_mlx.runtime.audio_worker import AudioWorkerBusyError

    dispatcher = AudioWorkerDispatcher()
    assert dispatcher.execute_sync("stt", "whisper", "infer", lambda: "fallback") == (
        "fallback"
    )
    assert dispatcher._fallback is not None

    handoff = dispatcher.begin_handoff()
    with pytest.raises(AudioWorkerBusyError, match="handoff is in progress"):
        dispatcher.bind(_RecordingWorker())
    with pytest.raises(AudioWorkerBusyError, match="handoff is in progress"):
        dispatcher.begin_handoff()
    with pytest.raises(RuntimeError, match="lease is not active"):
        dispatcher._finish_handoff(object(), None)

    handoff.commit(_RecordingWorker())
    assert dispatcher._fallback is None
    handoff.rollback()
    with pytest.raises(RuntimeError, match="already complete"):
        handoff.commit(None)


@pytest.mark.asyncio
async def test_bind_rejects_worker_change_during_active_audio():
    from vllm_mlx.runtime.audio_worker import AudioWorkerBusyError

    dispatcher = AudioWorkerDispatcher()
    old_worker = _ReplacementWorker("chat-old")
    started = threading.Event()
    release = threading.Event()

    def active_audio() -> str:
        started.set()
        assert release.wait(timeout=5)
        return "done"

    dispatcher.bind(old_worker)
    task = asyncio.create_task(
        dispatcher.execute("stt", "whisper", "infer", active_audio)
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        with pytest.raises(AudioWorkerBusyError, match="audio work is active"):
            dispatcher.bind(_ReplacementWorker("chat-new"))
    finally:
        release.set()
        assert await task == "done"
        dispatcher.bind(None)


def test_server_selects_isolated_fallback_for_non_batched_engine():
    from vllm_mlx import server

    assert server._bind_audio_worker_for_engine(object()) is False


@pytest.mark.asyncio
async def test_unbind_restores_dedicated_audio_only_worker():
    dispatcher = AudioWorkerDispatcher()
    dispatcher.bind(_RecordingWorker())
    dispatcher.bind(None)

    caller_thread = threading.get_ident()
    assert (
        await dispatcher.execute("stt", "whisper", "infer", threading.get_ident)
        != caller_thread
    )
    dispatcher.bind(None)


@pytest.mark.asyncio
async def test_batched_engine_async_dispatch_uses_owning_executor_thread():
    owner = object.__new__(BatchedEngine)
    caller_thread = threading.get_ident()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-model-worker"
    )
    owner._model_load_executor = executor
    try:
        worker_thread = await owner.execute_on_model_worker(threading.get_ident)
    finally:
        executor.shutdown(wait=True)

    assert worker_thread != caller_thread


def test_batched_engine_sync_dispatch_uses_owning_executor_thread():
    owner = object.__new__(BatchedEngine)
    caller_thread = threading.get_ident()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-model-worker"
    )
    owner._model_load_executor = executor
    try:
        worker_thread = owner.execute_on_model_worker_sync(threading.get_ident)
    finally:
        executor.shutdown(wait=True)

    assert worker_thread != caller_thread


@pytest.mark.asyncio
async def test_batched_engine_rejects_async_dispatch_when_stopped():
    owner = object.__new__(BatchedEngine)
    owner._model_load_executor = None

    with pytest.raises(RuntimeError, match="model worker is not running"):
        await owner.execute_on_model_worker(lambda: None)


def test_batched_engine_rejects_sync_dispatch_when_stopped():
    owner = object.__new__(BatchedEngine)
    owner._model_load_executor = None

    with pytest.raises(RuntimeError, match="model worker is not running"):
        owner.execute_on_model_worker_sync(lambda: None)


@pytest.mark.asyncio
async def test_lane_snapshot_reports_resident_model_after_successful_load():
    dispatcher = AudioWorkerDispatcher()

    await dispatcher.execute("stt", "whisper-small", "load", lambda: None)

    assert dispatcher.snapshot() == [
        {
            "lane": "stt",
            "role": "speech-input",
            "model": "whisper-small",
            "state": "resident",
            "active_requests": 0,
            "loaded_at": dispatcher.snapshot()[0]["loaded_at"],
            "idle_seconds": pytest.approx(0.0, abs=0.1),
            "last_error": None,
        }
    ]
    dispatcher.bind(None)


def test_lane_snapshot_records_failure_without_leaking_message():
    dispatcher = AudioWorkerDispatcher()

    with pytest.raises(ValueError, match="secret detail"):
        dispatcher.execute_sync(
            "tts",
            "kokoro",
            "load",
            lambda: (_ for _ in ()).throw(ValueError("secret detail")),
        )

    lane = dispatcher.snapshot()[0]
    assert lane["state"] == "failed"
    assert lane["active_requests"] == 0
    assert lane["last_error"] == "ValueError"
    assert "secret detail" not in repr(lane)
    dispatcher.bind(None)


@pytest.mark.asyncio
async def test_cancellation_drains_worker_before_releasing_lane_lease():
    dispatcher = AudioWorkerDispatcher()
    started = threading.Event()
    release = threading.Event()

    def blocking_transcription() -> str:
        started.set()
        assert release.wait(timeout=5)
        return "done"

    task = asyncio.create_task(
        dispatcher.execute("stt", "whisper", "infer", blocking_transcription)
    )
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0.05)

    assert not task.done()
    assert dispatcher.snapshot()[0]["active_requests"] == 1

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert dispatcher.snapshot()[0]["active_requests"] == 0
    dispatcher.bind(None)


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_a_failing_worker():
    dispatcher = AudioWorkerDispatcher()
    started = threading.Event()
    release = threading.Event()

    def failing_transcription() -> None:
        started.set()
        assert release.wait(timeout=5)
        raise RuntimeError("worker failed after cancellation")

    task = asyncio.create_task(
        dispatcher.execute("stt", "whisper", "infer", failing_transcription)
    )
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)

    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    lane = dispatcher.snapshot()[0]
    assert lane["active_requests"] == 0
    assert lane["state"] == "failed"
    assert lane["last_error"] == "RuntimeError"
    dispatcher.bind(None)


@pytest.mark.asyncio
async def test_server_shutdown_unloads_cached_audio_engines(monkeypatch):
    from vllm_mlx.routes import audio as audio_route

    class _Cached:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.unloaded = False

        def unload(self) -> None:
            self.unloaded = True

    stt = _Cached("whisper")
    aligner = _Cached("aligner")
    tts = _Cached("kokoro")
    monkeypatch.setattr(audio_route, "_stt_engine", stt)
    monkeypatch.setattr(audio_route, "_aligner_engine", aligner)
    monkeypatch.setattr(audio_route, "_tts_engine", tts)
    monkeypatch.setattr(audio_route, "_music_engine", object())

    await audio_route.shutdown_audio_lanes()

    assert stt.unloaded and aligner.unloaded and tts.unloaded
    assert audio_route._stt_engine is None
    assert audio_route._aligner_engine is None
    assert audio_route._tts_engine is None
    assert audio_route._music_engine is None


@pytest.mark.asyncio
async def test_server_shutdown_continues_after_audio_unload_failure(
    monkeypatch, caplog
):
    from vllm_mlx.routes import audio as audio_route

    class _Cached:
        def __init__(self, model_name: str, *, fails: bool = False) -> None:
            self.model_name = model_name
            self.fails = fails
            self.unloaded = False

        def unload(self) -> None:
            self.unloaded = True
            if self.fails:
                raise RuntimeError("damaged test backend")

    stt = _Cached("whisper", fails=True)
    aligner = _Cached("aligner")
    tts = _Cached("kokoro")
    monkeypatch.setattr(audio_route, "_stt_engine", stt)
    monkeypatch.setattr(audio_route, "_aligner_engine", aligner)
    monkeypatch.setattr(audio_route, "_tts_engine", tts)
    monkeypatch.setattr(audio_route, "_music_engine", object())

    await audio_route.shutdown_audio_lanes()

    assert stt.unloaded and aligner.unloaded and tts.unloaded
    assert audio_route._stt_engine is None
    assert audio_route._aligner_engine is None
    assert audio_route._tts_engine is None
    assert audio_route._music_engine is None
    assert "Failed to unload stt audio lane" in caplog.text
    assert "damaged test backend" in caplog.text


@pytest.mark.asyncio
async def test_server_shutdown_ignores_cached_objects_without_unload(monkeypatch):
    from vllm_mlx.routes import audio as audio_route

    monkeypatch.setattr(audio_route, "_stt_engine", object())
    monkeypatch.setattr(audio_route, "_aligner_engine", None)
    monkeypatch.setattr(audio_route, "_tts_engine", None)
    monkeypatch.setattr(audio_route, "_music_engine", None)

    await audio_route.shutdown_audio_lanes()

    assert audio_route._stt_engine is None


@pytest.mark.asyncio
async def test_async_stt_eviction_uses_async_model_worker(monkeypatch):
    from vllm_mlx.routes import audio as audio_route
    from vllm_mlx.runtime.audio_worker import bind_audio_worker

    class _CachedAligner:
        model_name = "aligner"

        def __init__(self) -> None:
            self.unloaded = False

        def unload(self) -> None:
            self.unloaded = True

    class _AsyncOnlyWorker:
        async def execute_on_model_worker(self, func, *args, **kwargs):
            await asyncio.sleep(0)
            return func(*args, **kwargs)

        def execute_on_model_worker_sync(self, func, *args, **kwargs):
            raise AssertionError("async STT eviction used the blocking boundary")

    aligner = _CachedAligner()
    monkeypatch.setattr(audio_route, "_aligner_engine", aligner)
    bind_audio_worker(_AsyncOnlyWorker())
    try:
        await audio_route._evict_other_lane("asr")
    finally:
        bind_audio_worker(None)

    assert aligner.unloaded
    assert audio_route._aligner_engine is None


@pytest.mark.asyncio
async def test_empty_stt_lanes_do_not_dispatch_eviction(monkeypatch):
    from vllm_mlx.routes import audio as audio_route

    monkeypatch.setattr(audio_route, "_stt_engine", None)
    monkeypatch.setattr(audio_route, "_aligner_engine", None)

    await audio_route._evict_other_lane("asr")
    audio_route._evict_other_lane_sync("aligner")


@pytest.mark.asyncio
async def test_stt_load_and_inference_use_audio_worker(monkeypatch):
    from vllm_mlx.audio import stt as stt_module
    from vllm_mlx.routes import audio as audio_route
    from vllm_mlx.runtime.audio_worker import bind_audio_worker

    operations: list[str] = []

    class _Upload:
        filename = "speech.wav"
        size = 4

        def __init__(self) -> None:
            self._read = False

        async def read(self, _size: int = -1) -> bytes:
            if self._read:
                return b""
            self._read = True
            return b"RIFF"

    class _OldAligner:
        model_name = "old-aligner"

        def unload(self) -> None:
            operations.append("unload-aligner")

    class _STT:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            operations.append("load-stt")

        def transcribe(self, path: str, **kwargs):
            operations.append("infer-stt")
            assert path.endswith(".wav")
            assert kwargs["context"] == "prompt"
            return SimpleNamespace(text="hello", language="en", duration=1.0)

    worker = _RecordingWorker()
    monkeypatch.setattr(stt_module, "STTEngine", _STT)
    monkeypatch.setattr(audio_route, "_stt_engine", None)
    monkeypatch.setattr(audio_route, "_aligner_engine", _OldAligner())
    bind_audio_worker(worker)
    try:
        response = await audio_route._run_stt_request(
            _Upload(),
            "whisper-small",
            "en",
            "json",
            "transcribe",
            context=" prompt ",
        )
    finally:
        bind_audio_worker(None)

    assert response == {"text": "hello", "language": "en", "duration": 1.0}
    assert operations == ["unload-aligner", "load-stt", "infer-stt"]
    assert len(worker.async_calls) == 3


def test_alignment_load_and_inference_use_audio_worker(monkeypatch):
    from vllm_mlx.audio import stt as stt_module
    from vllm_mlx.routes import audio as audio_route
    from vllm_mlx.runtime.audio_worker import bind_audio_worker

    operations: list[str] = []

    class _Aligner:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            operations.append("load-aligner")

        def align(self, path: str, text: str, **kwargs):
            operations.append("infer-aligner")
            assert (path, text, kwargs) == (
                "speech.wav",
                "known text",
                {"language": "en"},
            )
            return "aligned"

    worker = _RecordingWorker()
    monkeypatch.setattr(stt_module, "STTEngine", _Aligner)
    monkeypatch.setattr(audio_route, "_aligner_engine", None)
    monkeypatch.setattr(audio_route, "_stt_engine", None)
    bind_audio_worker(worker)
    try:
        result = audio_route._align_blocking(
            "aligner-model", "speech.wav", "known text", "en"
        )
    finally:
        bind_audio_worker(None)

    assert result == "aligned"
    assert operations == ["load-aligner", "infer-aligner"]
    assert len(worker.sync_calls) == 2


def test_sync_stt_eviction_uses_sync_model_worker(monkeypatch):
    from vllm_mlx.routes import audio as audio_route

    class _CachedSTT:
        model_name = "whisper"

        def __init__(self) -> None:
            self.unloaded = False

        def unload(self) -> None:
            self.unloaded = True

    stt = _CachedSTT()
    monkeypatch.setattr(audio_route, "_stt_engine", stt)

    audio_route._evict_other_lane_sync("aligner")

    assert stt.unloaded
    assert audio_route._stt_engine is None


def test_tts_replacement_and_reference_inference_use_audio_worker(monkeypatch):
    from vllm_mlx.audio import output_format
    from vllm_mlx.audio import tts as tts_module
    from vllm_mlx.routes import audio as audio_route

    operations: list[str] = []

    class _OldTTS:
        model_name = "old-tts"

        def unload(self) -> None:
            operations.append("unload")

    class _NewTTS:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            operations.append("load")

        def generate(self, text: str, **kwargs):
            operations.append("infer")
            assert text == "hello"
            if "ref_text" in kwargs:
                assert kwargs["ref_text"] == "reference"
            return SimpleNamespace(audio=b"pcm", sample_rate=24_000)

        def to_bytes(self, audio, format: str) -> bytes:
            assert format == "wav"
            return b"encoded"

    monkeypatch.setattr(tts_module, "TTSEngine", _NewTTS)
    monkeypatch.setattr(audio_route, "_tts_engine", _OldTTS())
    monkeypatch.setattr(
        output_format,
        "convert_audio_output",
        lambda audio, source_rate, **kwargs: (b"encoded", 24_000, 1),
    )

    payload, rate, channels = audio_route._generate_speech_blocking(
        model_name="new-tts",
        input_text="hello",
        response_format="wav",
        gen_kwargs={},
        ref_bytes=b"reference-audio",
        ref_text="reference",
        sample_rate=None,
        channels=None,
    )

    assert (payload, rate, channels) == (b"encoded", 24_000, 1)
    assert operations == ["unload", "load", "infer"]

    payload, rate, channels = audio_route._generate_speech_blocking(
        model_name="new-tts",
        input_text="hello",
        response_format="wav",
        gen_kwargs={},
        ref_bytes=None,
        ref_text=None,
        sample_rate=None,
        channels=None,
    )

    assert (payload, rate, channels) == (b"encoded", 24_000, 1)
    assert operations == ["unload", "load", "infer", "infer"]


@pytest.mark.asyncio
async def test_residency_snapshot_includes_audio_lane_truth(monkeypatch):
    from vllm_mlx.routes import residency
    from vllm_mlx.runtime.audio_worker import audio_worker

    class _Manager:
        def snapshot(self):
            return {"models": [{"id": "chat-model"}]}

    monkeypatch.setattr(residency, "_manager", lambda: _Manager())
    monkeypatch.setattr(
        audio_worker,
        "snapshot",
        lambda: [{"lane": "stt", "model": "whisper-small"}],
    )

    assert await residency.model_residency() == {
        "models": [{"id": "chat-model"}],
        "audio_lanes": [{"lane": "stt", "model": "whisper-small"}],
    }


@pytest.mark.asyncio
async def test_primary_replacement_rebinds_after_audio_work_finishes(monkeypatch):
    import vllm_mlx.server as server
    from vllm_mlx.runtime.audio_worker import (
        audio_worker,
        bind_audio_worker,
        run_audio_mlx,
    )

    old_worker = _ReplacementWorker("chat-old")
    _configure_server_primary(monkeypatch, server, old_worker)
    manager, registry, loaded = _replacement_manager(server, old_worker)
    bind_audio_worker(old_worker)
    try:
        assert (
            await run_audio_mlx("stt", "whisper", "infer", lambda: "before") == "before"
        )
        stt_before = next(
            lane for lane in audio_worker.snapshot() if lane["lane"] == "stt"
        )
        assert stt_before["model"] == "whisper"
        assert stt_before["state"] == "resident"

        replacement = await manager.load(
            "chat-new",
            estimated_bytes=1,
            replace_group="assistant",
        )

        assert old_worker.stopped is True
        assert old_worker.stopped_during_audio is False
        assert registry.default_name == "chat-new"
        assert server._engine is replacement.entry.engine
        assert await run_audio_mlx("stt", "whisper", "infer", lambda: "after") == (
            "after"
        )
        assert old_worker.async_calls == 1
        assert loaded["chat-new"].async_calls == 1
        stt_after = next(
            lane for lane in audio_worker.snapshot() if lane["lane"] == "stt"
        )
        assert stt_after["model"] == "whisper"
        assert stt_after["state"] == "resident"

        # Switching again, including back to the original assistant identity,
        # must only replace the assistant engine. Dictation is product-wide
        # process state and remains resident across every picker transition.
        switched_back = await manager.load(
            "chat-old",
            estimated_bytes=1,
            replace_group="assistant",
        )
        assert registry.default_name == "chat-old"
        assert server._engine is switched_back.entry.engine
        assert await run_audio_mlx("stt", "whisper", "infer", lambda: "back") == (
            "back"
        )
        stt_switched_back = next(
            lane for lane in audio_worker.snapshot() if lane["lane"] == "stt"
        )
        assert stt_switched_back["model"] == "whisper"
        assert stt_switched_back["state"] == "resident"
    finally:
        bind_audio_worker(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_mode", ["reject", "wait", "abort"])
async def test_primary_replacement_rejects_active_audio_without_stopping_old_worker(
    monkeypatch,
    replace_mode,
):
    import vllm_mlx.server as server
    from vllm_mlx.runtime.audio_worker import bind_audio_worker, run_audio_mlx
    from vllm_mlx.runtime.resident_models import ResidentModelBusyError

    old_worker = _ReplacementWorker("chat-old")
    _configure_server_primary(monkeypatch, server, old_worker)
    manager, registry, loaded = _replacement_manager(server, old_worker)
    started = threading.Event()
    release = threading.Event()

    def active_audio() -> str:
        started.set()
        assert release.wait(timeout=5)
        return "done"

    bind_audio_worker(old_worker)
    task = asyncio.create_task(run_audio_mlx("stt", "whisper", "infer", active_audio))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        with pytest.raises(ResidentModelBusyError, match="active audio work"):
            await manager.load(
                "chat-new",
                estimated_bytes=1,
                replace_group="assistant",
                replace_mode=replace_mode,
            )

        assert registry.default_name == "chat-old"
        assert server._engine is old_worker
        assert old_worker.stopped is False
        assert old_worker.stopped_during_audio is False
        assert loaded["chat-new"].stopped is True
    finally:
        release.set()
        assert await task == "done"
        bind_audio_worker(None)


@pytest.mark.asyncio
async def test_primary_reload_rejects_active_audio_without_stopping_worker(
    monkeypatch,
):
    import vllm_mlx.server as server
    from vllm_mlx.runtime.audio_worker import bind_audio_worker, run_audio_mlx
    from vllm_mlx.runtime.resident_models import (
        ResidentModelBusyError,
        ResidentPerformanceConfig,
    )

    old_worker = _ReplacementWorker("chat-old")
    _configure_server_primary(monkeypatch, server, old_worker)
    manager, registry, loaded = _replacement_manager(server, old_worker)
    started = threading.Event()
    release = threading.Event()

    def active_audio() -> str:
        started.set()
        assert release.wait(timeout=5)
        return "done"

    bind_audio_worker(old_worker)
    task = asyncio.create_task(run_audio_mlx("stt", "whisper", "infer", active_audio))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        with pytest.raises(ResidentModelBusyError, match="active audio work"):
            await manager.load(
                "chat-old",
                performance=ResidentPerformanceConfig(prefix_cache_enabled=True),
                reload_if_changed=True,
            )

        assert registry.default_name == "chat-old"
        assert server._engine is old_worker
        assert old_worker.stopped is False
        assert loaded == {}
    finally:
        release.set()
        assert await task == "done"
        bind_audio_worker(None)


@pytest.mark.asyncio
async def test_primary_reload_commits_audio_worker_handoff(monkeypatch):
    import vllm_mlx.server as server
    from vllm_mlx.runtime.audio_worker import (
        audio_worker,
        bind_audio_worker,
        run_audio_mlx,
    )
    from vllm_mlx.runtime.resident_models import ResidentPerformanceConfig

    old_worker = _ReplacementWorker("chat-old")
    _configure_server_primary(monkeypatch, server, old_worker)
    manager, registry, loaded = _replacement_manager(server, old_worker)
    bind_audio_worker(old_worker)
    try:
        assert await run_audio_mlx("stt", "whisper", "infer", lambda: "before") == (
            "before"
        )
        replacement = await manager.load(
            "chat-old",
            performance=ResidentPerformanceConfig(prefix_cache_enabled=True),
            reload_if_changed=True,
        )

        assert old_worker.stopped is True
        assert registry.default_name == "chat-old"
        assert server._engine is replacement.entry.engine
        assert await run_audio_mlx("stt", "whisper", "infer", lambda: "after") == (
            "after"
        )
        assert old_worker.async_calls == 1
        assert loaded["chat-old"].async_calls == 1
        stt_after = next(
            lane for lane in audio_worker.snapshot() if lane["lane"] == "stt"
        )
        assert stt_after["model"] == "whisper"
        assert stt_after["state"] == "resident"
    finally:
        bind_audio_worker(None)


@pytest.mark.asyncio
async def test_primary_reload_restores_old_config_after_publication_failure(
    monkeypatch,
):
    import vllm_mlx.server as server
    from vllm_mlx.runtime.audio_worker import bind_audio_worker, run_audio_mlx
    from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from vllm_mlx.runtime.resident_models import (
        ResidentModelManager,
        ResidentPerformanceConfig,
    )

    old_worker = _ReplacementWorker("chat-old")
    _configure_server_primary(monkeypatch, server, old_worker)
    registry = ModelRegistry()
    primary = ModelEntry(
        engine=old_worker,
        model_name="chat-old",
        model_path="repo/chat-old",
    )
    registry.add(primary, is_default=True)
    loaded: list[_ReplacementWorker] = []

    async def loader(name: str, path: str | None, performance=None):
        worker = _ReplacementWorker(f"{name}-reload-{len(loaded) + 1}")
        loaded.append(worker)
        return ModelEntry(
            engine=worker,
            model_name=name,
            model_path=path or f"repo/{name}",
        )

    def publish(entry: ModelEntry | None) -> None:
        server._set_resident_primary(entry)
        if entry is not None and entry.engine is loaded[0]:
            raise RuntimeError("primary publication failed")

    manager = ResidentModelManager(
        registry,
        loader,
        memory_reader=lambda: 0,
        on_primary_handoff=server._handoff_resident_primary_audio_worker,
        on_primary_changed=publish,
    )
    manager.register_primary(primary, estimated_bytes=1)
    bind_audio_worker(old_worker)
    try:
        with pytest.raises(RuntimeError, match="primary publication failed"):
            await manager.load(
                "chat-old",
                performance=ResidentPerformanceConfig(prefix_cache_enabled=True),
                reload_if_changed=True,
            )

        rejected, restored = loaded
        assert old_worker.stopped is True
        assert rejected.stopped is True
        assert restored.stopped is False
        assert registry.default_name == "chat-old"
        assert registry.get_entry("chat-old").engine is restored
        assert server._engine is restored
        assert await run_audio_mlx("stt", "whisper", "infer", lambda: "after") == (
            "after"
        )
        assert rejected.async_calls == 0
        assert restored.async_calls == 1
    finally:
        bind_audio_worker(None)


@pytest.mark.asyncio
async def test_primary_reload_rebuilds_handoff_when_old_stop_fails(monkeypatch):
    import vllm_mlx.server as server
    from vllm_mlx.runtime.audio_worker import bind_audio_worker, run_audio_mlx
    from vllm_mlx.runtime.resident_models import ResidentPerformanceConfig

    old_worker = _ReplacementWorker("chat-old", fail_stop=True)
    _configure_server_primary(monkeypatch, server, old_worker)
    manager, registry, loaded = _replacement_manager(server, old_worker)
    bind_audio_worker(old_worker)
    try:
        with pytest.raises(RuntimeError, match="old stop failed"):
            await manager.load(
                "chat-old",
                performance=ResidentPerformanceConfig(prefix_cache_enabled=True),
                reload_if_changed=True,
            )

        assert old_worker.stop_calls == 1
        assert old_worker.stopped is False
        restored = loaded["chat-old"]
        assert registry.default_name == "chat-old"
        assert registry.get_entry("chat-old").engine is restored
        assert server._engine is restored
        assert await run_audio_mlx("stt", "whisper", "infer", lambda: "after") == (
            "after"
        )
        assert old_worker.async_calls == 0
        assert restored.async_calls == 1
    finally:
        bind_audio_worker(None)


@pytest.mark.asyncio
async def test_primary_reload_restores_old_config_when_new_load_fails(monkeypatch):
    import vllm_mlx.server as server
    from vllm_mlx.runtime.audio_worker import bind_audio_worker, run_audio_mlx
    from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from vllm_mlx.runtime.resident_models import (
        ResidentModelManager,
        ResidentPerformanceConfig,
    )

    old_worker = _ReplacementWorker("chat-old")
    _configure_server_primary(monkeypatch, server, old_worker)
    registry = ModelRegistry()
    primary = ModelEntry(
        engine=old_worker,
        model_name="chat-old",
        model_path="repo/chat-old",
    )
    registry.add(primary, is_default=True)
    restored_workers: list[_ReplacementWorker] = []

    async def loader(name: str, path: str | None, performance=None):
        if performance is not None:
            raise RuntimeError("new config failed")
        restored = _ReplacementWorker("chat-old-restored")
        restored_workers.append(restored)
        return ModelEntry(
            engine=restored,
            model_name=name,
            model_path=path or f"repo/{name}",
        )

    manager = ResidentModelManager(
        registry,
        loader,
        memory_reader=lambda: 0,
        on_primary_handoff=server._handoff_resident_primary_audio_worker,
        on_primary_changed=server._set_resident_primary,
    )
    manager.register_primary(primary, estimated_bytes=1)
    bind_audio_worker(old_worker)
    try:
        with pytest.raises(RuntimeError, match="new config failed"):
            await manager.load(
                "chat-old",
                performance=ResidentPerformanceConfig(prefix_cache_enabled=True),
                reload_if_changed=True,
            )

        restored = restored_workers[0]
        assert old_worker.stopped is True
        assert registry.get_entry("chat-old").engine is restored
        assert server._engine is restored
        assert await run_audio_mlx("stt", "whisper", "infer", lambda: "after") == (
            "after"
        )
        assert restored.async_calls == 1
    finally:
        bind_audio_worker(None)


@pytest.mark.asyncio
async def test_primary_reload_releases_handoff_when_cleanup_and_restore_fail(
    monkeypatch, caplog
):
    import vllm_mlx.server as server
    from vllm_mlx.runtime.audio_worker import bind_audio_worker, run_audio_mlx
    from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from vllm_mlx.runtime.resident_models import (
        ResidentModelManager,
        ResidentPerformanceConfig,
    )

    old_worker = _ReplacementWorker("chat-old")
    _configure_server_primary(monkeypatch, server, old_worker)
    registry = ModelRegistry()
    primary = ModelEntry(
        engine=old_worker,
        model_name="chat-old",
        model_path="repo/chat-old",
    )
    registry.add(primary, is_default=True)
    rejected_workers: list[_ReplacementWorker] = []

    async def loader(name: str, path: str | None, performance=None):
        if rejected_workers:
            raise RuntimeError("restore failed")
        rejected = _ReplacementWorker("chat-old-rejected", fail_stop=True)
        rejected_workers.append(rejected)
        return ModelEntry(
            engine=rejected,
            model_name=name,
            model_path=path or f"repo/{name}",
        )

    def reject_publication(entry: ModelEntry | None) -> None:
        server._set_resident_primary(entry)
        if entry is not None:
            raise RuntimeError("primary publication failed")

    manager = ResidentModelManager(
        registry,
        loader,
        memory_reader=lambda: 0,
        on_primary_handoff=server._handoff_resident_primary_audio_worker,
        on_primary_changed=reject_publication,
    )
    manager.register_primary(primary, estimated_bytes=1)
    bind_audio_worker(old_worker)
    try:
        with pytest.raises(RuntimeError, match="primary publication failed"):
            await manager.load(
                "chat-old",
                performance=ResidentPerformanceConfig(prefix_cache_enabled=True),
                reload_if_changed=True,
            )

        rejected = rejected_workers[0]
        assert rejected.stop_calls == 1
        assert rejected.stopped is False
        assert registry.list_entries() == []
        assert "Failed to stop rejected resident model" in caplog.text
        assert "Failed to restore resident model" in caplog.text
        assert await run_audio_mlx("stt", "whisper", "infer", lambda: "fallback") == (
            "fallback"
        )
    finally:
        bind_audio_worker(None)


@pytest.mark.asyncio
async def test_primary_replacement_keeps_new_worker_after_old_stop_failure(
    monkeypatch,
):
    import vllm_mlx.server as server
    from vllm_mlx.runtime.audio_worker import bind_audio_worker, run_audio_mlx

    old_worker = _ReplacementWorker("chat-old", fail_stop=True)
    _configure_server_primary(monkeypatch, server, old_worker)
    manager, registry, loaded = _replacement_manager(server, old_worker)
    bind_audio_worker(old_worker)
    try:
        replacement = await manager.load(
            "chat-new",
            estimated_bytes=1,
            replace_group="assistant",
        )

        assert replacement.primary is True
        assert registry.default_name == "chat-new"
        assert [entry.model_name for entry in registry.list_entries()] == ["chat-new"]
        assert server._engine is loaded["chat-new"]
        assert old_worker.stopped is False
        assert loaded["chat-new"].stopped is False
        assert await run_audio_mlx("stt", "whisper", "infer", lambda: "after") == (
            "after"
        )
        assert old_worker.async_calls == 0
        assert loaded["chat-new"].async_calls == 1
    finally:
        bind_audio_worker(None)


@pytest.mark.asyncio
async def test_lifespan_binds_audio_worker_when_lane_is_enabled(monkeypatch):
    import vllm_mlx._signal_observability as signal_observability
    import vllm_mlx.server as server
    from vllm_mlx.routes import video as video_route

    class _Engine:
        _loaded = True

        def generate_warmup(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    class _Residency:
        async def start(self) -> None:
            pass

        async def shutdown(self) -> None:
            pass

        def contains(self, model_name: str) -> bool:
            return True

    engine = _Engine()
    bound: list[object] = []
    lifecycle_config = SimpleNamespace(
        ready=False,
        draining=False,
        bind_host=None,
        bind_port=None,
        bind_listen_fd=None,
        model_alias=None,
        model_name=None,
    )

    async def _shutdown_video_jobs() -> None:
        pass

    monkeypatch.setattr(
        signal_observability, "install_signal_observability", lambda: False
    )
    monkeypatch.setattr(video_route, "start_video_jobs", lambda: None)
    monkeypatch.setattr(video_route, "shutdown_video_jobs", _shutdown_video_jobs)
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_residency_manager", _Residency())
    monkeypatch.setattr(server, "_mcp_manager", None)
    monkeypatch.setattr(server, "_gc_control", False)
    monkeypatch.setattr(server, "_enable_audio_lane", True)
    monkeypatch.setattr(server, "_model_name", "chat-model")
    monkeypatch.setattr(server, "_model_alias", "chat-model")
    monkeypatch.setattr(server, "get_config", lambda: lifecycle_config)
    monkeypatch.setattr(
        server,
        "_bind_audio_worker_for_engine",
        lambda candidate: bound.append(candidate) or True,
    )

    lifespan = server.lifespan(server.app)
    await lifespan.__anext__()
    with pytest.raises(StopAsyncIteration):
        await lifespan.__anext__()

    assert bound == [engine]


# ---------------------------------------------------------------------------
# Alignment-role admission wiring (issue #2405). These exercise the shared
# residency ledger integration used by the forced-alignment lane — the
# ``_admitting_alignment`` context manager that resolves the aligner footprint
# from catalog metadata and admits it as the distinct ``alignment`` role with
# the typed 507 contract. Entirely MLX-free: a real ResidentModelManager with
# stubbed engines stands in for the server residency, and STTEngine is stubbed
# via ``vllm_mlx.audio.stt.STTEngine``.
# ---------------------------------------------------------------------------


def _aligner_catalog_bytes() -> int:
    # Real manifest footprint for qwen3-aligner /
    # mlx-community/Qwen3-ForcedAligner-0.6B-8bit.
    return 1276473392


def _make_role_manager(limit_gib: float):
    from vllm_mlx.runtime.model_registry import ModelEntry, ModelRegistry
    from vllm_mlx.runtime.resident_models import ResidentModelManager

    registry = ModelRegistry()
    manager = ResidentModelManager(
        registry,
        lambda name, path=None, perf=None: ModelEntry(
            engine=object(), model_name=name, model_path=f"repo/{name}"
        ),
        memory_limit_bytes=int(limit_gib * 1024**3),
        memory_reader=lambda: 0,
    )
    return manager


def _install_role_manager(monkeypatch, manager):
    """Point the audio route's residency lookup at a real manager so it
    performs actual role admission. ``monkeypatch`` restores the config field
    and the module function after the test."""
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import audio as audio_route

    cfg = get_config()
    monkeypatch.setattr(cfg, "residency_manager", manager)
    monkeypatch.setattr(audio_route, "_residency_manager", lambda: manager)


def test_noop_role_admission_commit_is_noop():
    """pr_validate codex NIT (round-8): ``_NoopRoleAdmission.commit()`` must
    exist and be a no-op so an unmanaged server (no residency manager) that
    publishes an engine and then gets cancelled surfaces the real
    ``CancelledError`` — not an ``AttributeError`` from a missing method.
    Synchronous: no asyncio marker, as a sync test under stricter
    pytest-asyncio configs would warn."""

    from vllm_mlx.routes.audio import _NoopRoleAdmission

    admission = _NoopRoleAdmission()
    admission.retire_previous()
    admission.retire_exclusive()  # must exist so unmanaged servers don't AttributeError
    admission.commit()  # must not raise


@pytest.mark.asyncio
async def test_admitting_alignment_releases_resident_speech_input_role(
    monkeypatch,
):
    """pr_validate codex BLOCKING (round-12/14): alignment and the dictation
    STT lane are mutually exclusive, so the resident ``speech-input``
    reservation must not be DOUBLE-CHARGED with the alignment role — otherwise
    the ledger can false-507 even though loading the aligner would immediately
    drop the ASR engine. The sibling is RETAINED through the admission but
    CREDITED against the alignment capacity check, and retired on success."""

    from vllm_mlx.routes import audio as audio_route

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)

    # A dictation speech-input reservation is resident (both weights present).
    async with manager.admit_role(
        role="speech-input",
        model_id="whisper-large",
        requested_bytes=_aligner_catalog_bytes(),
        capacity_source="catalog",
    ):
        pass
    assert [r for r in manager.snapshot()["roles"] if r["role"] == "speech-input"]

    # Entering alignment admission admits the aligner WITHOUT double-charging
    # (the retained speech-input bytes are credited, so no false 507) and
    # commits on success with the ASR-aligned sibling retired.
    async with audio_route._admitting_alignment("qwen3-aligner") as admission:
        assert admission is not None
        # Retained (still its engine's true reservation) but credited against
        # the aligner, so only the aligner's NET bytes consumed headroom.
        assert "alignment" in {r["role"] for r in manager.snapshot()["roles"]}
    # After success the aligner evicted the ASR engine, so the sibling retired.
    roles = {r["role"] for r in manager.snapshot()["roles"]}
    assert "speech-input" not in roles
    assert "alignment" in roles


@pytest.mark.asyncio
async def test_admitting_alignment_restores_speech_input_on_507(monkeypatch):
    """pr_validate codex BLOCKING (round-15): when alignment admission is
    REJECTED (507) before any load runs, the ASR engine is still resident, so
    the speech-input reservation released up front must be RESTORED — never
    leave the still-resident ASR engine unaccounted."""

    from fastapi import HTTPException

    from vllm_mlx.routes import audio as audio_route

    class _ResidentStt:
        model_name = "whisper-large"

    monkeypatch.setattr(audio_route, "_stt_engine", _ResidentStt())

    # Tight ceiling: fits the small speech-input reservation, but NOT the
    # large aligner footprint even after speech-input is released.
    manager = _make_role_manager(limit_gib=0.5)
    _install_role_manager(monkeypatch, manager)

    # Dictation role resident + ASR engine resident (small footprint).
    small = int(0.05 * 1024**3)
    async with manager.admit_role(
        role="speech-input",
        model_id="whisper-large",
        requested_bytes=small,
        capacity_source="catalog",
    ):
        pass
    assert [r for r in manager.snapshot()["roles"] if r["role"] == "speech-input"]

    # Alignment admission is rejected with 507 (over ceiling even after the
    # speech-input release) -> the speech-input reservation must be restored.
    with pytest.raises(HTTPException) as exc_info:
        async with audio_route._admitting_alignment("qwen3-aligner"):
            pass
    assert exc_info.value.status_code == 507

    roles = {r["role"] for r in manager.snapshot()["roles"]}
    assert "speech-input" in roles  # restored: the ASR engine is still resident
    assert "alignment" not in roles  # the rejected aligner left no reservation


@pytest.mark.asyncio
async def test_admitting_alignment_does_not_map_body_errors(monkeypatch):
    """pr_validate codex BLOCKING (round-7): the ``_admitting_alignment``
    exception mapping covers only ADMISSION ENTRY. A ``ResidentModelError``
    raised by the yielded LOAD BODY must propagate unchanged (a loader/runtime
    failure), never be rewritten as a 409 ``alignment_role_conflict``."""
    from vllm_mlx.routes import audio as audio_route
    from vllm_mlx.runtime.resident_models import ResidentModelError

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)

    with pytest.raises(ResidentModelError, match="loader blew up"):
        async with audio_route._admitting_alignment("qwen3-aligner"):
            raise ResidentModelError("loader blew up")


@pytest.mark.asyncio
async def test_admitting_alignment_maps_existing_role_conflict(monkeypatch):
    """An admission-entry invariant conflict is a typed 409 response."""
    from fastapi import HTTPException

    from vllm_mlx.routes import audio as audio_route

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)
    async with manager.admit_role(
        role="alignment",
        model_id="qwen3-aligner",
        requested_bytes=_aligner_catalog_bytes(),
        capacity_source="catalog",
    ):
        pass

    with pytest.raises(HTTPException) as exc_info:
        async with audio_route._admitting_alignment("qwen3-aligner"):
            pass

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["error"]["code"] == "alignment_role_conflict"


@pytest.mark.asyncio
async def test_admitting_alignment_drains_cancelled_rollback(monkeypatch):
    """Cancellation while rollback waits cannot strand a loading role."""
    from vllm_mlx.routes import audio as audio_route

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)
    original = manager.admit_role
    exit_started = asyncio.Event()
    release_exit = asyncio.Event()

    class DelayedExit:
        def __init__(self, inner):
            self._inner = inner

        async def __aenter__(self):
            return await self._inner.__aenter__()

        async def __aexit__(self, typ, exc, tb):
            exit_started.set()
            await release_exit.wait()
            return await self._inner.__aexit__(typ, exc, tb)

    monkeypatch.setattr(
        manager,
        "admit_role",
        lambda *args, **kwargs: DelayedExit(original(*args, **kwargs)),
    )

    async def fail_during_load():
        async with audio_route._admitting_alignment("qwen3-aligner"):
            raise RuntimeError("load failed")

    task = asyncio.create_task(fail_during_load())
    await exit_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    # A second cancellation must also be absorbed while the protected rollback
    # is still draining; only the original load failure is observable after
    # the ledger cleanup completes.
    task.cancel()
    await asyncio.sleep(0)
    release_exit.set()
    with pytest.raises(RuntimeError, match="load failed"):
        await task
    assert manager.snapshot()["roles"] == []


def test_canonical_model_id_resolves_default_aligner():
    from vllm_mlx.routes import audio as audio_route

    assert (
        audio_route._canonical_model_id("default")
        == audio_route.STT_MODEL_ALIASES[audio_route.DEFAULT_ALIGNER_ALIAS]
    )


@pytest.mark.asyncio
async def test_admitting_alignment_resolves_footprint_before_load(monkeypatch):
    """The alignment admission resolves the aligner footprint from catalog
    metadata (not blind) and is admitted through the shared ledger."""
    from vllm_mlx.routes import audio as audio_route

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)

    result = []
    async with audio_route._admitting_alignment("qwen3-aligner") as admission:
        result.append(("inside", admission))
        roles = manager.snapshot()["roles"]
        entry = next(r for r in roles if r["role"] == "alignment")
        # Footprint resolved from the catalog before load.
        assert entry["reserved_bytes"] == _aligner_catalog_bytes()
        assert entry["capacity_source"] == "catalog"
        assert entry["state"] == "loading"

    # After success the role commits resident.
    roles = manager.snapshot()["roles"]
    entry = next(r for r in roles if r["role"] == "alignment")
    assert entry["state"] == "resident"
    assert result[0][0] == "inside"


@pytest.mark.asyncio
async def test_admitting_alignment_returns_507_envelope_when_over_budget(
    monkeypatch,
):
    """Under a residency ceiling too small to hold the aligner, admission
    raises the typed 507 HTTP envelope (no blind load)."""
    from fastapi import HTTPException

    from vllm_mlx.routes import audio as audio_route

    manager = _make_role_manager(limit_gib=0.4)  # 0.4 GiB < aligner
    _install_role_manager(monkeypatch, manager)

    with pytest.raises(HTTPException) as exc_info:
        async with audio_route._admitting_alignment("qwen3-aligner"):
            pass

    assert exc_info.value.status_code == 507
    envelope = exc_info.value.detail["error"]
    assert envelope["type"] == "insufficient_capacity_error"
    assert envelope["code"] == "insufficient_capacity_error"
    assert envelope["reason"] == "role_capacity_alignment"
    assert envelope["param"] == "model"
    assert envelope["requested_bytes"] == _aligner_catalog_bytes()
    assert envelope["limit_bytes"] == int(0.4 * 1024**3)
    # No leaked reservation.
    assert manager.snapshot()["roles"] == []


@pytest.mark.asyncio
async def test_admitting_alignment_rolls_back_on_load_failure(monkeypatch):
    """A failed load inside the admission rolls back the reservation so the
    ledger holds no leaked charge."""
    from vllm_mlx.routes import audio as audio_route

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)

    with pytest.raises(RuntimeError, match="aligner load failed"):
        async with audio_route._admitting_alignment("qwen3-aligner"):
            raise RuntimeError("aligner load failed")

    assert manager.snapshot()["roles"] == []
    assert manager._accounted_usage() == 0


@pytest.mark.asyncio
async def test_aligner_load_failure_after_asr_eviction_does_not_restore_phantom_speech_input(
    monkeypatch,
):
    """pr_validate codex BLOCKING (round-18): ``_load_aligner_blocking`` evicts
    the ASR engine (fires ``retire_exclusive``) BEFORE constructing/loading the
    aligner. If that load then FAILS before publication, the retired
    ``speech-input`` reservation must NOT be restored — the ASR engine it
    guarded was already discarded, so restoring would charge a phantom engine
    that no longer exists. Drive the REAL route end-to-end."""
    import io

    from fastapi import HTTPException, UploadFile

    from vllm_mlx.routes import audio as audio_route
    from vllm_mlx.runtime.audio_worker import bind_audio_worker

    worker = _RecordingWorker()
    monkeypatch.setattr(audio_route, "_aligner_engine", None)
    monkeypatch.setattr(audio_route, "_stt_engine", None)
    bind_audio_worker(worker)

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)

    # A dictation speech-input reservation is resident (ASR engine present).
    small = int(0.05 * 1024**3)
    async with manager.admit_role(
        role="speech-input",
        model_id="whisper-large",
        requested_bytes=small,
        capacity_source="catalog",
    ):
        pass
    assert any(r["role"] == "speech-input" for r in manager.snapshot()["roles"])

    def _failing_load(model_name, on_discard_previous=None, on_discard_exclusive=None):
        # Mirror the real loader's ordering: evict the ASR sibling (fire
        # retire_exclusive) THEN fail before the aligner ever publishes — the
        # codex BLOCKING scenario for construction/load failure after eviction.
        if on_discard_exclusive is not None:
            on_discard_exclusive()
        raise RuntimeError("aligner construction failed after ASR eviction")

    monkeypatch.setattr(audio_route, "_load_aligner_blocking", _failing_load)

    with pytest.raises(HTTPException) as exc_info:
        await audio_route._run_alignment_request(
            UploadFile(filename="clip.wav", file=io.BytesIO(b"\x00" * 64)),
            model="qwen3-aligner",
            text="你好",
            language=None,
            response_format="json",
        )
    assert exc_info.value.status_code == 500

    # The failed load left NO alignment reservation (rolled back) AND the
    # evicted ASR engine means the retired speech-input reservation stays
    # retired — no phantom charge for a discarded engine.
    roles = {r["role"] for r in manager.snapshot()["roles"]}
    assert "alignment" not in roles
    assert "speech-input" not in roles


@pytest.mark.asyncio
async def test_admitting_alignment_rolls_back_on_cancellation(monkeypatch):
    """Cancellation inside the admission leaves no leaked reservation."""
    from vllm_mlx.routes import audio as audio_route

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)

    with pytest.raises(asyncio.CancelledError):
        async with audio_route._admitting_alignment("qwen3-aligner"):
            raise asyncio.CancelledError()

    assert manager.snapshot()["roles"] == []
    assert manager._accounted_usage() == 0


@pytest.mark.asyncio
async def test_evicting_aligner_releases_alignment_role(monkeypatch):
    """An ASR request evicting the aligner lane releases the alignment role
    so the ledger does not keep charging a dropped engine."""
    from vllm_mlx.routes import audio as audio_route

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)

    # Reserve the alignment role as if an aligner is resident.
    async with manager.admit_role(
        role="alignment",
        model_id="qwen3-aligner",
        requested_bytes=_aligner_catalog_bytes(),
        capacity_source="catalog",
    ):
        pass
    assert manager.snapshot()["roles"]

    # Simulate the aligner engine being resident in the cache.
    class _CachedAligner:
        model_name = "qwen3-aligner"

        def unload(self) -> None:
            pass

    monkeypatch.setattr(audio_route, "_aligner_engine", _CachedAligner())

    # ASR evicts the aligner lane.
    await audio_route._evict_other_lane("asr")

    assert audio_route._aligner_engine is None
    assert manager.snapshot()["roles"] == []


@pytest.mark.asyncio
async def test_shutdown_releases_alignment_role(monkeypatch):
    """Shutdown unloads the aligner and drops its role reservation."""
    from vllm_mlx.routes import audio as audio_route

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)

    async with manager.admit_role(
        role="alignment",
        model_id="qwen3-aligner",
        requested_bytes=_aligner_catalog_bytes(),
        capacity_source="catalog",
    ):
        pass

    class _Cached:
        model_name = "qwen3-aligner"

        def unload(self) -> None:
            pass

    monkeypatch.setattr(audio_route, "_stt_engine", None)
    monkeypatch.setattr(audio_route, "_aligner_engine", _Cached())
    monkeypatch.setattr(audio_route, "_tts_engine", None)
    monkeypatch.setattr(audio_route, "_music_engine", None)

    await audio_route.shutdown_audio_lanes()

    assert manager.snapshot()["roles"] == []


# ---------------------------------------------------------------------------
# Cached-model validation path (issue #2405): after an aligner is resident,
# a repeat request for the SAME model does not re-admit (no double charge),
# inferring against the cached engine, while the role stays charged.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cached_model_alignment_loads_once_and_stays_resident(monkeypatch):
    """pr_validate codex BLOCKING #5: drive THREE real ``_run_alignment_request``
    calls for the SAME model and assert one load, three inferences, and one
    resident role reservation.

    After admission + load, a repeat real request for the SAME model must not
    reload the aligner NOR re-admit the role — it infers against the cached
    engine while a single resident reservation stays in the ledger. Driving the
    full route (not ``_admitting_alignment``/``_load_aligner_blocking`` by
    hand) means a regression that reloads or re-admits per request fails here.
    """
    import io

    from fastapi import UploadFile

    from vllm_mlx.routes import audio as audio_route
    from vllm_mlx.runtime.audio_worker import bind_audio_worker

    operations: list[str] = []

    class _Aligner:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            operations.append("load-aligner")

        def align(self, path: str, text: str, **kwargs):
            operations.append("infer-aligner")
            return "aligned"

    worker = _RecordingWorker()
    monkeypatch.setattr("vllm_mlx.audio.stt.STTEngine", _Aligner)
    monkeypatch.setattr(audio_route, "_aligner_engine", None)
    monkeypatch.setattr(audio_route, "_stt_engine", None)
    bind_audio_worker(worker)

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)

    def _recording_load(
        model_name, on_discard_previous=None, on_discard_exclusive=None
    ):
        # Mirror the real loader's idempotency: once the engine for this model
        # is published, a repeat call is a no-op (no reload, no re-load).
        if (
            audio_route._aligner_engine is not None
            and audio_route._aligner_engine.model_name == model_name
        ):
            return
        operations.append("load-aligner")
        # Mirror the real loader's ordering: evicting the ASR sibling fires
        # retire_exclusive BEFORE the new aligner is published.
        if on_discard_exclusive is not None:
            on_discard_exclusive()
        audio_route._aligner_engine = _Aligner(model_name)
        if on_discard_previous is not None:
            on_discard_previous()

    monkeypatch.setattr(audio_route, "_load_aligner_blocking", _recording_load)

    def _request(i: int):
        return audio_route._run_alignment_request(
            UploadFile(filename=f"clip{i}.wav", file=io.BytesIO(b"\x00" * 64)),
            model="qwen3-aligner",
            text="你好",
            language=None,
            response_format="json",
        )

    try:
        for i in range(3):
            await _request(i)
    finally:
        bind_audio_worker(None)

    # Loaded exactly once across three real requests; three cached infer calls
    # against the reused engine.
    assert operations.count("load-aligner") == 1
    assert operations.count("infer-aligner") == 3
    # A single alignment role reservation remains resident in the ledger.
    roles = [r for r in manager.snapshot()["roles"] if r["role"] == "alignment"]
    assert len(roles) == 1
    assert roles[0]["state"] == "resident"
    assert roles[0]["reserved_bytes"] == _aligner_catalog_bytes()


@pytest.mark.asyncio
async def test_aligner_alias_and_canonical_request_reuse_same_engine(monkeypatch):
    """pr_validate codex BLOCKING (round-7): an alias and its canonical HF id
    name the same checkpoint, so alternating alias / canonical requests across
    the real ``_run_alignment_request`` must NOT reload or re-admit — the
    alignment stays loaded once and one role reservation stays resident."""

    import io

    from fastapi import UploadFile

    from vllm_mlx.routes import audio as audio_route
    from vllm_mlx.runtime.audio_worker import bind_audio_worker

    operations: list[str] = []

    class _Aligner:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            operations.append("load-aligner")

        def align(self, path: str, text: str, **kwargs):
            operations.append("infer-aligner")
            return "aligned"

    worker = _RecordingWorker()
    monkeypatch.setattr("vllm_mlx.audio.stt.STTEngine", _Aligner)
    monkeypatch.setattr(audio_route, "_aligner_engine", None)
    monkeypatch.setattr(audio_route, "_stt_engine", None)
    bind_audio_worker(worker)

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)

    def _recording_load(
        model_name, on_discard_previous=None, on_discard_exclusive=None
    ):
        if (
            audio_route._aligner_engine is not None
            and audio_route._aligner_engine.model_name == model_name
        ):
            return
        operations.append("load-aligner")
        if on_discard_exclusive is not None:
            on_discard_exclusive()
        audio_route._aligner_engine = _Aligner(model_name)
        if on_discard_previous is not None:
            on_discard_previous()

    monkeypatch.setattr(audio_route, "_load_aligner_blocking", _recording_load)

    def _request(i: int, model: str):
        return audio_route._run_alignment_request(
            UploadFile(filename=f"clip{i}.wav", file=io.BytesIO(b"\x00" * 64)),
            model=model,
            text="你好",
            language=None,
            response_format="json",
        )

    try:
        # Alternate alias and canonical id across requests; both resolve to the
        # same canonical checkpoint and must share one load.
        await _request(0, "qwen3-aligner")
        await _request(1, "mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
        await _request(2, "qwen3-forced-aligner")
    finally:
        bind_audio_worker(None)

    # Exactly one load across three requests (alias + canonical + long alias).
    assert operations.count("load-aligner") == 1
    assert operations.count("infer-aligner") == 3
    roles = [r for r in manager.snapshot()["roles"] if r["role"] == "alignment"]
    assert len(roles) == 1
    assert roles[0]["state"] == "resident"
    assert roles[0]["model"] == "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"


@pytest.mark.asyncio
async def test_aligner_load_retires_previous_only_after_actual_discard(monkeypatch):
    """pr_validate codex BLOCKING #1: the previous aligner's reservation must be
    retired only when the load has actually discarded the old engine — an
    import/ASR-unload failure BEFORE the drop must leave it restorable."""
    from vllm_mlx.routes import audio as audio_route
    from vllm_mlx.runtime.audio_worker import bind_audio_worker

    operations: list[str] = []
    retired = []

    class _Aligner:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            operations.append("load-aligner")

        def align(self, path, text, **kwargs):
            raise AssertionError("must not run after failed load")

    class _OldAligner:
        model_name = "old-aligner"

        def unload(self) -> None:
            operations.append("unload-old-aligner")

    worker = _RecordingWorker()
    previous = _OldAligner()
    monkeypatch.setattr("vllm_mlx.audio.stt.STTEngine", _Aligner)
    monkeypatch.setattr(audio_route, "_aligner_engine", previous)
    monkeypatch.setattr(audio_route, "_stt_engine", None)

    # Force the load to fail at the ASR-unload step, BEFORE the old aligner is
    # dropped (`_aligner_engine = None`). The retire callback must not fire.
    def boom_unload_sync(*a, **k):
        raise RuntimeError("asr unload failed")

    monkeypatch.setattr(audio_route, "_evict_other_lane_sync", boom_unload_sync)
    bind_audio_worker(worker)
    try:
        with pytest.raises(RuntimeError, match="asr unload failed"):
            audio_route._load_aligner_blocking(
                "new-aligner", on_discard_previous=lambda: retired.append(True)
            )
    finally:
        bind_audio_worker(None)

    # The old aligner was never dropped and its retire callback never fired.
    assert retired == []
    assert audio_route._aligner_engine is previous  # old stays resident


@pytest.mark.asyncio
async def test_aligner_retire_fires_after_previous_discard(monkeypatch):
    """The retire-previous callback fires only after the old aligner is
    actually discarded (set to None), before the new load publishes."""
    from vllm_mlx.routes import audio as audio_route
    from vllm_mlx.runtime.audio_worker import bind_audio_worker

    events: list[str] = []

    class _OldAligner:
        model_name = "old-aligner"

    class _Aligner:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def load(self) -> None:
            events.append("load")

    worker = _RecordingWorker()
    monkeypatch.setattr("vllm_mlx.audio.stt.STTEngine", _Aligner)
    monkeypatch.setattr(audio_route, "_aligner_engine", _OldAligner())
    monkeypatch.setattr(audio_route, "_stt_engine", None)
    monkeypatch.setattr(audio_route, "_evict_other_lane_sync", lambda _k: None)
    bind_audio_worker(worker)

    def on_discard():
        events.append("retired")
        # At this exact moment the old engine has been dropped.
        assert audio_route._aligner_engine is None

    try:
        audio_route._load_aligner_blocking(
            "new-aligner", on_discard_previous=on_discard
        )
    finally:
        bind_audio_worker(None)

    assert events == ["retired", "load"]
    assert audio_route._aligner_engine.model_name == "new-aligner"


@pytest.mark.asyncio
async def test_alignment_role_cancellation_keeps_published_engine_committed(
    monkeypatch,
):
    """pr_validate codex BLOCKING #2: drive cancellation through
    ``_run_alignment_request`` end-to-end.

    When the aligner load completes and publishes the engine but the request
    is cancelled before the next await returns, ``_run_alignment_request``'s
    ``admission.commit()`` branch must keep the (accounted) reservation
    resident — never roll it back into an unaccounted resident desync. This
    drives the REAL route + admission wiring (not ``admit_role`` directly), so
    deleting the ``admission.commit()`` branch would fail the test.
    """

    import io

    from fastapi import UploadFile

    from vllm_mlx.routes import audio as audio_route

    manager = _make_role_manager(limit_gib=4.0)
    _install_role_manager(monkeypatch, manager)

    class _FakeAlign:
        # Publish an ALIAS (not the canonical HF id) to prove the
        # cancellation-commit comparison canonicalizes: the request is for
        # ``qwen3-aligner`` and the route resolves it to the canonical id, but
        # the published engine may expose the alias. The commit must match on
        # canonical form or a successfully-published engine goes unaccounted.
        model_name = "qwen3-aligner"

    published = threading.Event()
    release = threading.Event()

    def _controlled_load_blocking(
        model_name: str, on_discard_previous=None, on_discard_exclusive=None
    ) -> None:
        # Runs on the worker thread via run_to_completion. Publish the engine
        # (the load "succeeded"), then block until the test releases us so the
        # load is still in-flight when the request is cancelled.
        audio_route._aligner_engine = _FakeAlign()
        if on_discard_exclusive is not None:
            on_discard_exclusive()
        if on_discard_previous is not None:
            on_discard_previous()
        published.set()
        release.wait()

    monkeypatch.setattr(
        audio_route, "_load_aligner_blocking", _controlled_load_blocking
    )
    monkeypatch.setattr(audio_route, "_aligner_engine", None)

    req = asyncio.create_task(
        audio_route._run_alignment_request(
            UploadFile(filename="clip.wav", file=io.BytesIO(b"\x00" * 64)),
            model="qwen3-aligner",
            text="你好",
            language=None,
            response_format="json",
        )
    )

    # Wait for the load worker to publish the engine, then cancel the request
    # mid-flight (client disconnect). run_to_completion drains the worker —
    # release it so the drain finishes.
    await asyncio.to_thread(published.wait)
    req.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await req

    # The engine IS published (the cancelled load finished on the worker).
    assert audio_route._aligner_engine is not None
    assert audio_route._aligner_engine.model_name == _FakeAlign.model_name

    # The admission-commit branch kept the reservation resident rather than
    # rolling it back. This is the branch under test: delete it and the entry
    # disappears.
    roles = [r for r in manager.snapshot()["roles"] if r["role"] == "alignment"]
    assert len(roles) == 1
    assert roles[0]["state"] == "resident"
    # The ledger records the canonical id for the ``qwen3-aligner`` alias.
    assert roles[0]["model"] == "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
