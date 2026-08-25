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
    from vllm_mlx.runtime.audio_worker import bind_audio_worker, run_audio_mlx

    old_worker = _ReplacementWorker("chat-old")
    _configure_server_primary(monkeypatch, server, old_worker)
    manager, registry, loaded = _replacement_manager(server, old_worker)
    bind_audio_worker(old_worker)
    try:
        assert (
            await run_audio_mlx("stt", "whisper", "infer", lambda: "before") == "before"
        )

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
    finally:
        bind_audio_worker(None)


@pytest.mark.asyncio
async def test_primary_replacement_rejects_active_audio_without_stopping_old_worker(
    monkeypatch,
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
    from vllm_mlx.runtime.audio_worker import bind_audio_worker, run_audio_mlx
    from vllm_mlx.runtime.resident_models import ResidentPerformanceConfig

    old_worker = _ReplacementWorker("chat-old")
    _configure_server_primary(monkeypatch, server, old_worker)
    manager, registry, loaded = _replacement_manager(server, old_worker)
    bind_audio_worker(old_worker)
    try:
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
        assert old_worker.async_calls == 0
        assert loaded["chat-old"].async_calls == 1
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

    def publish(entry: ModelEntry) -> None:
        server._set_resident_primary(entry)
        if entry.engine is loaded[0]:
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
async def test_primary_reload_rolls_back_handoff_when_old_stop_fails(monkeypatch):
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
        assert loaded == {}
        assert registry.default_name == "chat-old"
        assert registry.get_entry("chat-old").engine is old_worker
        assert server._engine is old_worker
        assert await run_audio_mlx("stt", "whisper", "infer", lambda: "after") == (
            "after"
        )
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

    def reject_publication(entry: ModelEntry) -> None:
        server._set_resident_primary(entry)
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
async def test_primary_replacement_rolls_back_after_old_worker_stop_failure(
    monkeypatch,
):
    import vllm_mlx.server as server
    from vllm_mlx.runtime.audio_worker import bind_audio_worker, run_audio_mlx

    old_worker = _ReplacementWorker("chat-old", fail_stop=True)
    _configure_server_primary(monkeypatch, server, old_worker)
    manager, registry, loaded = _replacement_manager(server, old_worker)
    bind_audio_worker(old_worker)
    try:
        with pytest.raises(RuntimeError, match="old stop failed"):
            await manager.load(
                "chat-new",
                estimated_bytes=1,
                replace_group="assistant",
            )

        assert registry.default_name == "chat-old"
        assert [entry.model_name for entry in registry.list_entries()] == ["chat-old"]
        assert server._engine is old_worker
        assert old_worker.stopped is False
        assert loaded["chat-new"].stopped is True
        assert await run_audio_mlx("stt", "whisper", "infer", lambda: "after") == (
            "after"
        )
        assert old_worker.async_calls == 1
        assert loaded["chat-new"].async_calls == 0
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
