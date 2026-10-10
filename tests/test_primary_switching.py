"""HTTP and lifecycle regressions for destructive, low-memory primary replacement."""

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from rapid_mlx.config import ServerConfig
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.runtime.audio_worker import AudioWorkerDispatcher
from rapid_mlx.runtime.model_registry import ModelEntry, ModelRegistry
from rapid_mlx.runtime.resident_models import ResidentModelManager

GIB = 1024**3


class Engine:
    is_mllm = False

    def __init__(self):
        self.stopped = False

    def get_stats(self):
        return {"num_running": 0, "num_waiting": 0}

    async def pause_generation(self, *args, **kwargs):
        pass

    async def resume_generation(self):
        pass

    async def stop(self):
        self.stopped = True

    async def execute_on_model_worker(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def execute_on_model_worker_sync(self, func, *args, **kwargs):
        return func(*args, **kwargs)


@pytest.fixture
def switching_server(monkeypatch):
    from rapid_mlx import server
    from rapid_mlx.config import server_config
    from rapid_mlx.routes import anthropic, audio, chat, completions, health, responses
    from rapid_mlx.runtime import audio_worker as audio_module

    cfg = ServerConfig()
    monkeypatch.setattr(server_config, "_config", cfg)
    # The production publisher also updates legacy globals. Restore every
    # touched global after the test, including the optional standby lifecycle.
    for name in (
        "_engine",
        "_model_name",
        "_model_alias",
        "_model_path",
        "_served_model_name_set",
        "_enable_auto_tool_choice",
        "_tool_call_parser",
        "_tool_parser_instance",
        "_reasoning_parser",
        "_reasoning_parser_name",
        "_primary_model_lifecycle",
    ):
        monkeypatch.setattr(server, name, getattr(server, name))
    monkeypatch.setattr(server, "_primary_model_lifecycle", None)
    monkeypatch.setattr(server, "_primary_lazy_load", False)
    monkeypatch.setattr(server, "_primary_idle_unload_seconds", 0)
    dispatcher = AudioWorkerDispatcher()
    monkeypatch.setattr(audio_module, "audio_worker", dispatcher)
    old = ModelEntry(Engine(), "chat-old", "repo/chat-old")
    registry = ModelRegistry()
    registry.add(old, is_default=True)
    cfg.model_registry = registry
    dispatcher.bind(old.engine)
    server._set_resident_primary(old)
    started, release = asyncio.Event(), asyncio.Event()
    outcome = SimpleNamespace(error=None)

    async def loader(name, path, performance=None):
        started.set()
        await release.wait()
        if outcome.error is not None:
            raise outcome.error
        return ModelEntry(Engine(), name, path or f"repo/{name}")

    manager = ResidentModelManager(
        registry,
        loader,
        memory_limit_bytes=6 * GIB,
        memory_reader=lambda: 0,
        on_primary_handoff=server._handoff_resident_primary_audio_worker,
        on_primary_changed=server._set_resident_primary,
    )
    cfg.residency_manager = manager
    manager.register_primary(old, estimated_bytes=4 * GIB)
    app = FastAPI()
    for router in (
        health.probe_router,
        chat.router,
        completions.router,
        anthropic.router,
        responses.router,
        audio.router,
    ):
        app.include_router(router)
    install_exception_handlers(app)
    # Reach the real STT dispatcher without audio weights or upload decoding.
    monkeypatch.setattr("rapid_mlx.audio.probe.require_mlx_audio_stt", lambda: None)
    monkeypatch.setattr(
        audio,
        "_stt_engine",
        SimpleNamespace(
            model_name="mlx-community/whisper-large-v3-turbo",
            transcribe=lambda *a, **kw: {"text": "ok"},
        ),
    )
    return SimpleNamespace(
        cfg=cfg,
        manager=manager,
        old=old,
        dispatcher=dispatcher,
        started=started,
        release=release,
        outcome=outcome,
        app=app,
    )


def start_switch(env):
    return asyncio.create_task(
        env.manager.load(
            "chat-new",
            estimated_bytes=4 * GIB,
            replace_group="assistant",
            replace_mode="wait",
            memory_policy="evict_first_if_needed",
            resolved_group="assistant",
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_blocked_low_memory_switch_has_typed_http_contract(
    switching_server, outcome
):
    env = switching_server
    task = start_switch(env)
    try:
        await asyncio.wait_for(env.started.wait(), timeout=2)
        assert env.old.engine.stopped
        assert env.cfg.engine is None
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=env.app), base_url="http://test"
        ) as client:
            health = await client.get("/health")
            assert health.json()["state"] == "switching"
            assert health.json()["ready"] is False
            requests = [
                client.get("/health/ready"),
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "chat-old",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                ),
                client.post(
                    "/v1/completions", json={"model": "chat-new", "prompt": "hi"}
                ),
                client.post(
                    "/v1/messages",
                    json={
                        "model": "chat-old",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                ),
                client.post("/v1/responses", json={"model": "chat-new", "input": "hi"}),
                client.post(
                    "/v1/audio/transcriptions",
                    data={"model": "whisper-large-v3-turbo"},
                    files={"file": ("speech.wav", b"audio", "audio/wav")},
                ),
                client.post(
                    "/v1/audio/translations",
                    data={"model": "whisper-large-v3-turbo"},
                    files={"file": ("speech.wav", b"audio", "audio/wav")},
                ),
                client.post(
                    "/v1/audio/speech",
                    json={"model": "kokoro", "input": "hello", "voice": "af_heart"},
                ),
            ]
            results = await asyncio.wait_for(asyncio.gather(*requests), timeout=3)
            for response in results:
                assert response.status_code == 503, response.text
                assert response.json()["error"]["code"] == "model_switching", (
                    response.text
                )
                assert response.headers["retry-after"] == "5"
            assert results[3].json()["type"] == "error"
            if outcome == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                if outcome == "failure":
                    env.outcome.error = RuntimeError("load failed")
                env.release.set()
                if outcome == "failure":
                    with pytest.raises(RuntimeError, match="load failed"):
                        await task
                else:
                    await task
            # Every terminal outcome releases the audio handoff and the
            # switching marker. Failure/cancel cannot revive an evicted engine.
            assert env.manager.primary_switching is False
            assert env.dispatcher.handoff_in_progress is False
            if outcome != "success":
                assert env.dispatcher._bound_worker() is None
                env.dispatcher.bind(Engine())
            assert (
                env.dispatcher.execute_sync("stt", "whisper", "infer", lambda: "ok")
                == "ok"
            )
            env.dispatcher.bind(None)
            ready = await client.get("/health/ready")
            assert ready.status_code == (200 if outcome == "success" else 503)
            assert (await client.get("/health")).json()["state"] != "switching"
            if outcome != "success":
                assert env.cfg.engine is None
                assert ready.json()["error"].get("code") != "model_switching"
    finally:
        env.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_switch_preserves_explicit_secondary_routing(switching_server):
    from rapid_mlx.service.helpers import _validate_model_name, get_engine

    env = switching_server
    secondary = ModelEntry(Engine(), "secondary", "repo/secondary", aliases={"spare"})
    env.cfg.model_registry.add(secondary)
    task = start_switch(env)
    try:
        await asyncio.wait_for(env.started.wait(), timeout=2)
        for name in ("secondary", "repo/secondary", "spare"):
            _validate_model_name(name)
            assert get_engine(name) is secondary.engine
        # Even if a registry mutation auto-promotes an unrelated model, the
        # default and absent names must never silently use it during switching.
        env.cfg.model_registry._default = "secondary"
        from fastapi import HTTPException

        for name in (None, "default", "chat-old", "chat-new", "unknown"):
            with pytest.raises(HTTPException) as error:
                get_engine(name)
            assert error.value.detail["error"]["code"] == "model_switching"
    finally:
        env.release.set()
        await task


@pytest.mark.asyncio
async def test_keep_both_load_keeps_primary_ready(switching_server):
    from rapid_mlx.service.helpers import get_engine

    env = switching_server
    env.manager.memory_limit_bytes = 12 * GIB
    task = start_switch(env)
    try:
        await asyncio.wait_for(env.started.wait(), timeout=2)
        assert env.manager.primary_switching is False
        assert env.dispatcher.handoff_in_progress is False
        assert env.cfg.ready is True
        assert get_engine() is env.old.engine
        assert env.old.engine.stopped is False
    finally:
        env.release.set()
        await task


@pytest.mark.asyncio
async def test_audio_wrappers_preserve_transient_error_after_entry_race(
    switching_server,
):
    from fastapi import HTTPException

    from rapid_mlx.runtime.audio_worker import run_audio_mlx, run_audio_mlx_sync

    env = switching_server
    handoff = env.dispatcher.begin_handoff()
    try:
        # The handoff can start after the HTTP dependency returned. Both
        # dispatcher entry points still classify that race as a retryable 503.
        with pytest.raises(HTTPException) as error:
            await run_audio_mlx("stt", "whisper", "infer", lambda: None)
        assert error.value.status_code == 503
        assert error.value.detail["error"]["code"] == "model_switching"
        with pytest.raises(HTTPException) as error:
            run_audio_mlx_sync("tts", "kokoro", "infer", lambda: None)
        assert error.value.status_code == 503
        assert error.value.detail["error"]["code"] == "model_switching"
        assert env.dispatcher.snapshot() == []
    finally:
        handoff.rollback()


@pytest.mark.asyncio
async def test_primary_stop_failure_clears_switching(switching_server):
    env = switching_server

    async def failed_stop():
        raise RuntimeError("stop failed")

    env.old.engine.stop = failed_stop
    with pytest.raises(RuntimeError, match="stop failed"):
        await start_switch(env)
    assert env.manager.primary_switching is False
    assert env.dispatcher.handoff_in_progress is False
    assert not env.started.is_set()


@pytest.mark.asyncio
async def test_switching_lasts_until_audio_commit(switching_server, monkeypatch):
    from fastapi import HTTPException

    from rapid_mlx.routes.health import health, health_ready
    from rapid_mlx.service.helpers import get_engine

    env = switching_server
    published, commit = asyncio.Event(), asyncio.Event()
    evict = env.manager._evict_for_locked

    async def delay_final_admission(required, **kwargs):
        await evict(required, **kwargs)
        if required == 0:
            published.set()
            await commit.wait()

    monkeypatch.setattr(env.manager, "_evict_for_locked", delay_final_admission)
    task = start_switch(env)
    try:
        await asyncio.wait_for(env.started.wait(), timeout=2)
        env.release.set()
        await asyncio.wait_for(published.wait(), timeout=2)
        assert env.cfg.ready is True  # Publisher ran; audio is not committed yet.
        assert (await health())["ready"] is False
        for name in (None, "default", "chat-new"):
            with pytest.raises(HTTPException) as error:
                get_engine(name)
            assert error.value.detail["error"]["code"] == "model_switching"
        with pytest.raises(HTTPException) as error:
            await health_ready()
        assert error.value.detail["error"]["code"] == "model_switching"
    finally:
        commit.set()
        env.release.set()
        await task
    assert env.manager.primary_switching is False
    assert get_engine() is env.cfg.engine


@pytest.mark.asyncio
async def test_audio_switch_guard_keeps_auth_first(switching_server):
    env = switching_server
    env.cfg.api_key = "test-key"
    task = start_switch(env)
    try:
        await asyncio.wait_for(env.started.wait(), timeout=2)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=env.app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/audio/speech", json={"model": "kokoro", "input": "hi"}
            )
            assert response.status_code == 401
            response = await client.post(
                "/v1/audio/speech",
                json={"model": "kokoro", "input": "hi"},
                headers={"Authorization": "Bearer test-key"},
            )
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "model_switching"
    finally:
        env.release.set()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", ["asr_upload", "alignment_capacity"])
async def test_audio_rechecks_handoff_before_residency_wait(
    switching_server, monkeypatch, barrier
):
    from rapid_mlx.routes import audio

    env = switching_server
    reached = asyncio.Event()
    upload_release = asyncio.Event()
    capacity_release = threading.Event()
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(audio, "_stt_engine", None)
    monkeypatch.setattr(audio, "_aligner_engine", None)
    weight_loads = []

    def forbidden_weight_load(*args, **kwargs):
        weight_loads.append(args)
        raise AssertionError("switching must reject before audio weight load")

    monkeypatch.setattr(
        "rapid_mlx.audio.stt.STTEngine",
        lambda name: SimpleNamespace(load=forbidden_weight_load),
    )
    monkeypatch.setattr(audio, "_load_aligner_blocking", forbidden_weight_load)

    async def upload(file, target):
        target.write(b"audio")
        if barrier == "asr_upload":
            reached.set()
            await upload_release.wait()

    def capacity(name):
        loop.call_soon_threadsafe(reached.set)
        assert capacity_release.wait(timeout=5)
        return SimpleNamespace(requested_bytes=GIB, source="test")

    monkeypatch.setattr(audio, "_stream_upload_to_tempfile", upload)
    monkeypatch.setattr("rapid_mlx.runtime.role_capacity.alignment_capacity", capacity)
    data = {"model": "whisper-large-v3-turbo"}
    if barrier == "alignment_capacity":
        data = {"model": audio.DEFAULT_ALIGNER_ALIAS, "text": "hello"}
    task = None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=env.app), base_url="http://test"
    ) as client:
        request = asyncio.create_task(
            client.post(
                "/v1/audio/transcriptions",
                data=data,
                files={"file": ("speech.wav", b"audio", "audio/wav")},
            )
        )
        try:
            # The route dependency has passed, then upload/capacity pauses.
            await asyncio.wait_for(reached.wait(), timeout=2)
            task = start_switch(env)
            await asyncio.wait_for(env.started.wait(), timeout=2)
            upload_release.set()
            capacity_release.set()
            response = await asyncio.wait_for(asyncio.shield(request), timeout=1)
            assert response.status_code == 503, response.text
            assert response.json()["error"]["code"] == "model_switching"
            assert response.headers["retry-after"] == "5"
            assert env.dispatcher.snapshot() == []
            assert env.manager._roles == {}
            assert weight_loads == []
            assert audio._stt_engine is None
            assert audio._aligner_engine is None
            assert not task.done()  # No need to wait for primary loading.
        finally:
            upload_release.set()
            capacity_release.set()
            env.release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await asyncio.gather(request, return_exceptions=True)
