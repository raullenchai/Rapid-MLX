"""Idle unload releases native state without churning the MLX stream worker."""

import asyncio
import concurrent.futures
import gc
import threading
import weakref
from collections import OrderedDict
from unittest.mock import AsyncMock, Mock

import pytest

from rapid_mlx.runtime.primary_lifecycle import PrimaryModelLifecycle


@pytest.mark.asyncio
async def test_unload_drains_grammar_build_before_releasing_matchers(monkeypatch):
    from rapid_mlx.api import tool_grammar
    from rapid_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._engine = None
    engine._mllm_scheduler = None
    engine._model_load_executor = None
    engine._tokenizer = object()
    engine._processor = None
    engine._model = None
    engine._mllm_instance = None
    engine._tool_grammar_futures_lock = threading.Lock()
    engine._tool_grammar_futures = set()
    monkeypatch.setattr(engine, "_abort_all_guided_requests", lambda: None)
    released = []
    monkeypatch.setattr(
        tool_grammar,
        "release_model_matchers",
        lambda tokenizer: released.append(tokenizer),
    )

    pending = concurrent.futures.Future()
    engine.track_tool_grammar_build(pending)
    unloading = asyncio.create_task(engine.stop())
    await asyncio.sleep(0)
    assert not unloading.done()
    assert released == []

    pending.set_result(None)
    await unloading
    assert len(released) == 1
    assert engine._tool_grammar_futures == set()


@pytest.mark.asyncio
async def test_cancelled_tool_build_cannot_recache_after_unload(monkeypatch):
    from rapid_mlx.api import tool_grammar
    from rapid_mlx.engine.batched import BatchedEngine
    from rapid_mlx.routes import chat

    class Tokenizer:
        pass

    class Matcher:
        def __init__(self, tokenizer, grammar):
            pass

        def get_error(self):
            return ""

        def deep_copy(self):
            return self

    tokenizer = Tokenizer()
    lltok = object()
    setattr(tokenizer, tool_grammar._LLTOKENIZER_ATTR, lltok)
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._engine = None
    engine._mllm_scheduler = None
    engine._model_load_executor = None
    engine._tokenizer = tokenizer
    engine._processor = None
    engine._model = None
    engine._mllm_instance = None
    engine._tool_grammar_futures_lock = threading.Lock()
    engine._tool_grammar_futures = set()
    monkeypatch.setattr(engine, "_abort_all_guided_requests", lambda: None)
    monkeypatch.setattr(tool_grammar, "LLMatcher", Matcher)
    monkeypatch.setattr(chat, "_tool_grammar_eligible", lambda *_: True)
    monkeypatch.setattr(chat, "_try_admit_tool_grammar_build", lambda: True)
    monkeypatch.setattr(chat, "_release_tool_grammar_build", lambda: None)

    entered = threading.Event()
    resume = threading.Event()

    def build(*args):
        entered.set()
        assert resume.wait(5)
        tool_grammar.get_request_matcher(lltok, "schema")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(chat, "_get_tool_grammar_build_executor", lambda: pool)
        monkeypatch.setattr(chat, "_maybe_build_tool_grammar_processor", build)
        request = asyncio.create_task(
            chat._offload_tool_grammar_build(engine, None, None)
        )
        assert await asyncio.to_thread(entered.wait, 5)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        unloading = asyncio.create_task(engine.stop())
        await asyncio.sleep(0)
        assert not unloading.done()
        resume.set()
        await unloading

    assert all(key[0] != id(lltok) for key in tool_grammar._compiled_matcher_cache)


@pytest.mark.asyncio
async def test_mllm_load_materializes_random_state_on_model_worker(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from rapid_mlx import engine_core
    from rapid_mlx.engine import batched as batched_mod
    from rapid_mlx.models import mllm as mllm_mod
    from rapid_mlx.utils import chat_template_registry

    calls = []

    class StopAfterLoadError(Exception):
        pass

    class Model:
        model = object()
        processor = object()

        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            calls.append(("load", threading.get_ident()))

    original_eval = mx.eval

    def record_eval(*arrays):
        calls.append(("eval", threading.get_ident()))
        return original_eval(*arrays)

    def stop_after_load(*args):
        raise StopAfterLoadError

    monkeypatch.setattr(mllm_mod, "MLXMultimodalLM", Model)
    monkeypatch.setattr(engine_core, "_init_mlx_step_thread", lambda: None)
    monkeypatch.setattr(mx, "eval", record_eval)
    monkeypatch.setattr(
        chat_template_registry, "resolve_chat_template", stop_after_load
    )
    engine = batched_mod.BatchedEngine("fake/model")
    engine._is_mllm = True

    try:
        with pytest.raises(StopAfterLoadError):
            await engine._start_mllm()
        assert [name for name, _ in calls] == ["load", "eval"]
        assert calls[0][1] == calls[1][1]
    finally:
        if engine._model_load_executor is not None:
            engine._model_load_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_idle_lifecycle_uses_idle_stop_when_available():
    class Engine:
        _loaded = True

        def __init__(self):
            self.calls = []

        async def idle_stop(self):
            self.calls.append("idle_stop")
            self._loaded = False

        async def stop(self):
            self.calls.append("stop")
            self._loaded = False

    engine = Engine()
    lifecycle = PrimaryModelLifecycle(engine, idle_unload_seconds=1, clock=lambda: 2)
    lifecycle._last_activity = 0

    assert await lifecycle.evict_if_idle() is True
    assert engine.calls == ["idle_stop"]


@pytest.mark.asyncio
async def test_idle_lifecycle_falls_back_to_stop():
    class Engine:
        _loaded = True

        async def stop(self):
            self._loaded = False

    engine = Engine()
    lifecycle = PrimaryModelLifecycle(engine, idle_unload_seconds=1, clock=lambda: 2)
    lifecycle._last_activity = 0

    assert await lifecycle.evict_if_idle() is True
    assert engine._loaded is False


@pytest.mark.asyncio
@pytest.mark.parametrize("dual_scheduler", [False, True])
async def test_mllm_idle_stop_releases_model_but_reuses_worker(
    monkeypatch, dual_scheduler
):
    from rapid_mlx.api import tool_grammar
    from rapid_mlx.engine.batched import BatchedEngine

    class Model:
        pass

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    text_engine = Mock(stop=AsyncMock(), engine=Mock(close=Mock()))
    engine._engine = text_engine if dual_scheduler else None
    engine._mllm_scheduler = Mock(stop=AsyncMock())
    engine._guided_requests_lock = threading.Lock()
    monkeypatch.setattr(engine, "_abort_all_guided_requests", lambda: None)
    engine._model_load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    worker = engine._model_load_executor
    model = Model()
    model_ref = weakref.ref(model)
    engine._model = model
    engine._mllm_instance = model
    engine._tokenizer = None

    class Tokenizer:
        pass

    tokenizer = Tokenizer()
    lltok = object()
    setattr(tokenizer, tool_grammar._LLTOKENIZER_ATTR, lltok)
    matcher_key = (id(lltok), "schema")
    monkeypatch.setattr(
        tool_grammar,
        "_compiled_matcher_cache",
        OrderedDict({matcher_key: (lltok, object(), 6)}),
    )
    monkeypatch.setattr(tool_grammar, "_compiled_matcher_cache_bytes", 6)
    engine._processor = Mock(tokenizer=tokenizer)
    engine._prompt_host_cache = Mock()
    engine._loaded = True

    try:
        del model
        await engine.idle_stop()
        gc.collect()

        assert model_ref() is None
        assert engine._model_load_executor is worker
        assert worker.submit(lambda: 1).result(timeout=2) == 1
        assert engine._mllm_scheduler is None
        assert matcher_key not in tool_grammar._compiled_matcher_cache
        assert tool_grammar._compiled_matcher_cache_bytes == 0
        if dual_scheduler:
            text_engine.stop.assert_awaited_once()
            text_engine.engine.close.assert_called_once()

        await engine.stop()
        assert engine._model_load_executor is None
        assert worker._shutdown is True
    finally:
        worker.shutdown(wait=True)
