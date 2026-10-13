"""Idle unload releases native state without churning the MLX stream worker."""

import ast
import asyncio
import concurrent.futures
import gc
import logging
import sys
import threading
import time
import types
import weakref
from collections import OrderedDict
from pathlib import Path
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
@pytest.mark.parametrize("missing", ["mlx", "mlx.core", "unrelated_backend"])
async def test_mllm_random_state_import_only_allows_missing_mlx(monkeypatch, missing):
    import builtins

    # The fake loader never uses tensors: keep its worker initializer isolated
    # from the optional runtime before testing the production import guard.
    engine_core = types.ModuleType("rapid_mlx.engine_core")
    engine_core._init_mlx_step_thread = lambda: None
    monkeypatch.setitem(sys.modules, "rapid_mlx.engine_core", engine_core)
    mllm_scheduler = types.ModuleType("rapid_mlx.mllm_scheduler")
    mllm_scheduler.MLLMScheduler = Mock()
    mllm_scheduler.MLLMSchedulerConfig = types.SimpleNamespace(
        __dataclass_fields__={"prefill_step_size": types.SimpleNamespace(default=4096)}
    )
    monkeypatch.setitem(sys.modules, "rapid_mlx.mllm_scheduler", mllm_scheduler)
    scheduler = types.ModuleType("rapid_mlx.scheduler")
    scheduler.SchedulerConfig = Mock()
    monkeypatch.setitem(sys.modules, "rapid_mlx.scheduler", scheduler)
    from rapid_mlx.engine import batched as batched_mod
    from rapid_mlx.models import mllm as mllm_mod
    from rapid_mlx.utils import chat_template_registry

    calls = []
    unavailable = ModuleNotFoundError("simulated missing dependency", name=missing)

    class StopAfterLoadError(Exception):
        pass

    class Model:
        model = object()
        processor = object()

        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            calls.append("load")

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "mlx.core"
            and (globals or {}).get("__name__") == batched_mod.__name__
        ):
            calls.append("random_state_import")
            raise unavailable
        return original_import(name, globals, locals, fromlist, level)

    def stop_after_load(*args):
        calls.append("template")
        raise StopAfterLoadError

    monkeypatch.setattr(mllm_mod, "MLXMultimodalLM", Model)
    monkeypatch.setattr(engine_core, "_init_mlx_step_thread", lambda: None)
    monkeypatch.setattr(
        chat_template_registry, "resolve_chat_template", stop_after_load
    )
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    engine = batched_mod.BatchedEngine("fake/model")
    engine._is_mllm = True

    try:
        if missing == "mlx":
            with pytest.raises(StopAfterLoadError):
                await engine._start_mllm()
            assert calls == ["load", "random_state_import", "template"]
            assert engine._mllm_instance is not None
        else:
            with pytest.raises(ModuleNotFoundError) as caught:
                await engine._start_mllm()
            assert caught.value is unavailable
            assert calls == ["load", "random_state_import"]
            assert engine._mllm_instance is None
            assert engine._model_load_executor is None
        assert engine.is_mllm is True
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


def _source_executor_lifecycle_classes():
    """Execute the actual tensor-free lifecycle methods on every CI host."""
    source = Path(__file__).parents[1] / "rapid_mlx" / "engine_core.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    methods = {
        "EngineCore": {"start", "stop", "_run_on_step_thread"},
        "AsyncEngineCore": {"stop"},
    }
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in methods:
            node.body = [
                method
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name in methods[node.name]
            ]
            assert {method.name for method in node.body} == methods[node.name]
            classes.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            )
        ]
        + classes,
        type_ignores=[],
    )
    namespace = {
        "asyncio": asyncio,
        "concurrent": concurrent,
        "time": time,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["EngineCore"], namespace["AsyncEngineCore"]


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
    engine._engine = None
    engine._mllm_scheduler = Mock(stop=AsyncMock())
    engine._guided_requests_lock = threading.Lock()
    monkeypatch.setattr(engine, "_abort_all_guided_requests", lambda: None)
    engine._model_load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    worker = engine._model_load_executor
    if dual_scheduler:
        EngineCore, AsyncEngineCore = _source_executor_lifecycle_classes()
        core = EngineCore.__new__(EngineCore)
        core._running = False
        core._task = None
        core.scheduler = Mock()
        core.close = Mock()

        async def idle_loop():
            await asyncio.Event().wait()

        core._engine_loop = idle_loop
        await core.start(executor=worker)
        assert core._owns_executor is False
        text_engine = AsyncEngineCore.__new__(AsyncEngineCore)
        text_engine.engine = core
        text_engine._executor = worker
        engine._engine = text_engine
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
            assert core._running is False
            assert core._task is None
            assert core._mlx_executor is None
            core.scheduler._close_batch_generator.assert_called_once()
            core.close.assert_called_once()
            assert worker._shutdown is False

        await engine.stop()
        assert engine._model_load_executor is None
        assert worker._shutdown is True
    finally:
        worker.shutdown(wait=True)


@pytest.mark.asyncio
async def test_text_idle_stop_keeps_stream_worker_until_final_stop(monkeypatch):
    from rapid_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._engine = None
    engine._mllm_scheduler = None
    engine._model_load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    worker = engine._model_load_executor
    engine._tokenizer = None
    engine._processor = None
    engine._mllm_instance = None
    engine._prompt_host_cache = Mock()
    engine._tool_grammar_futures_lock = threading.Lock()
    engine._tool_grammar_futures = set()
    monkeypatch.setattr(engine, "_abort_all_guided_requests", lambda: None)

    class Model:
        pass

    model = Model()
    model_ref = weakref.ref(model)
    engine._model = model
    try:
        del model
        await engine.idle_stop()
        gc.collect()
        assert model_ref() is None
        assert engine._model_load_executor is worker
        assert worker.submit(threading.get_ident).result(timeout=2)

        await engine.stop()
        assert engine._model_load_executor is None
        assert worker._shutdown is True
    finally:
        worker.shutdown(wait=True)


@pytest.mark.asyncio
async def test_text_reload_uses_existing_stream_worker(monkeypatch):
    _stub_text_start_dependencies(monkeypatch)
    from rapid_mlx.engine import batched as batched_mod
    from rapid_mlx.utils import tokenizer as tokenizer_mod

    class StopAfterDispatchError(Exception):
        pass

    engine = batched_mod.BatchedEngine.__new__(batched_mod.BatchedEngine)
    engine._model_name = "fake/model"
    engine._trust_remote_code = False
    engine._scheduler_config = None
    engine._model_load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    worker = engine._model_load_executor
    worker_ident = worker.submit(threading.get_ident).result(timeout=2)
    dispatched_on = []

    def stop_loader(*args, **kwargs):
        dispatched_on.append(threading.get_ident())
        raise StopAfterDispatchError

    monkeypatch.setattr(tokenizer_mod, "load_model_with_fallback", stop_loader)
    try:
        with pytest.raises(StopAfterDispatchError):
            await engine._start_llm()
        assert engine._model_load_executor is worker
        assert dispatched_on == [worker_ident]
    finally:
        engine._model_load_executor.shutdown(wait=True)
        if engine._model_load_executor is not worker:
            worker.shutdown(wait=True)


@pytest.mark.asyncio
async def test_text_load_retries_after_broken_worker(monkeypatch):
    from rapid_mlx.engine import batched as batched_mod
    from rapid_mlx.utils import tokenizer as tokenizer_mod

    class StopAfterDispatchError(Exception):
        pass

    attempts = []

    def initialize_worker():
        attempts.append(threading.get_ident())
        if len(attempts) == 1:
            raise RuntimeError("transient stream initialization failure")

    def stop_loader(*args, **kwargs):
        raise StopAfterDispatchError

    engine_core = types.ModuleType("rapid_mlx.engine_core")
    engine_core.AsyncEngineCore = Mock()
    engine_core.EngineConfig = Mock()
    engine_core._init_mlx_step_thread = initialize_worker
    monkeypatch.setitem(sys.modules, "rapid_mlx.engine_core", engine_core)
    scheduler = types.ModuleType("rapid_mlx.scheduler")
    scheduler.SchedulerConfig = Mock()
    monkeypatch.setitem(sys.modules, "rapid_mlx.scheduler", scheduler)
    monkeypatch.setattr(tokenizer_mod, "load_model_with_fallback", stop_loader)
    engine = batched_mod.BatchedEngine.__new__(batched_mod.BatchedEngine)
    engine._model_name = "fake/model"
    engine._trust_remote_code = False
    engine._scheduler_config = None
    engine._model_load_executor = None
    try:
        with pytest.raises(concurrent.futures.thread.BrokenThreadPool):
            await engine._start_llm()
        with pytest.raises(StopAfterDispatchError):
            await engine._start_llm()
        assert len(attempts) == 2
    finally:
        engine._model_load_executor.shutdown(wait=True)


def _stub_text_start_dependencies(monkeypatch):
    """Stop text startup before model loading without importing MLX."""
    engine_core = types.ModuleType("rapid_mlx.engine_core")
    engine_core.AsyncEngineCore = Mock()
    engine_core.EngineConfig = Mock()
    engine_core._init_mlx_step_thread = lambda: None
    monkeypatch.setitem(sys.modules, "rapid_mlx.engine_core", engine_core)
    scheduler = types.ModuleType("rapid_mlx.scheduler")
    scheduler.SchedulerConfig = Mock()
    monkeypatch.setitem(sys.modules, "rapid_mlx.scheduler", scheduler)


@pytest.mark.asyncio
async def test_text_reload_replaces_shutdown_worker(monkeypatch):
    _stub_text_start_dependencies(monkeypatch)
    from rapid_mlx.engine import batched as batched_mod
    from rapid_mlx.utils import tokenizer as tokenizer_mod

    class StopAfterDispatchError(Exception):
        pass

    engine = batched_mod.BatchedEngine.__new__(batched_mod.BatchedEngine)
    engine._model_name = "fake/model"
    engine._trust_remote_code = False
    engine._scheduler_config = None
    worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    worker.shutdown(wait=True)
    engine._model_load_executor = worker

    def stop_loader(*args, **kwargs):
        raise StopAfterDispatchError

    monkeypatch.setattr(tokenizer_mod, "load_model_with_fallback", stop_loader)
    try:
        with pytest.raises(StopAfterDispatchError):
            await engine._start_llm()
        assert engine._model_load_executor is not worker
        assert engine._model_load_executor.submit(lambda: 1).result(timeout=2) == 1
        assert worker._shutdown is True
    finally:
        engine._model_load_executor.shutdown(wait=True)
        worker.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("build_error", [False, True])
@pytest.mark.parametrize("drain_seconds", [0, 0.01])
async def test_stuck_grammar_build_does_not_block_unload_and_cleans_late_cache(
    monkeypatch, build_error, drain_seconds
):
    from rapid_mlx.api import tool_grammar
    from rapid_mlx.engine import batched as batched_mod

    engine = batched_mod.BatchedEngine.__new__(batched_mod.BatchedEngine)
    engine._is_mllm = False
    engine._engine = None
    engine._mllm_scheduler = None
    engine._model_load_executor = None
    tokenizer = object()
    engine._tokenizer = tokenizer
    engine._processor = None
    engine._model = None
    engine._mllm_instance = None
    engine._tool_grammar_futures_lock = threading.Lock()
    engine._tool_grammar_futures = set()
    monkeypatch.setattr(engine, "_abort_all_guided_requests", lambda: None)
    monkeypatch.setattr(
        batched_mod, "TOOL_GRAMMAR_DRAIN_SECONDS", drain_seconds, raising=False
    )
    released = []
    monkeypatch.setattr(tool_grammar, "release_model_matchers", released.append)
    pending = concurrent.futures.Future()
    assert pending.set_running_or_notify_cancel()
    engine.track_tool_grammar_build(pending)
    try:
        await asyncio.wait_for(engine.stop(), timeout=0.2)
        assert released == [tokenizer]
        assert engine._tokenizer is None and engine._loaded is False
        assert pending.running() and pending in engine._tool_grammar_futures
        if build_error:
            pending.set_exception(RuntimeError("late compiler failure"))
        else:
            pending.set_result(None)
        await asyncio.sleep(0)
        assert released == [tokenizer, tokenizer]
        assert not engine._tool_grammar_futures
    finally:
        if not pending.done():
            pending.set_result(None)
        await asyncio.sleep(0)
