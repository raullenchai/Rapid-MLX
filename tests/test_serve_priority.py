"""CPU-only priority contracts, exercising tensor-free production methods."""

import ast
import asyncio
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI

from rapid_mlx.middleware.priority import PriorityMiddleware, request_priority
from rapid_mlx.request import Request, SamplingParams
from tests.test_mllm_prompt_admission_cpu import _load_function


def _scheduler(requests, now, monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: now)
    namespace = {"time": time}
    methods = {
        name: _load_function("rapid_mlx/scheduler.py", name, namespace, "Scheduler")
        for name in (
            "_waiting_priority",
            "_select_waiting_request",
            "_pop_waiting_for_admission",
        )
    }
    scheduler = type("Selector", (), methods)()
    scheduler.waiting = deque(requests)
    scheduler.running = {}
    scheduler.config = SimpleNamespace(scheduling_max_deferrals=8)
    scheduler._request_is_generator_compatible = lambda r: True
    scheduler._validated_prompt_tail_cost = lambda r: len(r.prompt)
    scheduler._prefill_leader_for = lambda r: None
    return scheduler


def _request(name, priority="interactive"):
    request = Request(name, name, SamplingParams(), serve_priority=priority)
    request._arrival_monotonic = 100.0
    return request


@pytest.mark.parametrize("ranked", [False, True])
@pytest.mark.parametrize("now", [130.0, 130.001])
def test_priority_aging_fifo_and_running_untouched(ranked, now, monkeypatch):
    background = _request("background", "background")
    other = _request("other", "background")
    interactive = _request("interactive")
    scheduler = _scheduler([background, other, interactive], now, monkeypatch)
    # Equal costs preserve FIFO in the ranked path too; priority precedes cost.
    scheduler._validated_prompt_tail_cost = lambda r: 1
    running = _request("running", "background")
    scheduler.running = {running.request_id: running}
    selected = (
        scheduler._select_waiting_request()[0]
        if ranked
        else scheduler._pop_waiting_for_admission()
    )
    assert selected is (background if now > 130 else interactive)
    assert scheduler.running == {running.request_id: running}


@pytest.mark.parametrize("ranked", [False, True])
def test_default_fifo_and_priority_before_cost_and_deferral(ranked, monkeypatch):
    first, second = _request("a"), _request("b")
    scheduler = _scheduler([first, second], 101, monkeypatch)
    select = lambda: (
        scheduler._select_waiting_request()[0]
        if ranked
        else scheduler._pop_waiting_for_admission()
    )
    assert select() is first
    background = _request("x", "background")
    background._admission_deferrals = 8
    scheduler.waiting = deque([background, second])
    assert select() is second
    scheduler.waiting = deque([background])
    assert select() is background
    scheduler.waiting.clear()
    assert scheduler._select_waiting_request() is None
    assert scheduler._pop_waiting_for_admission() is None
    scheduler.waiting.extend([first, second])
    if ranked:
        scheduler._request_is_generator_compatible = lambda r: r is second
    else:
        first.prefix_wait_leader = "running"
        scheduler.running = {"running": SimpleNamespace(num_output_tokens=0)}
    assert select() is second


@pytest.mark.parametrize("value", [None, b"unknown", b"INTERACTIVE", b"BaCkGrOuNd"])
@pytest.mark.asyncio
async def test_header_context_and_engine_constructors(value):
    expected = "background" if value == b"BaCkGrOuNd" else "interactive"
    captured = []

    class AdmittedError(Exception):
        pass

    def admit(request):
        captured.append(request)
        raise AdmittedError

    async def endpoint(scope, receive, send):
        assert request_priority.get() == expected
        namespace = {
            "Request": Request,
            "SamplingParams": SamplingParams,
            "request_priority": request_priority,
            "time": time,
            "RequestOutputCollector": Mock(),
            "RequestStreamState": Mock(),
            "asyncio": asyncio,
        }
        engine = SimpleNamespace(
            scheduler=SimpleNamespace(add_request=admit),
            _hybrid_throttle=False,
            _output_collectors={},
            _stream_states={},
            _finished_events={},
            _mlx_executor=None,
            config=SimpleNamespace(stream_interval=1),
        )
        for name in ("add_request", "generate_batch_sync"):
            method = _load_function(
                "rapid_mlx/engine_core.py", name, namespace, "EngineCore"
            )
            with pytest.raises(AdmittedError):
                if name == "add_request":
                    await method(engine, "hello", request_id="text", images=["image"])
                else:
                    method(engine, ["hello"])
        assert all(r.serve_priority == expected for r in captured)

    # Execute the production registration with an isolated ASGI application.
    app = FastAPI()
    tree = ast.parse(Path("rapid_mlx/server.py").read_text())
    for path in ("rapid_mlx/server.py", "rapid_mlx/engine_core.py"):
        source = ast.parse(Path(path).read_text())
        imports = [
            n
            for n in source.body
            if isinstance(n, ast.ImportFrom) and n.module == "middleware.priority"
        ]
        exec(
            compile(ast.Module(imports, []), path, "exec"), {"__package__": "rapid_mlx"}
        )
    registration = next(
        n
        for n in tree.body
        if isinstance(n, ast.Expr)
        and ast.unparse(n) == "app.add_middleware(PriorityMiddleware)"
    )
    exec(
        compile(ast.Module([registration], []), "rapid_mlx/server.py", "exec"),
        {"app": app, "PriorityMiddleware": PriorityMiddleware},
    )
    assert any(m.cls is PriorityMiddleware for m in app.user_middleware)
    headers = [] if value is None else [(b"X-RaPiD-PrIoRiTy", value)]
    middleware = PriorityMiddleware(endpoint)
    await middleware({"type": "http", "headers": headers}, None, None)
    assert len(captured) == 2
    assert request_priority.get() == "interactive"

    # Non-HTTP scopes bypass parsing; failures must still restore the context.
    async def fail(scope, *args):
        if scope["type"] == "lifespan":
            return
        raise AdmittedError

    with pytest.raises(AdmittedError):
        await PriorityMiddleware(fail)({"type": "http", "headers": headers}, None, None)
    assert request_priority.get() == "interactive"
    await PriorityMiddleware(fail)({"type": "lifespan"}, None, None)
