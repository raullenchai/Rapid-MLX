# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.request import RequestOutput
from rapid_mlx.spec_decode.config import (
    SpeculativeConfigError,
    parse_speculative_config,
)
from rapid_mlx.speculative.tensorfold_qwen27_server import (
    TensorFoldRequestProvider,
    generation_kwargs,
    render_prompt,
    validate_http_request,
)


def test_dflash_tensorfold_config_is_explicit_and_method_scoped() -> None:
    config = parse_speculative_config(
        '{"method":"dflash","backend":"tensorfold","model":"/pinned/drafter"}'
    )
    assert config is not None and config.backend == "tensorfold"
    native = parse_speculative_config(
        '{"method":"dflash","backend":"native","model":"x"}'
    )
    assert native is not None and native.backend == "native"
    with pytest.raises(SpeculativeConfigError):
        parse_speculative_config('{"method":"mtp","backend":"tensorfold"}')


def test_http_gate_rejects_unqualified_features_and_maps_sampling() -> None:
    from fastapi import HTTPException

    request = SimpleNamespace(
        tools=[{"type": "function"}],
        response_format=None,
        messages=[SimpleNamespace(content="hi")],
        repetition_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        logit_bias=None,
    )
    with pytest.raises(HTTPException) as exc:
        validate_http_request(request)
    assert exc.value.status_code == 400

    request = SimpleNamespace(
        top_k=20,
        min_p=0.05,
        seed=42,
        stop=["END"],
        messages=[SimpleNamespace(content="hi")],
        tools=None,
        response_format=None,
        repetition_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        logit_bias=None,
    )
    validate_http_request(request)
    assert generation_kwargs(
        max_tokens=9, temperature=0.7, top_p=0.9, request=request
    ) == {
        "max_tokens": 9,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "seed": 42,
        "stop": ["END"],
    }

    request.messages = [SimpleNamespace(content=[{"type": "image_url"}])]
    with pytest.raises(HTTPException) as exc:
        validate_http_request(request)
    assert exc.value.status_code == 400

    request.messages = [SimpleNamespace(content="hi")]
    request.reasoning_max_tokens = 128
    with pytest.raises(HTTPException) as exc:
        validate_http_request(request)
    assert "reasoning_max_tokens" in exc.value.detail


def test_provider_preserves_exact_token_ids_and_request_outputs(
    monkeypatch, tmp_path
) -> None:
    class Cancellation:
        def cancel(self):
            pass

        def check(self):
            pass

    class ChatJob:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            import queue

            self.chunks = queue.Queue()
            self.cached_tokens = 3
            self.error = None
            self.stream = SimpleNamespace(finish_reason="stop")

    class StopPolicy:
        def __init__(self, fields, tokenizer, lock, eos_ids):
            self.ignore_eos = False
            self.eos_ids = eos_ids
            self.strings = tuple(fields.get("stop") or ())

        def visible(self, text, partial=False):
            return text.split("END", 1)[0]

    class SuffixLookupProposer:
        def __init__(self, min_match):
            self.min_match = min_match

    modules = {
        "tensorfold.engine.lane_engine": {"SuffixLookupProposer": SuffixLookupProposer},
        "tensorfold.server.cancellation": {"Cancellation": Cancellation},
        "tensorfold.server.scheduler": {"ChatJob": ChatJob},
        "tensorfold.server.stopping": {"StopPolicy": StopPolicy},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    class Tokenizer:
        def encode(self, text):
            return [10, 11]

        def decode(self, ids):
            return "".join({21: "A", 22: "B"}[i] for i in ids)

    class Scheduler:
        def submit(self, job):
            job.chunks.put([21, 22])
            job.chunks.put(None)

    app = SimpleNamespace(
        tokenizer=Tokenizer(),
        tokenizer_lock=__import__("threading").Lock(),
        stop_ids=frozenset({22}),
        min_match=3,
        scheduler=Scheduler(),
        _resolve_sampling=lambda fields, temperature, prompt: (fields, temperature),
    )
    audit = tmp_path / "tokens.ndjson"
    provider = TensorFoldRequestProvider(
        SimpleNamespace(_app=app), audit_path=str(audit)
    )
    chunks = list(provider.stream_generate(None, None, "prompt", max_tokens=8))

    assert [chunk.token for chunk in chunks] == [21, 22]
    assert [chunk.text for chunk in chunks] == ["A", ""]
    assert provider.last_token_ids == [21, 22]
    assert provider.last_outputs[-1].finished
    assert provider.last_outputs[-1].cached_tokens == 3
    import json

    record = json.loads(audit.read_text())
    assert record["token_ids"] == [21, 22]
    assert len(record["token_sha256"]) == 64


def test_provider_closed_timeout_error_and_generate(monkeypatch) -> None:
    # Reuse the provider fixture above with a scheduler that first times out,
    # then reports a worker error. This covers the cancellation polling path.
    import queue

    class Cancellation:
        def __init__(self):
            self.checks = 0

        def cancel(self):
            pass

        def check(self):
            self.checks += 1
            raise RuntimeError("worker stopped")

    class ChatJob:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.chunks = queue.Queue()
            self.error = None
            self.stream = None
            self.cached_tokens = 0

    class StopPolicy:
        def __init__(self, *_args):
            self.ignore_eos = False
            self.eos_ids = set()
            self.strings = ()

        def visible(self, text, partial=False):
            return text

    class Proposer:
        def __init__(self, min_match):
            self.min_match = min_match

    for name, attrs in {
        "tensorfold.engine.lane_engine": {"SuffixLookupProposer": Proposer},
        "tensorfold.server.cancellation": {"Cancellation": Cancellation},
        "tensorfold.server.scheduler": {"ChatJob": ChatJob},
        "tensorfold.server.stopping": {"StopPolicy": StopPolicy},
    }.items():
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    backend = SimpleNamespace(_app=None)
    provider = TensorFoldRequestProvider(backend)
    with pytest.raises(RuntimeError, match="closed"):
        list(provider._outputs("x"))
    tokenizer = SimpleNamespace(encode=lambda _p: [1])
    app = SimpleNamespace(
        tokenizer=tokenizer,
        tokenizer_lock=__import__("threading").Lock(),
        stop_ids=set(),
        min_match=1,
        scheduler=SimpleNamespace(submit=lambda _j: None),
        _resolve_sampling=lambda *a: {},
    )
    provider = TensorFoldRequestProvider(SimpleNamespace(_app=app))
    with pytest.raises(RuntimeError, match="worker stopped"):
        list(provider.generate(None, None, "x", max_tokens=1))


def test_provider_timeout_worker_error_and_generate_success(monkeypatch) -> None:
    import queue
    import threading

    class Cancellation:
        def cancel(self):
            pass

        def check(self):
            pass

    class ChatJob:
        next_error = None

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.chunks = queue.Queue()
            self.error = self.next_error
            self.stream = None
            self.cached_tokens = 0

    class StopPolicy:
        def __init__(self, *_args):
            self.ignore_eos = False
            self.eos_ids = set()
            self.strings = ()

        def visible(self, text, partial=False):
            return text

    class Proposer:
        def __init__(self, min_match):
            self.min_match = min_match

    for name, attrs in {
        "tensorfold.engine.lane_engine": {"SuffixLookupProposer": Proposer},
        "tensorfold.server.cancellation": {"Cancellation": Cancellation},
        "tensorfold.server.scheduler": {"ChatJob": ChatJob},
        "tensorfold.server.stopping": {"StopPolicy": StopPolicy},
    }.items():
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    class Scheduler:
        def submit(self, job):
            def finish():
                job.chunks.put([2])
                job.chunks.put(None)

            threading.Timer(0.06, finish).start()

    tokenizer = SimpleNamespace(
        encode=lambda _p: [1], decode=lambda ids: "x" * len(ids)
    )
    app = SimpleNamespace(
        tokenizer=tokenizer,
        tokenizer_lock=threading.Lock(),
        stop_ids=set(),
        min_match=1,
        scheduler=Scheduler(),
        _resolve_sampling=lambda *_a: {},
    )
    provider = TensorFoldRequestProvider(SimpleNamespace(_app=app))
    result = provider.generate(None, None, "x", max_tokens=1)
    assert result.text == "x"
    assert result.generation_tokens == 1

    ChatJob.next_error = RuntimeError("job failed")
    with pytest.raises(RuntimeError, match="job failed"):
        list(provider._outputs("x", max_tokens=1))

    terminal = RequestOutput(request_id="r", finished=True)
    provider._outputs = lambda *_a, **_k: iter([terminal])
    assert list(provider.stream_generate(None, None, "x")) == []


def test_render_prompt_and_server_bootstrap(monkeypatch) -> None:
    from rapid_mlx.speculative import tensorfold_qwen27_server as server

    request = SimpleNamespace(
        messages=[
            SimpleNamespace(model_dump=lambda **_k: {"role": "user", "content": "hi"})
        ]
    )
    processor = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: (messages, kwargs)
    )
    rendered = render_prompt(processor, None, request, enable_thinking=False)
    assert rendered[1]["add_generation_prompt"] is True

    with pytest.raises(RuntimeError, match="pinned local"):
        server.run_tensorfold_qwen27_server(
            main_model_repo="a",
            main_model_revision="x",
            drafter_repo="b",
            drafter_revision=None,
            host="h",
            port=1,
            port_explicit=True,
            served_model_name="m",
            default_max_tokens=1,
            cors_origins=[],
            uvicorn_log_level="info",
        )
    with pytest.raises(RuntimeError, match="tool parsing"):
        server.run_tensorfold_qwen27_server(
            main_model_repo="a",
            main_model_revision=None,
            drafter_repo="b",
            drafter_revision=None,
            host="h",
            port=1,
            port_explicit=True,
            served_model_name="m",
            default_max_tokens=1,
            cors_origins=[],
            uvicorn_log_level="info",
            tool_call_parser="qwen",
        )

    closed = []
    captured = {}
    fake_backend = SimpleNamespace(
        _app=SimpleNamespace(tokenizer=object()), close=lambda: closed.append(True)
    )
    monkeypatch.setattr(
        server.TensorFoldQwen27Backend, "load", lambda *a, **k: fake_backend
    )

    class App:
        def on_event(self, _name):
            return lambda fn: (fn(), fn)[1]

    dflash_server = ModuleType("rapid_mlx.speculative.dflash.server")
    dflash_server._build_app = lambda **kwargs: (captured.update(kwargs), App())[1]
    monkeypatch.setitem(
        sys.modules, "rapid_mlx.speculative.dflash.server", dflash_server
    )
    uvicorn = ModuleType("rapid_mlx._uvicorn")
    uvicorn.run_uvicorn = lambda app, **kwargs: captured.update(run=kwargs)
    monkeypatch.setitem(sys.modules, "rapid_mlx._uvicorn", uvicorn)
    server.run_tensorfold_qwen27_server(
        main_model_repo="a",
        main_model_revision=None,
        drafter_repo="b",
        drafter_revision=None,
        host="h",
        port=1,
        port_explicit=True,
        served_model_name="m",
        default_max_tokens=1,
        cors_origins=[],
        uvicorn_log_level="info",
        max_concurrent_requests=256,
    )
    assert closed == [True]
    assert captured["max_concurrent_requests"] == 1
    assert (
        captured["runtime_status_extra"]["profile"]["compatibility"]["state"] == "ready"
    )
    assert captured["run"]["port"] == 1


def test_tensorfold_rejects_second_in_flight_request_with_retry_after() -> None:
    from fastapi.testclient import TestClient

    from rapid_mlx.speculative.dflash.server import _build_app
    from rapid_mlx.speculative.tensorfold_qwen27_server import (
        _MAX_CONCURRENT_REQUESTS,
    )

    app = _build_app(
        model=None,
        processor=SimpleNamespace(),
        runtime=SimpleNamespace(
            algorithm="dflash2",
            drafter_repo="pinned-drafter",
            target_revision="a" * 40,
            drafter_revision="b" * 40,
        ),
        served_model_name="qwen27-tf",
        default_max_tokens=8,
        cors_origins=[],
        max_concurrent_requests=_MAX_CONCURRENT_REQUESTS,
    )
    first_request = app.state.dflash_admission.reserve()
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "qwen27-tf",
                "messages": [{"role": "user", "content": "queued"}],
            },
        )
    finally:
        first_request.release()

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.json()["error"]["code"] == "at_capacity"


def test_http_stream_and_nonstream_use_provider_and_reject_tools() -> None:
    from fastapi.testclient import TestClient

    from rapid_mlx.speculative.dflash.server import _build_app
    from rapid_mlx.speculative.tensorfold_qwen27_server import (
        ProviderChunk,
        ProviderResult,
    )

    class Processor:
        eos_token_id = 99
        chat_template = "template"
        tokenizer = None

        def __init__(self):
            self.tokenizer = self

        def apply_chat_template(self, messages, **kwargs):
            return "rendered prompt"

    def stream_generate(_model, _processor, _prompt, **_kwargs):
        yield ProviderChunk("hello", 7, 1, 4)

    def generate(_model, _processor, _prompt, **_kwargs):
        return ProviderResult("hello", [7], 1, 4)

    runtime = SimpleNamespace(
        algorithm="dflash2",
        drafter_repo="pinned-drafter",
        target_revision="a" * 40,
        drafter_revision="b" * 40,
    )
    app = _build_app(
        model=None,
        processor=Processor(),
        runtime=runtime,
        served_model_name="qwen27-tf",
        default_max_tokens=8,
        cors_origins=["*"],
        stream_generate_fn=stream_generate,
        generate_fn=generate,
        generation_kwargs_fn=generation_kwargs,
        render_prompt_fn=render_prompt,
        generation_kwargs_with_request=True,
        validate_request_fn=validate_http_request,
        backend_name="TensorFold Qwen3.8-27B",
        runtime_status_extra={"profile": {"mode": "accelerated"}},
    )
    client = TestClient(app)
    assert client.get("/healthz").json()["profile"]["mode"] == "accelerated"
    assert client.get("/v1/status").json()["profile"]["mode"] == "accelerated"
    body = {"model": "qwen27-tf", "messages": [{"role": "user", "content": "hi"}]}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    usage = response.json()["usage"]
    assert (
        usage["prompt_tokens"],
        usage["completion_tokens"],
        usage["total_tokens"],
    ) == (4, 1, 5)

    with client.stream(
        "POST", "/v1/chat/completions", json={**body, "stream": True}
    ) as response:
        wire = "".join(response.iter_text())
    assert response.status_code == 200
    assert '"content": "hello"' in wire
    assert "data: [DONE]" in wire

    rejected = client.post(
        "/v1/chat/completions",
        json={**body, "tools": [{"type": "function", "function": {"name": "x"}}]},
    )
    assert rejected.status_code == 400
