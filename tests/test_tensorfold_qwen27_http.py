# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.spec_decode.config import SpeculativeConfigError, parse_speculative_config
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
    native = parse_speculative_config('{"method":"dflash","backend":"native","model":"x"}')
    assert native is not None and native.backend == "native"
    with pytest.raises(SpeculativeConfigError):
        parse_speculative_config('{"method":"mtp","backend":"tensorfold"}')


def test_http_gate_rejects_unqualified_features_and_maps_sampling() -> None:
    from fastapi import HTTPException

    request = SimpleNamespace(
        tools=[{"type": "function"}], response_format=None,
        messages=[SimpleNamespace(content="hi")],
        repetition_penalty=None, presence_penalty=None,
        frequency_penalty=None, logit_bias=None,
    )
    with pytest.raises(HTTPException) as exc:
        validate_http_request(request)
    assert exc.value.status_code == 400

    request = SimpleNamespace(
        top_k=20, min_p=0.05, seed=42, stop=["END"],
        messages=[SimpleNamespace(content="hi")],
        tools=None, response_format=None, repetition_penalty=None,
        presence_penalty=None, frequency_penalty=None, logit_bias=None,
    )
    validate_http_request(request)
    assert generation_kwargs(
        max_tokens=9, temperature=0.7, top_p=0.9, request=request
    ) == {
        "max_tokens": 9, "temperature": 0.7, "top_p": 0.9,
        "top_k": 20, "min_p": 0.05, "seed": 42, "stop": ["END"],
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


def test_provider_preserves_exact_token_ids_and_request_outputs(monkeypatch) -> None:
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
        tokenizer=Tokenizer(), tokenizer_lock=__import__("threading").Lock(),
        stop_ids=frozenset(), min_match=3, scheduler=Scheduler(),
        _resolve_sampling=lambda fields, temperature, prompt: (fields, temperature),
    )
    provider = TensorFoldRequestProvider(SimpleNamespace(_app=app))
    chunks = list(provider.stream_generate(None, None, "prompt", max_tokens=8))

    assert [chunk.token for chunk in chunks] == [21, 22]
    assert [chunk.text for chunk in chunks] == ["A", "B"]
    assert provider.last_token_ids == [21, 22]
    assert provider.last_outputs[-1].finished
    assert provider.last_outputs[-1].cached_tokens == 3


def test_http_stream_and_nonstream_use_provider_and_reject_tools() -> None:
    from fastapi.testclient import TestClient
    from rapid_mlx.speculative.dflash.server import _build_app
    from rapid_mlx.speculative.tensorfold_qwen27_server import ProviderChunk, ProviderResult

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
        algorithm="dflash2", drafter_repo="pinned-drafter",
        target_revision="a" * 40, drafter_revision="b" * 40,
    )
    app = _build_app(
        model=None, processor=Processor(), runtime=runtime,
        served_model_name="qwen27-tf", default_max_tokens=8,
        cors_origins=["*"], stream_generate_fn=stream_generate,
        generate_fn=generate, generation_kwargs_fn=generation_kwargs,
        render_prompt_fn=render_prompt,
        generation_kwargs_with_request=True,
        validate_request_fn=validate_http_request,
        backend_name="TensorFold Qwen3.8-27B",
    )
    client = TestClient(app)
    body = {"model": "qwen27-tf", "messages": [{"role": "user", "content": "hi"}]}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    usage = response.json()["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (4, 1, 5)

    with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as response:
        wire = "".join(response.iter_text())
    assert response.status_code == 200
    assert '"content": "hello"' in wire
    assert "data: [DONE]" in wire

    rejected = client.post(
        "/v1/chat/completions",
        json={**body, "tools": [{"type": "function", "function": {"name": "x"}}]},
    )
    assert rejected.status_code == 400
