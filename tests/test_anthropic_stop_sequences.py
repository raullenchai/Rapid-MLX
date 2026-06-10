# SPDX-License-Identifier: Apache-2.0
"""Regression test for /v1/messages dropping ``stop_sequences``.

Bug: the Anthropic route's ``_resolved_sampling_kwargs`` helper forwarded
``temperature``, ``top_p``, and extended sampling params to the engine but
NOT ``stop`` — so ``stop_sequences`` from the request flowed through
``anthropic_to_openai`` into ``openai_request.stop`` and then died at the
route boundary. Engine ran uncapped, model emitted past the user's stop
tokens. Surfaced by the iter8 onboarding sweep on gpt-oss-20b-mxfp4-q8: same
prompt + ``stop_sequences:["STOPHERE"]`` returned full text including
"STOPHERE", finish_reason=end_turn. Identical prompt via /v1/chat/
completions with ``stop:["STOPHERE"]`` stopped correctly. Fix: include
``stop`` in the kwargs returned by ``_resolved_sampling_kwargs`` so both
the non-stream and stream branches forward it.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.anthropic import router as anthropic_router


class _RecordingEngine:
    """Mock engine that records the kwargs passed to ``chat`` / ``stream_chat``.

    The point of the test isn't whether the model stops — it's that the
    request's ``stop_sequences`` reaches the engine. The previous bug was
    the route dropping the field silently.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self):
        self.chat_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text="ok",
            new_text="ok",
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        yield GenerationOutput(
            text="ok",
            new_text="ok",
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _make_client(engine: _RecordingEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.tool_parser = None

    app = FastAPI()
    app.include_router(anthropic_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_config()


def test_stop_sequences_forwarded_to_engine_non_stream():
    engine = _RecordingEngine()
    client = _make_client(engine)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stop_sequences": ["STOPHERE", "ALSO_STOP"],
            "messages": [{"role": "user", "content": "say STOPHERE then more"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(engine.chat_calls) == 1
    forwarded_stop = engine.chat_calls[0]["kwargs"].get("stop")
    assert forwarded_stop == ["STOPHERE", "ALSO_STOP"], (
        f"stop_sequences must be forwarded as ``stop`` to engine.chat(); "
        f"got {forwarded_stop!r}"
    )


def test_stop_sequences_forwarded_to_engine_streaming():
    engine = _RecordingEngine()
    client = _make_client(engine)

    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "say END then more"}],
        },
    ) as resp:
        assert resp.status_code == 200
        # drain the stream
        for _ in resp.iter_lines():
            pass

    assert len(engine.stream_calls) == 1
    forwarded_stop = engine.stream_calls[0]["kwargs"].get("stop")
    assert forwarded_stop == ["END"], (
        f"stop_sequences must be forwarded as ``stop`` to engine.stream_chat(); "
        f"got {forwarded_stop!r}"
    )


def test_stop_sequences_none_when_omitted_non_stream():
    """Sanity: when client omits stop_sequences, we don't synthesize one."""
    engine = _RecordingEngine()
    client = _make_client(engine)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    forwarded_stop = engine.chat_calls[0]["kwargs"].get("stop")
    assert forwarded_stop is None, (
        f"omitted stop_sequences must forward as None; got {forwarded_stop!r}"
    )
