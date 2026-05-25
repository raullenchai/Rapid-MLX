# SPDX-License-Identifier: Apache-2.0
"""Route-level Anthropic streaming regressions."""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.routes.anthropic import router


class _ThinkingTemplateTokenizer:
    chat_template = "{% if add_generation_prompt %}<think>{% endif %}"


class _StreamingEngine:
    preserve_native_tool_format = False
    tokenizer = _ThinkingTemplateTokenizer()

    def __init__(self, deltas: list[str]):
        self._deltas = deltas
        self.calls = []

    async def stream_chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        for i, text in enumerate(self._deltas, start=1):
            yield SimpleNamespace(
                new_text=text,
                prompt_tokens=5,
                completion_tokens=i,
            )


def _make_client(engine: _StreamingEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.no_thinking = True
    cfg.reasoning_parser_name = None
    cfg.model_registry = None

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _parse_sse_data(response_text: str) -> list[dict]:
    events = []
    for raw_event in response_text.split("\n\n"):
        data_line = next(
            (line for line in raw_event.splitlines() if line.startswith("data: ")),
            None,
        )
        if not data_line:
            continue
        data = data_line.removeprefix("data: ")
        if data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


@pytest.fixture(autouse=True)
def _reset_server_config():
    reset_config()
    yield
    reset_config()


def test_anthropic_stream_route_no_thinking_template_answers_as_text():
    """Server no-thinking mode should keep direct answers as text blocks."""
    engine = _StreamingEngine(["Direct ", "answer"])
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "answer directly"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert engine.calls[0]["kwargs"]["enable_thinking"] is False

    events = _parse_sse_data(response.text)
    block_starts = [e for e in events if e.get("type") == "content_block_start"]
    assert [e["content_block"]["type"] for e in block_starts] == ["text"]

    text_deltas = [
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    ]
    thinking_deltas = [
        e
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "thinking_delta"
    ]

    assert "".join(text_deltas) == "Direct answer"
    assert thinking_deltas == []
    assert any(
        e.get("type") == "message_delta"
        and e.get("delta", {}).get("stop_reason") == "end_turn"
        for e in events
    )


def test_anthropic_stream_route_reasoning_parser_with_no_thinking_answers_as_text():
    """Closes #223. Server has --reasoning-parser qwen3 active AND the
    request opts out of thinking. The qwen3 parser's implicit-think
    heuristic routes any text without a <think> tag to ``reasoning``;
    pre-fix that meant every direct-answer token landed in
    ``thinking_delta`` blocks and ``text_delta`` was empty. The fix
    bypasses the reasoning parser whenever enable_thinking=False so the
    answer flows through the same think_router path as the
    no-parser-configured case.

    This test is the regression guard PR #213 missed: that PR added the
    bypass for the parser-less path but left the parser-configured path
    unchanged — surfaced by post-merge audit on 2026-05-05.
    """
    engine = _StreamingEngine(["Direct ", "answer"])

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.no_thinking = True
    # The exact scenario #223 catches: reasoning parser configured at
    # server start, then a per-request enable_thinking=False arrives.
    cfg.reasoning_parser_name = "qwen3"
    cfg.model_registry = None

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "answer directly"}],
        },
    )

    assert response.status_code == 200
    assert engine.calls[0]["kwargs"]["enable_thinking"] is False

    events = _parse_sse_data(response.text)

    text_deltas = [
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    ]
    thinking_deltas = [
        e
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "thinking_delta"
    ]

    # Pre-fix this assertion failed: thinking_deltas would have ALL the
    # text and text_deltas would be []. Post-fix the answer streams as
    # text and the thinking channel stays empty.
    assert "".join(text_deltas) == "Direct answer", (
        f"answer should stream as text_delta, got {text_deltas!r}; "
        f"thinking_deltas={thinking_deltas!r}"
    )
    assert thinking_deltas == []


def test_anthropic_stream_route_reasoning_parser_with_thinking_default_still_works():
    """Inverse guard: when enable_thinking is NOT explicitly False (i.e.
    default thinking-on for a reasoning model), the reasoning parser
    must still be exercised so the existing #185 fix isn't regressed.
    The model emits a <think>…</think> block followed by the answer;
    the parser splits them, and the route emits thinking_delta then
    text_delta.
    """
    # Model output: a thinking block + a real answer.
    engine = _StreamingEngine(["<think>scratch</think>", "real answer"])

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    # Server is NOT in no_thinking mode; client doesn't override.
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
    cfg.model_registry = None

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "what is 6*7"}],
        },
    )

    assert response.status_code == 200
    # No enable_thinking override on the request → kwargs absent or None.
    assert engine.calls[0]["kwargs"].get("enable_thinking") is not False

    events = _parse_sse_data(response.text)

    text_deltas = [
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    ]
    thinking_deltas = [
        e["delta"]["thinking"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "thinking_delta"
    ]

    # The reasoning parser path is engaged. The qwen3 parser splits
    # <think>…</think> from the rest, so thinking and text both carry
    # content. Asserting non-empty on each side guards the parser path
    # without binding to specific token boundaries the parser chooses.
    assert "real answer" in "".join(text_deltas), text_deltas
    assert "scratch" in "".join(thinking_deltas), thinking_deltas


class _CacheReportingEngine:
    """Streaming engine that reports a prefix-cache hit count, like
    the prefix-cache scheduler does in production. Mirrors
    ``_StreamingEngine`` but adds ``cached_tokens`` on each chunk so
    the route's ``message_delta`` usage can pick it up.
    """

    preserve_native_tool_format = False
    tokenizer = _ThinkingTemplateTokenizer()

    def __init__(self, deltas: list[str], *, prompt_tokens: int, cached_tokens: int):
        self._deltas = deltas
        self._prompt_tokens = prompt_tokens
        self._cached_tokens = cached_tokens

    async def stream_chat(self, messages, **kwargs):
        for i, text in enumerate(self._deltas, start=1):
            yield SimpleNamespace(
                new_text=text,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=i,
                cached_tokens=self._cached_tokens,
            )


def _find_message_delta(events: list[dict]) -> dict:
    deltas = [e for e in events if e.get("type") == "message_delta"]
    assert deltas, "message_delta event missing from stream"
    return deltas[-1]


def test_anthropic_stream_emits_cache_read_when_engine_reports_hit():
    """When the underlying engine surfaces a prefix-cache hit on its
    stream chunks, the Anthropic ``message_delta`` usage block must
    populate ``cache_read_input_tokens`` and adjust ``input_tokens``
    down by the cached share so Anthropic's spec identity
    (``total_input = input + cache_read + cache_creation``) holds.
    """
    engine = _CacheReportingEngine(
        ["Direct ", "answer"], prompt_tokens=100, cached_tokens=30
    )
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "answer directly"}],
        },
    )
    assert response.status_code == 200

    events = _parse_sse_data(response.text)
    usage = _find_message_delta(events)["usage"]
    assert usage["input_tokens"] == 70  # 100 prompt - 30 cached
    assert usage["cache_read_input_tokens"] == 30
    # cache_creation is intentionally absent (Anthropic's billing
    # category has no local-engine analog).
    assert "cache_creation_input_tokens" not in usage


def test_anthropic_stream_omits_cache_fields_without_hit():
    """When the engine reports no hit (``cached_tokens=0``), the
    ``message_delta`` usage block must NOT include cache fields, and
    ``input_tokens`` must reflect the full prompt. Mirrors the
    non-streaming adapter's "engine doesn't report" semantic.
    """
    engine = _CacheReportingEngine(
        ["Direct ", "answer"], prompt_tokens=100, cached_tokens=0
    )
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "answer directly"}],
        },
    )
    assert response.status_code == 200

    events = _parse_sse_data(response.text)
    usage = _find_message_delta(events)["usage"]
    assert usage["input_tokens"] == 100
    assert "cache_read_input_tokens" not in usage
    assert "cache_creation_input_tokens" not in usage
