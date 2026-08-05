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


def _make_reasoning_client(engine: _StreamingEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
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
            "max_tokens": 2048,
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


@pytest.mark.parametrize(
    "request_extension",
    [
        {},
        {"enable_thinking": False},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"thinking": {"type": "disabled"}},
    ],
    ids=["casual-default", "top-level", "template-kwargs", "native"],
)
def test_anthropic_stream_route_disables_thinking(request_extension):
    """Every supported opt-out reaches the engine, including shared defaults."""
    engine = _StreamingEngine(["Direct answer"])
    client = _make_reasoning_client(engine)
    payload = {
        "model": "test-model",
        "max_tokens": 32,
        "stream": True,
        "messages": [{"role": "user", "content": "answer directly"}],
    }
    payload.update(request_extension)

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200, response.text
    assert engine.calls[0]["kwargs"]["enable_thinking"] is False
    events = _parse_sse_data(response.text)
    assert not any(
        event.get("delta", {}).get("type") == "thinking_delta" for event in events
    )


def test_anthropic_stream_route_tools_default_disables_thinking(monkeypatch):
    """The tools policy reaches the engine independently of casual-chat logic."""
    monkeypatch.setattr(
        "vllm_mlx.routes.anthropic.maybe_auto_disable_thinking_for_casual_chat",
        lambda request: False,
    )
    engine = _StreamingEngine(["Direct answer"])
    client = _make_reasoning_client(engine)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "Inspect the scheduler and correlate cache misses.",
                }
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Look something up",
                    "input_schema": {"type": "object"},
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert engine.calls[0]["kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("with_tools", [False, True], ids=["casual", "tools"])
def test_anthropic_stream_route_effort_preserves_thinking_intent(with_tools):
    """Anthropic output_config.effort must override both auto-disable policies."""
    engine = _StreamingEngine(["<think>scratch</think>", "answer"])
    client = _make_reasoning_client(engine)
    payload = {
        "model": "test-model",
        "max_tokens": 2048,
        "stream": True,
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": "answer directly"}],
    }
    if with_tools:
        payload["tools"] = [
            {
                "name": "lookup",
                "description": "Look something up",
                "input_schema": {"type": "object"},
            }
        ]

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200, response.text
    assert engine.calls[0]["kwargs"]["enable_thinking"] is True


def test_anthropic_stream_route_reasoning_parser_with_explicit_thinking_still_works():
    """Inverse guard: when thinking is explicitly enabled, the reasoning parser
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
            "thinking": {"type": "enabled", "budget_tokens": 16},
            "messages": [{"role": "user", "content": "what is 6*7"}],
        },
    )

    assert response.status_code == 200
    assert engine.calls[0]["kwargs"].get("enable_thinking") is True

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


class _ThinkThenToolEngine:
    """Streams reasoning, then the template's ``</think>`` → ``<tool_call>``
    separator ("\n\n") as a standalone delta, then a hermes tool call — the
    exact shape that used to open a blank text block between the thinking and
    tool_use blocks."""

    preserve_native_tool_format = False
    tokenizer = _ThinkingTemplateTokenizer()
    _tokenizer = None

    async def stream_chat(self, messages, **kwargs):
        deltas = [
            "<think>Let me check ",
            "the weather.</think>",
            "\n\n",
            '<tool_call>\n{"name": "get_weather", ',
            '"arguments": {"city": "SF"}}\n</tool_call>',
        ]
        for i, text in enumerate(deltas, start=1):
            yield SimpleNamespace(new_text=text, prompt_tokens=14, completion_tokens=i)


def test_anthropic_stream_no_blank_text_block_between_thinking_and_tool_use():
    """Regression: the whitespace-only ``</think>``→``<tool_call>`` separator
    must NOT stream as its own ``text`` content block. Streaming should match
    non-stream ([thinking, tool_use]); previously it emitted
    [thinking, text("\\n\\n"), tool_use]."""
    from vllm_mlx.reasoning import get_parser

    cfg = reset_config()
    cfg.engine = _ThinkThenToolEngine()
    cfg.model_name = "test-model"
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = get_parser("qwen3")()
    cfg.tool_call_parser = "hermes"
    cfg.enable_auto_tool_choice = True
    cfg.model_registry = None

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 256,
            "stream": True,
            "messages": [{"role": "user", "content": "weather in SF?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        },
    )
    assert response.status_code == 200

    events = _parse_sse_data(response.text)
    block_types = [
        e["content_block"]["type"]
        for e in events
        if e.get("type") == "content_block_start"
    ]
    text_deltas = [
        e["delta"].get("text", "")
        for e in events
        if e.get("type") == "content_block_delta"
        and e["delta"].get("type") == "text_delta"
    ]
    assert block_types == ["thinking", "tool_use"]
    assert text_deltas == []
