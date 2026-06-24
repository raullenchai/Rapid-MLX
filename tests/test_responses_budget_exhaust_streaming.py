# SPDX-License-Identifier: Apache-2.0
"""R11-B / R11-M-F1 regression guard — streaming ``/v1/responses`` emits a
``reasoning`` output item when ``max_output_tokens`` cuts off mid-think.

Background: Mira R1 F1 captured the streaming ``/v1/responses`` path
silently shipping only three SSE events (``response.created``,
``response.in_progress``, ``response.completed``) with ``output:[]`` and
``status:"completed"`` when ``max_output_tokens`` truncated the model while
still inside ``<think>...</think>``. The non-streaming surface on the SAME
prompt correctly returned ``status:"incomplete"`` + a ``reasoning`` output
item via ``openai_to_responses``. This cross-path asymmetry was
unrecoverable by Codex CLI / openai-python clients — the streaming
``output[]`` was empty so nothing displayable reached the user.

The fix (vllm_mlx/routes/responses.py ``_stream_responses``):
  * Accumulate reasoning bytes from BOTH the channel-routed (gemma4 /
    harmony) and parser-routed (qwen3 / deepseek / glm4 / minimax) paths.
  * Emit a ``response.output_item.added`` → ``response.output_item.done``
    pair for a ``reasoning`` item before ``response.completed`` whenever
    reasoning text was accumulated.
  * Mark the reasoning item ``status:"incomplete"`` when
    ``last_finish_reason == "length"`` (mirrors the non-stream surface's
    ``_build_reasoning_output_item`` invocation under R11-B).
  * Emit ``response.completed`` with ``status:"incomplete"`` and
    ``incomplete_details:{reason:"max_output_tokens"}`` instead of
    ``status:"completed"`` whenever the engine reported
    ``finish_reason="length"``.

These tests pin those invariants and the streaming/non-streaming parity
contract.
"""

import json
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Engine stub: streams a reasoning-only output that ends with
# ``finish_reason="length"``. Mirrors the live-server shape Mira R1 F1
# captured on Qwen3-0.6B with ``max_output_tokens=32`` — the model emitted
# ``<think>\n...thoughts...`` and was cut off before the closing tag.
# ---------------------------------------------------------------------------


class _Tokenizer:
    # Qwen3-shaped chat template: includes both ``<think>`` and the
    # ``add_generation_prompt`` marker so
    # ``service.helpers._should_start_in_thinking`` returns True. That
    # routes the parser's no-tag Case-3 stream to ``reasoning`` and lets
    # the qwen3 finalize correctly classify the cut-off thought trace as
    # ``DeltaMessage(reasoning=...)`` instead of leaking to ``content``.
    chat_template = (
        "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}"
    )

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class _BaseEngine:
    pass


@dataclass
class _GenerationOutput:
    text: str
    raw_text: str = ""
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    new_text: str = ""
    finished: bool = True
    logprobs: Any = None
    channel: str | None = None
    tool_calls: list | None = None
    reasoning_text: str = ""
    matched_stop: str | None = None
    cached_tokens: int = 0


# Realistic mid-think prefix the model emits when ``<think>`` is in the
# prompt (Qwen3 chat template injects ``<think>\n``) and ``max_output_tokens``
# trips before ``</think>`` arrives. NO ``</think>`` token here.
_REASONING_CHUNKS = [
    "Okay, the user said",
    ' "Hi". I should respond',
    " politely. Let me",
    " start with a greeting.",
]


class _ReasoningCutoffEngine:
    """Streams the reasoning chunks above, then a final chunk with
    ``finish_reason="length"`` to mimic the ``max_output_tokens`` cutoff."""

    preserve_native_tool_format = False

    def __init__(self):
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        full = "".join(_REASONING_CHUNKS)
        return _GenerationOutput(
            text="",
            raw_text=full,
            reasoning_text=full,
            prompt_tokens=4,
            completion_tokens=len(_REASONING_CHUNKS),
            finish_reason="length",
        )

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        for i, chunk in enumerate(_REASONING_CHUNKS):
            accumulated += chunk
            is_last = i == len(_REASONING_CHUNKS) - 1
            yield _GenerationOutput(
                text=accumulated,
                new_text=chunk,
                prompt_tokens=4 if i == 0 else 0,
                completion_tokens=i + 1,
                # ``length`` only on the terminal chunk — the engine
                # reports the truncation signal once it runs out of
                # budget, the same shape the live MLX scheduler emits.
                finish_reason="length" if is_last else None,
            )


class _StubbedShortReasoningEngine:
    """Codex r1 (R12-8): emits reasoning text but reports
    ``completion_tokens=0`` throughout the stream — the stubbed-short /
    speculative-decode-empty edge case where the engine streams
    reasoning bytes but the token accountant hasn't credited any
    output. Used to pin the invariant that ``reasoning_tokens`` MUST
    NOT exceed ``output_tokens`` even when both are zero.
    """

    preserve_native_tool_format = False

    def __init__(self):
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        full = "".join(_REASONING_CHUNKS)
        return _GenerationOutput(
            text="",
            raw_text=full,
            reasoning_text=full,
            prompt_tokens=4,
            completion_tokens=0,
            finish_reason="length",
        )

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        for i, chunk in enumerate(_REASONING_CHUNKS):
            accumulated += chunk
            is_last = i == len(_REASONING_CHUNKS) - 1
            yield _GenerationOutput(
                text=accumulated,
                new_text=chunk,
                prompt_tokens=4 if i == 0 else 0,
                completion_tokens=0,
                finish_reason="length" if is_last else None,
            )


# Reasoning-then-message engine for codex r1 HIGH #1 regression: the
# model closes ``</think>`` mid-stream then emits an assistant message
# and STILL hits ``finish_reason="length"`` before finishing the
# message. This is the mixed case that exposed the original
# ``output_index`` collision (reasoning wire-index = 1, but array
# position = 0).
_REASONING_THEN_MESSAGE_CHUNKS = [
    "Let me think.",
    "</think>",
    "\n\nHello there!",
    " How can I help",
]


class _ReasoningThenMessageEngine:
    """Streams a complete <think> block followed by a partial assistant
    message, terminating with ``finish_reason="length"``. The qwen3
    parser splits the bytes before/after ``</think>`` into reasoning
    vs content."""

    preserve_native_tool_format = False

    def __init__(self):
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        full = "".join(_REASONING_THEN_MESSAGE_CHUNKS)
        return _GenerationOutput(
            text=" How can I help",
            raw_text=full,
            reasoning_text="Let me think.",
            prompt_tokens=4,
            completion_tokens=len(_REASONING_THEN_MESSAGE_CHUNKS),
            finish_reason="length",
        )

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        for i, chunk in enumerate(_REASONING_THEN_MESSAGE_CHUNKS):
            accumulated += chunk
            is_last = i == len(_REASONING_THEN_MESSAGE_CHUNKS) - 1
            yield _GenerationOutput(
                text=accumulated,
                new_text=chunk,
                prompt_tokens=4 if i == 0 else 0,
                completion_tokens=i + 1,
                finish_reason="length" if is_last else None,
            )


# Reasoning + message + tool_call engine for codex r3 NIT regression:
# verify ``tool_output_index = len(completed_output)`` accounts for
# BOTH the message AND the reasoning items already appended. Pre-fix
# (before r1 HIGH #1) ``tool_output_index = (message_output_index + 1)
# if message_open else 0`` collided with the reasoning slot.
_REASONING_THEN_MESSAGE_THEN_TOOL_CHUNKS = [
    "Let me think.",
    "</think>",
    "\n\nI'll check that for you.",
]


class _ReasoningMessageToolEngine:
    """Streams reasoning, then a brief assistant message, then emits a
    tool_call on the final chunk — the full mixed shape that
    exercises ``tool_output_index`` after both message and reasoning
    have been appended to ``completed_output``."""

    preserve_native_tool_format = False

    def __init__(self):
        self.tokenizer = _Tokenizer()
        # Flat-dict shape the engine surfaces to
        # ``_parse_tool_calls_with_parser`` — see
        # ``tests/test_responses_bundle.py::_make_function_call``.
        self._tool_calls = [
            {
                "id": "call_test_001",
                "name": "get_weather",
                "arguments": '{"city":"Pittsburgh"}',
            }
        ]

    async def chat(self, messages, **kwargs):
        full = "".join(_REASONING_THEN_MESSAGE_THEN_TOOL_CHUNKS)
        return _GenerationOutput(
            text="I'll check that for you.",
            raw_text=full,
            reasoning_text="Let me think.",
            prompt_tokens=4,
            completion_tokens=len(_REASONING_THEN_MESSAGE_THEN_TOOL_CHUNKS),
            finish_reason="tool_calls",
            tool_calls=self._tool_calls,
        )

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        for i, chunk in enumerate(_REASONING_THEN_MESSAGE_THEN_TOOL_CHUNKS):
            accumulated += chunk
            is_last = i == len(_REASONING_THEN_MESSAGE_THEN_TOOL_CHUNKS) - 1
            yield _GenerationOutput(
                text=accumulated,
                new_text=chunk,
                prompt_tokens=4 if i == 0 else 0,
                completion_tokens=i + 1,
                finish_reason="tool_calls" if is_last else None,
                tool_calls=self._tool_calls if is_last else None,
            )


# Reasoning + tool_call + length cutoff engine for codex r5 BLOCKING:
# the model closes ``</think>``, emits a function_call, and hits
# ``finish_reason="length"`` on the tool args. ``accumulated_text``
# is empty BUT reasoning is completed (closed ``</think>`` to reach
# the tool emit). Pre-fix the streaming path flagged reasoning
# ``incomplete`` because the check only looked at ``accumulated_text``.
_REASONING_THEN_TOOL_CHUNKS = [
    "Let me think.",
    "</think>",
]


class _ReasoningToolCallLengthEngine:
    """Streams reasoning, closes ``</think>``, emits a tool_call on
    the last chunk with ``finish_reason="length"``. No assistant
    text body — only reasoning + tool_call before the budget is
    exhausted on the tool arguments."""

    preserve_native_tool_format = False

    def __init__(self):
        self.tokenizer = _Tokenizer()
        self._tool_calls = [
            {
                "id": "call_test_002",
                "name": "get_weather",
                "arguments": '{"city":"Pittsburgh"}',
            }
        ]

    async def chat(self, messages, **kwargs):
        full = "".join(_REASONING_THEN_TOOL_CHUNKS)
        return _GenerationOutput(
            text="",
            raw_text=full,
            reasoning_text="Let me think.",
            prompt_tokens=4,
            completion_tokens=len(_REASONING_THEN_TOOL_CHUNKS),
            finish_reason="length",
            tool_calls=self._tool_calls,
        )

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        for i, chunk in enumerate(_REASONING_THEN_TOOL_CHUNKS):
            accumulated += chunk
            is_last = i == len(_REASONING_THEN_TOOL_CHUNKS) - 1
            yield _GenerationOutput(
                text=accumulated,
                new_text=chunk,
                prompt_tokens=4 if i == 0 else 0,
                completion_tokens=i + 1,
                finish_reason="length" if is_last else None,
                tool_calls=self._tool_calls if is_last else None,
            )


# ---------------------------------------------------------------------------
# Test fixture — mirrors ``tests/test_responses_sse_event_order.py``'s
# lightweight engine swap so we don't need to load a real MLX model.
# ---------------------------------------------------------------------------


_IMPORTED = (
    "vllm_mlx.config",
    "vllm_mlx.config.server_config",
    "vllm_mlx.engine",
    "vllm_mlx.engine.base",
    "vllm_mlx.middleware.auth",
    "vllm_mlx.service.helpers",
    "vllm_mlx.routes.responses",
)
_PARENT_ATTRS = (
    ("vllm_mlx", "config"),
    ("vllm_mlx", "engine"),
    ("vllm_mlx.config", "server_config"),
    ("vllm_mlx.engine", "base"),
    ("vllm_mlx.middleware", "auth"),
    ("vllm_mlx.service", "helpers"),
    ("vllm_mlx.routes", "responses"),
)
_MISSING = object()


def _install_lightweight_engine_modules(monkeypatch):
    engine_pkg = types.ModuleType("vllm_mlx.engine")
    engine_pkg.BaseEngine = _BaseEngine
    engine_pkg.GenerationOutput = _GenerationOutput

    base_mod = types.ModuleType("vllm_mlx.engine.base")
    base_mod.BaseEngine = _BaseEngine
    base_mod.GenerationOutput = _GenerationOutput

    monkeypatch.setitem(sys.modules, "vllm_mlx.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm_mlx.engine.base", base_mod)


@pytest.fixture
def responses_client(monkeypatch):
    previous_modules = {n: sys.modules.get(n, _MISSING) for n in _IMPORTED}
    previous_attrs = {}
    for module_name, attr in _PARENT_ATTRS:
        module = sys.modules.get(module_name)
        previous_attrs[(module_name, attr)] = (
            getattr(module, attr, _MISSING) if module is not None else _MISSING
        )

    _install_lightweight_engine_modules(monkeypatch)

    from vllm_mlx.config import reset_config
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes.responses import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.engine = _ReasoningCutoffEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    # Wire the Qwen3 reasoning parser — that's the path the bug repro
    # used (the live Mira R1 F1 server ran ``--reasoning-parser qwen3``
    # auto-configured from the model family detector).
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = "qwen3"

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), engine=cfg.engine)

    reset_config()
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    for name, previous in previous_modules.items():
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    for (module_name, attr), previous in previous_attrs.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if previous is _MISSING:
            if hasattr(module, attr):
                delattr(module, attr)
        else:
            setattr(module, attr, previous)


@pytest.fixture
def reasoning_then_message_client(monkeypatch):
    """Sibling fixture for the reasoning+message cutoff regression. Same
    wiring as ``responses_client`` but swaps in
    ``_ReasoningThenMessageEngine`` so the stream emits BOTH a
    ``reasoning`` and a ``message`` item before hitting
    ``finish_reason="length"``."""
    previous_modules = {n: sys.modules.get(n, _MISSING) for n in _IMPORTED}
    previous_attrs = {}
    for module_name, attr in _PARENT_ATTRS:
        module = sys.modules.get(module_name)
        previous_attrs[(module_name, attr)] = (
            getattr(module, attr, _MISSING) if module is not None else _MISSING
        )

    _install_lightweight_engine_modules(monkeypatch)

    from vllm_mlx.config import reset_config
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes.responses import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.engine = _ReasoningThenMessageEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = "qwen3"

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), engine=cfg.engine)

    reset_config()
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    for name, previous in previous_modules.items():
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    for (module_name, attr), previous in previous_attrs.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if previous is _MISSING:
            if hasattr(module, attr):
                delattr(module, attr)
        else:
            setattr(module, attr, previous)


@pytest.fixture
def reasoning_message_tool_client(monkeypatch):
    """Fixture for the reasoning+message+tool_call regression. Same
    wiring as ``responses_client`` but swaps in
    ``_ReasoningMessageToolEngine`` so the stream emits a reasoning
    item, a message item, AND a tool_call — exercising
    ``tool_output_index`` after both have been appended to
    ``completed_output``."""
    previous_modules = {n: sys.modules.get(n, _MISSING) for n in _IMPORTED}
    previous_attrs = {}
    for module_name, attr in _PARENT_ATTRS:
        module = sys.modules.get(module_name)
        previous_attrs[(module_name, attr)] = (
            getattr(module, attr, _MISSING) if module is not None else _MISSING
        )

    _install_lightweight_engine_modules(monkeypatch)

    from vllm_mlx.config import reset_config
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes.responses import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.engine = _ReasoningMessageToolEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = "qwen3"

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), engine=cfg.engine)

    reset_config()
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    for name, previous in previous_modules.items():
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    for (module_name, attr), previous in previous_attrs.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if previous is _MISSING:
            if hasattr(module, attr):
                delattr(module, attr)
        else:
            setattr(module, attr, previous)


@pytest.fixture
def reasoning_tool_length_client(monkeypatch):
    """Fixture for the reasoning+tool_call+length cutoff regression
    (codex r5 BLOCKING). Same wiring as ``responses_client`` but
    swaps in ``_ReasoningToolCallLengthEngine`` so reasoning closes,
    a tool_call lands, and ``finish_reason="length"`` fires. Pre-fix
    this shape flipped reasoning to ``incomplete`` because the
    ``mid_think_cutoff`` check only inspected ``accumulated_text``."""
    previous_modules = {n: sys.modules.get(n, _MISSING) for n in _IMPORTED}
    previous_attrs = {}
    for module_name, attr in _PARENT_ATTRS:
        module = sys.modules.get(module_name)
        previous_attrs[(module_name, attr)] = (
            getattr(module, attr, _MISSING) if module is not None else _MISSING
        )

    _install_lightweight_engine_modules(monkeypatch)

    from vllm_mlx.config import reset_config
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes.responses import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.engine = _ReasoningToolCallLengthEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = "qwen3"

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), engine=cfg.engine)

    reset_config()
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    for name, previous in previous_modules.items():
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    for (module_name, attr), previous in previous_attrs.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if previous is _MISSING:
            if hasattr(module, attr):
                delattr(module, attr)
        else:
            setattr(module, attr, previous)


def _parse_sse(body_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = None
        data_text = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_text = line[len("data:") :].strip()
        if event_name and data_text is not None:
            events.append((event_name, json.loads(data_text)))
    return events


HEADERS = {"Authorization": "Bearer test-secret"}


def _stream_payload(**overrides):
    # R12-T2F-276 (PR #XXX): the casual-chat auto-disable flips the
    # default ``enable_thinking`` to False on a thinking-capable model
    # when the request has no explicit reasoning signal. These tests
    # are EXPLICITLY about the reasoning-emission path, so they opt
    # into reasoning via the OpenAI-spec ``reasoning.effort`` knob —
    # the helper's no-auto-disable contract for explicit reasoning
    # intent then keeps the engine kwarg unset (template default =
    # True for non-coder Qwen3) and the reasoning streaming path runs.
    base = {
        "model": "test-model",
        "input": "Hi",
        "stream": True,
        "max_output_tokens": 32,
        "reasoning": {"effort": "medium"},
    }
    base.update(overrides)
    return base


def _non_stream_payload(**overrides):
    # See _stream_payload — same R12-T2F-276 opt-in.
    base = {
        "model": "test-model",
        "input": "Hi",
        "max_output_tokens": 32,
        "reasoning": {"effort": "medium"},
    }
    base.update(overrides)
    return base


# =============================================================================
# R11-B (R11-M-F1) — streaming emits a reasoning item under max_output_tokens
# =============================================================================


class TestStreamingBudgetExhaustEmitsReasoningItem:
    def test_reasoning_output_item_added_under_length_cutoff(self, responses_client):
        """Pre-fix the streaming path shipped ONLY ``response.created`` +
        ``response.in_progress`` + ``response.completed`` with ``output:[]``
        when ``</think>`` never closed within budget. Post-fix the
        accumulated reasoning bytes ship as a ``reasoning`` output item via
        the ``output_item.added`` → ``output_item.done`` event pair."""
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        names = [n for n, _ in events]

        # The reasoning item must be emitted via both lifecycle events.
        assert "response.output_item.added" in names, (
            f"reasoning ``output_item.added`` missing under max_output_tokens "
            f"cutoff. Events seen: {names}"
        )
        assert "response.output_item.done" in names, (
            f"reasoning ``output_item.done`` missing under max_output_tokens "
            f"cutoff. Events seen: {names}"
        )

        # Locate the added/done pair and verify the item shape.
        added = next(d for n, d in events if n == "response.output_item.added")
        done = next(d for n, d in events if n == "response.output_item.done")
        assert added["item"]["type"] == "reasoning"
        assert done["item"]["type"] == "reasoning"
        # Shared item id across added/done (SDK contract).
        assert added["item"]["id"] == done["item"]["id"]
        assert added["item"]["id"].startswith("rs_")

    def test_reasoning_item_done_status_is_incomplete(self, responses_client):
        """The reasoning item itself flips to ``status:"incomplete"`` when
        the model was still mid-think when its budget ran out — the text
        we ship is partial, flagging it as such matches the non-stream
        surface's ``_build_reasoning_output_item(status="incomplete")``
        invocation."""
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        done = next(d for n, d in events if n == "response.output_item.done")
        assert done["item"]["status"] == "incomplete"

    def test_completed_status_is_incomplete_with_max_output_tokens_reason(
        self, responses_client
    ):
        """Pre-fix ``response.completed`` reported ``status:"completed"``
        regardless of the underlying truncation. Post-fix:
        ``finish_reason="length"`` → ``status:"incomplete"`` +
        ``incomplete_details:{reason:"max_output_tokens"}``."""
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        completed = next(d for n, d in events if n == "response.completed")
        assert completed["response"]["status"] == "incomplete"
        assert completed["response"]["incomplete_details"] == {
            "reason": "max_output_tokens"
        }

    def test_completed_output_array_carries_reasoning_item(self, responses_client):
        """The terminal ``response.completed.response.output[]`` array
        must include the reasoning item so SDK consumers reading the
        Response object see the same shape as the per-event walk. Pre-fix
        ``output:[]`` shipped here."""
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        completed = next(d for n, d in events if n == "response.completed")
        output = completed["response"]["output"]
        assert len(output) >= 1, f"completed.output is empty: {output}"
        # Pure-reasoning cutoff: no message item was opened, so reasoning
        # lands at array index 0. (Mixed reasoning+message streams are
        # covered separately under R11-B codex r1 HIGH #1 — see
        # ``test_output_index_aligns_with_completed_output_position``.)
        assert output[0]["type"] == "reasoning"
        assert output[0]["status"] == "incomplete"
        assert output[0]["summary"], "summary[] is empty"
        summary_text = output[0]["summary"][0]["text"]
        # The accumulated reasoning text is non-empty AND not duplicated.
        # The expected payload is the concatenation of the engine's
        # reasoning chunks (no leading ``\n`` from the chat template
        # since the test engine emits chunks without the template
        # injecting one into ``raw_text``).
        expected_reasoning = "".join(_REASONING_CHUNKS)
        assert summary_text == expected_reasoning, (
            f"summary text mismatch:\n  got: {summary_text!r}\n  "
            f"expected: {expected_reasoning!r}"
        )
        # Pre-fix duplication regression guard: the reasoning text must
        # appear EXACTLY ONCE — the original bug had finalize and the
        # in-loop accumulator both contributing the same bytes.
        assert summary_text.count("Okay, the user said") == 1, (
            f"reasoning text duplicated:\n  {summary_text!r}"
        )

    def test_output_index_aligns_with_completed_output_position(self, responses_client):
        """R11-B codex r1 HIGH #1 regression guard. ``output_index`` on a
        streaming event is the position of that item in the terminal
        ``Response.output[]`` array — NOT just a wire-event ordinal.
        Pre-fix the reasoning emit used ``(message_output_index + 1)``
        as its wire index but was inserted at ``completed_output[0]``,
        which broke SDK consumers that index into ``response.output[]``
        by event ``output_index``. This test pins the invariant that
        ``output[i].id == output_item.done[output_index=i].item.id``
        for every emitted item."""
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        completed = next(d for n, d in events if n == "response.completed")
        output = completed["response"]["output"]

        for ev_name, payload in events:
            if ev_name != "response.output_item.done":
                continue
            idx = payload["output_index"]
            ev_item_id = payload["item"]["id"]
            assert 0 <= idx < len(output), (
                f"output_index={idx} out of range for output[] of len {len(output)}"
            )
            assert output[idx]["id"] == ev_item_id, (
                f"output_index/array mismatch at idx={idx}: "
                f"event item id={ev_item_id!r}, output[{idx}].id="
                f"{output[idx]['id']!r}"
            )

    def test_reasoning_tokens_credited_under_cutoff(self, responses_client):
        """``usage.output_tokens_details.reasoning_tokens`` MUST be > 0
        when reasoning bytes were accumulated. Pre-fix the streaming path
        always reported ``reasoning_tokens=0`` because the reasoning
        accumulator didn't exist."""
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        completed = next(d for n, d in events if n == "response.completed")
        usage = completed["response"]["usage"]
        reasoning_tokens = usage["output_tokens_details"]["reasoning_tokens"]
        assert reasoning_tokens > 0, (
            f"reasoning_tokens={reasoning_tokens} but reasoning text was "
            f"accumulated — usage credit missing."
        )

    def test_streaming_responses_emits_rescue_output_text_under_cutoff(
        self, responses_client, monkeypatch
    ):
        """Codex r2 (R12-8) MED #4: cross-path parity. Non-stream
        Responses materializes the H-01 rescue text into an
        ``output_text`` message item via ``openai_to_responses``.
        The streaming surface must mirror this — emit a synthetic
        message item carrying the rescue payload so clients that
        render only ``output_text`` see the same retry signal +
        reasoning tail the non-stream surface ships. Pre-fix the
        streaming surface only emitted the reasoning item (status
        incomplete) and silently dropped the rescue text.
        """
        monkeypatch.delenv("RAPID_MLX_REASONING_RESCUE", raising=False)
        monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        # The completed.response.output[] must carry BOTH the reasoning
        # item AND a synthetic message item with the rescue payload.
        completed = next(d for n, d in events if n == "response.completed")
        output = completed["response"]["output"]
        types = [item["type"] for item in output]
        assert "reasoning" in types, (
            f"reasoning item missing from output; got types={types}"
        )
        assert "message" in types, (
            f"streaming rescue message item missing; got types={types} "
            f"(non-stream surface emits one for the same cutoff shape)"
        )
        message_item = next(item for item in output if item["type"] == "message")
        assert message_item["content"], "rescue message must have content"
        rescue_text = message_item["content"][0]["text"]
        from vllm_mlx.service.helpers import REASONING_CUTOFF_SENTINEL

        assert rescue_text.startswith(REASONING_CUTOFF_SENTINEL), (
            f"rescue message must lead with the sentinel; got {rescue_text!r}"
        )
        # And the SSE ladder MUST emit the canonical message-item
        # event sequence — added → content_part.added → output_text.delta
        # → output_text.done → content_part.done → output_item.done.
        event_types_after_reasoning_done = []
        seen_reasoning_done = False
        for name, _ in events:
            if name == "response.output_item.done" and not seen_reasoning_done:
                seen_reasoning_done = True
                continue
            if seen_reasoning_done:
                event_types_after_reasoning_done.append(name)
                if name == "response.output_item.done":
                    break
        canonical_ladder = [
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]
        assert event_types_after_reasoning_done == canonical_ladder, (
            f"rescue message ladder missing or out-of-order; expected "
            f"{canonical_ladder}, got {event_types_after_reasoning_done}"
        )

    def test_reasoning_tokens_not_credited_when_completion_tokens_zero(
        self, monkeypatch
    ):
        """Codex r1 (R12-8) MED: when an engine streams reasoning text
        but the token accountant reports ``completion_tokens=0`` (the
        stubbed-short / spec-decode-empty edge case), the streaming
        path MUST emit ``reasoning_tokens=0`` — NOT ``reasoning_tokens=1``
        with ``output_tokens=0``. The invariant ``reasoning_tokens <=
        output_tokens`` is what SDK consumers depend on for usage
        arithmetic.

        Pre-fix the clamp lived inside the ``if completion_tokens:``
        branch, so the zero-completion case skipped clamping entirely
        and emitted ``output_tokens: 0`` + ``reasoning_tokens: 1``.
        """
        previous_modules = {n: sys.modules.get(n, _MISSING) for n in _IMPORTED}
        previous_attrs = {}
        for module_name, attr in _PARENT_ATTRS:
            module = sys.modules.get(module_name)
            previous_attrs[(module_name, attr)] = (
                getattr(module, attr, _MISSING) if module is not None else _MISSING
            )
        _install_lightweight_engine_modules(monkeypatch)
        from vllm_mlx.config import reset_config
        from vllm_mlx.middleware.auth import rate_limiter
        from vllm_mlx.middleware.exception_handlers import install_exception_handlers
        from vllm_mlx.routes.responses import router

        cfg = reset_config()
        cfg.api_key = "test-secret"
        cfg.engine = _StubbedShortReasoningEngine()
        cfg.model_name = "test-model"
        cfg.model_registry = None
        cfg.reasoning_parser_name = "qwen3"
        cfg.reasoning_parser = "qwen3"
        rate_limiter.enabled = False
        rate_limiter.requests_per_minute = 60
        rate_limiter._requests.clear()
        app = FastAPI()
        install_exception_handlers(app)
        app.include_router(router)
        client = TestClient(app)

        try:
            with client.stream(
                "POST",
                "/v1/responses",
                json=_stream_payload(),
                headers=HEADERS,
            ) as resp:
                body = "".join(resp.iter_text())
            events = _parse_sse(body)
            completed = next(d for n, d in events if n == "response.completed")
            usage = completed["response"]["usage"]
            output_tokens = usage["output_tokens"]
            reasoning_tokens = usage["output_tokens_details"]["reasoning_tokens"]
            assert output_tokens == 0, (
                f"engine reported completion_tokens=0; expected "
                f"output_tokens=0, got {output_tokens}"
            )
            assert reasoning_tokens <= output_tokens, (
                f"invariant violated: reasoning_tokens={reasoning_tokens} > "
                f"output_tokens={output_tokens}"
            )
            assert reasoning_tokens == 0, (
                f"with completion_tokens=0 the reasoning credit must be 0; "
                f"got {reasoning_tokens}"
            )
        finally:
            reset_config()
            rate_limiter.enabled = False
            rate_limiter.requests_per_minute = 60
            rate_limiter._requests.clear()
            for name, previous in previous_modules.items():
                if previous is _MISSING:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
            for (module_name, attr), previous in previous_attrs.items():
                module = sys.modules.get(module_name)
                if module is None:
                    continue
                if previous is _MISSING:
                    if hasattr(module, attr):
                        delattr(module, attr)
                else:
                    setattr(module, attr, previous)


# =============================================================================
# Cross-path parity — streaming + non-streaming converge on the same shape
# =============================================================================


class TestStreamingNonStreamingParity:
    def test_same_status_and_incomplete_details(self, responses_client):
        """Stream and non-stream on the same prompt + same
        ``max_output_tokens`` must report the same ``status`` and the
        same ``incomplete_details`` block."""
        # Non-streaming call
        non_stream_resp = responses_client.client.post(
            "/v1/responses",
            json=_non_stream_payload(),
            headers=HEADERS,
        )
        assert non_stream_resp.status_code == 200
        non_stream_body = non_stream_resp.json()

        # Streaming call
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            stream_body = "".join(resp.iter_text())
        stream_events = _parse_sse(stream_body)
        stream_completed = next(
            d for n, d in stream_events if n == "response.completed"
        )

        assert non_stream_body["status"] == "incomplete"
        assert stream_completed["response"]["status"] == "incomplete"
        assert non_stream_body["status"] == stream_completed["response"]["status"]

        assert non_stream_body.get("incomplete_details") == {
            "reason": "max_output_tokens"
        }
        assert stream_completed["response"].get("incomplete_details") == {
            "reason": "max_output_tokens"
        }

    def test_same_output_shape_under_cutoff(self, responses_client):
        """Both surfaces must ship ``output[0].type == "reasoning"`` with
        the same item ``status``. Pre-fix the streaming surface shipped
        ``output:[]`` here — exactly the cross-path asymmetry R11-M-F1
        documented."""
        non_stream_resp = responses_client.client.post(
            "/v1/responses",
            json=_non_stream_payload(),
            headers=HEADERS,
        )
        non_stream_body = non_stream_resp.json()

        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            stream_body = "".join(resp.iter_text())
        stream_events = _parse_sse(stream_body)
        stream_completed = next(
            d for n, d in stream_events if n == "response.completed"
        )
        stream_output = stream_completed["response"]["output"]
        non_stream_output = non_stream_body["output"]

        assert len(non_stream_output) >= 1
        assert len(stream_output) >= 1
        assert non_stream_output[0]["type"] == "reasoning"
        assert stream_output[0]["type"] == "reasoning"
        assert non_stream_output[0]["status"] == stream_output[0]["status"]
        assert non_stream_output[0]["status"] == "incomplete"


# =============================================================================
# R11-B codex r1 HIGH #1 — mixed reasoning + message cutoff
# =============================================================================


class TestReasoningPlusMessageOutputIndexAlignment:
    """The bug surfaced in codex round 1: when the stream emits BOTH a
    message item (because ``</think>`` closed mid-stream) AND a
    reasoning item (because we accumulated the pre-``</think>`` bytes
    and hit ``finish_reason="length"`` before the message finished),
    the wire ``output_index`` must equal the array position of that
    item in ``response.output[]``."""

    def test_message_then_reasoning_indices_align(self, reasoning_then_message_client):
        with reasoning_then_message_client.client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hi",
                "stream": True,
                "max_output_tokens": 32,
            },
            headers=HEADERS,
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        completed = next(d for n, d in events if n == "response.completed")
        output = completed["response"]["output"]

        # Sanity: at least one message and one reasoning item shipped.
        types_seen = [item["type"] for item in output]
        assert "reasoning" in types_seen, (
            f"reasoning item missing from output[]; types={types_seen}"
        )
        assert "message" in types_seen, (
            f"message item missing from output[]; types={types_seen}"
        )

        # R11-B codex r2 BLOCKING regression guard: when message body
        # shipped, reasoning ``status`` must be ``completed`` even
        # under ``finish_reason="length"`` — only the MESSAGE was
        # truncated, the model already closed ``</think>``. This
        # mirrors the non-stream gating in
        # ``openai_to_responses`` (responses_adapter.py L487).
        reasoning_item = next(item for item in output if item["type"] == "reasoning")
        assert reasoning_item["status"] == "completed", (
            f"reasoning item must be ``completed`` in the mixed case "
            f"(message body shipped → reasoning is NOT mid-think), "
            f"got status={reasoning_item['status']!r}"
        )

        # And cross-path parity: the non-stream surface on the same
        # engine shape must also report ``status="completed"``.
        non_stream_resp = reasoning_then_message_client.client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hi",
                "max_output_tokens": 32,
            },
            headers=HEADERS,
        )
        assert non_stream_resp.status_code == 200
        non_stream_output = non_stream_resp.json()["output"]
        non_stream_reasoning = next(
            item for item in non_stream_output if item["type"] == "reasoning"
        )
        assert non_stream_reasoning["status"] == reasoning_item["status"], (
            f"streaming vs non-streaming reasoning status diverged: "
            f"stream={reasoning_item['status']!r}, "
            f"non-stream={non_stream_reasoning['status']!r}"
        )

        # Walk every ``output_item.done`` and verify its
        # ``output_index`` matches its position in ``output[]`` by id.
        # Pre-fix the reasoning event reported ``output_index=1`` while
        # being inserted at array position 0 — that's the collision
        # this test pins.
        for ev_name, payload in events:
            if ev_name != "response.output_item.done":
                continue
            idx = payload["output_index"]
            ev_item_id = payload["item"]["id"]
            assert 0 <= idx < len(output), (
                f"output_index={idx} out of range for output[] of len "
                f"{len(output)} on item {ev_item_id!r}"
            )
            assert output[idx]["id"] == ev_item_id, (
                f"output_index/array mismatch at idx={idx}: "
                f"event item id={ev_item_id!r}, "
                f"output[{idx}].id={output[idx]['id']!r}"
            )

        # No two ``output_item.done`` events share an output_index.
        done_indices = [
            payload["output_index"]
            for ev_name, payload in events
            if ev_name == "response.output_item.done"
        ]
        assert len(done_indices) == len(set(done_indices)), (
            f"duplicate output_index in output_item.done events: {done_indices}"
        )


# =============================================================================
# R11-B codex r3 NIT — reasoning + tool_call output_index coverage
# (renamed per codex r5 NIT #3 — the route folds the message body into
# the function_call envelope, so this exercise is really reasoning+
# tool_call, not three-way reasoning+message+tool_call)
# =============================================================================


class TestReasoningToolOutputIndexAlignment:
    """Codex round-3 NIT regression guard. The
    ``tool_output_index = len(completed_output)`` change in r1 was
    motivated by the message+reasoning+tool_call case, but the test
    suite only exercised reasoning+message. This test pins the
    reasoning+tool_call shape (the route folds short message bodies
    into the function_call envelope): every emitted
    ``output_item.done`` event's ``output_index`` matches its
    position in the terminal ``response.output[]`` by id, and no
    two events share an index."""

    def test_reasoning_tool_indices_align(self, reasoning_message_tool_client):
        with reasoning_message_tool_client.client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "What's the weather?",
                "stream": True,
                # R12-T1F: opt back INTO thinking on a tools request —
                # the test exercises the reasoning+message+tool_call
                # output_index alignment, which by construction
                # requires thinking to be ON. Pre-R12-T1F the no-
                # preference path defaulted thinking on; the new
                # auto-disable turns it off for tools, so the test
                # must declare its intent.
                "chat_template_kwargs": {"enable_thinking": True},
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ],
            },
            headers=HEADERS,
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        completed = next(d for n, d in events if n == "response.completed")
        output = completed["response"]["output"]

        # Sanity: reasoning AND function_call both shipped. (The
        # message body in this engine is short and the route may
        # roll it into the function_call envelope depending on
        # ``preserve_native_tool_format``; we don't pin its
        # presence — the important invariant is that BOTH the
        # reasoning slot and the tool slot land without index
        # collision.)
        types_seen = [item["type"] for item in output]
        assert "reasoning" in types_seen, (
            f"reasoning item missing from output[]; types={types_seen}"
        )
        assert "function_call" in types_seen, (
            f"function_call item missing from output[]; types={types_seen}"
        )

        # Every ``output_item.done`` event's ``output_index`` matches
        # the array position of its ``item.id`` in the terminal
        # ``output[]``. Pre-fix the tool_call's output_index could
        # collide with the reasoning slot's output_index.
        for ev_name, payload in events:
            if ev_name != "response.output_item.done":
                continue
            idx = payload["output_index"]
            ev_item_id = payload["item"]["id"]
            assert 0 <= idx < len(output), (
                f"output_index={idx} out of range for output[] of len "
                f"{len(output)} on item {ev_item_id!r}"
            )
            assert output[idx]["id"] == ev_item_id, (
                f"output_index/array mismatch at idx={idx}: "
                f"event item id={ev_item_id!r}, "
                f"output[{idx}].id={output[idx]['id']!r}"
            )

        # No two ``output_item.done`` events share an output_index.
        done_indices = [
            payload["output_index"]
            for ev_name, payload in events
            if ev_name == "response.output_item.done"
        ]
        assert len(done_indices) == len(set(done_indices)), (
            f"duplicate output_index in output_item.done events: {done_indices}"
        )


# =============================================================================
# R11-B codex r5 BLOCKING — reasoning + tool_call + length cutoff
# =============================================================================


class TestReasoningCompletedWhenToolCallOnlyAfterThink:
    """Codex round-5 BLOCKING regression guard. When the model closes
    ``</think>`` and then emits ONLY a tool_call (no text) before
    ``finish_reason="length"`` fires on the tool args, the reasoning
    item must still be ``completed`` — the model already left the
    thinking block to reach the tool emit. Pre-fix the streaming
    ``mid_think_cutoff`` check only inspected ``accumulated_text``,
    so this shape incorrectly flipped reasoning to ``incomplete``
    (text was empty even though reasoning was complete).

    R12-T1F (0.8.16 operator dogfood): the auto-disable-thinking-for-
    tools route gate now turns thinking OFF by default whenever the
    client declares ``tools`` without expressing a thinking
    preference. This regression guard exercises the OPPOSITE corner
    (reasoning + tool_call + length cutoff), so it must explicitly
    opt INTO thinking via ``chat_template_kwargs.enable_thinking=true``
    — otherwise the engine reasoning-text never reaches the
    reasoning_parser branch on the streaming surface and the
    ``reasoning`` item is never assembled. The non-stream surface
    surfaces the reasoning item via ``output.reasoning_text``
    regardless, so the explicit opt-in is only load-bearing for the
    streaming case below.
    """

    def test_reasoning_completed_with_tool_call_under_length(
        self, reasoning_tool_length_client
    ):
        with reasoning_tool_length_client.client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Weather?",
                "stream": True,
                "max_output_tokens": 4,
                # R12-T1F: opt back into thinking on a tools request —
                # the test is exercising the reasoning + tool_call +
                # length corner, so it needs thinking ON to be
                # representative.
                "chat_template_kwargs": {"enable_thinking": True},
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ],
            },
            headers=HEADERS,
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        completed = next(d for n, d in events if n == "response.completed")
        output = completed["response"]["output"]

        # Sanity: reasoning + function_call shipped.
        types_seen = [item["type"] for item in output]
        assert "reasoning" in types_seen, f"reasoning item missing; types={types_seen}"
        assert "function_call" in types_seen, (
            f"function_call item missing; types={types_seen}"
        )

        # The headline assertion: reasoning is ``completed`` even
        # though ``finish_reason="length"`` and ``accumulated_text``
        # is empty — the tool_call landing proves the model closed
        # ``</think>``.
        reasoning_item = next(item for item in output if item["type"] == "reasoning")
        assert reasoning_item["status"] == "completed", (
            f"reasoning must be ``completed`` when ``</think>`` closed "
            f"before tool_call emit (text empty but tool_calls present); "
            f"got status={reasoning_item['status']!r}"
        )

    def test_non_stream_reasoning_completed_with_tool_call_under_length(
        self, reasoning_tool_length_client
    ):
        """R11-B codex r6 BLOCKING regression guard. Non-stream
        ``openai_to_responses`` must apply the SAME
        ``downstream_output_seen`` gate as the streaming path —
        otherwise the two surfaces report divergent reasoning
        status for the closed-``</think>`` + truncated tool_call
        shape.

        R12-T1F: opt back INTO thinking on this tools+reasoning corner
        for cross-surface parity with the streaming test above. The
        non-stream finalize path emits the reasoning item from
        ``output.reasoning_text`` regardless, but pinning the same
        request shape keeps the two cases comparable.
        """
        resp = reasoning_tool_length_client.client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Weather?",
                "max_output_tokens": 4,
                "chat_template_kwargs": {"enable_thinking": True},
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ],
            },
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        output = body["output"]
        types_seen = [item["type"] for item in output]
        assert "reasoning" in types_seen, (
            f"non-stream reasoning item missing; types={types_seen}"
        )
        assert "function_call" in types_seen, (
            f"non-stream function_call item missing; types={types_seen}"
        )
        reasoning_item = next(item for item in output if item["type"] == "reasoning")
        assert reasoning_item["status"] == "completed", (
            f"non-stream reasoning must be ``completed`` when "
            f"``</think>`` closed before tool_call emit; got "
            f"status={reasoning_item['status']!r}"
        )
