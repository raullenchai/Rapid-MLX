# SPDX-License-Identifier: Apache-2.0
"""HTTP-level tests for the 0.8.5 dogfood ``/v1/responses`` bundle fixes.

Covers Yuki F4, F6, F8, F13 + Yuki R6, R7, R10 + Ana C-06. See
``0.8TODO.md`` r4 section and ``/tmp/dogfood-085/yuki-r{1,2}.md`` for
the original evidence. Each test class names its finding so a future
regression triages to the right report.

Same lightweight-engine harness shape as ``test_responses_route.py`` —
no MLX import — so the tests stay fast and CI-portable.
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
# Lightweight engine harness — copy of routes/test_responses_route.py's
# fixture with a richer mock so the new tests can exercise reasoning
# content, tool_calls, and Computer-Use translation in one place.
# ---------------------------------------------------------------------------


class _Tokenizer:
    chat_template = ""

    def __init__(self):
        self.calls = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
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


def _make_function_call(name: str, args: str, call_id: str = "call_test"):
    """Build a structured tool_call dict in the shape the engine surfaces
    to ``_parse_tool_calls_with_parser`` (flat ``name``/``arguments``/
    ``id`` — see service/helpers.py L1816)."""
    return {
        "id": call_id,
        "name": name,
        "arguments": args,
    }


class _Engine:
    preserve_native_tool_format = False

    def __init__(
        self,
        *,
        text: str = "hello world",
        reasoning_text: str = "",
        tool_calls: list | None = None,
        finish_reason: str = "stop",
    ):
        self.calls: list[SimpleNamespace] = []
        self.stream_calls: list[SimpleNamespace] = []
        self.tokenizer = _Tokenizer()
        self._text = text
        self._reasoning_text = reasoning_text
        self._tool_calls = tool_calls
        self._finish_reason = finish_reason

    async def chat(self, messages, **kwargs):
        self.calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        return _GenerationOutput(
            text=self._text,
            raw_text=self._text,
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason=self._finish_reason,
            tool_calls=self._tool_calls,
            reasoning_text=self._reasoning_text,
        )

    async def stream_chat(self, messages, **kwargs):
        """Emit a tiny synthetic stream: three text chunks then EOS, with
        optional reasoning_text and tool_calls on the final chunk."""
        self.stream_calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        chunks = ["Hello", " from", " rapid"]
        for i, c in enumerate(chunks):
            yield _GenerationOutput(
                text="".join(chunks[: i + 1]),
                new_text=c,
                prompt_tokens=3 if i == 0 else 0,
                completion_tokens=i + 1,
                finish_reason=None if i < len(chunks) - 1 else self._finish_reason,
                tool_calls=self._tool_calls if i == len(chunks) - 1 else None,
                reasoning_text=self._reasoning_text if i == len(chunks) - 1 else "",
            )


def _install_lightweight_engine_modules(monkeypatch):
    engine_pkg = types.ModuleType("vllm_mlx.engine")
    engine_pkg.BaseEngine = _BaseEngine
    engine_pkg.GenerationOutput = _GenerationOutput

    base_mod = types.ModuleType("vllm_mlx.engine.base")
    base_mod.BaseEngine = _BaseEngine
    base_mod.GenerationOutput = _GenerationOutput

    monkeypatch.setitem(sys.modules, "vllm_mlx.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm_mlx.engine.base", base_mod)


_IMPORTED_UNDER_LIGHTWEIGHT_ENGINE = (
    "vllm_mlx.config",
    "vllm_mlx.config.server_config",
    "vllm_mlx.engine",
    "vllm_mlx.engine.base",
    "vllm_mlx.middleware.auth",
    "vllm_mlx.service.helpers",
    "vllm_mlx.routes.responses",
)
_PARENT_ATTRS_UNDER_LIGHTWEIGHT_ENGINE = (
    ("vllm_mlx", "config"),
    ("vllm_mlx", "engine"),
    ("vllm_mlx.config", "server_config"),
    ("vllm_mlx.engine", "base"),
    ("vllm_mlx.middleware", "auth"),
    ("vllm_mlx.service", "helpers"),
    ("vllm_mlx.routes", "responses"),
)
_MISSING = object()


def _make_client(monkeypatch, engine: _Engine) -> SimpleNamespace:
    previous_modules = {
        name: sys.modules.get(name, _MISSING)
        for name in _IMPORTED_UNDER_LIGHTWEIGHT_ENGINE
    }
    previous_attrs = {}
    for module_name, attr in _PARENT_ATTRS_UNDER_LIGHTWEIGHT_ENGINE:
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
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)

    return SimpleNamespace(
        client=TestClient(app),
        engine=engine,
        previous_modules=previous_modules,
        previous_attrs=previous_attrs,
        reset_config=reset_config,
        rate_limiter=rate_limiter,
    )


def _teardown_client(state: SimpleNamespace):
    state.reset_config()
    state.rate_limiter.enabled = False
    state.rate_limiter.requests_per_minute = 60
    state.rate_limiter._requests.clear()

    for name, previous in state.previous_modules.items():
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    for (module_name, attr), previous in state.previous_attrs.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if previous is _MISSING:
            if hasattr(module, attr):
                delattr(module, attr)
        else:
            setattr(module, attr, previous)


@pytest.fixture
def make_responses_client(monkeypatch):
    """Factory fixture so each test can supply its own configured engine."""
    states: list[SimpleNamespace] = []

    def _factory(**engine_kwargs):
        engine = _Engine(**engine_kwargs)
        state = _make_client(monkeypatch, engine)
        states.append(state)
        return state

    yield _factory

    for state in reversed(states):
        _teardown_client(state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_AUTH = {"Authorization": "Bearer test-secret"}


def _payload(**overrides) -> dict:
    base = {
        "model": "test-model",
        "input": "Hello, world",
    }
    base.update(overrides)
    return base


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse Codex-shape ``event: ...\\ndata: {...}\\n\\n`` framing."""
    events = []
    for raw in body.split("\n\n"):
        if not raw.strip():
            continue
        lines = raw.split("\n")
        event_name = None
        data_text = None
        for line in lines:
            if line.startswith("event: "):
                event_name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                data_text = line[len("data: ") :].strip()
        if event_name and data_text is not None:
            events.append((event_name, json.loads(data_text)))
    return events


# ---------------------------------------------------------------------------
# Yuki F4 / R10 — reasoning output item emitted on /v1/responses
# ---------------------------------------------------------------------------


class TestF4ReasoningOutputItem:
    """Yuki F4 (0.8.5 dogfood): the prior shim dropped reasoning
    content entirely on /v1/responses even though /v1/chat/completions
    returned ``message.reasoning_content``. The fix emits a top-level
    ``reasoning`` output item with ``summary[].text`` populated.
    """

    def test_reasoning_emitted_when_engine_returns_reasoning_text(
        self, make_responses_client
    ):
        state = make_responses_client(
            text="The answer is 4.",
            reasoning_text="Two plus two equals four because addition is commutative.",
        )

        resp = state.client.post(
            "/v1/responses",
            json=_payload(input="What is 2+2?"),
            headers=_AUTH,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        reasoning_items = [o for o in body["output"] if o["type"] == "reasoning"]
        message_items = [o for o in body["output"] if o["type"] == "message"]

        assert len(reasoning_items) == 1, (
            f"expected 1 reasoning item, got output={body['output']}"
        )
        assert len(message_items) == 1, body["output"]

        # Reasoning item ships before the message item per OpenAI spec.
        types_ = [o["type"] for o in body["output"]]
        assert types_.index("reasoning") < types_.index("message"), types_

        # Summary text carries the engine's reasoning_text verbatim.
        summary = reasoning_items[0]["summary"]
        assert summary and summary[0]["type"] == "summary_text"
        assert "Two plus two equals four" in summary[0]["text"]

    def test_no_reasoning_item_when_engine_returns_none(self, make_responses_client):
        """Sanity: when reasoning_text is empty, no ``reasoning`` item
        appears — back-compat with the pre-0.8.5 envelope shape for
        non-reasoning models."""
        state = make_responses_client(text="plain reply", reasoning_text="")

        resp = state.client.post("/v1/responses", json=_payload(), headers=_AUTH)
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert not [o for o in body["output"] if o["type"] == "reasoning"]


# ---------------------------------------------------------------------------
# Yuki R10 — cross-lane reasoning parity (chat-completions ⇄ responses)
# ---------------------------------------------------------------------------


class TestR10CrossLaneReasoningParity:
    """Yuki R10 (0.8.5 dogfood): same model + prompt sent to
    /v1/chat/completions and /v1/responses returned reasoning on the
    chat lane and nothing on the responses lane. After the F4 fix the
    semantic content matches: chat ``message.reasoning_content`` ==
    responses ``output[i].summary[0].text`` (for type=='reasoning').
    """

    def test_responses_reasoning_summary_matches_chat_reasoning_content(
        self, make_responses_client
    ):
        reasoning = (
            "I'll work through this step by step. First, 2 is the number two. "
            "Adding two and two yields four. So 2+2=4."
        )
        state = make_responses_client(text="4.", reasoning_text=reasoning)

        # Drive the same engine through the Responses adapter.
        resp = state.client.post(
            "/v1/responses",
            json=_payload(input="What is 2+2? Think briefly."),
            headers=_AUTH,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        reasoning_items = [o for o in body["output"] if o["type"] == "reasoning"]
        assert reasoning_items, body["output"]
        summary_text = reasoning_items[0]["summary"][0]["text"]

        # Same engine's reasoning_text — adapter must carry the bytes
        # over verbatim (or as a strict prefix when summarisation
        # lands). For the no-summarisation release, bytes match.
        assert reasoning in summary_text


# ---------------------------------------------------------------------------
# Yuki F6 — tool_choice enforcement on /v1/responses
# ---------------------------------------------------------------------------


class TestF6ToolChoiceEnforcement:
    """Yuki F6 (0.8.5 dogfood): ``tool_choice="required"`` and the
    named-function form returned a plain message instead of a forced
    ``function_call``. The fix mirrors the chat-route post-parse
    synthesis: when the model returns no tool_calls under a forced
    choice, synthesise a stub call so the OpenAI guarantee holds.
    """

    _PING_TOOL = {
        "type": "function",
        "name": "ping",
        "parameters": {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
    }

    def test_required_with_single_tool_synthesises_function_call(
        self, make_responses_client
    ):
        state = make_responses_client(text="just chatty text", tool_calls=None)

        resp = state.client.post(
            "/v1/responses",
            json=_payload(
                tools=[self._PING_TOOL],
                tool_choice="required",
            ),
            headers=_AUTH,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        function_calls = [o for o in body["output"] if o["type"] == "function_call"]
        assert function_calls, body["output"]
        assert function_calls[0]["name"] == "ping"

    def test_named_function_choice_synthesises_call_to_named_tool(
        self, make_responses_client
    ):
        state = make_responses_client(text="text-only reply", tool_calls=None)

        resp = state.client.post(
            "/v1/responses",
            json=_payload(
                tools=[
                    self._PING_TOOL,
                    {"type": "function", "name": "pong", "parameters": {}},
                ],
                tool_choice={"type": "function", "name": "pong"},
            ),
            headers=_AUTH,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        function_calls = [o for o in body["output"] if o["type"] == "function_call"]
        assert function_calls, body["output"]
        assert function_calls[0]["name"] == "pong"

    def test_required_without_tools_is_400(self, make_responses_client):
        state = make_responses_client(text="hi")

        resp = state.client.post(
            "/v1/responses",
            json=_payload(tool_choice="required"),
            headers=_AUTH,
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "required" in body["error"]["message"].lower()

    def test_named_function_not_in_tools_is_400(self, make_responses_client):
        state = make_responses_client(text="hi")

        resp = state.client.post(
            "/v1/responses",
            json=_payload(
                tools=[self._PING_TOOL],
                tool_choice={"type": "function", "name": "not_present"},
            ),
            headers=_AUTH,
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "not_present" in body["error"]["message"]

    def test_auto_does_not_synthesise_when_model_returns_text(
        self, make_responses_client
    ):
        """Sanity: only the forced shapes coerce — ``auto`` lets the
        model decide and we don't inject phantom tool_calls."""
        state = make_responses_client(text="hello", tool_calls=None)

        resp = state.client.post(
            "/v1/responses",
            json=_payload(tools=[self._PING_TOOL], tool_choice="auto"),
            headers=_AUTH,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert not [o for o in body["output"] if o["type"] == "function_call"]

    def test_required_with_multiple_tools_no_call_returns_422(
        self, make_responses_client
    ):
        """Codex r1 BLOCKING #1 (PR #817): with multiple submitted
        tools, ``required`` can't pick a target deterministically. The
        route 422s (parity with chat.py) instead of silently violating
        the contract.
        """
        state = make_responses_client(text="text-only", tool_calls=None)

        resp = state.client.post(
            "/v1/responses",
            json=_payload(
                tools=[
                    self._PING_TOOL,
                    {"type": "function", "name": "pong", "parameters": {}},
                ],
                tool_choice="required",
            ),
            headers=_AUTH,
        )
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert "tool_choice_required_unfulfilled" in str(body)

    def test_required_streaming_with_synthesis_suppresses_message_item(
        self, make_responses_client
    ):
        """Codex r1 BLOCKING #2 (PR #817): when the streaming model
        emits text under forced choice AND no tool_call, the message
        item must NOT be emitted (the synthesised tool_call is the
        only legitimate output for the contract). Pre-fix the client
        saw BOTH ``response.output_text.delta`` for the model's prose
        AND a synthesised ``function_call`` — violating the
        tool_call-guaranteed contract.
        """
        state = make_responses_client(text="Hello from rapid", tool_calls=None)

        with state.client.stream(
            "POST",
            "/v1/responses",
            json=_payload(
                stream=True,
                tools=[self._PING_TOOL],
                tool_choice="required",
            ),
            headers=_AUTH,
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        names = [n for (n, _) in events]

        # No text deltas should reach the client.
        assert "response.output_text.delta" not in names, names
        # Function-call item IS emitted via the synthesis path.
        assert any(
            d.get("item", {}).get("type") == "function_call"
            for (n, d) in events
            if n == "response.output_item.added"
        ), events

    def test_required_streaming_multi_tool_unfulfilled_emits_response_failed(
        self, make_responses_client
    ):
        """Codex r2 BLOCKING (PR #817): the non-stream path 422s for
        multi-tool ``required`` with no model call, but the streaming
        path cannot raise mid-stream — the SSE headers are already
        committed. Emit a ``response.failed`` event with the same
        error code/message so clients see a clean shutdown.
        """
        state = make_responses_client(text="text-only", tool_calls=None)

        with state.client.stream(
            "POST",
            "/v1/responses",
            json=_payload(
                stream=True,
                tools=[
                    self._PING_TOOL,
                    {"type": "function", "name": "pong", "parameters": {}},
                ],
                tool_choice="required",
            ),
            headers=_AUTH,
        ) as resp:
            assert resp.status_code == 200, resp.text
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        failed = [d for (n, d) in events if n == "response.failed"]
        assert failed, [n for (n, _) in events]
        assert failed[0]["response"]["error"]["code"] == (
            "tool_choice_required_unfulfilled"
        )

    def test_required_streaming_with_real_tool_call_keeps_text(
        self, make_responses_client
    ):
        """When the model produces a real tool_call AND text under
        forced choice, the buffered text is flushed (no synthesis
        fires, so the assistant's prose stays as legitimate context).
        """
        state = make_responses_client(
            text="Hello from rapid",
            tool_calls=[
                _make_function_call(
                    "ping", json.dumps({"msg": "hi"}), call_id="call_real"
                )
            ],
            finish_reason="tool_calls",
        )

        with state.client.stream(
            "POST",
            "/v1/responses",
            json=_payload(
                stream=True,
                tools=[self._PING_TOOL],
                tool_choice="required",
            ),
            headers=_AUTH,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        names = [n for (n, _) in events]

        # Text delta flushed because the model produced a real call.
        assert "response.output_text.delta" in names, names
        # And the real call is shipped.
        assert any(
            d.get("item", {}).get("type") == "function_call"
            for (n, d) in events
            if n == "response.output_item.added"
        ), events


# ---------------------------------------------------------------------------
# Ana C-06 — Computer-Use (UI-TARS) reachability via /v1/responses
# ---------------------------------------------------------------------------


class TestC06ComputerUseReachability:
    """Ana C-06 (0.8.5 dogfood): UI-TARS was unreachable via
    /v1/responses. The documented input shape returned 400 and
    ``computer_call`` output items were never emitted. The fix:

    * The route accepts ``tools=[{"type":"computer_20251022", ...}]``.
    * When the underlying parser emits ``function.name == "computer"``,
      the adapter translates it to a ``computer_call`` output item with
      an OpenAI-spec ``action`` envelope.
    """

    _COMPUTER_TOOL = {
        "type": "computer_20251022",
        "name": "computer",
        "display_width": 1280,
        "display_height": 800,
        "environment": "linux",
    }

    def test_computer_use_tool_type_accepted_no_400(self, make_responses_client):
        state = make_responses_client(text="ok, ready")

        resp = state.client.post(
            "/v1/responses",
            json=_payload(tools=[self._COMPUTER_TOOL]),
            headers=_AUTH,
        )
        assert resp.status_code == 200, resp.text

    def test_computer_call_emitted_when_parser_returns_computer_function(
        self, make_responses_client
    ):
        """Simulate UI-TARS parser output (function.name=='computer',
        canonical JSON args ``{"action": "click", "start_box": [128, 128]}``)
        — the response envelope must emit a ``computer_call`` item with
        the action shape per the OpenAI Computer-Use SDK contract."""
        state = make_responses_client(
            text="",
            tool_calls=[
                _make_function_call(
                    "computer",
                    json.dumps({"action": "click", "start_box": [128, 128]}),
                    call_id="call_ui_tars",
                )
            ],
            finish_reason="tool_calls",
        )

        resp = state.client.post(
            "/v1/responses",
            json=_payload(tools=[self._COMPUTER_TOOL]),
            headers=_AUTH,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        computer_calls = [o for o in body["output"] if o["type"] == "computer_call"]
        assert computer_calls, body["output"]
        cc = computer_calls[0]
        assert cc["action"]["type"] == "click"
        assert cc["action"]["start_box"] == [128, 128]

    def test_function_tool_without_computer_use_still_emits_function_call(
        self, make_responses_client
    ):
        """When the request did NOT submit a computer_20251022 tool, a
        ``function`` call named ``"computer"`` is NOT mistaken for a
        Computer-Use action — it stays a regular ``function_call``."""
        state = make_responses_client(
            text="",
            tool_calls=[
                _make_function_call("computer", '{"action":"click"}', call_id="call_x")
            ],
            finish_reason="tool_calls",
        )

        resp = state.client.post(
            "/v1/responses",
            json=_payload(
                tools=[{"type": "function", "name": "computer", "parameters": {}}],
            ),
            headers=_AUTH,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert [o for o in body["output"] if o["type"] == "function_call"]
        assert not [o for o in body["output"] if o["type"] == "computer_call"]


# ---------------------------------------------------------------------------
# Yuki F8 — streaming SSE event coverage
# ---------------------------------------------------------------------------


class TestF8StreamingSseEvents:
    """Yuki F8 (0.8.5 dogfood): the streaming /v1/responses pipeline
    was missing ``response.content_part.added`` and
    ``response.output_text.done`` per OpenAI Responses SSE spec.
    Clients gating UI state on those events never saw them.
    """

    def test_stream_emits_content_part_added_and_output_text_done(
        self, make_responses_client
    ):
        state = make_responses_client(text="Hello from rapid")

        with state.client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers=_AUTH,
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())

        events = _parse_sse(body)
        event_names = [e[0] for e in events]

        assert "response.content_part.added" in event_names, event_names
        assert "response.output_text.done" in event_names, event_names

        # Spec order: created → output_item.added (message) →
        # content_part.added → output_text.delta → output_text.done →
        # content_part.done → output_item.done → completed.
        added_idx = event_names.index("response.output_item.added")
        cpa_idx = event_names.index("response.content_part.added")
        delta_idx = event_names.index("response.output_text.delta")
        otd_idx = event_names.index("response.output_text.done")
        item_done_idx = event_names.index("response.output_item.done")
        completed_idx = event_names.index("response.completed")
        assert (
            added_idx < cpa_idx < delta_idx < otd_idx < item_done_idx < completed_idx
        ), event_names

    def test_output_text_done_carries_full_assistant_text(self, make_responses_client):
        state = make_responses_client()  # default emits "Hello from rapid"

        with state.client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers=_AUTH,
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse(body)
        done_events = [d for (name, d) in events if name == "response.output_text.done"]
        assert done_events
        assert done_events[0]["text"] == "Hello from rapid"


# ---------------------------------------------------------------------------
# Yuki R6 / R7 — truncation + service_tier echo
# ---------------------------------------------------------------------------


class TestR6R7TruncationAndServiceTierEcho:
    """Yuki R6 / R7 (0.8.5 dogfood): ``truncation="auto"`` and
    ``service_tier="flex"`` were silently dropped — neither echoed on
    the response envelope. The fix echoes both; truncation behaviour
    is a no-op for this release (operator preference, see
    ResponsesRequest docstring).
    """

    def test_truncation_echoed_on_response_envelope(self, make_responses_client):
        state = make_responses_client()

        resp = state.client.post(
            "/v1/responses",
            json=_payload(truncation="auto"),
            headers=_AUTH,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("truncation") == "auto"

    def test_service_tier_echoed_on_response_envelope(self, make_responses_client):
        state = make_responses_client()

        resp = state.client.post(
            "/v1/responses",
            json=_payload(service_tier="flex"),
            headers=_AUTH,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("service_tier") == "flex"


# ---------------------------------------------------------------------------
# Yuki F13 — declining unimplemented tool types
# ---------------------------------------------------------------------------


class TestF13UnsupportedToolTypes:
    """Yuki F13 (0.8.5 dogfood): ``web_search`` / ``file_search`` /
    ``code_interpreter`` were silently accepted with 200 and the tools
    were never actually invoked. The fix returns a clean 400 listing
    the supported types.
    """

    @pytest.mark.parametrize(
        "unsupported",
        ["web_search", "file_search", "code_interpreter", "image_generation"],
    )
    def test_unsupported_tool_type_returns_400(
        self, make_responses_client, unsupported
    ):
        state = make_responses_client()

        resp = state.client.post(
            "/v1/responses",
            json=_payload(tools=[{"type": unsupported}]),
            headers=_AUTH,
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert "unsupported_tool_type" in str(body)
        # The error message lists the allowlist so callers can fix
        # their request without guessing.
        assert "function" in body["error"]["message"]
        assert "computer_20251022" in body["error"]["message"]

    def test_function_tool_type_still_accepted(self, make_responses_client):
        state = make_responses_client()

        resp = state.client.post(
            "/v1/responses",
            json=_payload(
                tools=[{"type": "function", "name": "ok_tool", "parameters": {}}]
            ),
            headers=_AUTH,
        )
        assert resp.status_code == 200, resp.text

    def test_computer_20251022_tool_type_still_accepted(self, make_responses_client):
        state = make_responses_client()

        resp = state.client.post(
            "/v1/responses",
            json=_payload(
                tools=[
                    {
                        "type": "computer_20251022",
                        "name": "computer",
                        "display_width": 1280,
                        "display_height": 800,
                        "environment": "linux",
                    }
                ]
            ),
            headers=_AUTH,
        )
        assert resp.status_code == 200, resp.text
