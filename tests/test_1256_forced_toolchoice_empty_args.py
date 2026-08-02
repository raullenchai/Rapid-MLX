# SPDX-License-Identifier: Apache-2.0
"""#1256 — forced ``tool_choice`` must never report a successful tool call
whose synthesised arguments fail the tool's ``required`` schema.

Repro (from the issue): a single ``add`` tool whose schema requires ``a`` and
``b``, prompt "Call add with a=7 and b=8", sent with ``tool_choice="required"``.
The text parser surfaces no call, the forced-choice fallback synthesises one
with empty ``"{}"`` arguments, and the response reports
``finish_reason="tool_calls"`` — so a client executes a call the server already
knows is schema-invalid.

Fix (``routes/chat._forced_synth_schema_error`` + call sites in chat.py /
responses.py): when the synthesised arguments don't provide every ``required``
property, fail EXPLICITLY — 422 on the non-stream paths, drop-the-synth
(``finish_reason="stop"``) on the streaming chat surface, ``response.failed``
on the streaming responses surface — instead of shipping the bad call. A tool
with NO required fields still synthesises ``"{}"`` as before.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import _forced_synth_schema_error
from vllm_mlx.routes.chat import router as chat_router


def _tool(name: str, *, required: list[str] | None, props: dict | None = None):
    """Build a request-tool-shaped object (``.function`` is a dict, matching the
    pydantic ``Tool`` the routes access via ``t.function.get(...)``)."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": props or {"a": {"type": "integer"}, "b": {"type": "integer"}},
    }
    if required is not None:
        schema["required"] = required
    return SimpleNamespace(
        type="function", function={"name": name, "parameters": schema}
    )


# =====================================================================
# Unit: _forced_synth_schema_error decision core (shared by all sites)
# =====================================================================


class TestForcedSynthSchemaError:
    def test_missing_required_all(self):
        tools = [_tool("add", required=["a", "b"])]
        err = _forced_synth_schema_error("add", "{}", tools)
        assert err is not None
        # The draft-aware validator surfaces the first missing required property.
        assert "required" in err.lower()

    def test_missing_required_partial(self):
        tools = [_tool("add", required=["a", "b"])]
        err = _forced_synth_schema_error("add", '{"a": 7}', tools)
        assert err is not None
        assert "b" in err
        # ``a`` was provided so it should not be reported as missing.
        assert "'a'" not in err

    def test_all_required_provided_returns_none(self):
        tools = [_tool("add", required=["a", "b"])]
        assert _forced_synth_schema_error("add", '{"a": 7, "b": 8}', tools) is None

    def test_no_required_fields_returns_none(self):
        tools = [_tool("ping", required=None)]
        assert _forced_synth_schema_error("ping", "{}", tools) is None

    def test_empty_required_list_returns_none(self):
        tools = [_tool("ping", required=[])]
        assert _forced_synth_schema_error("ping", "{}", tools) is None

    def test_unknown_tool_name_returns_none(self):
        tools = [_tool("add", required=["a", "b"])]
        assert _forced_synth_schema_error("other", "{}", tools) is None

    def test_malformed_arguments_treated_as_empty(self):
        tools = [_tool("add", required=["a", "b"])]
        # Non-JSON / non-object args provide nothing → required unmet.
        assert _forced_synth_schema_error("add", "not json", tools) is not None
        assert _forced_synth_schema_error("add", "[1, 2]", tools) is not None

    def test_no_tools_returns_none(self):
        assert _forced_synth_schema_error("add", "{}", None) is None
        assert _forced_synth_schema_error("add", "{}", []) is None

    def test_dict_shaped_tool_supported(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}},
                        "required": ["a"],
                    },
                },
            }
        ]
        assert _forced_synth_schema_error("add", "{}", tools) is not None
        assert _forced_synth_schema_error("add", '{"a": 1}', tools) is None

    def _tool_with_schema(self, name, schema):
        return SimpleNamespace(
            type="function", function={"name": name, "parameters": schema}
        )

    def test_composed_required_under_allof_is_caught(self):
        # Tier 2: `required` nested under allOf has no top-level `required`,
        # so only full draft-aware validation catches the empty synth.
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "allOf": [{"required": ["city"]}],
        }
        tool = self._tool_with_schema("weather", schema)
        assert _forced_synth_schema_error("weather", "{}", [tool]) is not None
        assert _forced_synth_schema_error("weather", '{"city": "SF"}', [tool]) is None

    def test_enum_violation_in_recovered_args_is_caught(self):
        # Tier 2: no required fields, but a recovered value violates an enum.
        schema = {
            "type": "object",
            "properties": {"unit": {"enum": ["c", "f"]}},
        }
        tool = self._tool_with_schema("weather", schema)
        assert _forced_synth_schema_error("weather", '{"unit": "kelvin"}', [tool])
        assert _forced_synth_schema_error("weather", '{"unit": "c"}', [tool]) is None
        # No enum-constrained field present → empty synth is fine.
        assert _forced_synth_schema_error("weather", "{}", [tool]) is None

    def test_malformed_tool_schema_fails_open(self):
        # A tool schema our validator can't evaluate must NOT block the synth
        # (fail-open): return None rather than 422 on our own limitation.
        bad = self._tool_with_schema("x", {"type": "not-a-real-type"})
        assert _forced_synth_schema_error("x", "{}", [bad]) is None

    def test_malformed_required_entry_does_not_crash(self):
        # codex r4 MAJOR: a non-string `required` entry (``[["a"]]``) must not
        # raise TypeError on the Tier-1 membership test; it fails open.
        bad = self._tool_with_schema("x", {"type": "object", "required": [["a"]]})
        assert _forced_synth_schema_error("x", "{}", [bad]) is None

    def test_non_object_instance_validated_against_its_own_schema(self):
        # codex r2 MAJOR: validate the ACTUAL decoded value, not a coerced {}.
        # An array instance is valid against an array schema (no false reject)...
        arr_schema = {"type": "array", "items": {"type": "integer"}}
        arr_tool = self._tool_with_schema("nums", arr_schema)
        assert _forced_synth_schema_error("nums", "[1, 2]", [arr_tool]) is None
        # ...and invalid against an object schema (no false accept).
        obj_schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        obj_tool = self._tool_with_schema("obj", obj_schema)
        assert _forced_synth_schema_error("obj", "[1, 2]", [obj_tool]) is not None
        # Literal JSON ``null`` decodes to a real value, not a parse failure —
        # invalid against an object schema (codex r3 MAJOR).
        assert _forced_synth_schema_error("obj", "null", [obj_tool]) is not None


# =====================================================================
# Non-stream chat route: repro + control (TestClient + fake engine)
# =====================================================================

_ADD_TOOL_REQUIRED = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
]

_PING_TOOL_NO_REQUIRED = [
    {
        "type": "function",
        "function": {
            "name": "ping",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


class _PlainTextEngine:
    """Fake engine whose model returns plain prose with NO tool-call markers —
    the text parser extracts nothing, so the forced-choice fallback path runs."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None
    supports_tool_calls = True

    def __init__(self, raw: str = "The sum is 15."):
        self._text = raw
        self._raw_text = raw

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text=self._text,
            raw_text=self._raw_text,
            prompt_tokens=4,
            completion_tokens=8,
            finished=True,
            finish_reason="stop",
        )

    async def stream_chat(self, **kwargs):
        for i, tok in enumerate(self._text.split()):
            yield _FakeStreamingOutput(
                (tok + " "), finished=(i == len(self._text.split()) - 1)
            )


def _make_client(engine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "qwen3-0.6b-4bit"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.tool_call_parser = "hermes"
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def _body(tool_choice, tools) -> dict:
    return {
        "model": "qwen3-0.6b-4bit",
        "messages": [{"role": "user", "content": "Call add with a=7 and b=8."}],
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": 64,
    }


def test_nonstream_required_missing_args_is_422_not_empty_call():
    """``tool_choice="required"`` + a schema with required fields the model
    didn't provide → 422, NOT a ``finish_reason="tool_calls"`` with ``{}``."""
    client = _make_client(_PlainTextEngine())
    resp = client.post(
        "/v1/chat/completions", json=_body("required", _ADD_TOOL_REQUIRED)
    )
    assert resp.status_code == 422, resp.text
    assert "1256" in resp.text or "required" in resp.text.lower()


def test_nonstream_named_pin_missing_args_is_422():
    """Named-function ``tool_choice`` with required fields unmet → 422."""
    client = _make_client(_PlainTextEngine())
    resp = client.post(
        "/v1/chat/completions",
        json=_body(
            {"type": "function", "function": {"name": "add"}}, _ADD_TOOL_REQUIRED
        ),
    )
    assert resp.status_code == 422, resp.text


def test_nonstream_no_required_still_synthesizes_empty_call():
    """Control: a tool with NO required fields still synthesises a ``{}`` call
    (the #571/#447 forced-choice guarantee is preserved for that case)."""
    client = _make_client(_PlainTextEngine())
    resp = client.post(
        "/v1/chat/completions", json=_body("required", _PING_TOOL_NO_REQUIRED)
    )
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "ping"
    assert tcs[0]["function"]["arguments"] in ("{}", "")


# =====================================================================
# Streaming chat route: repro + control
# =====================================================================


class _FakeStreamingOutput:
    def __init__(self, new_text: str, finished: bool):
        self.new_text = new_text
        self.text = new_text
        self.finished = finished
        self.finish_reason = "stop" if finished else None
        self.channel = None
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.cached_tokens = 0
        self.tokens = []
        self.logprobs = None
        self.tool_calls = None
        self.matched_stop = None
        self.raw_text = new_text


class _FakeStreamEngine:
    def __init__(self, deltas: list[str]):
        self._deltas = deltas
        self.tokenizer = None
        self.is_mllm = False
        self.supports_tool_calls = True
        self.supports_guided_generation = False

    async def stream_chat(self, **kwargs):
        for i, d in enumerate(self._deltas):
            yield _FakeStreamingOutput(d, finished=(i == len(self._deltas) - 1))

    def build_prompt(self, *args, **kwargs):
        return "prompt"


def _drive_stream(engine, request) -> tuple[list[dict], str | None]:
    from vllm_mlx.routes.chat import stream_chat_completion

    chunks: list[dict] = []

    async def _run():
        gen = stream_chat_completion(
            engine, [{"role": "user", "content": "hi"}], request
        )
        async for sse in gen:
            line = sse.strip()
            if not line.startswith("data: "):
                continue
            body = line[len("data: ") :]
            if body == "[DONE]":
                break
            try:
                chunks.append(json.loads(body))
            except json.JSONDecodeError:
                continue

    asyncio.run(_run())
    finish = None
    for c in reversed(chunks):
        choices = c.get("choices") or []
        if choices and choices[0].get("finish_reason"):
            finish = choices[0]["finish_reason"]
            break
    return chunks, finish


def _stream_request(tool_choice, tools):
    return ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Call add with a=7 and b=8."}],
        tools=tools,
        tool_choice=tool_choice,
        stream=True,
        max_tokens=50,
        chat_template_kwargs={"enable_thinking": False},
    )


@pytest.fixture()
def _qwen_hermes_cfg(monkeypatch):
    from vllm_mlx.config import server_config

    cfg = server_config.get_config()
    monkeypatch.setattr(cfg, "tool_call_parser", "hermes", raising=False)
    monkeypatch.setattr(cfg, "reasoning_parser_name", "qwen3", raising=False)
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", True, raising=False)
    monkeypatch.setattr(cfg, "gc_control", False, raising=False)
    yield


def _emitted_tool_calls(chunks) -> list[dict]:
    out: list[dict] = []
    for c in chunks:
        for ch in c.get("choices") or []:
            tcs = (ch.get("delta") or {}).get("tool_calls")
            if tcs:
                out.extend(tcs)
    return out


def test_stream_required_missing_args_does_not_fabricate_call(_qwen_hermes_cfg):
    """Streaming ``required`` + required-field schema the model didn't fill →
    NO synthesised ``delta.tool_calls`` (headers are out so we can't 422);
    finish is ``stop``, never ``tool_calls`` with ``{}``."""
    # Plain prose — the parser detects no call and cannot recover arguments.
    deltas = ["The ", "sum ", "is ", "15."]
    chunks, finish = _drive_stream(
        _FakeStreamEngine(deltas), _stream_request("required", _ADD_TOOL_REQUIRED)
    )
    assert _emitted_tool_calls(chunks) == [], (
        "streaming forced synth must not fabricate a schema-invalid empty call"
    )
    assert finish != "tool_calls"


def test_stream_no_required_still_synthesizes(_qwen_hermes_cfg):
    """Control: streaming ``required`` on a NO-required tool still synthesises
    the guaranteed call (preserves #447)."""
    deltas = ["The ", "answer", "."]
    chunks, finish = _drive_stream(
        _FakeStreamEngine(deltas), _stream_request("required", _PING_TOOL_NO_REQUIRED)
    )
    emitted = _emitted_tool_calls(chunks)
    assert emitted, "no-required forced synth must still fire"
    assert emitted[0]["function"]["name"] == "ping"
    assert finish == "tool_calls"


# =====================================================================
# Responses route: _enforce_responses_tool_choice raises on unsatisfiable
# =====================================================================


def test_responses_enforce_raises_on_required_schema_unmet():
    from fastapi import HTTPException

    from vllm_mlx.api.responses_models import ResponsesRequest
    from vllm_mlx.routes.responses import _enforce_responses_tool_choice

    openai_req = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "Call add with a=7 and b=8."}],
        tools=_ADD_TOOL_REQUIRED,
        tool_choice="required",
    )
    resp_req = ResponsesRequest(model="test", input="x", tool_choice="required")

    with pytest.raises(HTTPException) as ei:
        _enforce_responses_tool_choice([], resp_req, openai_req)
    assert ei.value.status_code == 422


def test_responses_enforce_synthesizes_when_no_required():
    from vllm_mlx.api.responses_models import ResponsesRequest
    from vllm_mlx.routes.responses import _enforce_responses_tool_choice

    openai_req = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "ping"}],
        tools=_PING_TOOL_NO_REQUIRED,
        tool_choice="required",
    )
    resp_req = ResponsesRequest(model="test", input="x", tool_choice="required")

    out = _enforce_responses_tool_choice([], resp_req, openai_req)
    assert out and out[0].function.name == "ping"
