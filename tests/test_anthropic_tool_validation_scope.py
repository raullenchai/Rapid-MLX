# SPDX-License-Identifier: Apache-2.0
"""H-05: ``/v1/messages`` tool-arg validation scope.

PR #742 (F-220) wired ``_validate_tool_call_params`` into ``/v1/messages``
so a model emitting schema-violating tool arguments would 400 instead of
silently returning a broken ``tool_use`` block. Sergei (r2) immediately
hit a regression: with two tools defined and
``tool_choice={"type":"tool","name":"get_weather"}``, the validator
fired on the un-pinned tool (``lookup_zip``) and 400-ed the entire
request — even though the user had pinned ``get_weather`` and the model
DID emit a correct ``get_weather`` call alongside the bad
``lookup_zip`` call.

The fix has two parts:

1. ``_validate_tool_call_params`` is refactored to per-call iteration:
   for each emitted tool_call, find the matching tool spec by name and
   validate that call's arguments against THAT spec only. Tool specs the
   model did not call are never consulted. (See
   ``service/helpers.py:_validate_tool_call_params``.)

2. ``/v1/messages`` (both branches) drops tool_calls that don't match a
   pinned ``tool_choice={"type":"tool","name":X}`` BEFORE running the
   validator. This realises the user-visible contract: "you asked for
   X, the response carries X" — and keeps the validator scoped to the
   pinned tool even when the model emits extras. (See
   ``routes/anthropic.py:_filter_tool_calls_by_tool_choice``.)

These tests lock the five cases the H-05 ticket enumerates plus a
regression for the original Sergei repro (two tools, pinned one,
model defies the pin and fires both).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.routes.anthropic import router


class _StubTokenizer:
    chat_template = ""


class _MultiCallEngine:
    """Stub engine whose ``chat()`` returns a pre-canned tool_calls payload.

    The Anthropic non-stream branch reads ``output.tool_calls`` first
    via ``getattr(output, "tool_calls", None)`` and feeds it into
    ``_parse_tool_calls_with_parser`` as ``structured_tool_calls`` —
    bypassing the text-based parser entirely. By handing back a list
    of pre-shaped ``{name, arguments}`` dicts we drive the route
    through the same path a real harmony / structured-engine surface
    would.
    """

    preserve_native_tool_format = False
    tokenizer = _StubTokenizer()

    def __init__(self, tool_calls_payload: list[dict] | None, text: str = ""):
        self._tool_calls = tool_calls_payload
        self._text = text

    async def chat(self, messages, **kwargs):  # noqa: ARG002
        return SimpleNamespace(
            text=self._text,
            raw_text=self._text,
            tool_calls=self._tool_calls,
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="tool_calls" if self._tool_calls else "stop",
            reasoning_text="",
        )

    async def stream_chat(self, messages, **kwargs):  # noqa: ARG002
        # Emit text first (mirrors how a real streaming engine
        # surfaces text tokens chunk-by-chunk before any tool_calls
        # are finalised by the harmony parser). The route's chunk
        # loop short-circuits with ``continue`` on any chunk that
        # carries ``tool_calls``, so text + tool_calls must be on
        # SEPARATE chunks for both paths to exercise.
        if self._text:
            yield SimpleNamespace(
                new_text=self._text,
                prompt_tokens=10,
                completion_tokens=1,
                tool_calls=None,
            )
        if self._tool_calls:
            yield SimpleNamespace(
                new_text="",
                prompt_tokens=10,
                completion_tokens=1,
                tool_calls=self._tool_calls,
            )


def _make_client(engine: _MultiCallEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.no_thinking = True
    cfg.reasoning_parser_name = None
    cfg.model_registry = None

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_server_config():
    reset_config()
    yield
    reset_config()


# Two-tool fixture: ``get_weather`` (open string ``location``) +
# ``lookup_zip`` (strictly-typed ``zip`` with min/maxLength=5). The
# ``lookup_zip`` schema is the one Sergei's repro tripped over — the
# model emits an integer for ``zip`` instead of the declared string.
_GET_WEATHER = {
    "name": "get_weather",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}
_GET_WEATHER_OPTIONAL = {
    "name": "get_weather",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
    },
}
_LOOKUP_ZIP = {
    "name": "lookup_zip",
    "input_schema": {
        "type": "object",
        "properties": {
            "zip": {"type": "string", "minLength": 5, "maxLength": 5},
        },
        "required": ["zip"],
    },
}


def _call(name: str, arguments: dict) -> dict:
    """Flat ``{name, arguments}`` shape — matches harmony's
    ``StreamableParser`` output and is what
    ``_parse_tool_calls_with_parser`` wraps into ``ToolCall``."""
    return {"name": name, "arguments": json.dumps(arguments)}


def _post_messages(
    client: TestClient, tool_choice=None, *, stream: bool = False
) -> dict:
    body = {
        "model": "test-model",
        "max_tokens": 32,
        "stream": stream,
        "messages": [{"role": "user", "content": "weather + zip lookup"}],
        "tools": [_GET_WEATHER, _LOOKUP_ZIP],
    }
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    return body


# ---------------------------------------------------------------------------
# H-05 scope tests
# ---------------------------------------------------------------------------


def test_no_tool_choice_called_tool_args_ok_returns_200():
    """Two tools declared, model calls ``get_weather`` correctly → 200.

    The validator must not consult ``lookup_zip``'s schema because the
    model never called it.
    """
    engine = _MultiCallEngine([_call("get_weather", {"location": "SF"})])
    client = _make_client(engine)

    response = client.post("/v1/messages", json=_post_messages(client))

    assert response.status_code == 200, response.text
    body = response.json()
    tool_uses = [b for b in body["content"] if b["type"] == "tool_use"]
    assert len(tool_uses) == 1
    assert tool_uses[0]["name"] == "get_weather"
    assert tool_uses[0]["input"] == {"location": "SF"}


def test_no_tool_choice_called_tool_args_bad_returns_400_about_called_tool():
    """Two tools, model calls ``get_weather`` with a schema-violating
    argument → 400 about ``get_weather`` (NOT ``lookup_zip``).

    Lock down: when the validator does fire, the error message must
    name the tool the model actually called. Anything mentioning the
    un-called tool would mean we leaked schema scope.
    """
    # ``location`` declared as string; emit an integer to trip the type check.
    engine = _MultiCallEngine([_call("get_weather", {"location": 42})])
    client = _make_client(engine)

    response = client.post("/v1/messages", json=_post_messages(client))

    assert response.status_code == 400, response.text
    body = response.json()
    message = body.get("detail") or body.get("error", {}).get("message", "")
    assert "get_weather" in message, message
    assert "lookup_zip" not in message, message


def test_no_tool_choice_text_only_no_validation_returns_200():
    """Two tools declared, model emits text and no tool_use → no
    validation, 200. Validator must not fire when ``tool_calls`` is
    empty even if ``request.tools`` is non-empty.
    """
    engine = _MultiCallEngine(None, text="No tool needed for this.")
    client = _make_client(engine)

    response = client.post("/v1/messages", json=_post_messages(client))

    assert response.status_code == 200, response.text
    body = response.json()
    # Either no tool_use blocks at all, or — with no tool_calls
    # surfaced — Anthropic's normal text-content response.
    assert not any(b["type"] == "tool_use" for b in body["content"])


def test_tool_choice_pinned_called_tool_ok_returns_200():
    """``tool_choice={type:tool,name:get_weather}`` + model calls
    ``get_weather`` correctly → 200. Sergei's expected happy path.
    """
    engine = _MultiCallEngine([_call("get_weather", {"location": "SF"})])
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json=_post_messages(
            client, tool_choice={"type": "tool", "name": "get_weather"}
        ),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    tool_uses = [b for b in body["content"] if b["type"] == "tool_use"]
    assert len(tool_uses) == 1
    assert tool_uses[0]["name"] == "get_weather"


def test_tool_choice_pinned_called_tool_bad_args_returns_400_about_pinned_tool():
    """``tool_choice={type:tool,name:get_weather}`` + model calls
    ``get_weather`` with schema-violating arguments → 400 about
    ``get_weather``. The pinned tool's own schema is still enforced.
    """
    engine = _MultiCallEngine([_call("get_weather", {"location": 42})])
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json=_post_messages(
            client, tool_choice={"type": "tool", "name": "get_weather"}
        ),
    )

    assert response.status_code == 400, response.text
    body = response.json()
    message = body.get("detail") or body.get("error", {}).get("message", "")
    assert "get_weather" in message
    assert "lookup_zip" not in message


# ---------------------------------------------------------------------------
# Sergei's original repro — model defies tool_choice and fires both tools.
# Pre-fix this returned 400 about ``lookup_zip.zip``; post-fix the un-pinned
# call is dropped and the user gets the pinned tool back at 200.
# ---------------------------------------------------------------------------


def test_sergei_repro_pinned_get_weather_model_emits_both_returns_200_with_pinned_only():
    """The bug headline:

      * ``tool_choice={"type":"tool","name":"get_weather"}``
      * Model emits ``get_weather`` (good) + ``lookup_zip`` with an
        integer ``zip`` (bad — string declared).

    Pre-fix: 400 with detail "Tool call 'lookup_zip' parameter 'zip'
    violates declared schema: Expected string, got int." (validation
    leaked onto the un-pinned tool).

    Post-fix: 200 with a single ``tool_use`` block for ``get_weather``.
    The un-pinned ``lookup_zip`` call is dropped server-side and the
    validator never sees its schema.
    """
    engine = _MultiCallEngine(
        [
            _call("get_weather", {"location": "SF"}),
            _call("lookup_zip", {"zip": 941}),
        ]
    )
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json=_post_messages(
            client, tool_choice={"type": "tool", "name": "get_weather"}
        ),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    tool_uses = [b for b in body["content"] if b["type"] == "tool_use"]
    # Only the pinned tool survives; the un-pinned ``lookup_zip`` is dropped.
    assert len(tool_uses) == 1
    assert tool_uses[0]["name"] == "get_weather"
    assert tool_uses[0]["input"] == {"location": "SF"}
    assert body["stop_reason"] == "tool_use"


def test_sergei_repro_streaming_pinned_get_weather_model_emits_both_returns_200_with_pinned_only():
    """Streaming variant of the Sergei repro.

    Pre-fix: SSE ``event: error`` with the lookup_zip schema-violation
    message (the F-220 streaming branch routed the validator's
    HTTPException through the error-event path).

    Post-fix: the un-pinned lookup_zip is dropped before validation,
    so the stream emits a clean ``tool_use`` content block for
    ``get_weather`` and terminates with ``stop_reason=tool_use``.
    """
    engine = _MultiCallEngine(
        [
            _call("get_weather", {"location": "SF"}),
            _call("lookup_zip", {"zip": 941}),
        ]
    )
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json=_post_messages(
            client,
            tool_choice={"type": "tool", "name": "get_weather"},
            stream=True,
        ),
    )

    assert response.status_code == 200
    raw = response.text
    # No error event — the un-pinned call was dropped silently.
    assert "event: error" not in raw, raw
    # The pinned tool's content block IS emitted.
    assert '"type": "tool_use"' in raw
    assert "get_weather" in raw
    # And the un-pinned tool's name MUST NOT survive into the stream
    # (otherwise we'd be shipping the dropped call to the client).
    assert "lookup_zip" not in raw, raw
    assert '"stop_reason": "tool_use"' in raw


# ---------------------------------------------------------------------------
# Direct unit tests on the refactored ``_validate_tool_call_params`` so the
# "per-call scope" property is locked at the helper boundary, independent
# of the route plumbing.
# ---------------------------------------------------------------------------


def _tool_def(name: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": properties},
        },
    }


def test_validator_only_checks_called_tools_schema():
    """Validator must not 400 on a non-called tool's bad-looking
    schema. We pass two tools but the model only called the first
    — the second tool's schema must be ignored entirely.
    """
    from vllm_mlx.api.models import FunctionCall, ToolCall
    from vllm_mlx.service.helpers import _validate_tool_call_params

    tools = [
        _tool_def("get_weather", {"location": {"type": "string"}}),
        _tool_def(
            "lookup_zip",
            {"zip": {"type": "string", "minLength": 5, "maxLength": 5}},
        ),
    ]
    calls = [
        ToolCall(
            id="c1",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"location": "SF"}'),
        )
    ]

    # No exception — only ``get_weather``'s schema is consulted.
    _validate_tool_call_params(calls, tools)


def test_validator_called_unknown_tool_skips_validation():
    """If the model calls a tool that isn't in the ``tools`` list at
    all (parser hallucination), the validator skips it — there is no
    schema to validate against, and the upstream parser layers own
    that failure mode.
    """
    from vllm_mlx.api.models import FunctionCall, ToolCall
    from vllm_mlx.service.helpers import _validate_tool_call_params

    tools = [_tool_def("get_weather", {"location": {"type": "string"}})]
    calls = [
        ToolCall(
            id="c1",
            type="function",
            function=FunctionCall(name="ghost_tool", arguments='{"x": 1}'),
        )
    ]

    _validate_tool_call_params(calls, tools)  # no raise


def test_validator_multiple_calls_each_validated_against_own_schema():
    """Two emitted calls to two different tools, both with valid args
    → no raise. Locks that the per-call loop is using each call's own
    tool spec rather than mixing schemas between calls.
    """
    from vllm_mlx.api.models import FunctionCall, ToolCall
    from vllm_mlx.service.helpers import _validate_tool_call_params

    tools = [
        _tool_def("get_weather", {"location": {"type": "string"}}),
        _tool_def(
            "lookup_zip",
            {"zip": {"type": "string", "minLength": 5, "maxLength": 5}},
        ),
    ]
    calls = [
        ToolCall(
            id="c1",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"location": "SF"}'),
        ),
        ToolCall(
            id="c2",
            type="function",
            function=FunctionCall(name="lookup_zip", arguments='{"zip": "94105"}'),
        ),
    ]

    _validate_tool_call_params(calls, tools)


def test_validator_multiple_calls_bad_one_raises_about_bad_one_only():
    """Same as above but the second call's args are bad. The 400 must
    name the bad-call's tool, NOT mix names in the message.
    """
    from fastapi import HTTPException

    from vllm_mlx.api.models import FunctionCall, ToolCall
    from vllm_mlx.service.helpers import _validate_tool_call_params

    tools = [
        _tool_def("get_weather", {"location": {"type": "string"}}),
        _tool_def(
            "lookup_zip",
            {"zip": {"type": "string", "minLength": 5, "maxLength": 5}},
        ),
    ]
    calls = [
        ToolCall(
            id="c1",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"location": "SF"}'),
        ),
        ToolCall(
            id="c2",
            type="function",
            function=FunctionCall(name="lookup_zip", arguments='{"zip": 941}'),
        ),
    ]

    with pytest.raises(HTTPException) as exc_info:
        _validate_tool_call_params(calls, tools)
    assert exc_info.value.status_code == 400
    assert "lookup_zip" in exc_info.value.detail
    assert "get_weather" not in exc_info.value.detail


# ---------------------------------------------------------------------------
# PR #763 codex round-1 BLOCKING #1 — when ``tool_choice`` pins X but the
# filter empties the call list (model emitted zero X calls), the route
# MUST 422 instead of silently 200-ing with no tool_use. Two emit-shapes
# of this failure mode:
#
#   * model emitted text only (count == 0): pin defied, text returned.
#   * model emitted only wrong-tool calls (count > 0): pin defied, every
#     call dropped by the filter.
#
# Both leave the user with no ``tool_use`` for the tool they pinned,
# which violates the forced-tool contract.
# ---------------------------------------------------------------------------


def test_pinned_tool_required_schema_model_emits_no_tool_calls_returns_422():
    """Pinned required-schema tool + model returns text only → 422.

    Empty synthesized input would violate the tool schema, so this remains
    an explicit error instead of shipping an invalid ``tool_use``.
    """
    engine = _MultiCallEngine(None, text="I can't help with weather right now.")
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json=_post_messages(
            client, tool_choice={"type": "tool", "name": "get_weather"}
        ),
    )

    assert response.status_code == 422, response.text
    assert "tool_choice" in response.json()["detail"]
    assert "get_weather" in response.json()["detail"]


def test_pinned_tool_optional_schema_model_emits_no_tool_calls_synthesizes_200():
    """Pinned optional-schema tool can synthesize an empty ``tool_use``."""
    engine = _MultiCallEngine(None, text="I can't help with weather right now.")
    client = _make_client(engine)
    body = _post_messages(client, tool_choice={"type": "tool", "name": "get_weather"})
    body["tools"] = [_GET_WEATHER_OPTIONAL, _LOOKUP_ZIP]

    response = client.post("/v1/messages", json=body)

    assert response.status_code == 200, response.text
    tool_blocks = [b for b in response.json()["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "get_weather"
    assert tool_blocks[0]["input"] == {}


def test_pinned_tool_required_schema_model_emits_only_wrong_tool_returns_422():
    """Pinned required-schema tool + model fires only wrong tool → 422.

    The un-pinned ``lookup_zip`` call is dropped by the filter, then the
    required pinned call cannot be safely synthesized with empty input.
    """
    engine = _MultiCallEngine([_call("lookup_zip", {"zip": "94105"})])
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json=_post_messages(
            client, tool_choice={"type": "tool", "name": "get_weather"}
        ),
    )

    assert response.status_code == 422, response.text
    assert "tool_choice" in response.json()["detail"]
    assert "get_weather" in response.json()["detail"]


def test_pinned_tool_required_schema_streaming_no_calls_emits_sse_error():
    """Streaming required-schema variant: pinned tool, text only → SSE error.

    PR #771 round-2 BLOCKING #1 invariant remains in force: the
    chunk loop's text deltas must NEVER reach the wire when the
    contract was defied.
    """
    distinctive_text = "ANTHROPIC_TEXT_LEAK_CANARY_xyzzy"
    engine = _MultiCallEngine(None, text=distinctive_text)
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json=_post_messages(
            client,
            tool_choice={"type": "tool", "name": "get_weather"},
            stream=True,
        ),
    )

    assert response.status_code == 200, response.text
    raw = response.text
    assert "event: error" in raw, raw
    assert "tool_choice" in raw
    # No fabricated tool_use for the pinned tool is emitted.
    assert '"type": "tool_use"' not in raw, raw
    assert "get_weather" in raw
    assert '"stop_reason": "tool_use"' not in raw
    # The model's forbidden text MUST NOT appear anywhere in the
    # stream — neither as a ``text_delta``, ``content_block_start``
    # of type ``text``, nor in the raw SSE payload. The distinctive
    # text token makes the leak detectable independent of framing
    # (PR #771 round-2 BLOCKING #1 invariant survives F8).
    assert distinctive_text not in raw, raw
    assert '"type": "text_delta"' not in raw, raw
    # ``stop_reason`` includes the literal string ``"text"`` only
    # when synthesizing failed; under F8 we never emit a content
    # block of type "text" on this code path.
    assert (
        '"content_block":' not in raw
        or '"text"' not in raw.split('"content_block":')[1].split("}")[0]
        or '"tool_use"' in raw
    )


def test_pinned_tool_required_schema_streaming_only_wrong_tool_emits_sse_error():
    """Streaming required-schema variant: model fires only wrong tool → SSE error.

    Locks PR #763 round-1 BLOCKING #2 invariant under F8: the stream
    must NOT have shipped any ``tool_use`` content_block_start for
    the dropped tool before the filter ran. Tool_use SSE emits only
    AFTER the filter and the F8 synthesis run, so the only tool_use
    on the wire names the pinned tool.
    """
    engine = _MultiCallEngine([_call("lookup_zip", {"zip": "94105"})])
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json=_post_messages(
            client,
            tool_choice={"type": "tool", "name": "get_weather"},
            stream=True,
        ),
    )

    assert response.status_code == 200, response.text
    raw = response.text
    assert "event: error" in raw, raw
    assert "tool_choice" in raw
    assert '"type": "tool_use"' not in raw, raw
    # The un-pinned tool name MUST NOT appear anywhere — the filter
    # dropped it before the emit loop.
    assert "lookup_zip" not in raw, raw


# ---------------------------------------------------------------------------
# Direct unit tests on the helpers added in PR #763 codex round-1.
# ---------------------------------------------------------------------------


def test_filter_returns_input_unchanged_for_non_named_tool_choice():
    """``tool_choice="auto"`` / ``"required"`` / ``"none"`` / unset has
    no defined "wrong tool" case. The filter must pass calls through
    unchanged so the validator runs against the full set.
    """
    from vllm_mlx.routes.anthropic import _filter_tool_calls_by_tool_choice

    calls = [_call("a", {}), _call("b", {})]
    # ``None`` (unset).
    assert _filter_tool_calls_by_tool_choice(calls, None) == calls
    # ``"auto"`` (string form survives if a caller passed it through).
    assert _filter_tool_calls_by_tool_choice(calls, "auto") == calls
    # ``{"type":"function"}`` with no name → no target → passthrough.
    assert _filter_tool_calls_by_tool_choice(calls, {"type": "function"}) == calls
    # ``{"type":"function","function":{"name":""}}`` → no target.
    assert (
        _filter_tool_calls_by_tool_choice(
            calls, {"type": "function", "function": {"name": ""}}
        )
        == calls
    )


def test_enforce_named_tool_choice_present_noop_for_non_named_choice():
    """The enforcer must pass calls through unchanged when
    ``tool_choice`` doesn't pin a specific tool — there's nothing to
    enforce. Locks against a future bug where adding the enforcer
    would break ``tool_choice="auto"`` flows that the model resolved
    as text-only.

    F8 contract change: the helper now RETURNS ``(tool_calls,
    synthesized)`` instead of raising. The non-pinned case must
    return its input verbatim with ``synthesized=False``.
    """
    from vllm_mlx.routes.anthropic import _enforce_named_tool_choice_present

    # Returns ``([], False)`` verbatim for ``None``, ``"auto"``, or a
    # ``{"type":"function"}`` shape without a name.
    assert _enforce_named_tool_choice_present([], None, original_call_count=0) == (
        [],
        False,
        None,
    )
    assert _enforce_named_tool_choice_present([], "auto", original_call_count=0) == (
        [],
        False,
        None,
    )
    assert _enforce_named_tool_choice_present(
        [], {"type": "function"}, original_call_count=0
    ) == ([], False, None)


def test_enforce_named_tool_choice_present_noop_when_pinned_call_survives():
    """When the filter kept the pinned-tool call, the enforcer must
    pass through silently — the contract is satisfied.

    F8 contract: returns the input list unchanged with
    ``synthesized=False`` (no synthesis fires because the contract is
    already met).
    """
    from vllm_mlx.api.models import FunctionCall, ToolCall
    from vllm_mlx.routes.anthropic import _enforce_named_tool_choice_present

    pinned_call = ToolCall(
        id="c1",
        type="function",
        function=FunctionCall(name="get_weather", arguments='{"location": "SF"}'),
    )
    calls_out, synthesized, err = _enforce_named_tool_choice_present(
        [pinned_call],
        {"type": "function", "function": {"name": "get_weather"}},
        original_call_count=1,
    )
    assert calls_out == [pinned_call]
    assert synthesized is False
    assert err is None


def test_enforce_named_tool_choice_present_synthesizes_when_pinned_call_missing():
    """When the pinned tool call is missing, the helper synthesizes the
    pinned call. Call sites then validate that empty input against the
    tool schema before deciding whether to ship it or surface 422.
    """
    from vllm_mlx.routes.anthropic import _enforce_named_tool_choice_present

    calls_out, synthesized, err = _enforce_named_tool_choice_present(
        [],
        {"type": "function", "function": {"name": "get_weather"}},
        original_call_count=0,
    )
    assert len(calls_out) == 1
    assert calls_out[0].function.name == "get_weather"
    assert calls_out[0].function.arguments == "{}"
    assert synthesized is True
    assert err is None


def test_enforce_named_tool_choice_present_synthesizes_when_only_wrong_tool_emitted():
    """F8 disambiguation: the model emitted only WRONG-tool calls
    (filter dropped them all). The helper synthesizes the pinned call and
    uses ``original_call_count > 0`` only to log a different warning for
    operator debugging. Wire shape is identical to the no-calls case.
    """
    from vllm_mlx.routes.anthropic import _enforce_named_tool_choice_present

    calls_out, synthesized, err = _enforce_named_tool_choice_present(
        [],
        {"type": "function", "function": {"name": "get_weather"}},
        original_call_count=3,  # model emitted 3 wrong-tool calls
    )
    assert len(calls_out) == 1
    assert calls_out[0].function.name == "get_weather"
    assert calls_out[0].function.arguments == "{}"
    assert synthesized is True
    assert err is None


def test_pinned_tool_with_required_field_returns_422_not_synthesized_200():
    """A pinned tool whose schema has ``required`` fields must still
    return the named-tool 422 when the model emits no pinned call, not
    synthesize an empty placeholder and then fail schema validation.
    """
    engine = _MultiCallEngine(None, text="I'd rather not.")
    client = _make_client(engine)
    body = {
        "model": "test-model",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "weather please"}],
        # ``get_weather`` declares ``location`` as ``required``.
        "tools": [_GET_WEATHER],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }

    response = client.post("/v1/messages", json=body)
    assert response.status_code == 422, response.text
    assert "tool_choice" in response.json()["detail"]


def test_pinned_tool_streaming_text_replays_when_enforcement_passes():
    """Happy-path buffer-replay regression: pinned tool, model emits
    BOTH the pinned tool AND incidental text. The text must arrive
    in the SSE stream (it's legitimate output adjacent to the
    pinned-tool call), AND it must arrive AFTER the buffer-replay —
    i.e. the order ``message_start`` → text content_block → tool_use
    is preserved.

    Locks the round-2 fix's "replay buffer on success" branch: a
    naive implementation that dropped the buffer on every named
    ``tool_choice`` would lose legitimate text output adjacent to a
    correct pinned-tool call.
    """
    distinctive_text = "ANTHROPIC_STREAM_REPLAY_CANARY_42"
    engine = _MultiCallEngine(
        [_call("get_weather", {"location": "SF"})],
        text=distinctive_text,
    )
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json=_post_messages(
            client,
            tool_choice={"type": "tool", "name": "get_weather"},
            stream=True,
        ),
    )

    assert response.status_code == 200
    raw = response.text
    # No error — enforcement passed.
    assert "event: error" not in raw, raw
    # Both the text payload AND the pinned tool's tool_use must reach
    # the client. The buffered text is replayed after the enforcement
    # check, preserving streaming UX for legitimate adjacent text.
    assert distinctive_text in raw, raw
    assert '"type": "tool_use"' in raw
    assert "get_weather" in raw
    # Order check: the buffered text content_block_start/stop pair
    # is replayed BEFORE the tool_use content_block_start. A reversed
    # order would mean we leaked tool_use first then text — clients
    # would see the tool result before the surrounding narration.
    text_block_pos = raw.find(distinctive_text)
    tool_use_pos = raw.find('"type": "tool_use"')
    assert text_block_pos < tool_use_pos, (
        f"text block at {text_block_pos} should precede tool_use at {tool_use_pos}"
    )
    assert '"stop_reason": "tool_use"' in raw
