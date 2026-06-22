# SPDX-License-Identifier: Apache-2.0
"""
r5-E F-DGF-V080-B-8: gemma-4-12b-4bit tool-call regression.

Trace verdict (see commit body):
  Neither parser-pattern miss NOR template tool-injection miss. The
  chat template renders ``<|tool>declaration:NAME{...}<tool|>`` for
  every tool in the request (verified against
  ``mlx-community/gemma-4-12B-it-4bit`` 4bit), and the existing
  structured pattern catches every emission that hits the
  ``<|tool_call>`` channel form. The intermittent failure is a model
  decoding edge case — at low temperature (~0.1) the model
  occasionally describes the tool intent in prose ("I should call
  the `add` tool with a=13 and b=29.") instead of channel-routing
  through ``<|tool_call>``.

Defence-in-depth fix: ``_try_prose_recover_tool_call`` runs after a
structured-match miss and recovers a tool call from prose when:

  1. The request carried a ``tools`` array.
  2. The prose mentions a tool by its exact name.
  3. The prose includes ``key=value`` (or ``key: value``) assignments
     for EVERY required parameter on that tool.

Recovery returns the standard structured shape; a miss leaves the
prose in ``content`` unchanged. The conservative gating means a
chat that just discusses ``add`` and an unrelated ``a=`` mention is
NOT falsely recovered.
"""
import json

import pytest

from vllm_mlx.tool_parsers.gemma4_tool_parser import (
    Gemma4ToolParser,
    _try_prose_recover_tool_call,
)

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers and return their sum.",
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

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


# ---------------------------------------------------------------------------
# Direct prose-recovery helper
# ---------------------------------------------------------------------------


def test_prose_recovers_canonical_dogfood_case():
    """Exact verbatim prose from the cycle-DGF-v080 agent-B B-8 repro."""
    prose = (
        "The user wants to add 13 and 29 using the `add` tool. "
        "I should call the `add` tool with a=13 and b=29."
    )
    out = _try_prose_recover_tool_call(prose, [ADD_TOOL])
    assert out is not None
    assert out["name"] == "add"
    args = json.loads(out["arguments"])
    assert args == {"a": 13, "b": 29}


def test_prose_recovers_with_colon_assignments():
    """Some prose variants use ``a: 13`` not ``a=13``."""
    prose = "Calling add with a: 13, b: 29."
    out = _try_prose_recover_tool_call(prose, [ADD_TOOL])
    assert out is not None
    assert json.loads(out["arguments"]) == {"a": 13, "b": 29}


def test_prose_recovers_quoted_string_value():
    prose = 'I will call get_weather with location="Palo Alto".'
    out = _try_prose_recover_tool_call(prose, [WEATHER_TOOL])
    assert out is not None
    assert out["name"] == "get_weather"
    assert json.loads(out["arguments"]) == {"location": "Palo Alto"}


def test_prose_recovers_backticked_value():
    prose = "I should call get_weather with location=`Tokyo`."
    out = _try_prose_recover_tool_call(prose, [WEATHER_TOOL])
    assert out is not None
    assert json.loads(out["arguments"]) == {"location": "Tokyo"}


def test_prose_no_tool_named_returns_none():
    """Tool name absent from the prose → don't recover."""
    prose = "I should compute the sum a=13 and b=29."
    out = _try_prose_recover_tool_call(prose, [ADD_TOOL])
    assert out is None


def test_prose_partial_required_params_returns_none():
    """Required param missing → confidence too low, leave as content."""
    prose = "I'll call the `add` tool with a=13."
    out = _try_prose_recover_tool_call(prose, [ADD_TOOL])
    assert out is None


def test_prose_no_tools_in_request_returns_none():
    """Empty tools list → no recovery (gate condition 1)."""
    prose = "I should call the `add` tool with a=13 and b=29."
    out = _try_prose_recover_tool_call(prose, [])
    assert out is None


def test_prose_unrelated_natural_text_returns_none():
    """A normal chat reply that doesn't mention the tool's name
    must not be collaterally captured even with assignments
    elsewhere."""
    prose = "Sure! The answer is 42."
    out = _try_prose_recover_tool_call(prose, [ADD_TOOL])
    assert out is None


def test_prose_first_value_wins_on_self_correction():
    """``a=13 ... a=14`` → take the earliest commitment (the model's
    initial reasoning, not a later "correction" mid-thought)."""
    prose = "I'll call the `add` tool with a=13 and b=29 (or maybe a=14)."
    out = _try_prose_recover_tool_call(prose, [ADD_TOOL])
    assert out is not None
    assert json.loads(out["arguments"]) == {"a": 13, "b": 29}


def test_prose_multiple_tools_first_named_wins():
    """When both tools are mentioned, the earliest mention wins."""
    prose = (
        "I considered get_weather but actually I should call the `add` "
        "tool with a=13 and b=29."
    )
    out = _try_prose_recover_tool_call(prose, [ADD_TOOL, WEATHER_TOOL])
    # ``get_weather`` is mentioned first but its required ``location``
    # has no assignment in the prose, so it falls through to ``add``.
    assert out is not None
    assert out["name"] == "add"


# ---------------------------------------------------------------------------
# extract_tool_calls integration — runs the full parser path
# ---------------------------------------------------------------------------


def test_extract_tool_calls_recovers_prose_with_request_tools():
    parser = Gemma4ToolParser()
    prose = (
        "The user wants to add 13 and 29 using the `add` tool. "
        "I should call the `add` tool with a=13 and b=29."
    )
    request = {"tools": [ADD_TOOL]}
    res = parser.extract_tool_calls(prose, request)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["name"] == "add"
    assert json.loads(tc["arguments"]) == {"a": 13, "b": 29}
    # Content is dropped — OpenAI spec is content OR tool_calls
    # on a single response choice, not both verbatim.
    assert res.content is None


def test_extract_tool_calls_recovery_skipped_without_tools():
    """No ``tools`` in the request → recovery never fires; prose
    stays in content (existing pre-fix contract on the non-tools
    path)."""
    parser = Gemma4ToolParser()
    prose = "I should call the `add` tool with a=13 and b=29."
    res = parser.extract_tool_calls(prose, request=None)
    assert res.tools_called is False
    assert res.tool_calls == []
    assert res.content == prose


def test_extract_tool_calls_recovery_skipped_when_structured_form_present():
    """When the model DOES emit the structured form, the recovery
    must not also fire — the recovery is a fallback path."""
    parser = Gemma4ToolParser()
    structured = "<|tool_call>call:add{a:13,b:29}<tool_call|>"
    request = {"tools": [ADD_TOOL]}
    res = parser.extract_tool_calls(structured, request)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    assert json.loads(res.tool_calls[0]["arguments"]) == {"a": 13, "b": 29}


def test_extract_tool_calls_no_recovery_on_natural_chat():
    """Pure natural-language reply with no tool name mention → no
    recovery, content preserved."""
    parser = Gemma4ToolParser()
    text = "Hello! How can I help you today?"
    request = {"tools": [ADD_TOOL]}
    res = parser.extract_tool_calls(text, request)
    assert res.tools_called is False
    assert res.content == text
