# SPDX-License-Identifier: Apache-2.0
"""r10 G/H bundle — wire-scrub completion + validation tests.

Covers the six findings the bundle PR landed:

* **R10-C8** (Mira r10-R1): UI-TARS-style ``"Tool: <name>\\nParameters: ..."``
  prose preamble leaked into ``delta.content`` before the structured
  ``delta.tool_calls`` chunk. The streaming postprocessor now buffers
  content matching the tool-prose prefix pattern and discards the
  buffer when a tool_call event arrives.
* **R10-M4** (Mira r10-R1): trailing ``\\n\\n`` whitespace after the
  scrubbed ``<tool_call>`` literal leaked into ``delta.content``. The
  tool-prose hold-back folds the whitespace into the prose buffer so
  the same tool_call discard takes care of it.
* **R10-H2** (Sven r10-R1): forced ``tool_choice={"type":"function",
  "function":{"name":X}}`` admitted two tool_call indices with
  distinct ``call_id`` values on qwen3 streaming. Single-call
  enforcement now drops every anchor after the first under the
  forced-name latch.
* **R10-H3** (Sven r10-R1): ``tool_choice="required"`` streamed raw
  token-sequence ``arguments`` (``"20230805"``) that violated the
  declared JSON-object schema. The forced-tool-choice filter now
  applies the same arguments-must-be-object gate to ``required``
  mode.
* **R10-H4** (Vlad r10-R1 / Bo r10-R1, R9-H2 carry):
  ``/v1/completions`` with ``response_format={"type":"json_object"}``
  silently dropped the field on both sync and streaming paths. The
  field is now declared on ``CompletionRequest`` and the route
  applies the same ``extract_json_from_response`` peel the chat lane
  uses.
* **R10-H5** (Sven r10-R1 / Vlad r10-R1, R9-H3 carry):
  ``reasoning_effort`` accepted any value (int, list, null,
  ``"banana"``) with HTTP 200 because the field was undeclared. The
  field is now ``Literal[...]``-enforced on both
  ``/v1/chat/completions`` AND ``/v1/responses`` (including the
  nested ``reasoning.effort`` shape on Responses).
* **R10-H6** (Mira r10-R1, R9-H4 carry): chat-lane
  ``tools=[{"type":"computer_use"}]`` shorthand 400'd because
  ``ToolDefinition`` required the ``function`` field. The model
  validator now synthesises the canonical ``computer`` function
  tool when ``type`` is a Computer-Use alias — parity with the
  ``/v1/responses`` ``_convert_tools`` normalisation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vllm_mlx.api.models import (
    ChatCompletionRequest,
    CompletionRequest,
    ToolDefinition,
)
from vllm_mlx.api.responses_models import ResponsesRequest

# ---------------------------------------------------------------------------
# R10-H5 — reasoning_effort enum on chat + responses
# ---------------------------------------------------------------------------


def _base_chat_kwargs():
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize(
    "value",
    ["none", "minimal", "low", "medium", "high"],
)
def test_r10_h5_chat_reasoning_effort_valid(value):
    """Every documented effort string is accepted on /v1/chat/completions."""
    req = ChatCompletionRequest(**_base_chat_kwargs(), reasoning_effort=value)
    assert req.reasoning_effort == value


def test_r10_h5_chat_reasoning_effort_null_accepted():
    """``reasoning_effort=None`` (the field default) is the unset signal."""
    req = ChatCompletionRequest(**_base_chat_kwargs(), reasoning_effort=None)
    assert req.reasoning_effort is None


@pytest.mark.parametrize(
    "bad",
    [
        "banana",
        "NONE",  # case variant; OpenAI spec uses lowercase only
        "Low",
        42,
        [],
        {},
        True,  # bool is an int subclass; must still 400
    ],
)
def test_r10_h5_chat_reasoning_effort_invalid_rejected(bad):
    """Sven r10-R1 + Vlad r10-R1: garbage values must 400 with the spec set."""
    with pytest.raises(ValidationError) as excinfo:
        ChatCompletionRequest(**_base_chat_kwargs(), reasoning_effort=bad)
    msg = str(excinfo.value)
    assert "reasoning_effort" in msg


@pytest.mark.parametrize(
    "value",
    ["none", "minimal", "low", "medium", "high"],
)
def test_r10_h5_responses_top_level_reasoning_effort_valid(value):
    """The top-level shorthand surface accepts the same set as chat."""
    req = ResponsesRequest(model="m", input="hi", reasoning_effort=value)
    assert req.reasoning_effort == value


@pytest.mark.parametrize(
    "bad",
    ["banana", 42, [], {}, True],
)
def test_r10_h5_responses_top_level_reasoning_effort_invalid_rejected(bad):
    with pytest.raises(ValidationError):
        ResponsesRequest(model="m", input="hi", reasoning_effort=bad)


@pytest.mark.parametrize(
    "value",
    ["none", "minimal", "low", "medium", "high"],
)
def test_r10_h5_responses_nested_reasoning_effort_valid(value):
    """Canonical OpenAI Responses spec uses ``reasoning.effort`` (nested)."""
    req = ResponsesRequest(model="m", input="hi", reasoning={"effort": value})
    assert req.reasoning == {"effort": value}


@pytest.mark.parametrize(
    "bad",
    ["banana", 42, [], True],
)
def test_r10_h5_responses_nested_reasoning_effort_invalid_rejected(bad):
    with pytest.raises(ValidationError) as excinfo:
        ResponsesRequest(model="m", input="hi", reasoning={"effort": bad})
    assert "reasoning.effort" in str(excinfo.value)


def test_r10_h5_responses_reasoning_dict_without_effort_passes():
    """Other ``reasoning`` keys (``summary``, ``encrypted_content``) flow."""
    req = ResponsesRequest(
        model="m",
        input="hi",
        reasoning={"summary": "auto"},
    )
    assert req.reasoning == {"summary": "auto"}


# ---------------------------------------------------------------------------
# R10-H6 — chat-lane computer_use shorthand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias",
    ["computer_use", "computer_use_preview", "computer_20251022"],
)
def test_r10_h6_computer_use_shorthand_accepted(alias):
    """Mira r10-R1: chat-lane ToolDefinition now mirrors the Responses
    alias map. The shorthand is normalised to a synthetic ``computer``
    function tool so the UI-TARS tool parser sees a matching entry.
    """
    t = ToolDefinition.model_validate({"type": alias})
    assert t.type == "function"
    assert isinstance(t.function, dict)
    assert t.function["name"] == "computer"
    assert "parameters" in t.function


def test_r10_h6_computer_use_shorthand_preserves_geometry_hints():
    """``display_width`` / ``display_height`` / ``environment`` flow into
    the synthetic function's ``parameters._computer_use`` block — same
    plumbing the Responses lane uses.
    """
    t = ToolDefinition.model_validate(
        {
            "type": "computer_use_preview",
            "display_width": 1280,
            "display_height": 800,
            "environment": "linux",
        }
    )
    cu = t.function["parameters"]["_computer_use"]
    assert cu == {
        "display_width": 1280,
        "display_height": 800,
        "environment": "linux",
    }


def test_r10_h6_function_classic_still_required():
    """The canonical ``{"type":"function","function":{"name":"x"}}`` shape
    keeps working (F-035 contract preserved).
    """
    t = ToolDefinition.model_validate(
        {"type": "function", "function": {"name": "x", "parameters": {}}}
    )
    assert t.function["name"] == "x"


def test_r10_h6_missing_function_for_function_type_rejected():
    """``{"type":"function"}`` with no ``function`` field still 400s."""
    with pytest.raises(ValidationError):
        ToolDefinition.model_validate({"type": "function"})


def test_r10_h6_in_chat_completion_request():
    """End-to-end: shorthand inside a request parses without 400."""
    req = ChatCompletionRequest(
        **_base_chat_kwargs(),
        tools=[{"type": "computer_use_preview", "display_width": 1024}],
    )
    assert req.tools is not None
    assert len(req.tools) == 1
    assert req.tools[0].function["name"] == "computer"


# ---------------------------------------------------------------------------
# R10-H4 — /v1/completions response_format declared + validated
# ---------------------------------------------------------------------------


def test_r10_h4_completions_response_format_json_object_accepted():
    """Vlad r10-R1: the field was previously silently dropped."""
    req = CompletionRequest(
        model="m", prompt="hi", response_format={"type": "json_object"}
    )
    assert req.response_format is not None
    # Typed arm wins because of the union arm dispatch.
    assert getattr(req.response_format, "type", None) == "json_object"


def test_r10_h4_completions_response_format_invalid_type_rejected():
    """Same closed-set check the chat lane runs."""
    with pytest.raises(ValidationError):
        CompletionRequest(
            model="m", prompt="hi", response_format={"type": "xml"}
        )


def test_r10_h4_completions_response_format_missing_type_rejected():
    with pytest.raises(ValidationError):
        CompletionRequest(model="m", prompt="hi", response_format={})


# ---------------------------------------------------------------------------
# R10-H2 + R10-H3 — forced tool_choice single-call + args-must-be-object
# ---------------------------------------------------------------------------


def _make_postprocessor(*, tool_choice):
    """Build a StreamingPostProcessor with a minimal ServerConfig shim."""
    from vllm_mlx.service.postprocessor import StreamingPostProcessor

    class _Cfg:
        reasoning_parser_name = None
        reasoning_parser = None
        tool_call_parser = None
        tool_parser_instance = None
        enable_auto_tool_choice = False
        engine = None
        no_thinking = False

    req = {
        "tool_choice": tool_choice,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    return StreamingPostProcessor(_Cfg(), tools_requested=True, request=req)


def test_r10_h2_forced_named_choice_single_call_enforced():
    """Sven r10-R1: forced named choice on qwen3 admitted index 0 AND
    index 1 with two distinct call_ids. The single-call latch now drops
    every anchor after the first.
    """
    pp = _make_postprocessor(
        tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )
    anchors = [
        {
            "index": 0,
            "id": "call_0d8c5338",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
        },
        {
            "index": 1,
            "id": "call_fbe75ec9",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
        },
    ]
    filtered = pp._apply_forced_tool_choice_filter(anchors)
    assert len(filtered) == 1
    assert filtered[0]["id"] == "call_0d8c5338"


def test_r10_h2_forced_named_choice_continuation_fragments_pass():
    """Argument-fragment deltas (no anchor metadata) STILL flow through —
    they belong to the admitted call so its arguments JSON can complete.
    """
    pp = _make_postprocessor(
        tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )
    # First the anchor lands.
    pp._apply_forced_tool_choice_filter(
        [
            {
                "index": 0,
                "id": "call_a",
                "type": "function",
                "function": {"name": "get_weather"},
            }
        ]
    )
    # Then a continuation arrives in a separate batch.
    continuations = [
        {"index": 0, "function": {"arguments": '"Paris"}'}},
    ]
    out = pp._apply_forced_tool_choice_filter(continuations)
    assert len(out) == 1


def test_r10_h3_required_mode_arguments_must_be_json_object():
    """Sven r10-R1: ``tool_choice="required"`` streamed raw token-sequence
    ``arguments="20230805"`` — valid JSON, but NOT a JSON object, so the
    declared tool schema can never be satisfied. Drop the anchor.
    """
    pp = _make_postprocessor(tool_choice="required")
    anchors = [
        {
            "index": 0,
            "id": "call_a",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "20230805",  # bare integer; non-object root
            },
        }
    ]
    filtered = pp._apply_forced_tool_choice_filter(anchors)
    assert filtered == []


def test_r10_h3_required_mode_object_args_pass():
    pp = _make_postprocessor(tool_choice="required")
    anchors = [
        {
            "index": 0,
            "id": "call_a",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
        }
    ]
    filtered = pp._apply_forced_tool_choice_filter(anchors)
    assert len(filtered) == 1


def test_r10_h3_required_mode_allows_parallel_calls():
    """``required`` mode does NOT trigger the single-call latch (parallel
    multi-tool dispatch is legal). Forced NAMED choice IS capped — see
    test_r10_h2_forced_named_choice_single_call_enforced.
    """
    pp = _make_postprocessor(tool_choice="required")
    anchors = [
        {
            "index": 0,
            "id": "call_a",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
        },
        {
            "index": 1,
            "id": "call_b",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Berlin"}',
            },
        },
    ]
    filtered = pp._apply_forced_tool_choice_filter(anchors)
    assert len(filtered) == 2


# ---------------------------------------------------------------------------
# R10-C8 + R10-M4 — tool-prose prefix + trailing whitespace scrub
# ---------------------------------------------------------------------------


def _content_chunks(events):
    from vllm_mlx.domain.events import StreamEvent  # noqa: F401

    return [ev.content for ev in events if ev.type == "content" and ev.content]


def test_r10_c8_tool_prose_prefix_is_buffered():
    """Mira r10-R1 evidence: when a request declares tools, content
    starting with ``Tool: <name>`` MUST be held back. The buffer is
    discarded when a tool_call event arrives later in the same turn.
    """
    pp = _make_postprocessor(tool_choice="auto")
    from vllm_mlx.domain.events import StreamEvent

    # Simulate the 12 prose chunks Mira observed.
    held = pp._filter_events_for_tool_prose(
        [StreamEvent(type="content", content="Tool")]
    )
    assert _content_chunks(held) == []
    held = pp._filter_events_for_tool_prose(
        [StreamEvent(type="content", content=":")]
    )
    assert _content_chunks(held) == []
    held = pp._filter_events_for_tool_prose(
        [StreamEvent(type="content", content=" get_weather")]
    )
    assert _content_chunks(held) == []


def test_r10_c8_tool_call_discards_buffered_prose():
    """When the parser surfaces a tool_call, the buffered prose IS
    discarded — clients see only the structured call, never the
    preamble.
    """
    pp = _make_postprocessor(tool_choice="auto")
    from vllm_mlx.domain.events import StreamEvent

    pp._filter_events_for_tool_prose(
        [StreamEvent(type="content", content="Tool: get_weather\n")]
    )
    pp._filter_events_for_tool_prose(
        [StreamEvent(type="content", content="Parameters: location=Paris\n")]
    )
    # Now the structured tool_call arrives.
    out = pp._filter_events_for_tool_prose(
        [
            StreamEvent(
                type="tool_call",
                content=None,
                tool_calls=[{"index": 0, "id": "call_x"}],
            )
        ]
    )
    # The tool_call passes through; no content leaked.
    assert any(ev.type == "tool_call" for ev in out)
    assert _content_chunks(out) == []
    # Buffer cleared.
    assert pp._tool_prose_buffer == ""


def test_r10_m4_trailing_whitespace_folded_into_prose_buffer():
    """The `\\n\\n` trailing whitespace Mira called out is folded into the
    buffer; when the tool_call lands, the whitespace is discarded too.
    """
    pp = _make_postprocessor(tool_choice="auto")
    from vllm_mlx.domain.events import StreamEvent

    pp._filter_events_for_tool_prose(
        [StreamEvent(type="content", content="Tool: get_weather\n\n")]
    )
    assert "\n\n" in pp._tool_prose_buffer
    pp._filter_events_for_tool_prose(
        [
            StreamEvent(
                type="tool_call",
                content=None,
                tool_calls=[{"index": 0, "id": "call_x"}],
            )
        ]
    )
    assert pp._tool_prose_buffer == ""


def test_r10_c8_non_prose_content_passes_through():
    """A request that legitimately starts with ``"Hello"`` is NOT held —
    only the narrow ``Tool|Action|Function:`` prefix triggers the
    buffer. Defense against over-eager censoring.
    """
    pp = _make_postprocessor(tool_choice="auto")
    from vllm_mlx.domain.events import StreamEvent

    out = pp._filter_events_for_tool_prose(
        [StreamEvent(type="content", content="Hello there!")]
    )
    assert _content_chunks(out) == ["Hello there!"]


def test_r10_c8_no_op_when_tools_not_requested():
    """If the client never declared tools, the filter is a pass-through.
    Non-tool requests pay zero overhead.
    """
    pp = _make_postprocessor(tool_choice="auto")
    pp.tools_requested = False  # simulate a tool-less request
    from vllm_mlx.domain.events import StreamEvent

    out = pp._filter_events_for_tool_prose(
        [StreamEvent(type="content", content="Tool: foo")]
    )
    assert _content_chunks(out) == ["Tool: foo"]


def test_r10_c8_held_buffer_released_at_stream_end_without_tool_call():
    """If the model legitimately ended the turn with ``Tool:`` prose AND
    no tool_call was detected, the buffer IS released (silent drop
    would be wrong).
    """
    pp = _make_postprocessor(tool_choice="auto")
    from vllm_mlx.domain.events import StreamEvent

    pp._filter_events_for_tool_prose(
        [StreamEvent(type="content", content="Tool:")]
    )
    # No tool_calls_detected yet.
    flushed = pp._flush_tool_prose_buffer()
    assert flushed == "Tool:"


def test_r10_c8_held_buffer_dropped_at_stream_end_with_tool_call():
    """If a tool_call WAS detected, the held buffer IS dropped at
    finalize — it was the dispatch preamble.
    """
    pp = _make_postprocessor(tool_choice="auto")
    from vllm_mlx.domain.events import StreamEvent

    pp._filter_events_for_tool_prose(
        [StreamEvent(type="content", content="Tool: get_weather")]
    )
    pp.tool_calls_detected = True
    flushed = pp._flush_tool_prose_buffer()
    assert flushed == ""


def test_r10_c8_buffer_releases_when_growing_past_cap():
    """Soft cap protects against indefinite hold on a model legitimately
    discussing the word ``Tool:`` in long prose.
    """
    pp = _make_postprocessor(tool_choice="auto")
    from vllm_mlx.domain.events import StreamEvent

    long_text = "Tool: " + "x" * (pp._TOOL_PROSE_MAX_HOLD + 8)
    out = pp._filter_events_for_tool_prose(
        [StreamEvent(type="content", content=long_text)]
    )
    # The buffer is released as content.
    assert any(ev.type == "content" and ev.content for ev in out)
    assert pp._tool_prose_buffer == ""


# ---------------------------------------------------------------------------
# Cross-cut: full request shape still validates
# ---------------------------------------------------------------------------


def test_r10_bundle_full_request_with_all_fields_accepted():
    """One request exercising every new field at once."""
    req = ChatCompletionRequest(
        **_base_chat_kwargs(),
        reasoning_effort="low",
        tools=[{"type": "computer_use_preview", "display_width": 1280}],
        tool_choice={"type": "function", "function": {"name": "computer"}},
    )
    assert req.reasoning_effort == "low"
    assert req.tools[0].function["name"] == "computer"
    assert req.tool_choice == {"type": "function", "function": {"name": "computer"}}
