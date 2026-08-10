# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer ATEM tool parser + recipient-channel reasoning parser.

Wire shapes are pinned against ``chat_template.jinja`` on
``meta-models/Muse-Glimmer-30B`` (2026-08-10): tool calls are
Anthropic-style ``<atem:...>`` XML blocks, reasoning rides a
``to=self`` channel, and the template itself warns the output "is not
expected to be valid XML and is parsed with regular expressions" — so
the delimiter-ambiguity cases here (#1730's bug class) are not
paranoia, they are the documented contract.

Streaming cases run per-character where it matters: every sentinel in
this format spans many deltas at char granularity, which is exactly
where prefix-leak bugs (#444/#480) live.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.reasoning import get_parser
from vllm_mlx.tool_parsers import ToolParserManager

from .parsers.dispatch import run_reasoning_extraction, run_tool_extraction

BOTH_MODES = pytest.mark.parametrize(
    "streaming", [False, True], ids=["nonstream", "stream"]
)


def _tool_parser():
    parser = ToolParserManager.get_tool_parser("muse")(None)
    parser.reset()
    return parser


def _reasoning_parser():
    return get_parser("muse")()


def _chars(text: str) -> list[str]:
    return list(text)


def _block(*invokes: str) -> str:
    return "<atem:function_calls>\n" + "\n".join(invokes) + "\n</atem:function_calls>"


def _invoke(name: str, params: dict[str, str]) -> str:
    lines = [f'<atem:invoke name="{name}">']
    for k, v in params.items():
        lines.append(f'<atem:parameter name="{k}">{v}</atem:parameter>')
    lines.append("</atem:invoke>")
    return "\n".join(lines)


def _request(name: str, properties: dict) -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {"type": "object", "properties": properties},
                },
            }
        ]
    }


# ---------------------------------------------------------------------------
# Tool parser — extraction
# ---------------------------------------------------------------------------


@BOTH_MODES
def test_single_call_bare_string_value(streaming):
    text = _block(_invoke("get_weather", {"city": "Paris"}))
    content, calls = run_tool_extraction(
        _tool_parser(), _chars(text), streaming=streaming
    )
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [
        ("get_weather", {"city": "Paris"})
    ]
    assert not (content or "").strip()


@BOTH_MODES
def test_multiple_invokes_in_one_block(streaming):
    text = _block(
        _invoke("read_file", {"path": "/a"}),
        _invoke("read_file", {"path": "/b"}),
    )
    # Both calls close with the same ``</atem:function_calls>`` delta, so
    # the stream finalizes them together — the documented parallel-tool
    # finalization shape.
    _, calls = run_tool_extraction(
        _tool_parser(),
        _chars(text),
        streaming=streaming,
        assert_one_tool_per_delta=False,
    )
    assert [json.loads(c.arguments)["path"] for c in calls] == ["/a", "/b"]


@BOTH_MODES
def test_content_before_block_survives(streaming):
    text = "Let me check.\n" + _block(_invoke("get_weather", {"city": "Oslo"}))
    content, calls = run_tool_extraction(
        _tool_parser(), _chars(text), streaming=streaming
    )
    assert len(calls) == 1
    assert (content or "").strip() == "Let me check."


def test_string_whitespace_is_preserved():
    # The template: "spaces for string values are not stripped."
    parser = _tool_parser()
    text = _block(_invoke("echo", {"text": "  padded  "}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": "  padded  "}


def test_multiline_string_value_kept_verbatim():
    value = 'line one\ncan span\n"multiple" lines\n'
    parser = _tool_parser()
    text = _block(_invoke("echo", {"text": value}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value}


def test_schema_typing_scalars_and_containers():
    parser = _tool_parser()
    props = {
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
        "items": {"type": "array"},
        "meta": {"type": "object"},
        "note": {"type": "string"},
    }
    text = _block(
        _invoke(
            "configure",
            {
                "count": "5",
                "ratio": "0.5",
                "flag": "true",
                "items": '["a", "b"]',
                "meta": '{"k": 1}',
                "note": "true",
            },
        )
    )
    result = parser.extract_tool_calls(text, request=_request("configure", props))
    args = json.loads(result.tool_calls[0]["arguments"])
    assert args == {
        "count": 5,
        "ratio": 0.5,
        "flag": True,
        "items": ["a", "b"],
        "meta": {"k": 1},
        # A declared string stays a string even when it spells a boolean.
        "note": "true",
    }


def test_undeclared_param_stays_string_unless_container():
    # No schema: bare scalars must stay strings ("5" could be a real
    # string), containers self-identify as JSON per the template.
    parser = _tool_parser()
    text = _block(_invoke("f", {"a": "5", "b": '{"x": 1}'}))
    result = parser.extract_tool_calls(text, request=None)
    assert json.loads(result.tool_calls[0]["arguments"]) == {"a": "5", "b": {"x": 1}}


def test_literal_closer_inside_string_value():
    # #1730's bug class: a value containing a literal closer must not be
    # truncated at the first one — the value runs to the LAST closer.
    parser = _tool_parser()
    value = "before </atem:parameter> after"
    text = _block(_invoke("echo", {"text": value}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value}


def test_fake_opener_inside_value_filtered_by_schema():
    # An opener whose name the schema does not declare is value text.
    parser = _tool_parser()
    value = 'x <atem:parameter name="fake">y</atem:parameter> z'
    text = _block(_invoke("echo", {"text": value}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value}


def test_null_only_when_schema_allows_it():
    # Codex r4 #1: null -> None ONLY for nullable schemas; against a
    # non-nullable type the raw string survives so strict validation
    # rejects it visibly instead of receiving a forged None.
    parser = _tool_parser()
    text = _block(_invoke("f", {"limit": "null", "count": "null"}))
    result = parser.extract_tool_calls(
        text,
        request=_request(
            "f",
            {
                "limit": {"type": ["integer", "null"]},
                "count": {"type": "integer"},
            },
        ),
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "limit": None,
        "count": "null",
    }


def test_duplicate_name_fake_opener_cannot_overwrite_value():
    # Codex r4 #2: a fake opener REUSING the declared name must not
    # become a second parameter that overwrites the real value.
    parser = _tool_parser()
    value = 'real start <atem:parameter name="text">evil</atem:parameter> real end'
    text = _block(_invoke("echo", {"text": value}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value}


def test_literal_invoke_closer_inside_value_does_not_truncate():
    # Codex r4 #3: a literal </atem:invoke> inside a parameter value
    # leaves the parameter structure unbalanced at that point, so the
    # invoke scan extends to the real closer.
    parser = _tool_parser()
    value = "docs mention </atem:invoke> literally"
    text = _block(_invoke("echo", {"text": value}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value}


@BOTH_MODES
def test_no_tools_plain_text_passthrough(streaming):
    content, calls = run_tool_extraction(
        _tool_parser(), _chars("Just an answer."), streaming=streaming
    )
    assert calls == []
    assert content == "Just an answer."


@BOTH_MODES
def test_channel_wrapped_call_extracts_and_strips_plumbing(streaming):
    # Standalone use (no reasoning parser): raw channel output.
    text = (
        " to=self<|message|>Consider the request.<|eom|>"
        "<|start|>assistant to=get_weather<|message|>"
        + _block(_invoke("get_weather", {"city": "Lima"}))
        + "<|eot|>"
    )
    content, calls = run_tool_extraction(
        _tool_parser(), _chars(text), streaming=streaming
    )
    assert [(c.name,) for c in calls] == [("get_weather",)]
    # Neither plumbing nor the to=self reasoning may leak into content —
    # exact comparison, because a substring check let a 2-byte " to"
    # header fragment through during development.
    assert not (content or "").strip()


@BOTH_MODES
def test_content_after_block_survives(streaming):
    # Codex r1 BLOCKING #1: streaming used to swallow everything after
    # the first opener; content between/after blocks must reach the wire.
    text = (
        "Checking.\n"
        + _block(_invoke("get_weather", {"city": "Oslo"}))
        + "\nDone checking."
    )
    content, calls = run_tool_extraction(
        _tool_parser(), _chars(text), streaming=streaming
    )
    assert len(calls) == 1
    combined = content or ""
    assert "Checking." in combined
    assert "Done checking." in combined


def test_invoke_outside_block_is_literal_content():
    # Codex r1 BLOCKING #2: invoke-shaped text OUTSIDE a completed block
    # (the model quoting an example) must never become an executable call.
    quoted = _invoke("rm_rf", {"path": "/"})
    text = (
        "For example you would write:\n"
        + quoted
        + "\n"
        + _block(_invoke("get_weather", {"city": "Oslo"}))
    )
    parser = _tool_parser()
    result = parser.extract_tool_calls(text)
    assert [c["name"] for c in result.tool_calls] == ["get_weather"]
    assert "rm_rf" in (result.content or "")


@BOTH_MODES
def test_reasoning_whitespace_parity(streaming):
    # Codex r1 BLOCKING #3: non-stream used to strip() segment bodies
    # while streaming preserved them — same input must give the same
    # output in both modes, whitespace included.
    text = " to=self<|message|>  padded thought \n<|eom|><|start|>assistant to=user<|message|> spaced answer <|eot|>"
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars(text), streaming=streaming
    )
    assert reasoning == "  padded thought \n"
    assert content == " spaced answer "


def test_close_and_trailing_text_in_one_delta():
    # Codex r2 #1: a single delta that completes the block AND carries
    # trailing text must not lose that text — the tool_calls return wins
    # the delta, and the cursor picks the text up on the next call.
    parser = _tool_parser()
    block = _block(_invoke("f", {"a": "1"}))
    deltas = ["Hi ", block[:-5], block[-5:] + " tail", " end"]
    contents: list[str] = []
    calls = 0
    prev = ""
    for d in deltas:
        curr = prev + d
        out = parser.extract_tool_calls_streaming(prev, curr, d)
        if out:
            if out.get("content"):
                contents.append(out["content"])
            calls += len(out.get("tool_calls") or [])
        prev = curr
    contents.append(parser.flush_held_content(prev))
    assert calls == 1
    assert "".join(contents) == "Hi  tail end"


def test_valid_call_then_literal_opener_released_at_flush():
    # Codex r2 #3: after a real call, a trailing literal/truncated opener
    # can never complete — its bytes are content and must be released at
    # end of stream, matching the non-streaming path (#1766 principle).
    parser = _tool_parser()
    text = _block(_invoke("f", {"a": "1"})) + "\nsee <atem:function_calls> docs"
    emitted: list[str] = []
    calls = 0
    prev = ""
    for ch in text:
        curr = prev + ch
        out = parser.extract_tool_calls_streaming(prev, curr, ch)
        if out:
            if out.get("content"):
                emitted.append(out["content"])
            calls += len(out.get("tool_calls") or [])
        prev = curr
    emitted.append(parser.flush_held_content(prev))
    assert calls == 1
    assert "".join(emitted) == "\nsee <atem:function_calls> docs"
    # And non-stream agrees byte-for-byte.
    nonstream = _tool_parser().extract_tool_calls(text)
    assert nonstream.content == "\nsee <atem:function_calls> docs"


def test_literal_block_closer_inside_value_does_not_truncate_call():
    # Codex r3 #1: a value containing the literal BLOCK closer leaves the
    # invoke structure unbalanced at that point, so the block scan must
    # extend to the next closer instead of truncating the real call.
    parser = _tool_parser()
    value = "docs mention </atem:function_calls> literally"
    text = _block(_invoke("echo", {"text": value}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value}


@BOTH_MODES
def test_malformed_completed_block_bytes_survive(streaming):
    # Codex r3 #2: a completed block with no parseable invoke is model
    # output the client must see — in BOTH modes.
    text = (
        "Before <atem:function_calls>\ngarbage, no invoke\n</atem:function_calls> after"
    )
    content, calls = run_tool_extraction(
        _tool_parser(), _chars(text), streaming=streaming
    )
    assert calls == []
    combined = content or ""
    assert "garbage, no invoke" in combined
    assert "Before " in combined and " after" in combined


def test_nullable_list_type_schema_coerces():
    # Codex r3 #3: {"type": ["integer", "null"]} must behave as integer.
    parser = _tool_parser()
    text = _block(_invoke("f", {"n": "5", "m": "null"}))
    result = parser.extract_tool_calls(
        text,
        request=_request(
            "f",
            {
                "n": {"type": ["integer", "null"]},
                "m": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            },
        ),
    )
    assert json.loads(result.tool_calls[0]["arguments"]) == {"n": 5, "m": None}


def test_unmatched_literal_invoke_opener_in_value_still_parses():
    # Codex r5 #2: an UNMATCHED literal "<atem:invoke" inside a value
    # must not stop the real block from completing — inside a value the
    # scanner examines nothing until a boundary-valid closer.
    parser = _tool_parser()
    value = "the wire uses <atem:invoke tags for calls"
    text = _block(_invoke("echo", {"text": value}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value}


def test_unmatched_fake_param_opener_in_value_still_parses():
    # Codex r5 #3: an unmatched fake parameter opener inside a value
    # must not make a valid call read as malformed.
    parser = _tool_parser()
    value = 'see <atem:parameter name="x"> for details'
    text = _block(_invoke("echo", {"text": value}))
    result = parser.extract_tool_calls(
        text, request=_request("echo", {"text": {"type": "string"}})
    )
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == {"text": value}


def test_param_without_closer_never_becomes_a_call():
    # Codex r5 #1: a parameter that never closes must not execute with
    # a truncated value — the whole structure is unparseable content.
    parser = _tool_parser()
    text = (
        "<atem:function_calls>\n"
        '<atem:invoke name="rm">\n'
        '<atem:parameter name="path">/tmp/x\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    result = parser.extract_tool_calls(text)
    assert not result.tools_called
    assert "/tmp/x" in (result.content or "")


def test_non_finite_number_stays_raw():
    # Codex r5 #4: NaN/Infinity/1e309 would make json.dumps emit tokens
    # that are not valid JSON — keep the raw string for the validator.
    parser = _tool_parser()
    text = _block(_invoke("f", {"a": "NaN", "b": "1e309", "c": "2.5"}))
    result = parser.extract_tool_calls(
        text,
        request=_request(
            "f",
            {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "c": {"type": "number"},
            },
        ),
    )
    args = json.loads(result.tool_calls[0]["arguments"])
    assert args == {"a": "NaN", "b": "1e309", "c": 2.5}


@BOTH_MODES
def test_quoted_opener_in_prose_does_not_swallow_later_call(streaming):
    # Codex r6 #2: a definitively malformed (quoted-in-prose) opener
    # must not stop the scan — a real call later in the response fires.
    text = "The wire wraps calls in <atem:function_calls> as you know.\n" + _block(
        _invoke("get_weather", {"city": "Oslo"})
    )
    content, calls = run_tool_extraction(
        _tool_parser(), _chars(text), streaming=streaming
    )
    assert [c.name for c in calls] == ["get_weather"]
    assert "as you know." in (content or "")


@BOTH_MODES
def test_reasoning_trailing_partial_sentinel_parity(streaming):
    # Codex r6 #3: output ending in a partial sentinel must not lose
    # those bytes in streaming — finalize releases the hold.
    text = " to=user<|message|>answer <|eo"
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars(text), streaming=streaming
    )
    assert reasoning is None
    assert content == "answer <|eo"


@BOTH_MODES
def test_literal_terminator_mid_prose_survives(streaming):
    # Codex r7 #3/#4: only STRUCTURAL terminators (segment boundary /
    # end) are plumbing; a literal one mid-prose is model output.
    text = " to=user<|message|>the <|eot|> token ends a turn<|eot|>"
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars(text), streaming=streaming
    )
    assert reasoning is None
    assert content == "the <|eot|> token ends a turn"


def test_same_delta_content_and_call_keep_wire_order():
    # Codex r7 #2: content arriving in the same delta as the block
    # close rides in the SAME response as the calls.
    parser = _tool_parser()
    block = _block(_invoke("f", {"a": "1"}))
    deltas = ["Hi", " there\n" + block]
    prev = ""
    outs = []
    for d in deltas:
        curr = prev + d
        out = parser.extract_tool_calls_streaming(prev, curr, d)
        if out:
            outs.append(out)
        prev = curr
    last = outs[-1]
    assert last.get("tool_calls") and len(last["tool_calls"]) == 1
    assert last.get("content") == " there\n"


def test_truncated_block_keeps_bytes_as_content():
    # An opener with no parseable invoke must not vanish silently.
    parser = _tool_parser()
    text = '<atem:function_calls>\n<atem:invoke name="get_w'
    result = parser.extract_tool_calls(text)
    assert not result.tools_called
    assert "atem:invoke" in (result.content or "")


def test_streaming_emits_no_partial_sentinel_bytes():
    parser = _tool_parser()
    text = "Hello " + _block(_invoke("f", {"a": "1"}))
    emitted: list[str] = []
    prev = ""
    for ch in text:
        curr = prev + ch
        delta = parser.extract_tool_calls_streaming(prev, curr, ch)
        if delta and delta.get("content"):
            emitted.append(delta["content"])
        prev = curr
    assert "".join(emitted) == "Hello "


# ---------------------------------------------------------------------------
# Reasoning parser
# ---------------------------------------------------------------------------


@BOTH_MODES
def test_reasoning_then_answer(streaming):
    text = (
        " to=self<|message|>Two plus two is four.<|eom|>"
        "<|start|>assistant to=user<|message|>4<|eot|>"
    )
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars(text), streaming=streaming
    )
    assert reasoning == "Two plus two is four."
    assert content == "4"


@BOTH_MODES
def test_bare_message_header_is_user_content(streaming):
    text = "<|message|>Plain answer.<|eot|>"
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars(text), streaming=streaming
    )
    assert reasoning is None
    assert content == "Plain answer."


@BOTH_MODES
def test_no_channel_markers_degrades_to_content(streaming):
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars("No plumbing at all."), streaming=streaming
    )
    assert reasoning is None
    assert content == "No plumbing at all."


@BOTH_MODES
def test_tool_segment_passes_through_as_content(streaming):
    # The ATEM block must SURVIVE the reasoning split — the tool parser
    # downstream consumes it from the content channel (harmony division).
    block = _block(_invoke("get_weather", {"city": "Rome"}))
    text = (
        " to=self<|message|>Need the weather.<|eom|>"
        "<|start|>assistant to=get_weather<|message|>" + block + "<|eot|>"
    )
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars(text), streaming=streaming
    )
    assert reasoning == "Need the weather."
    assert content is not None and block in content


@BOTH_MODES
def test_plain_text_before_explicit_header_is_content(streaming):
    # Codex r4 #4: with no implicit header, text before the first
    # explicit header is model output, not discardable plumbing.
    text = "Hello<|start|>assistant to=user<|message|>Hi<|eot|>"
    reasoning, content = run_reasoning_extraction(
        _reasoning_parser(), _chars(text), streaming=streaming
    )
    assert reasoning is None
    assert content == "HelloHi"


def test_streaming_never_leaks_header_or_terminator_bytes():
    text = (
        " to=self<|message|>thinking<|eom|>"
        "<|start|>assistant to=user<|message|>answer<|eot|>"
    )
    parser = _reasoning_parser()
    parser.reset_state()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    prev = ""
    for ch in text:
        curr = prev + ch
        msg = parser.extract_reasoning_streaming(prev, curr, ch)
        if msg is not None:
            if msg.reasoning:
                reasoning_parts.append(msg.reasoning)
            if msg.content:
                content_parts.append(msg.content)
        prev = curr
    assert "".join(reasoning_parts) == "thinking"
    assert "".join(content_parts) == "answer"


def test_reasoning_parser_survives_thinking_disabled():
    """The muse parser must stay in the demux path when a request
    resolves ``enable_thinking=False`` (R12-T2F casual-chat auto-disable).

    Muse's template has no thinking switch — the model always emits
    channel plumbing. Without ``sanitize_when_thinking_disabled`` the
    postprocessor bypassed the parser and ``strip_special_tokens`` ate
    the wire markers while leaking `` to=self`` header bytes into
    ``delta.content`` (observed on real 30B weights, 2026-08-10 smoke).
    """
    parser = _reasoning_parser()
    assert parser.sanitize_when_thinking_disabled is True


def test_postprocessor_demuxes_with_thinking_disabled():
    """End-to-end postprocessor regression for the mini smoke failure:
    exact per-token deltas observed from the real 30B checkpoint, with
    ``enable_thinking=False`` as injected by the casual-chat auto-disable."""
    from unittest.mock import MagicMock

    from vllm_mlx.service.postprocessor import StreamingPostProcessor

    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = "muse"
    cfg.enable_auto_tool_choice = False
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = None

    pp = StreamingPostProcessor(cfg, enable_thinking=False)
    pp.reset()

    deltas = [
        " to",
        "=self",
        "<|message|>",
        "hi",
        "\n\n",
        "We",
        " need",
        " to",
        " respond",
        ".",
        "<|eom|>",
        "<|start|>",
        "assistant",
        " to",
        "=user",
        "<|message|>",
        "Hello",
        "!",
        "<|eot|>",
    ]
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for i, ch in enumerate(deltas):
        out = MagicMock()
        out.new_text = ch
        out.finished = i == len(deltas) - 1
        out.channel = None
        out.finish_reason = "stop" if out.finished else None
        out.prompt_tokens = 10
        out.completion_tokens = 5
        out.tokens = []
        out.logprobs = None
        out.tool_calls = None
        for e in pp.process_chunk(out):
            if e.type == "content" and e.content:
                content_parts.append(e.content)
            if getattr(e, "reasoning", None):
                reasoning_parts.append(e.reasoning)

    assert "".join(reasoning_parts) == "hi\n\nWe need to respond."
    assert "".join(content_parts) == "Hello!"


def test_finalize_uses_raw_wire_content_for_muse():
    """Non-streaming counterpart of the demux regression: the route's
    ``clean_output_text`` strips channel markers WITHOUT extracting
    channels, so ``_finalize_content_and_reasoning``'s first parse sees
    markerless mush. The raw-text retry must supply BOTH halves for
    muse — before the fix the mush (header bytes + duplicated
    reasoning) shipped as ``content`` (real-weights mini smoke,
    2026-08-10, non-streaming surface).
    """
    from vllm_mlx.api.utils import clean_output_text
    from vllm_mlx.service.helpers import _finalize_content_and_reasoning

    raw = (
        " to=self<|message|>We need to respond.<|eom|>"
        "<|start|>assistant to=user<|message|>Hello!<|eot|>"
    )
    cleaned = clean_output_text(raw)
    assert "<|message|>" not in cleaned  # the generic regex ate the wire

    content, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=cleaned,
        tool_calls=[],
        reasoning_parser=_reasoning_parser(),
    )
    assert reasoning == "We need to respond."
    assert content == "Hello!"


def test_finalize_truncated_all_reasoning_muse():
    """finish_reason=length mid-reasoning: everything is to=self, no
    content channel ever opened — content must be empty, not the mush."""
    from vllm_mlx.api.utils import clean_output_text
    from vllm_mlx.service.helpers import _finalize_content_and_reasoning

    raw = " to=self<|message|>Thinking hard about the answer"
    cleaned = clean_output_text(raw)

    content, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=cleaned,
        tool_calls=[],
        reasoning_parser=_reasoning_parser(),
        finish_reason="length",
    )
    assert reasoning == "Thinking hard about the answer"
    assert content == ""


def test_clean_output_text_extracts_muse_channels():
    """``clean_output_text`` must demux muse wire (harmony-precedent
    branch), not regex-strip it into header mush — its output feeds the
    non-streaming tool parser and the finalize first-parse."""
    from vllm_mlx.api.utils import clean_output_text

    raw = (
        " to=self<|message|>plan the call<|eom|>"
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    )
    assert clean_output_text(raw, muse_wire=True) == "Done."
    # Without the model-identity gate the branch must NOT engage, even
    # on structurally perfect wire (codex r6 #1: gate on the serving
    # model, never on output bytes).
    assert "to=self" in clean_output_text(raw)

    # Tool-addressed segments pass through as content so the ATEM
    # block reaches the tool parser.
    raw_tool = (
        " to=self<|message|>need weather<|eom|>"
        "<|start|>assistant to=get_weather<|message|>"
        '<atem:function_calls><atem:invoke name="get_weather">'
        '<atem:parameter name="city">Tokyo</atem:parameter>'
        "</atem:invoke></atem:function_calls><|eot|>"
    )
    cleaned = clean_output_text(raw_tool, muse_wire=True)
    assert "<atem:invoke" in cleaned
    assert "to=self" not in cleaned
    assert "<|message|>" not in cleaned

    # Non-muse text with a literal mid-prose mention must NOT enter
    # the muse branch: only the generic token strip applies, the
    # surrounding prose survives byte-exact (codex r5 #3).
    prose = "The wire uses <|message|> as a separator."
    assert clean_output_text(prose, muse_wire=True) == "The wire uses  as a separator."
    prose2 = "Historically <|start|>assistant marked a header."
    assert (
        clean_output_text(prose2, muse_wire=True)
        == "Historically assistant marked a header."
    )
