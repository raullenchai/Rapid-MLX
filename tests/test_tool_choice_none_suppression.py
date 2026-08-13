# SPDX-License-Identifier: Apache-2.0
"""``tool_choice="none"`` must surface ZERO tool calls, on every parser
and on both the streaming and non-streaming paths.

Reported bug (deterministic 5/5 on ``lfm2.5-2.6b-4bit``): a request that
explicitly sets ``tool_choice="none"`` still came back with a tool call —
and one whose name/arguments were never declared (``GetWeather`` /
``location`` against a declared ``get_weather`` / ``city, unit``). An
OpenAI-SDK client that has explicitly turned tools OFF for a turn received
a phantom call it cannot execute.

Root cause: ``"none"`` only had a best-effort *prompt-level* lever
(dropping ``request.tools`` before rendering in ``routes/chat.py``). A
tool-trained model still echoes ``[name({...})]``-style markup it was
never shown, and the text parser promoted it — the declared-name / choice
enforcement ran ONLY for the ``qwen3_coder_xml`` parser, and the
post-parse ``routes/chat.py`` enforcement short-circuited on the very
``request.tools = None`` the prompt-level lever had just set.

Fix (``tool_choice_is_none`` gate, parser-agnostic): the parser still
RUNS under ``none`` — that is what strips the wire markup out of content
(the R12 sanitizer invariant) — but the resolved call is DROPPED at the
emission points rather than forwarded as a phantom:
  * non-streaming ``_parse_tool_calls_with_parser`` wraps ``_run_tool_parser``
    and drops the calls (text parsers AND the structured harmony/gemma4
    channel), keeping the cleaned content;
  * streaming ``_detect_tool_calls`` drops the resolved call and surfaces
    only co-emitted prose, leaving ``tool_calls_detected`` False;
  * the streaming channel-routed path suppresses the whole ``tool_call``
    channel and normalizes the terminal ``finish_reason`` to ``"stop"``.
In every case the wire markup is stripped from content, never leaked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm_mlx.service.helpers as helpers
from vllm_mlx.config import get_config
from vllm_mlx.service.helpers import tool_choice_is_none
from vllm_mlx.service.postprocessor import StreamingPostProcessor

# The exact wire markup the weak model emitted despite the prompt-level
# ``tools`` drop — an undeclared name/param, the reported reproduction.
LFM_WIRE = '[GetWeather({"location": "Rome"})]'

DECLARED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]


def _request(tool_choice):
    """A minimal chat request exposing ``model_dump`` + ``tool_choice``."""
    return SimpleNamespace(
        tool_choice=tool_choice,
        tools=DECLARED_TOOLS,
        model_dump=lambda: {"tool_choice": tool_choice, "tools": DECLARED_TOOLS},
    )


# =====================================================================
# The parser-agnostic predicate
# =====================================================================


class TestToolChoiceIsNonePredicate:
    def test_string_none_object(self):
        assert tool_choice_is_none(SimpleNamespace(tool_choice="none")) is True

    def test_string_none_dict(self):
        assert tool_choice_is_none({"tool_choice": "none"}) is True

    @pytest.mark.parametrize("choice", ["auto", "required", None])
    def test_other_string_modes_are_not_none(self, choice):
        assert tool_choice_is_none({"tool_choice": choice}) is False

    def test_named_function_choice_is_not_none(self):
        # ``{"type":"function",...}`` is a FORCED call, the opposite of none.
        assert (
            tool_choice_is_none(
                {"tool_choice": {"type": "function", "function": {"name": "x"}}}
            )
            is False
        )

    def test_none_request_object(self):
        assert tool_choice_is_none(None) is False

    def test_request_without_tool_choice_attr(self):
        assert tool_choice_is_none(SimpleNamespace()) is False


# =====================================================================
# Non-streaming path — _parse_tool_calls_with_parser
# =====================================================================


class TestNonStreamingSuppression:
    def _run(self, parser, request, *, structured=None):
        cfg = get_config()
        saved = (cfg.enable_auto_tool_choice, cfg.tool_call_parser)
        cfg.enable_auto_tool_choice = True
        cfg.tool_call_parser = parser
        try:
            return helpers._parse_tool_calls_with_parser(
                LFM_WIRE if structured is None else "plain answer",
                request,
                structured_tool_calls=structured,
            )
        finally:
            cfg.enable_auto_tool_choice, cfg.tool_call_parser = saved

    def test_none_suppresses_text_parser_call(self):
        # The reported reproduction: lfm markup under ``none`` must NOT
        # surface a call. The parser still runs, so the wire markup is
        # STRIPPED from content (R12 sanitizer invariant) rather than leaked.
        content, calls = self._run("lfm", _request("none"))
        assert calls is None
        assert "GetWeather" not in content  # markup did not leak into content

    def test_none_suppresses_structured_channel_call(self):
        # Harmony/gemma4 surface calls via ``structured_tool_calls``; the
        # none drop runs after parsing, so the call is dropped and the
        # already-clean channel content is preserved.
        content, calls = self._run(
            "harmony",
            _request("none"),
            structured=[{"name": "GetWeather", "arguments": '{"location":"Rome"}'}],
        )
        assert calls is None
        assert content == "plain answer"

    def test_auto_still_parses_the_same_markup(self):
        # Guard against over-suppression: the identical markup under
        # ``auto`` MUST still yield the call.
        _content, calls = self._run("lfm", _request("auto"))
        assert calls and calls[0].function.name == "GetWeather"

    def test_unset_tool_choice_still_parses(self):
        _content, calls = self._run("lfm", _request(None))
        assert calls and calls[0].function.name == "GetWeather"


# =====================================================================
# Streaming path — StreamingPostProcessor
# =====================================================================


def _cfg(parser="lfm"):
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = parser
    cfg.tool_parser_instance = None
    return cfg


def _output(text, finished, *, channel=None, tool_calls=None):
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = channel
    out.finish_reason = ("tool_calls" if tool_calls else "stop") if finished else None
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    out.tool_calls = tool_calls
    return out


def _drive(pp, outputs):
    events = []
    for out in outputs:
        events.extend(pp.process_chunk(out))
    tool_calls = [
        tc
        for e in events
        if e.type == "tool_call" and e.tool_calls
        for tc in e.tool_calls
    ]
    content = "".join(
        (getattr(e, "content", "") or "")
        for e in events
        if e.type in ("content", "finish")
    )
    finish_reasons = [
        e.finish_reason for e in events if e.type == "finish" and e.finish_reason
    ]
    return tool_calls, content, finish_reasons


def _make_pp(tool_choice, parser="lfm"):
    request = {"tool_choice": tool_choice, "tools": DECLARED_TOOLS}
    pp = StreamingPostProcessor(
        _cfg(parser), tools_requested=True, enable_thinking=False, request=request
    )
    pp.reset()
    return pp


class TestStreamingTextParserSuppression:
    def test_none_suppresses_text_parser_stream(self):
        pp = _make_pp("none")
        calls, content, finish = _drive(pp, [_output(LFM_WIRE, finished=True)])
        assert calls == []
        # Parser consumed the block, so the markup did not leak into content.
        assert "GetWeather" not in content
        # No call reached the wire → the terminal must not claim tool_calls.
        assert "tool_calls" not in finish

    def test_auto_still_emits_the_same_stream(self):
        pp = _make_pp("auto")
        calls, _content, _finish = _drive(pp, [_output(LFM_WIRE, finished=True)])
        assert [c["function"]["name"] for c in calls] == ["GetWeather"]

    @pytest.mark.parametrize("tool_choice", ["auto", "none"])
    def test_north_finalize_cannot_repromote_undeclared_raw_json(self, tool_choice):
        pp = _make_pp(tool_choice, parser="north")
        pp.tool_accumulated_text = (
            '<|START_ACTION|>{"name":"delete_everything","arguments":{}}<|END_ACTION|>'
        )

        events = pp.finalize()

        assert [event for event in events if event.type == "tool_call"] == []


class TestStreamingChannelRoutedSuppression:
    """Harmony/gemma4 surface structured calls on a dedicated channel that
    bypasses the text ``tool_parser`` — it needs its own none gate."""

    def _channel_output(self):
        return _output(
            "tool",
            finished=True,
            channel="tool_call",
            tool_calls=[
                {
                    "name": "GetWeather",
                    "arguments": '{"location":"Rome"}',
                    "id": "call_x",
                }
            ],
        )

    def test_none_suppresses_channel_routed_call(self):
        pp = _make_pp("none", parser="harmony")
        calls, content, finish = _drive(pp, [self._channel_output()])
        assert calls == []
        # The raw tool-call-channel delta must not leak into content either.
        assert "tool" not in content
        # R11-A: zero calls on the wire ⇒ the terminal is normalized to stop,
        # never a phantom finish_reason="tool_calls" (codex #1761 BLOCKING).
        assert finish == ["stop"]

    def test_none_suppresses_partial_then_separate_finish(self):
        # codex #1761: a partial ``tool_call`` channel chunk (no structured
        # call yet) must not leak as content, and a SEPARATE finish-only
        # chunk that still carries ``finish_reason="tool_calls"`` must be
        # normalized to ``"stop"`` — the suppression can't be gated on this
        # chunk carrying ``engine_tool_calls``.
        pp = _make_pp("none", parser="harmony")
        partial = _output("<call>get_", finished=False, channel="tool_call")
        finish = _output("", finished=True, channel="tool_call")
        finish.finish_reason = "tool_calls"  # engine's terminal, no calls set
        calls, content, finish_reasons = _drive(pp, [partial, finish])
        assert calls == []
        assert "get_" not in content  # partial channel markup did not leak
        assert finish_reasons == ["stop"]

    def test_auto_still_emits_channel_routed_call(self):
        pp = _make_pp("auto", parser="harmony")
        calls, _content, _finish = _drive(pp, [self._channel_output()])
        assert [c["function"]["name"] for c in calls] == ["GetWeather"]
