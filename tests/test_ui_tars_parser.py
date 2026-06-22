# SPDX-License-Identifier: Apache-2.0
"""Tests for UI-TARS Computer-Use action parser and reasoning parser.

Covers:
- All nine Computer-Use action verbs (click, left_double, right_single,
  drag, hotkey, type, scroll, wait, finished)
- Mobile-extension verbs (long_press, open_app, press_home, press_back)
- Leading ``Thought: ...`` preamble routed to reasoning_content
- ``Reflection: ... Action_Summary: ...`` (UI-TARS-1.5 reflective shape)
- Multi-action responses (one tool_call per action, sequential indices)
- Streaming dedup (don't double-emit already-yielded actions)
- Partial mid-stream action (no balanced close ``)``) doesn't crash
- Tokenizer detection regression guard — no ``Ġ`` / ``Ċ`` BPE markers
  in reasoning_content
- Anthropic adapter: same input → tool_use blocks with name="computer"
- Alias registry resolves to the ``ui_tars`` parsers
- Auto-config regex resolves bare HF paths to the ``ui_tars`` parsers

Tests deliberately use deterministic small inputs (no real model weights,
no real screenshots) and assert PARSER behavior, not model accuracy.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.model_auto_config import detect_model_config
from vllm_mlx.reasoning import get_parser as get_reasoning_parser
from vllm_mlx.reasoning.ui_tars_parser import UiTarsReasoningParser
from vllm_mlx.tool_parsers import ToolParserManager, UiTarsToolParser
from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
    _find_balanced_close,
    _normalize_action,
    _parse_kwargs,
    _parse_point,
)

# ---------------------------------------------------------------------------
# Registry / wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_tool_parser_registered_under_canonical_name(self):
        cls = ToolParserManager.get_tool_parser("ui_tars")
        assert cls is UiTarsToolParser

    @pytest.mark.parametrize("alias", ["ui-tars", "uitars"])
    def test_tool_parser_alias_names(self, alias: str):
        assert ToolParserManager.get_tool_parser(alias) is UiTarsToolParser

    def test_reasoning_parser_registered(self):
        cls = get_reasoning_parser("ui_tars")
        assert cls is UiTarsReasoningParser

    def test_tool_parser_declares_wire_format(self):
        # Audit guard: every parser must declare at least one wire format.
        # Adding a new label here also requires updating
        # ``WIRE_FORMAT_LABELS`` in abstract_tool_parser.
        assert UiTarsToolParser.EXPECTED_WIRE_FORMATS == ("ui_tars_action",)


class TestAliases:
    @pytest.mark.parametrize(
        "alias,hf_path",
        [
            ("ui-tars-1.5-7b-4bit", "mlx-community/UI-TARS-1.5-7B-4bit"),
            ("ui-tars-1.5-7b-6bit", "mlx-community/UI-TARS-1.5-7B-6bit"),
            ("ui-tars-1.5-7b-8bit", "mlx-community/UI-TARS-1.5-7B-8bit"),
            ("ui-tars-7b-dpo-4bit", "mlx-community/UI-TARS-7B-DPO-4bit"),
            ("ui-tars-7b-dpo-6bit", "mlx-community/UI-TARS-7B-DPO-6bit"),
            ("ui-tars-7b-dpo-8bit", "mlx-community/UI-TARS-7B-DPO-8bit"),
            ("ui-tars-7b-sft-4bit", "mlx-community/UI-TARS-7B-SFT-4bit"),
            ("ui-tars-7b-sft-8bit", "mlx-community/UI-TARS-7B-SFT-8bit"),
            ("ui-tars-72b-dpo-4bit", "mlx-community/UI-TARS-72B-DPO-4bit"),
        ],
    )
    def test_alias_resolves_to_ui_tars_parsers(self, alias: str, hf_path: str):
        profile = resolve_profile(alias)
        assert profile is not None, f"alias {alias} not in aliases.json"
        assert profile.hf_path == hf_path
        assert profile.tool_call_parser == "ui_tars"
        assert profile.reasoning_parser == "ui_tars"

    @pytest.mark.parametrize(
        "path",
        [
            "ByteDance-Seed/UI-TARS-1.5-7B",
            "ByteDance/UI-TARS-7B-DPO",
            "ByteDance/UI-TARS-72B-SFT",
            "uitars-experimental-7b",  # underscore-elided variant
            "ui_tars_local_checkpoint",
        ],
    )
    def test_regex_fallback_resolves_bare_hf_paths(self, path: str):
        cfg = detect_model_config(path)
        assert cfg is not None, f"no config for {path}"
        assert cfg.tool_call_parser == "ui_tars"
        assert cfg.reasoning_parser == "ui_tars"


# ---------------------------------------------------------------------------
# Coordinate normalization
# ---------------------------------------------------------------------------


class TestParsePoint:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("<point>200 300</point>", [200, 300]),
            ("<point>0 0</point>", [0, 0]),
            ("<point>999 1000</point>", [999, 1000]),
            ("<point>500.0 400.5</point>", [500, 400.5]),  # subpixel preserved
            ("<bbox>10 20 30 40</bbox>", [10, 20, 30, 40]),
            # UI-TARS-1.5 sentinel — verified against the live
            # mlx-community/UI-TARS-1.5-7B-4bit checkpoint (2026-06-21).
            ("<|box_start|>(233,45)<|box_end|>", [233, 45]),
            ("<|box_start|>(10,20),(30,40)<|box_end|>", [10, 20, 30, 40]),
            ("<200, 300>", [200, 300]),
            ("<200 300>", [200, 300]),
            ("[100, 200]", [100, 200]),
            ("(100, 200)", [100, 200]),
            ("100,200", [100, 200]),
        ],
    )
    def test_parse_recognized_shapes(self, raw: str, expected: list):
        assert _parse_point(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "not a point",
            "<point></point>",
            "garbage",
        ],
    )
    def test_unrecognized_returns_none(self, raw: str):
        assert _parse_point(raw) is None

    def test_non_string_returns_none(self):
        assert _parse_point(None) is None  # type: ignore[arg-type]
        assert _parse_point(123) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Kwarg parsing
# ---------------------------------------------------------------------------


class TestParseKwargs:
    def test_single_kwarg(self):
        assert _parse_kwargs("point='<point>1 2</point>'") == {
            "point": "<point>1 2</point>"
        }

    def test_multi_kwarg(self):
        out = _parse_kwargs(
            "start_point='<point>1 2</point>', end_point='<point>3 4</point>'"
        )
        assert out == {
            "start_point": "<point>1 2</point>",
            "end_point": "<point>3 4</point>",
        }

    def test_double_quote_kwarg(self):
        assert _parse_kwargs('content="hello"') == {"content": "hello"}

    def test_escaped_newline(self):
        # Escape sequences inside quoted args should round-trip the
        # literal newline (Python string semantics), not the ``\n``
        # token.
        assert _parse_kwargs("content='line1\\nline2'") == {"content": "line1\nline2"}

    def test_paren_inside_string(self):
        # ``)`` inside the string must not truncate the body.
        out = _parse_kwargs("content='hello (world)'")
        assert out == {"content": "hello (world)"}

    def test_empty_body(self):
        assert _parse_kwargs("") == {}

    def test_malformed_falls_through_to_lenient(self):
        # Unclosed quote — ast.parse fails, lenient regex picks it up.
        # (We accept best-effort here; no crash is the contract.)
        out = _parse_kwargs("key='unclosed")
        # Could be {} or {"key": "unclosed"} depending on fallback —
        # the load-bearing assertion is: no exception.
        assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# Action normalization
# ---------------------------------------------------------------------------


class TestNormalizeAction:
    def test_click(self):
        args = _normalize_action("click", {"point": "<point>200 300</point>"})
        assert args == {"action": "click", "point": [200, 300]}

    def test_drag_two_points(self):
        args = _normalize_action(
            "drag",
            {
                "start_point": "<point>1 2</point>",
                "end_point": "<point>3 4</point>",
            },
        )
        assert args == {
            "action": "drag",
            "start_point": [1, 2],
            "end_point": [3, 4],
        }

    def test_hotkey_passthrough(self):
        # Dogfood F-R1-06 fix: UI-TARS trains on space-separated chord
        # syntax (``"ctrl c"``) but the documented Computer-Use spec —
        # and every downstream computer-use runtime (xdotool,
        # pyautogui, the Anthropic computer-tool harness) — expects
        # plus form (``"ctrl+c"``). Normalize at the parser boundary
        # so SDK consumers don't have to.
        args = _normalize_action("hotkey", {"key": "ctrl c"})
        assert args == {"action": "hotkey", "key": "ctrl+c"}

    def test_hotkey_plus_form_preserved(self):
        # Already plus-shaped chord (rare — model usually emits space
        # form) stays untouched so a runtime that accepts BOTH forms
        # keeps working.
        args = _normalize_action("hotkey", {"key": "ctrl+c"})
        assert args == {"action": "hotkey", "key": "ctrl+c"}

    def test_unknown_verb_emitted_verbatim(self):
        # Future UI-TARS verb — preserve so we don't silently drop calls.
        args = _normalize_action(
            "tap_with_pressure",
            {"point": "<point>10 20</point>", "pressure": 0.5},
        )
        assert args == {
            "action": "tap_with_pressure",
            "point": [10, 20],
            "pressure": 0.5,
        }

    def test_unparseable_point_kept_as_string(self):
        # If the coord body is junk, preserve the raw string in args so
        # the downstream consumer can decide what to do.
        args = _normalize_action("click", {"point": "garbage"})
        assert args == {"action": "click", "point": "garbage"}


# ---------------------------------------------------------------------------
# Balanced-paren close
# ---------------------------------------------------------------------------


class TestFindBalancedClose:
    def test_simple(self):
        s = "abc)"
        assert _find_balanced_close(s, 0) == 3

    def test_nested(self):
        s = "a(b)c)"
        assert _find_balanced_close(s, 0) == 5

    def test_paren_in_string(self):
        s = "x='hi)', y=1)"
        assert _find_balanced_close(s, 0) == len(s) - 1

    def test_unbalanced_returns_minus_one(self):
        assert _find_balanced_close("no close here", 0) == -1

    def test_escaped_quote(self):
        s = "x='it\\'s', y=2)"
        assert _find_balanced_close(s, 0) == len(s) - 1


# ---------------------------------------------------------------------------
# Tool parser — complete-response
# ---------------------------------------------------------------------------


def _decode(tc: dict) -> dict:
    """Round-trip the JSON arguments back to a dict for assertions."""
    return json.loads(tc["arguments"])


class TestToolParserComplete:
    def setup_method(self) -> None:
        self.p = UiTarsToolParser()

    def test_click(self):
        text = (
            "Thought: I need to click search.\n"
            "Action: click(point='<point>200 300</point>')"
        )
        r = self.p.extract_tool_calls(text)
        assert r.tools_called is True
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0]["name"] == "computer"
        args = _decode(r.tool_calls[0])
        assert args == {"action": "click", "point": [200, 300]}

    def test_left_double(self):
        text = "Action: left_double(point='<point>10 20</point>')"
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {
            "action": "left_double",
            "point": [10, 20],
        }

    def test_right_single(self):
        text = "Action: right_single(point='<point>10 20</point>')"
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {
            "action": "right_single",
            "point": [10, 20],
        }

    def test_drag(self):
        text = (
            "Action: drag(start_point='<point>1 2</point>', "
            "end_point='<point>3 4</point>')"
        )
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {
            "action": "drag",
            "start_point": [1, 2],
            "end_point": [3, 4],
        }

    def test_hotkey(self):
        text = "Action: hotkey(key='ctrl c')"
        # Dogfood F-R1-06 fix: parser normalizes space-form chord to
        # plus-form so downstream computer-use runtimes receive the
        # spec shape.
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {"action": "hotkey", "key": "ctrl+c"}

    def test_type(self):
        text = "Action: type(content='hello world\\n')"
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {
            "action": "type",
            "content": "hello world\n",
        }

    def test_scroll(self):
        text = "Action: scroll(point='<point>500 400</point>', direction='down')"
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {
            "action": "scroll",
            "point": [500, 400],
            "direction": "down",
        }

    def test_wait(self):
        text = "Action: wait()"
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {"action": "wait"}

    def test_finished(self):
        text = "Action: finished(content='done')"
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {"action": "finished", "content": "done"}

    def test_mobile_long_press(self):
        text = "Action: long_press(point='<point>50 60</point>')"
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {
            "action": "long_press",
            "point": [50, 60],
        }

    def test_mobile_press_home(self):
        text = "Action: press_home()"
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {"action": "press_home"}

    def test_mobile_open_app(self):
        text = "Action: open_app(app_name='Calendar')"
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {
            "action": "open_app",
            "app_name": "Calendar",
        }

    def test_multi_action_sequential_indices(self):
        text = (
            "Thought: Click then type.\n"
            "Action: click(point='<point>1 2</point>')\n"
            "Action: type(content='hi')"
        )
        r = self.p.extract_tool_calls(text)
        assert len(r.tool_calls) == 2
        first = _decode(r.tool_calls[0])
        second = _decode(r.tool_calls[1])
        assert first == {"action": "click", "point": [1, 2]}
        assert second == {"action": "type", "content": "hi"}

    def test_actions_stripped_from_content(self):
        text = "Thought: Click search.\nAction: click(point='<point>200 300</point>')"
        r = self.p.extract_tool_calls(text)
        # Both Action: and Thought: preamble are stripped — the reasoning
        # parser owns the chain-of-thought channel, the tool parser owns
        # the action channel, content should be empty (or whatever sits
        # between Action: lines). Stripping Thought: here is a defensive
        # mirror so the chain-of-thought doesn't double-render when the
        # reasoning parser also extracts it.
        assert "Action:" not in (r.content or "")
        assert "Thought:" not in (r.content or "")

    def test_box_start_sentinel_normalized(self):
        # UI-TARS-1.5 emits ``start_box='<|box_start|>(x,y)<|box_end|>'``
        # — verified against the live mlx-community/UI-TARS-1.5-7B-4bit
        # checkpoint (2026-06-21). Coords are absolute pixel offsets.
        # Dogfood F-R1-02 fix: the parser renames the verb-specific
        # ``start_box`` (UI-TARS-1.5 internal) → spec ``point`` for
        # single-point verbs like ``click``, so downstream consumers
        # see the documented PR #812 contract regardless of which
        # UI-TARS checkpoint produced the bytes.
        text = "Action: click(start_box='<|box_start|>(233,45)<|box_end|>')"
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {
            "action": "click",
            "point": [233, 45],
        }

    def test_nested_action_in_string_arg_not_double_parsed(self):
        # Codex r1 BLOCKING #1 (2026-06-21): ``Action: type(content=
        # 'Action: wait()')`` previously parsed as TWO calls because the
        # outer ``finditer`` matched the sentinel inside the string arg.
        # The scanner now advances past each matched ``)`` before
        # searching for the next ``Action:``, so the inner sentinel
        # stays as plain string content.
        text = "Action: type(content='Action: wait()')"
        r = self.p.extract_tool_calls(text)
        assert len(r.tool_calls) == 1
        assert _decode(r.tool_calls[0]) == {
            "action": "type",
            "content": "Action: wait()",
        }

    def test_no_action_passes_through(self):
        text = "Just a thought, no action."
        r = self.p.extract_tool_calls(text)
        assert r.tools_called is False
        assert r.content == text

    def test_empty_input(self):
        r = self.p.extract_tool_calls("")
        assert r.tools_called is False
        assert r.content == ""

    def test_unbalanced_paren_skipped_gracefully(self):
        # Partial mid-stream (no balanced ``)``) — must not crash, no
        # tool_calls emitted.
        text = "Action: click(point='<point>1 2</point>'"
        r = self.p.extract_tool_calls(text)
        assert r.tools_called is False

    def test_double_quote_args(self):
        # Per UI-TARS prompt, kwargs are single-quoted, but some
        # quantized variants emit double-quoted — accept both.
        text = 'Action: click(point="<point>10 20</point>")'
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {"action": "click", "point": [10, 20]}

    def test_paren_inside_string_arg(self):
        # ``type(content='paste (here)')`` — ``)`` inside string must
        # not terminate the action body early.
        text = "Action: type(content='paste (here)')"
        r = self.p.extract_tool_calls(text)
        assert _decode(r.tool_calls[0]) == {
            "action": "type",
            "content": "paste (here)",
        }

    def test_call_id_uses_call_prefix(self):
        # Coordinated with D-ANTHRO-SPEC-POLISH: every parser stays on
        # the OpenAI ``call_`` convention; the Anthropic adapter rewrites
        # to ``toolu_`` at the /v1/messages boundary.
        text = "Action: wait()"
        r = self.p.extract_tool_calls(text)
        assert r.tool_calls[0]["id"].startswith("call_")


class TestToolParserStreaming:
    def setup_method(self) -> None:
        self.p = UiTarsToolParser()

    def _stream(self, deltas: list[str]) -> list:
        prev = ""
        out = []
        for d in deltas:
            cur = prev + d
            msg = self.p.extract_tool_calls_streaming(prev, cur, d)
            if msg is not None:
                out.append(msg)
            prev = cur
        return out

    def test_partial_action_no_double_emit(self):
        deltas = [
            "Thought: hi\n",
            "Action: click(",
            "point='<point>1 2</point>'",
            ")",  # close arrives in a separate delta
        ]
        events = self._stream(deltas)
        # Only one tool_call emitted, on the delta that closes the paren.
        tool_events = [e for e in events if "tool_calls" in e]
        assert len(tool_events) == 1
        assert tool_events[0]["tool_calls"][0]["index"] == 0
        args = json.loads(tool_events[0]["tool_calls"][0]["function"]["arguments"])
        assert args == {"action": "click", "point": [1, 2]}

    def test_multi_action_streaming(self):
        deltas = [
            "Action: click(point='<point>1 2</point>')\n",
            "Action: type(content='hi')",
        ]
        events = self._stream(deltas)
        tool_events = [e for e in events if "tool_calls" in e]
        assert len(tool_events) == 2
        assert tool_events[0]["tool_calls"][0]["index"] == 0
        assert tool_events[1]["tool_calls"][0]["index"] == 1

    def test_thought_passes_through_as_content(self):
        # Before any Action: token arrives, deltas stream through as
        # content (the reasoning parser claims them downstream).
        prev = ""
        msg = self.p.extract_tool_calls_streaming(prev, "Thought: hi", "Thought: hi")
        assert msg == {"content": "Thought: hi"}

    def test_has_pending_tool_call_partial(self):
        # An open ``Action: foo(`` should keep the postprocessor in
        # "hold" mode.
        assert self.p.has_pending_tool_call("Action: click(") is True
        # A completed action — not pending anymore.
        assert self.p.has_pending_tool_call("Action: wait()") is False
        # No action at all — not pending.
        assert self.p.has_pending_tool_call("Thought: hi") is False

    def test_residual_content_after_action_in_same_delta(self):
        # codex r2 BLOCKING #1 + NIT #3: when a delta contains a
        # completed action followed by ordinary text (e.g. the model
        # appends ``  done`` after ``Action: wait()`` in a single chunk),
        # the parser MUST emit both the ``tool_calls`` and the trailing
        # text as ``content`` — otherwise the trailing bytes are lost
        # from the response.
        events = self._stream(["Action: wait() done"])
        tool_events = [e for e in events if "tool_calls" in e]
        assert len(tool_events) == 1
        assert tool_events[0]["tool_calls"][0]["function"]["name"] == "computer"
        # Content side must carry the trailing `` done`` bytes — note
        # leading space is preserved verbatim (no normalization).
        assert tool_events[0].get("content") == " done"

    def test_residual_content_between_two_actions_in_same_delta(self):
        # Same regression as above but exercises the leading/trailing
        # residual paths simultaneously: ``prefix Action: wait() between
        # Action: finished() trailing`` packs prefix/inter-action/trailing
        # non-action text alongside two completed actions in one delta.
        # All three text spans must surface in the ``content`` field.
        delta = "leading Action: wait() between Action: finished() trailing"
        events = self._stream([delta])
        tool_events = [e for e in events if "tool_calls" in e]
        assert len(tool_events) == 1
        assert len(tool_events[0]["tool_calls"]) == 2
        residual = tool_events[0].get("content", "")
        assert "leading " in residual
        assert " between " in residual
        assert " trailing" in residual


# ---------------------------------------------------------------------------
# Reasoning parser — Thought:/Reflection: preamble split
# ---------------------------------------------------------------------------


class TestReasoningParserComplete:
    def setup_method(self) -> None:
        self.p = UiTarsReasoningParser()

    def test_thought_preamble(self):
        text = (
            "Thought: I need to click search.\n"
            "Action: click(point='<point>200 300</point>')"
        )
        reasoning, content = self.p.extract_reasoning(text)
        assert reasoning is not None
        assert "I need to click search" in reasoning
        assert content is not None
        assert content.startswith("Action:")

    def test_reflection_action_summary_preamble(self):
        text = (
            "Reflection: Last action missed.\n"
            "Action_Summary: Click the correct button.\n"
            "Action: click(point='<point>100 200</point>')"
        )
        reasoning, content = self.p.extract_reasoning(text)
        assert reasoning is not None
        assert "Reflection:" in reasoning
        assert "Action_Summary:" in reasoning
        assert content is not None
        assert content.startswith("Action:")

    def test_no_preamble(self):
        # Grounding template: action-only.
        text = "Action: click(point='<point>1 2</point>')"
        reasoning, content = self.p.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_empty_input(self):
        reasoning, content = self.p.extract_reasoning("")
        assert reasoning is None
        assert content == ""

    def test_bpe_markers_pass_through_untouched(self):
        # Regression guard: if a buggy upstream pipeline forwards raw
        # byte-level BPE artifacts (``Ġ`` for space, ``Ċ`` for newline)
        # to the reasoning parser, we must NOT silently strip them — we
        # also must not add them where they didn't exist. The parser is
        # a string-level slicer; markers should round-trip verbatim into
        # the reasoning channel so the caller can detect the upstream
        # detokenizer bug instead of having it hidden here.
        # (codex r1 NIT #3: the previous version of this test claimed to
        # exercise marker handling but never fed markers in.)
        text = "Thought:ĠThisĠthoughtĠhasĠBPEĠmarkers.ĊAction: wait()"
        reasoning, content = self.p.extract_reasoning(text)
        # Marker bytes survive untouched in reasoning (no covert strip).
        assert reasoning is not None
        assert "Ġ" in reasoning
        assert "Ċ" in reasoning
        # And the parser doesn't smear markers into the content side
        # where the source text had none.
        assert content is not None
        assert "Ġ" not in content
        assert "Ċ" not in content

        # Companion check: with NO markers in source, none appear in
        # output either (parser never invents BPE artifacts).
        clean = "Thought: a clean thought.\nAction: wait()"
        r2, c2 = self.p.extract_reasoning(clean)
        assert r2 is not None and "Ġ" not in r2 and "Ċ" not in r2
        assert c2 is not None and "Ġ" not in c2 and "Ċ" not in c2


class TestReasoningParserStreaming:
    def setup_method(self) -> None:
        self.p = UiTarsReasoningParser()

    def _stream(self, deltas: list[str]) -> list:
        prev = ""
        out = []
        for d in deltas:
            cur = prev + d
            msg = self.p.extract_reasoning_streaming(prev, cur, d)
            if msg is not None:
                out.append(msg)
            prev = cur
        return out

    def test_thought_streamed_as_reasoning(self):
        events = self._stream(["Thought: ", "I'm thinking.\n", "Action: wait()"])
        reasoning_concat = "".join(e.reasoning or "" for e in events)
        content_concat = "".join(e.content or "" for e in events)
        assert "I'm thinking" in reasoning_concat
        assert content_concat.startswith("Action:")

    def test_boundary_split_in_single_delta(self):
        events = self._stream(["Thought: hi.\nAction: wait()"])
        reasoning_concat = "".join(e.reasoning or "" for e in events)
        content_concat = "".join(e.content or "" for e in events)
        assert "hi" in reasoning_concat
        # Boundary delta must NOT have ``Action:`` in reasoning side.
        assert "Action:" not in reasoning_concat
        assert content_concat.startswith("Action:")

    def test_no_preamble_routes_to_content(self):
        # Buffer reaches Action_Summary: length (15 chars) without a
        # preamble opener → flips to content.
        events = self._stream(["Action: wait() done extra padding bytes"])
        content_concat = "".join(e.content or "" for e in events)
        reasoning_concat = "".join(e.reasoning or "" for e in events)
        assert reasoning_concat == ""
        assert content_concat.startswith("Action:")

    def test_short_action_only_response_routes_promptly(self):
        # codex r1 BLOCKING #2: when the response is a short action-only
        # ack (Grounding template, no Thought:) and the bytes arrive
        # split across tiny deltas, the parser must flip to content as
        # soon as ``Action:`` is complete — NOT wait until the buffer
        # reaches ``Action_Summary:`` length (15 chars). Otherwise a
        # bare ``Action: wait()`` (13 chars) gets held forever and the
        # dispatcher emits an empty assistant message.
        events = self._stream(
            [
                "Action",  # 6 chars — could still be Action: prefix
                ":",  # now ``Action:`` complete at 7 chars
                " wait()",
            ]
        )
        reasoning_concat = "".join(e.reasoning or "" for e in events)
        content_concat = "".join(e.content or "" for e in events)
        # All bytes route to content (no preamble).
        assert reasoning_concat == ""
        # Content stream contains the full ``Action: wait()`` sentinel
        # ready for the tool parser to extract.
        assert "Action:" in content_concat
        assert "wait()" in content_concat
        # And we actually emitted at least one content event before
        # end-of-stream — i.e. the bytes weren't held until the buffer
        # reached 15 chars (the response is only 13 chars total).
        assert sum(1 for e in events if e.content) >= 1

    def test_partial_action_opener_held_back(self):
        # Regression for the live-stream symptom: model emits
        # ``Action`` then ``:`` in separate deltas. Without the
        # partial-opener hold, the ``Action`` token leaks into the
        # reasoning channel and the tool parser sees ``: click(...)``
        # without the leading ``Action:`` sentinel — extracts no
        # tool_calls. The hold keeps the trailing ``Action`` in
        # reasoning's buffer until the ``:`` arrives, then releases
        # ``Action:`` as a single content event so the tool parser
        # sees the full token.
        events = self._stream(
            [
                "Thought: ",
                "click.\n",
                "Action",  # partial opener
                ":",  # boundary resolves
                " click(point='<point>1 2</point>')",
            ]
        )
        reasoning_concat = "".join(e.reasoning or "" for e in events)
        content_concat = "".join(e.content or "" for e in events)
        # No ``Action`` token leaked into reasoning.
        assert "Action" not in reasoning_concat
        # Tool parser sees the complete ``Action:`` opener.
        assert "Action:" in content_concat


# ---------------------------------------------------------------------------
# Anthropic adapter end-to-end mapping
# ---------------------------------------------------------------------------


class TestAnthropicAdapter:
    """UI-TARS tool_calls should round-trip cleanly to Anthropic tool_use blocks."""

    def setup_method(self) -> None:
        self.p = UiTarsToolParser()

    def _to_openai_choice(self, text: str):
        """Build a minimal OpenAI-style ChatCompletion response with parser output."""
        # Import inside the test to avoid pulling the adapter into module
        # import paths used by smaller parser-only tests.
        from vllm_mlx.api.models import (
            AssistantMessage,
            ChatCompletionChoice,
            ChatCompletionResponse,
            FunctionCall,
            ToolCall,
        )

        r = self.p.extract_tool_calls(text)
        oai_tool_calls = [
            ToolCall(
                id=tc["id"],
                type="function",
                function=FunctionCall(name=tc["name"], arguments=tc["arguments"]),
            )
            for tc in r.tool_calls
        ]
        return ChatCompletionResponse(
            id="chatcmpl-test",
            object="chat.completion",
            created=0,
            model="ui-tars-1.5-7b-4bit",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=r.content,
                        tool_calls=oai_tool_calls or None,
                    ),
                    finish_reason="tool_calls" if r.tools_called else "stop",
                )
            ],
        )

    def test_click_maps_to_tool_use_with_computer_name(self):
        from vllm_mlx.api.anthropic_adapter import openai_to_anthropic

        text = "Thought: Click search.\nAction: click(point='<point>200 300</point>')"
        openai_resp = self._to_openai_choice(text)
        anth = openai_to_anthropic(openai_resp, "ui-tars-1.5-7b-4bit")
        tool_use_blocks = [b for b in anth.content if b.type == "tool_use"]
        assert len(tool_use_blocks) == 1
        tu = tool_use_blocks[0]
        assert tu.name == "computer"
        # R6-M2: Anthropic adapter translates UI-TARS canonical ``point``
        # to spec ``coordinate`` on the ``/v1/messages`` boundary.
        assert tu.input == {"action": "click", "coordinate": [200, 300]}

    def test_multi_action_emits_multiple_tool_use_blocks(self):
        from vllm_mlx.api.anthropic_adapter import openai_to_anthropic

        text = (
            "Thought: Click then type.\n"
            "Action: click(point='<point>1 2</point>')\n"
            "Action: type(content='hi')"
        )
        openai_resp = self._to_openai_choice(text)
        anth = openai_to_anthropic(openai_resp, "ui-tars-1.5-7b-4bit")
        tool_use_blocks = [b for b in anth.content if b.type == "tool_use"]
        assert len(tool_use_blocks) == 2
        assert tool_use_blocks[0].input["action"] == "click"
        assert tool_use_blocks[1].input["action"] == "type"
