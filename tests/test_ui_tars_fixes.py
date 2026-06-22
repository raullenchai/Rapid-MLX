# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for 0.8.5 UI-TARS dogfood findings (Ana R1+R2).

Bugs covered:
- C-05 (CRIT): UI-TARS Computer-Use system prompt not auto-prepended on
  the chat lane; parser silently no-ops on raw model output. Fixed by
  ``maybe_inject_ui_tars_system_prompt`` wired into ``routes/chat.py``
  and ``routes/anthropic.py``.
- C-07 (CRIT): ``tool_choice="none"`` ignored — UI-TARS still emits a
  ``computer`` tool_call. Fixed by (a) skipping the sysprompt inject
  when ``tool_choice="none"`` (REQUEST-time) and (b) defensive
  short-circuit in the parser (RESPONSE-time) so even an
  operator-supplied sysprompt path produces no ``tool_calls``.
- F-R1-02 (HIGH): Parser emits ``start_box``/``end_box`` instead of
  spec ``point``/``start_point``/``end_point``. Fixed by verb-aware
  key normalization in ``_normalize_action`` / ``_spec_key_for``.
- F-R1-04 (HIGH): Streaming OpenAI lane leaks ``Thought:`` / ``Action:``
  markers into ``delta.content`` while non-streaming strips them.
  Fixed by tightening the partial-opener hold-back gate in the
  streaming reasoning parser (``"Thought"`` prefix no longer flips
  to content when buffer reaches 7 chars).
- F-R1-06 (HIGH): ``hotkey.key`` emitted as space-separated chord
  (``"ctrl c"``) instead of plus-form (``"ctrl+c"``). Fixed by
  normalizing the chord at the parser boundary.
- F-R2-04 (HIGH): OpenAI lane and Anthropic lane produce different
  coords on identical input. Root cause: only the OAI lane was
  injecting the canonical UI-TARS sysprompt — the Anthropic lane
  never did, so the model received different prompts and emitted
  different coords. Fixed by wiring the SAME shared helper into
  BOTH routes (parser-level emit identity assert kept here; the
  end-to-end cross-lane assert lives in the dogfood replay).
- F-R2-05 (HIGH): ``[SSE-TC]`` INFO log leaked tool_call arguments
  (user-action coords) into the server log on every Computer-Use
  turn. Dropped to DEBUG so the PII path is opt-in.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from vllm_mlx.reasoning.ui_tars_parser import UiTarsReasoningParser
from vllm_mlx.tool_parsers import UiTarsToolParser
from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
    UI_TARS_COMPUTER_USE_SYSTEM_PROMPT,
    _is_tool_choice_none,
    _normalize_action,
    has_ui_tars_system_prompt,
    maybe_inject_ui_tars_system_prompt,
)

# ---------------------------------------------------------------------------
# C-05: sysprompt auto-wire
# ---------------------------------------------------------------------------


class TestSysPromptAutoWire:
    def test_canonical_sysprompt_contains_action_space(self):
        # Sanity check that the canonical sysprompt actually mentions
        # the action-API contract the model is post-trained on.
        assert "## Action Space" in UI_TARS_COMPUTER_USE_SYSTEM_PROMPT
        assert "## Output Format" in UI_TARS_COMPUTER_USE_SYSTEM_PROMPT
        assert "Thought: ..." in UI_TARS_COMPUTER_USE_SYSTEM_PROMPT
        assert "Action: ..." in UI_TARS_COMPUTER_USE_SYSTEM_PROMPT
        assert "click(point=" in UI_TARS_COMPUTER_USE_SYSTEM_PROMPT
        assert "drag(start_point=" in UI_TARS_COMPUTER_USE_SYSTEM_PROMPT

    def test_inject_when_no_user_sysprompt(self):
        # C-05 repro: a UI-TARS request with no system message — the
        # canonical sysprompt MUST be prepended so the parser actually
        # sees the ``Action:`` lines the model is post-trained on.
        messages = [{"role": "user", "content": "Click the search button."}]
        out = maybe_inject_ui_tars_system_prompt(
            messages, tool_call_parser="ui_tars", tool_choice=None
        )
        assert out[0]["role"] == "system"
        assert out[0]["content"] == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT
        assert out[1]["role"] == "user"
        # Original list was NOT mutated in place — caller can keep
        # using their own reference safely.
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_inject_with_user_sysprompt_lands_first(self):
        # When the user ALSO supplies a system message, the auto-
        # injected sysprompt lands FIRST and the user content is
        # preserved verbatim AFTER (additive, not overriding).
        # Operator design choice — extends, doesn't override.
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Click search."},
        ]
        out = maybe_inject_ui_tars_system_prompt(
            messages, tool_call_parser="ui_tars", tool_choice=None
        )
        # Auto-injected sysprompt FIRST, user system PRESERVED.
        assert out[0]["content"] == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT
        assert out[1] == {"role": "system", "content": "Be concise."}
        assert out[2]["role"] == "user"

    def test_skip_inject_when_user_pasted_canonical(self):
        # If the operator already pasted (a variant of) the canonical
        # sysprompt, we DON'T double-inject — respect their wording.
        # Detection requires a STRONG UI-TARS-specific marker (codex
        # r2): use the canonical ``## Action Space`` heading here.
        user_sys = (
            "You are a GUI agent.\n## Action Space\n"
            "click(point='<point>x y</point>')\nfinished()"
        )
        messages = [
            {"role": "system", "content": user_sys},
            {"role": "user", "content": "Click."},
        ]
        out = maybe_inject_ui_tars_system_prompt(
            messages, tool_call_parser="ui_tars", tool_choice=None
        )
        # No new sysprompt prepended.
        assert out == messages
        assert len(out) == 2

    @pytest.mark.parametrize(
        "marker",
        [
            # Header-level marker — unique to UI-TARS-class prompts.
            "## Action Space",
            # Action-verb call signatures — model-API kwarg shape.
            "click(point=",
            "click(start_box=",
            "drag(start_point=",
            "drag(start_box=",
            # Joint Output Format / Action skeleton — catches
            # forks that strip the markdown headers.
            "Thought: ...\nAction: ...",
        ],
    )
    def test_detect_canonical_via_strong_marker(self, marker: str):
        # Detection requires a STRONG, UI-TARS-specific marker:
        # either the canonical section heading
        # (``## Action Space``), a model-API kwarg call signature
        # (``click(point=`` etc.), or the joint Output-Format
        # skeleton. Generic markers were rejected per codex r2.
        messages = [{"role": "system", "content": f"prefix\n{marker}\nsuffix"}]
        assert has_ui_tars_system_prompt(messages) is True

    @pytest.mark.parametrize(
        "generic_sys",
        [
            # Codex r2 BLOCKING regression: a generic system
            # prompt that happens to mention ``## Output Format``
            # for JSON / markdown formatting MUST NOT skip the
            # UI-TARS auto-inject — the pre-tightening detector
            # would have falsely treated this as "operator already
            # pasted UI-TARS sysprompt".
            "## Output Format\nReturn answers as JSON.",
            "You are a GUI agent for browser-based tasks.",
            # A generic system prompt about agents / instructions
            # without the UI-TARS structural markers.
            "Follow the user's instructions step by step.",
            "## Tools available\nclick, drag, scroll",
        ],
    )
    def test_generic_system_prompt_does_not_false_positive(self, generic_sys: str):
        messages = [{"role": "system", "content": generic_sys}]
        assert has_ui_tars_system_prompt(messages) is False
        # And the auto-inject still fires for a UI-TARS request that
        # carries one of these generic system messages.
        out = maybe_inject_ui_tars_system_prompt(
            messages, tool_call_parser="ui_tars", tool_choice="auto"
        )
        assert len(out) == 2
        assert out[0]["content"] == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT

    def test_skip_inject_when_parser_is_not_ui_tars(self):
        # Wrong model family — no inject. Avoids polluting
        # non-UI-TARS aliases (every Qwen / Hermes / etc.) with the
        # Computer-Use action-API contract.
        messages = [{"role": "user", "content": "Hi."}]
        out = maybe_inject_ui_tars_system_prompt(
            messages, tool_call_parser="hermes", tool_choice=None
        )
        assert out == messages

    def test_skip_inject_when_tool_choice_none(self):
        # C-07: when ``tool_choice="none"`` the caller is opting OUT
        # of tool emission. Skip the sysprompt inject so the model
        # produces plain prose, NOT ``Action: ...`` lines.
        messages = [{"role": "user", "content": "What time is it?"}]
        out = maybe_inject_ui_tars_system_prompt(
            messages, tool_call_parser="ui_tars", tool_choice="none"
        )
        assert out == messages

    def test_inject_matches_message_object_shape(self):
        # Codex r1 BLOCKING #1: when ``messages`` is a list of
        # pydantic Message objects (defensive — production code
        # paths normalize to dicts via ``extract_multimodal_content``,
        # but some test paths and future surfaces may not), the
        # helper must NOT prepend a plain dict that would produce a
        # mixed-shape list downstream. Mirror the object shape via
        # ``model_copy(update=...)`` when available.
        from vllm_mlx.api.models import Message

        msgs = [Message(role="user", content="Click.")]
        out = maybe_inject_ui_tars_system_prompt(
            msgs, tool_call_parser="ui_tars", tool_choice="auto"
        )
        assert len(out) == 2
        # Inserted message is a Message object (same shape as the
        # original list entries), NOT a plain dict.
        assert isinstance(out[0], Message)
        assert out[0].role == "system"
        assert out[0].content == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT
        # Original user message preserved.
        assert isinstance(out[1], Message)
        assert out[1].role == "user"

    def test_anthropic_shape_system_content_blocks_detected(self):
        # System message with Anthropic-style content blocks must
        # also flow through the detector — otherwise the
        # multimodal Anthropic lane would double-inject.
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "## Action Space\nclick(point=...)"}
                ],
            }
        ]
        assert has_ui_tars_system_prompt(messages) is True


# ---------------------------------------------------------------------------
# C-07: tool_choice="none" honored
# ---------------------------------------------------------------------------


def _decode_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    return json.loads(tool_call["arguments"])


class TestToolChoiceNone:
    def setup_method(self) -> None:
        self.p = UiTarsToolParser()

    def test_helper_recognizes_string_none(self):
        assert _is_tool_choice_none({"tool_choice": "none"}) is True
        assert _is_tool_choice_none({"tool_choice": "auto"}) is False
        assert _is_tool_choice_none({"tool_choice": None}) is False
        assert _is_tool_choice_none(None) is False

    def test_non_streaming_suppresses_tool_calls(self):
        # Ana F-R2-02 repro: model output contains ``Action: click(...)``
        # but the request specified ``tool_choice="none"``. Per OpenAI
        # spec the response MUST be text-only with no ``tool_calls``.
        text = "Thought: Want to click.\nAction: click(point='<point>100 200</point>')"
        r = self.p.extract_tool_calls(text, request={"tool_choice": "none"})
        assert r.tools_called is False
        assert r.tool_calls == []
        # Raw bytes surface as content so the caller still sees the
        # model output (no silent drop).
        assert "Action: click" in (r.content or "")

    def test_non_streaming_emits_tool_calls_for_auto(self):
        # Same input, ``tool_choice="auto"`` → tool_call emitted.
        text = "Action: click(point='<point>100 200</point>')"
        r = self.p.extract_tool_calls(text, request={"tool_choice": "auto"})
        assert r.tools_called is True
        assert len(r.tool_calls) == 1
        assert _decode_args(r.tool_calls[0]) == {
            "action": "click",
            "point": [100, 200],
        }

    def test_streaming_suppresses_tool_calls(self):
        # Streaming variant: deltas with Action: bytes get passed
        # through as content (not tool_call deltas).
        delta = "Action: click(point='<point>10 20</point>')"
        result = self.p.extract_tool_calls_streaming(
            previous_text="",
            current_text=delta,
            delta_text=delta,
            request={"tool_choice": "none"},
        )
        assert result is not None
        assert "tool_calls" not in result
        assert result.get("content") == delta


# ---------------------------------------------------------------------------
# Parser key normalization: start_box → point, end_box → end_point
# ---------------------------------------------------------------------------


class TestParserKeyNormalization:
    def setup_method(self) -> None:
        self.p = UiTarsToolParser()

    def test_click_start_box_renamed_to_point(self):
        # Ana F-R1-02: model emits ``start_box`` for single-point
        # verbs; the spec says ``point``. Parser renames.
        text = "Action: click(start_box='(112,126)')"
        r = self.p.extract_tool_calls(text)
        args = _decode_args(r.tool_calls[0])
        assert args == {"action": "click", "point": [112, 126]}
        assert "start_box" not in args

    def test_left_double_start_box_renamed_to_point(self):
        text = "Action: left_double(start_box='(100,200)')"
        r = self.p.extract_tool_calls(text)
        args = _decode_args(r.tool_calls[0])
        assert args["action"] == "left_double"
        assert args["point"] == [100, 200]

    def test_right_single_start_box_renamed_to_point(self):
        text = "Action: right_single(start_box='(50,75)')"
        r = self.p.extract_tool_calls(text)
        args = _decode_args(r.tool_calls[0])
        assert args["action"] == "right_single"
        assert args["point"] == [50, 75]

    def test_scroll_start_box_renamed_to_point(self):
        text = "Action: scroll(start_box='(500,400)', direction='down')"
        r = self.p.extract_tool_calls(text)
        args = _decode_args(r.tool_calls[0])
        assert args == {
            "action": "scroll",
            "point": [500, 400],
            "direction": "down",
        }

    def test_drag_start_box_end_box_renamed_to_start_end_point(self):
        # Two-point verb: spec is ``start_point`` / ``end_point``,
        # NOT ``start_box`` / ``end_box``.
        text = "Action: drag(start_box='(100,100)', end_box='(300,500)')"
        r = self.p.extract_tool_calls(text)
        args = _decode_args(r.tool_calls[0])
        assert args == {
            "action": "drag",
            "start_point": [100, 100],
            "end_point": [300, 500],
        }
        assert "start_box" not in args
        assert "end_box" not in args

    def test_drag_native_start_end_point_pass_through(self):
        # UI-TARS-1.0 already emits ``start_point`` / ``end_point`` —
        # pass through unchanged.
        text = (
            "Action: drag(start_point='<point>1 2</point>', "
            "end_point='<point>3 4</point>')"
        )
        r = self.p.extract_tool_calls(text)
        args = _decode_args(r.tool_calls[0])
        assert args == {
            "action": "drag",
            "start_point": [1, 2],
            "end_point": [3, 4],
        }

    def test_single_point_verb_with_end_box_collapses_to_point(self):
        # Codex r1 BLOCKING #2: a defensively-emitted ``end_box`` on a
        # single-point verb (model variance; rare but observed) MUST
        # collapse to the spec ``point`` key — never leak the
        # two-point-only ``end_point`` key on a single-point verb.
        args = _normalize_action("click", {"end_box": "(100,200)"})
        assert args == {"action": "click", "point": [100, 200]}
        assert "end_point" not in args
        assert "end_box" not in args

    def test_single_point_verb_first_write_wins(self):
        # If the model emits BOTH ``point`` and ``start_box`` on a
        # single-point verb, the first-write-wins semantics keep
        # the dict-iteration-first kwarg and drop the duplicate.
        args = _normalize_action(
            "click",
            {"point": "(50,60)", "start_box": "(70,80)"},
        )
        assert args == {"action": "click", "point": [50, 60]}

    def test_box_sentinel_renamed_per_verb(self):
        # UI-TARS-1.5 sentinel-token form: ``start_box='<|box_start|>
        # (x,y)<|box_end|>'`` — same rename per verb applies.
        text = "Action: click(start_box='<|box_start|>(233,45)<|box_end|>')"
        r = self.p.extract_tool_calls(text)
        args = _decode_args(r.tool_calls[0])
        assert args == {"action": "click", "point": [233, 45]}


# ---------------------------------------------------------------------------
# hotkey.key normalization: "ctrl c" → "ctrl+c"
# ---------------------------------------------------------------------------


class TestHotkeyNormalization:
    def setup_method(self) -> None:
        self.p = UiTarsToolParser()

    def test_space_form_normalized_to_plus(self):
        # Ana F-R1-06: UI-TARS trains on space form; spec is plus.
        args = _normalize_action("hotkey", {"key": "ctrl c"})
        assert args == {"action": "hotkey", "key": "ctrl+c"}

    def test_multi_modifier_chord(self):
        args = _normalize_action("hotkey", {"key": "ctrl shift v"})
        assert args == {"action": "hotkey", "key": "ctrl+shift+v"}

    def test_already_plus_form_preserved(self):
        # If the model already emitted plus form (rare), preserve.
        args = _normalize_action("hotkey", {"key": "ctrl+c"})
        assert args == {"action": "hotkey", "key": "ctrl+c"}

    def test_single_key_no_chord(self):
        # Single-key "hotkey" stays as-is (no spaces to collapse).
        args = _normalize_action("hotkey", {"key": "enter"})
        assert args == {"action": "hotkey", "key": "enter"}

    def test_full_pipeline_via_extract_tool_calls(self):
        text = "Action: hotkey(key='ctrl c')"
        r = self.p.extract_tool_calls(text)
        args = _decode_args(r.tool_calls[0])
        assert args == {"action": "hotkey", "key": "ctrl+c"}


# ---------------------------------------------------------------------------
# Streaming Thought:/Action: hold-back (F-R1-04)
# ---------------------------------------------------------------------------


class TestStreamingThoughtHoldback:
    """Cover the regression where the 7-char heuristic flipped the parser
    to content on ``"Thought"`` (still a prefix of ``"Thought:"``) and
    leaked the entire ``Thought:`` preamble into ``delta.content`` instead
    of routing it to ``reasoning_content`` (dogfood F-R1-04).
    """

    def setup_method(self) -> None:
        self.p = UiTarsReasoningParser()

    def _stream(self, deltas: list[str]):
        prev = ""
        out = []
        for d in deltas:
            cur = prev + d
            msg = self.p.extract_reasoning_streaming(prev, cur, d)
            if msg is not None:
                out.append(msg)
            prev = cur
        return out

    def test_thought_7chars_no_colon_yet_holds(self):
        # Token-by-token simulation of the live SSE stream: the model
        # emits ``"Thought"`` (7 chars) BEFORE the colon arrives in
        # the next delta. Old parser flipped to content here and
        # leaked. New parser holds.
        events = self._stream(["Thought"])
        # Held back: no events emitted yet.
        assert events == []

    def test_thought_held_then_released_as_reasoning(self):
        # Once the colon arrives the held bytes flow to reasoning
        # alongside the new delta (not lost, not content-leaked).
        events = self._stream(
            [
                "Thought",
                ":",
                " I need to click.\n",
                "Action: ",
                "click(point='<point>1 2</point>')",
            ]
        )
        reasoning = "".join(e.reasoning or "" for e in events)
        content = "".join(e.content or "" for e in events)
        # The full Thought: preamble lands in reasoning, NOT content.
        assert "Thought:" in reasoning
        assert "I need to click" in reasoning
        assert "Thought:" not in content
        # Action: arrives in content for the tool parser.
        assert "Action:" in content
        # No bytes dropped.
        assert reasoning + content == (
            "Thought: I need to click.\nAction: click(point='<point>1 2</point>')"
        )

    @pytest.mark.parametrize(
        "opener_prefix",
        ["Thought", "Reflection", "Action_Summa"],
    )
    def test_opener_prefix_at_seven_chars_does_not_leak(self, opener_prefix: str):
        # Any opener prefix that's longer than ``"Action:"`` (7 chars)
        # — ``"Thought"`` 7, ``"Reflection"`` 10, ``"Action_Summa"`` 12 —
        # must stay held until the disambiguating colon arrives.
        # Pre-fix the 7-char gate flipped to content at this exact
        # boundary.
        events = self._stream([opener_prefix])
        # Held — no events. No content leak.
        assert events == []

    def test_truly_non_opener_seven_char_buffer_flips_to_content(self):
        # ``"Hello, w"`` is 8 chars, not a prefix of any opener — must
        # flip to content immediately so non-preamble responses don't
        # stall forever waiting for an opener that will never come.
        events = self._stream(["Hello, w"])
        assert len(events) == 1
        assert events[0].content == "Hello, w"
        assert events[0].reasoning is None

    def test_thought_complete_in_single_delta_routes_to_reasoning(self):
        # The fast path: a single delta carries the entire
        # ``Thought: ...`` preamble plus the ``Action:`` boundary.
        # All Thought bytes go to reasoning; Action bytes go to
        # content for the tool parser.
        events = self._stream(["Thought: I want to click.\nAction: click()"])
        reasoning = "".join(e.reasoning or "" for e in events)
        content = "".join(e.content or "" for e in events)
        assert "Thought:" in reasoning
        assert "Action:" not in reasoning
        assert content.startswith("Action:")

    def test_no_token_dropped_under_partial_opener_holdback(self):
        # End-to-end invariant: every byte the model emitted ends up
        # in EITHER ``reasoning`` OR ``content`` (never lost,
        # never duplicated). Mirrors the dogfood replay where the
        # streamed text was concatenated and asserted byte-for-byte
        # against the non-streaming response.
        chunks = ["Th", "oug", "ht", ":", " ok.\n", "Ac", "tion", ":", " wait()"]
        full = "".join(chunks)
        events = self._stream(chunks)
        reasoning = "".join(e.reasoning or "" for e in events)
        content = "".join(e.content or "" for e in events)
        assert reasoning + content == full
        assert "Thought:" in reasoning
        assert "Action:" not in reasoning
        assert "Action:" in content


# ---------------------------------------------------------------------------
# Lane parity (parser-level): same emit shape regardless of caller
# ---------------------------------------------------------------------------


class TestLaneInjectionParity:
    """Dogfood F-R2-04 root-cause: pre-fix the OAI and Anthropic lanes
    received DIFFERENT prompts (only the OAI lane saw a UI-TARS
    sysprompt — and only when the test client pasted it themselves),
    so the model emitted different coords across surfaces.

    The fix is the SHARED ``maybe_inject_ui_tars_system_prompt``
    helper wired into both ``routes/chat.py`` AND
    ``routes/anthropic.py``. So long as both routes feed the helper
    the same UI-TARS-bound request, the prompts they pass to the
    engine are byte-identical AT the sysprompt level.

    Codex r1 NIT #3: the prior draft of this test ran the same
    parser instance twice on the same raw bytes — which can't
    catch a route/adapter regression, only a parser-state
    regression. Replace with a helper-layer test that proves both
    routes prepend byte-identical sysprompts; end-to-end image-
    embed parity is out of unit scope (needs a live MLX engine
    running the actual checkpoint, covered by the dogfood replay).
    """

    def test_both_lanes_get_identical_sysprompt(self):
        # Simulate the two route paths' first call: both routes
        # produce the SAME extracted-message dict shape via the
        # shared ``extract_multimodal_content`` upstream. The
        # helper produces the SAME prepended messages,
        # byte-for-byte, on both calls.
        user_msg = {"role": "user", "content": "Click the search button."}
        oai_inputs = [user_msg]
        anth_inputs = [user_msg]
        oai_out = maybe_inject_ui_tars_system_prompt(
            oai_inputs, tool_call_parser="ui_tars", tool_choice="auto"
        )
        anth_out = maybe_inject_ui_tars_system_prompt(
            anth_inputs, tool_call_parser="ui_tars", tool_choice="auto"
        )
        assert oai_out == anth_out
        assert oai_out[0]["role"] == "system"
        assert oai_out[0]["content"] == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT

    def test_inject_parity_under_user_sysprompt(self):
        # Operator extends with a custom system message: same
        # injection on both lanes — auto-injected sysprompt FIRST,
        # user system PRESERVED, no double-injection.
        msgs = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Click."},
        ]
        oai_out = maybe_inject_ui_tars_system_prompt(
            list(msgs), tool_call_parser="ui_tars", tool_choice="auto"
        )
        anth_out = maybe_inject_ui_tars_system_prompt(
            list(msgs), tool_call_parser="ui_tars", tool_choice="auto"
        )
        assert oai_out == anth_out
        # Three messages: auto-injected + user system + user.
        assert len(oai_out) == 3
        assert oai_out[0]["content"] == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT
        assert oai_out[1] == {"role": "system", "content": "Be concise."}

    def test_inject_parity_skips_on_tool_choice_none(self):
        # ``tool_choice="none"`` short-circuit fires identically on
        # both lanes — neither gets a UI-TARS sysprompt, so the
        # model emits plain prose with no Action: lines.
        msgs = [{"role": "user", "content": "What time is it?"}]
        oai_out = maybe_inject_ui_tars_system_prompt(
            list(msgs), tool_call_parser="ui_tars", tool_choice="none"
        )
        anth_out = maybe_inject_ui_tars_system_prompt(
            list(msgs), tool_call_parser="ui_tars", tool_choice="none"
        )
        assert oai_out == anth_out == msgs


# ---------------------------------------------------------------------------
# Anthropic adapter: tool_use.input carries the spec'd ``point`` shape
# ---------------------------------------------------------------------------


class TestAnthropicAdapterPointShape:
    """The Anthropic adapter ``openai_to_anthropic`` just ``json.loads``
    the OAI tool_call arguments and stuffs them into ``tool_use.input``.
    With our parser-level rename, the Anthropic ``tool_use.input``
    naturally carries ``point`` / ``start_point`` / ``end_point``
    instead of ``start_box`` / ``end_box`` — closing dogfood F-R1-02
    for the Anthropic lane.
    """

    def test_click_tool_use_input_uses_point_not_start_box(self):
        from vllm_mlx.api.anthropic_adapter import openai_to_anthropic
        from vllm_mlx.api.models import (
            AssistantMessage,
            ChatCompletionChoice,
            ChatCompletionResponse,
            FunctionCall,
            ToolCall,
        )

        tc = ToolCall(
            id="call_abc12345",
            type="function",
            function=FunctionCall(
                name="computer",
                arguments=json.dumps({"action": "click", "point": [128, 128]}),
            ),
        )
        response = ChatCompletionResponse(
            id="chatcmpl-test",
            object="chat.completion",
            created=1,
            model="ui-tars-1.5-7b-4bit",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant", content="", tool_calls=[tc]
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
        anth = openai_to_anthropic(response, model="ui-tars-1.5-7b-4bit")
        # Find the tool_use block.
        tool_use_blocks = [b for b in anth.content if b.type == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0].name == "computer"
        # Anthropic ``tool_use.input`` carries the spec key.
        assert tool_use_blocks[0].input == {
            "action": "click",
            "point": [128, 128],
        }
        assert "start_box" not in tool_use_blocks[0].input
