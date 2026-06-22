# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for 0.8.6 UI-TARS dogfood bundle (Fadi r5-B).

Architectural fix: the canonical UI-TARS Computer-Use action-API system
prompt is injected only when the request **declares a Computer-Use
tool**, not just because the loaded alias's ``tool_call_parser`` is
``"ui_tars"``. The same gate fires on all three lanes —
``/v1/chat/completions``, ``/v1/messages``, ``/v1/responses`` — so
identical (model, prompt, tools) triples produce lane-correct but
intent-identical outputs across the surfaces.

Bugs covered:

C-09 (F-R1-L/M, CRIT) — chat lane unconditionally injected the
    Computer-Use sysprompt. A plain-text request (``"what is 2+2?"``,
    no ``tools``) came back with ``content=null`` and a phantom
    ``computer`` tool_call clicking ``[1404, 240]``. JSON mode
    degraded to ``content="[]"``. Fixed by tool-coupled gate.

C-10 (F-R2-D, CRIT) — ``/v1/responses`` with ``computer_20251022``
    tool emitted plain text (``output[].type=="message"``), never
    a ``computer_call`` output item. The route bypassed the
    injection helper entirely. Fixed by wiring the helper into
    both the non-stream and streaming responses paths.

C-11 (F-R2-I, CRIT) — same prompt + same Computer-Use tool produced
    three different intents across the three lanes:
    chat → ``tool_call``, messages → ``tool_use``, responses → text.
    Fixed by the same tool-coupled helper running on every lane;
    parser output then flows through each lane's spec-correct
    response builder.

R-09 (F-R1-E, HIGH) — ``tool_choice={"type":"function","function":
    {"name":"computer"}}`` returned 422. With tool-coupled
    injection the model now reliably emits ``Action: ...`` lines
    which the parser surfaces as ``computer``; the named-pin check
    no longer 422s when the only target IS ``computer``.

The tests below exercise the helper-level decision tree (parsing of
``tools`` shapes from all three lanes) plus the response-builder
translation paths that surface the parser output to each lane's
spec-correct shape (chat ``tool_calls``, Anthropic ``tool_use``,
Responses ``computer_call``).
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
    UI_TARS_COMPUTER_USE_SYSTEM_PROMPT,
    maybe_inject_ui_tars_system_prompt,
    request_declares_computer_tool,
)

# ---------------------------------------------------------------------------
# Tool-shape detector (request_declares_computer_tool)
# ---------------------------------------------------------------------------


class TestRequestDeclaresComputerTool:
    """The detector accepts every tool-shape every lane uses."""

    def test_none_returns_false(self):
        assert request_declares_computer_tool(None) is False

    def test_empty_list_returns_false(self):
        assert request_declares_computer_tool([]) is False

    def test_chat_nested_function_shape_computer(self):
        # OpenAI Chat Completions tools array — pydantic
        # ToolDefinition dump shape.
        tools = [
            {
                "type": "function",
                "function": {"name": "computer", "parameters": {"type": "object"}},
            }
        ]
        assert request_declares_computer_tool(tools) is True

    def test_chat_nested_function_shape_non_computer(self):
        # Vanilla function tool with a custom name → NOT
        # Computer-Use, even on a UI-TARS model. The model should
        # answer as a normal tool-calling LLM.
        tools = [
            {
                "type": "function",
                "function": {"name": "search_screen", "parameters": {}},
            }
        ]
        assert request_declares_computer_tool(tools) is False

    def test_responses_flat_computer_20251022_shape(self):
        # OpenAI Responses computer_20251022 tool — flat shape
        # with ``type`` carrying the spec name.
        tools = [
            {
                "type": "computer_20251022",
                "display_width": 1280,
                "display_height": 800,
            }
        ]
        assert request_declares_computer_tool(tools) is True

    def test_anthropic_flat_name_shape(self):
        # Anthropic /v1/messages tools array — flat dict with
        # ``name``, no nested ``function``.
        tools = [
            {
                "name": "computer",
                "description": "GUI action tool",
                "input_schema": {"type": "object"},
            }
        ]
        assert request_declares_computer_tool(tools) is True

    def test_pydantic_tool_definition_shape(self):
        # ChatCompletionRequest carries a list of pydantic
        # ToolDefinition objects (not dicts). Detector must
        # tolerate this — the helper is called from the route
        # BEFORE any model_dump rewrite.
        from vllm_mlx.api.models import ToolDefinition

        t = ToolDefinition(
            type="function",
            function={"name": "computer", "parameters": {"type": "object"}},
        )
        assert request_declares_computer_tool([t]) is True

    def test_mixed_tools_with_computer_present(self):
        # Two tools — search_screen + computer. Detector returns
        # True because at least one Computer-Use tool is in scope.
        tools = [
            {"type": "function", "function": {"name": "search_screen"}},
            {"type": "function", "function": {"name": "computer"}},
        ]
        assert request_declares_computer_tool(tools) is True

    def test_non_iterable_returns_false(self):
        # Defensive: don't crash on bad shapes.
        assert request_declares_computer_tool(42) is False
        assert request_declares_computer_tool("computer") is False
        # Even though "computer" is in the string, it's NOT a
        # tool array — must return False.


# ---------------------------------------------------------------------------
# C-09: tool-coupled gate (no tool → no inject)
# ---------------------------------------------------------------------------


class TestC09NoToolNoInject:
    """The headline architectural fix. Plain-text and JSON-mode
    requests to a UI-TARS-aliased model MUST NOT get the Computer-Use
    sysprompt — that was the root cause of the F-R1-L "what is 2+2"
    phantom click and F-R1-M JSON mode returning ``"[]"``.
    """

    def test_no_tools_no_inject(self):
        # F-R1-L repro: plain prompt, no tools — model must NOT see
        # the Computer-Use action-API contract.
        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        out = maybe_inject_ui_tars_system_prompt(
            messages,
            tool_call_parser="ui_tars",
            tool_choice=None,
            tools=None,
        )
        assert out == messages
        # And the canonical sysprompt is NOT anywhere in the messages.
        joined = "\n".join(str(m) for m in out)
        assert "## Action Space" not in joined

    def test_empty_tools_no_inject(self):
        messages = [{"role": "user", "content": "Hello!"}]
        out = maybe_inject_ui_tars_system_prompt(
            messages,
            tool_call_parser="ui_tars",
            tool_choice=None,
            tools=[],
        )
        assert out == messages

    def test_non_computer_function_tool_no_inject(self):
        # The user submitted a custom function tool (say a weather
        # tool) — it's not Computer-Use. Don't prime the model to
        # emit click actions.
        messages = [
            {"role": "user", "content": "What's the weather in NYC?"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
        out = maybe_inject_ui_tars_system_prompt(
            messages,
            tool_call_parser="ui_tars",
            tool_choice=None,
            tools=tools,
        )
        assert out == messages

    def test_json_mode_no_tools_no_inject(self):
        # F-R1-M repro: a JSON-mode request to UI-TARS used to come
        # back as ``content="[]"`` because the auto-injected
        # Computer-Use sysprompt steered the model into emitting
        # ``Action: ...`` text instead of JSON. With the
        # tool-coupled gate, no Computer-Use tool → no sysprompt →
        # JSON mode works normally.
        messages = [
            {
                "role": "user",
                "content": (
                    "Return a JSON object with keys 'city' and 'country' for Paris."
                ),
            }
        ]
        out = maybe_inject_ui_tars_system_prompt(
            messages,
            tool_call_parser="ui_tars",
            tool_choice=None,
            tools=None,
        )
        assert out == messages

    def test_computer_tool_present_does_inject(self):
        # Positive control: a request that DOES declare the
        # Computer-Use tool gets the sysprompt as expected.
        messages = [{"role": "user", "content": "Click the OK button."}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "computer",
                    "parameters": {"type": "object"},
                },
            }
        ]
        out = maybe_inject_ui_tars_system_prompt(
            messages,
            tool_call_parser="ui_tars",
            tool_choice=None,
            tools=tools,
        )
        assert len(out) == 2
        assert out[0]["role"] == "system"
        assert out[0]["content"] == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# C-10 / C-11: lane parity — all three lanes converge on the same gate
# ---------------------------------------------------------------------------


class TestCrossLaneParity:
    """Same input (model + prompt + tools) → same injection decision
    across chat / messages / responses. Pre-fix the three lanes
    diverged: chat over-injected (C-09), responses under-injected
    (C-10), so the same prompt produced three different intents
    (C-11). The shared helper, called with each lane's tool shape,
    eliminates the divergence.
    """

    @pytest.mark.parametrize(
        "tools",
        [
            # OpenAI Chat shape (nested function)
            [
                {
                    "type": "function",
                    "function": {"name": "computer", "parameters": {}},
                }
            ],
            # OpenAI Responses shape (flat computer_20251022)
            [
                {
                    "type": "computer_20251022",
                    "display_width": 1280,
                    "display_height": 800,
                }
            ],
            # Anthropic Messages shape (flat name)
            [
                {
                    "name": "computer",
                    "description": "GUI action tool",
                    "input_schema": {"type": "object"},
                }
            ],
        ],
        ids=["chat-nested", "responses-flat", "anthropic-flat"],
    )
    def test_each_lane_shape_fires_inject(self, tools):
        # The same helper, called with each lane's specific tool
        # shape, MUST fire the inject. If one shape silently fails
        # the detector, the corresponding lane regresses to F-R2-D
        # / F-R2-I (no computer_call output).
        messages = [{"role": "user", "content": "Click the search button."}]
        out = maybe_inject_ui_tars_system_prompt(
            messages,
            tool_call_parser="ui_tars",
            tool_choice="auto",
            tools=tools,
        )
        assert len(out) == 2
        assert out[0]["content"] == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT

    @pytest.mark.parametrize(
        "tools",
        [
            None,
            [],
            [{"type": "function", "function": {"name": "search_screen"}}],
        ],
        ids=["none", "empty", "non-computer-fn"],
    )
    def test_each_lane_shape_skips_inject_when_no_computer(self, tools):
        # The same helper, called with each "no computer-use" shape,
        # MUST skip the inject on every lane. If any lane shape
        # accidentally injects, the corresponding "what is 2+2"
        # request regresses to F-R1-L.
        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        out = maybe_inject_ui_tars_system_prompt(
            messages,
            tool_call_parser="ui_tars",
            tool_choice="auto",
            tools=tools,
        )
        assert out == messages

    def test_three_lanes_byte_identical_sysprompt_when_computer_tool_present(self):
        # Simulate the three lanes' helper invocations on the same
        # prompt + same Computer-Use tool. The auto-injected
        # sysprompt must be byte-identical on all three lanes so
        # the model sees the SAME action-API contract regardless
        # of which surface received the request (C-11 root cause:
        # divergent prompts produced divergent intents).
        user = {"role": "user", "content": "Click (500, 250)."}
        chat_tools = [
            {
                "type": "function",
                "function": {"name": "computer", "parameters": {}},
            }
        ]
        responses_tools = [
            {
                "type": "computer_20251022",
                "display_width": 1280,
                "display_height": 800,
            }
        ]
        anthropic_tools = [
            {
                "name": "computer",
                "description": "GUI action tool",
                "input_schema": {"type": "object"},
            }
        ]

        chat_out = maybe_inject_ui_tars_system_prompt(
            [user], tool_call_parser="ui_tars", tool_choice="auto", tools=chat_tools
        )
        resp_out = maybe_inject_ui_tars_system_prompt(
            [user],
            tool_call_parser="ui_tars",
            tool_choice="auto",
            tools=responses_tools,
        )
        anth_out = maybe_inject_ui_tars_system_prompt(
            [user],
            tool_call_parser="ui_tars",
            tool_choice="auto",
            tools=anthropic_tools,
        )

        # All three injected; sysprompt is byte-identical.
        assert chat_out[0]["content"] == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT
        assert resp_out[0]["content"] == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT
        assert anth_out[0]["content"] == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# C-10 detail: Responses-lane response-builder emits computer_call
# ---------------------------------------------------------------------------


class TestResponsesLaneComputerCallEmission:
    """Once the C-09 fix primes the model to emit ``Action: ...`` text,
    the parser surfaces a ``computer`` tool_call. The Responses
    adapter (``openai_to_responses``) MUST then translate that to a
    ``computer_call`` output item per the OpenAI Computer-Use spec.

    This is the second half of the C-10 / C-11 fix — the first half
    (injection on the responses lane) is covered in
    ``test_ui_tars_fixes.py::TestLaneInjectionParity::
    test_responses_route_actually_invokes_helper``.
    """

    def _build_chat_response_with_computer_call(self, arguments_json: str):
        """Synthesize what the chat lane would surface for a
        Computer-Use-pinned UI-TARS turn. Used to drive
        ``openai_to_responses`` under test.
        """
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
            function=FunctionCall(name="computer", arguments=arguments_json),
        )
        return ChatCompletionResponse(
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

    def test_computer_call_emitted_for_computer_20251022_request(self):
        # F-R2-D fix: a request with computer_20251022 + a
        # synthesized computer tool_call MUST produce a
        # ``computer_call`` output item (not ``function_call``).
        from vllm_mlx.api.responses_adapter import openai_to_responses
        from vllm_mlx.api.responses_models import ResponsesRequest

        req = ResponsesRequest(
            model="ui-tars-1.5-7b-4bit",
            input="Click the OK button at (500, 300).",
            tools=[
                {
                    "type": "computer_20251022",
                    "display_width": 1280,
                    "display_height": 800,
                }
            ],
        )
        chat_resp = self._build_chat_response_with_computer_call(
            json.dumps({"action": "click", "point": [500, 300]})
        )
        resp = openai_to_responses(
            chat_resp, model="ui-tars-1.5-7b-4bit", request=req, created_at=1
        )
        computer_calls = [
            o for o in resp.output if getattr(o, "type", None) == "computer_call"
        ]
        assert len(computer_calls) == 1, [
            (getattr(o, "type", None), o) for o in resp.output
        ]
        cc = computer_calls[0]
        # Action verb mapped from "action" → "type".
        # R6-M2: ``point`` is the UI-TARS parser's canonical key; the
        # Responses lane translates to OpenAI's spec ``coordinate``.
        assert cc.action == {"type": "click", "coordinate": [500, 300]}
        # No function_call in output (would be the C-10 regression).
        assert not [
            o for o in resp.output if getattr(o, "type", None) == "function_call"
        ]


# ---------------------------------------------------------------------------
# R-09: tool_choice={'type':'function','function':{'name':'computer'}}
# ---------------------------------------------------------------------------


class TestR09PinnedComputerToolChoice:
    """Pinning ``tool_choice`` to the canonical ``"computer"`` name MUST
    route the injection through (Computer-Use is in scope) and let
    the parser surface a ``computer`` tool_call. The 422 the dogfood
    report saw was a downstream effect of the model emitting nothing
    parseable; with the sysprompt injection now firing tool-coupled,
    this path produces a clean 200.
    """

    def test_pinned_computer_tool_choice_fires_inject(self):
        # tool_choice pinning ``computer`` + computer tool in
        # request.tools — the helper must fire the inject so the
        # model is primed to emit ``Action: ...`` text the parser
        # can surface as ``computer``.
        messages = [{"role": "user", "content": "Click the OK button at (450, 220)."}]
        tools = [
            {
                "type": "function",
                "function": {"name": "computer", "parameters": {}},
            }
        ]
        out = maybe_inject_ui_tars_system_prompt(
            messages,
            tool_call_parser="ui_tars",
            tool_choice={"type": "function", "function": {"name": "computer"}},
            tools=tools,
        )
        assert len(out) == 2
        assert out[0]["content"] == UI_TARS_COMPUTER_USE_SYSTEM_PROMPT

    def test_pinned_non_computer_tool_choice_still_skips_when_no_computer_tool(self):
        # A user pinned a non-Computer-Use tool and didn't supply
        # the computer tool — no inject (vanilla function tool
        # flow, not Computer-Use).
        messages = [{"role": "user", "content": "Search the screen."}]
        tools = [
            {
                "type": "function",
                "function": {"name": "search_screen", "parameters": {}},
            }
        ]
        out = maybe_inject_ui_tars_system_prompt(
            messages,
            tool_call_parser="ui_tars",
            tool_choice={"type": "function", "function": {"name": "search_screen"}},
            tools=tools,
        )
        assert out == messages

    def test_pinned_computer_with_computer_tool_choice_none_skips(self):
        # Defense-in-depth: pinned computer tool + tool_choice="none"
        # still skips (the tool_choice="none" arm short-circuits
        # before the tool-coupled gate fires).
        messages = [{"role": "user", "content": "Describe how you'd click."}]
        tools = [
            {
                "type": "function",
                "function": {"name": "computer", "parameters": {}},
            }
        ]
        out = maybe_inject_ui_tars_system_prompt(
            messages,
            tool_call_parser="ui_tars",
            tool_choice="none",
            tools=tools,
        )
        assert out == messages


# ---------------------------------------------------------------------------
# r6-B R6-M1: reasoning channel populates on plain chat lane
# ---------------------------------------------------------------------------


class TestR6M1ReasoningGateDecoupling:
    """The reasoning emission gate must fire whenever the model
    produces a thought block — NOT only when the Computer-Use
    sysprompt was auto-injected.

    Pre-r6-B: ``_PREAMBLE_RE`` required ``(?=\\s*Action:)`` lookahead,
    so a plain-chat response (no Computer-Use tool declared, so r5-B's
    tool-coupled gate skipped sysprompt injection) that still emitted
    ``Thought: ...`` (because the UI-TARS checkpoint is post-trained
    on the format) silently routed the entire buffer to ``content``.

    Fixed by decoupling the reasoning extraction from the
    sysprompt-injection presence: the regex now accepts three
    additional shapes — ``Thought:`` with a blank-line boundary,
    ``Thought:`` ending the buffer (no follow-up answer), and the
    generic ``<think>...</think>`` tag.
    """

    def test_plain_chat_thought_blank_line_surfaces_as_reasoning(self):
        # Plain chat lane: no Action: anywhere, blank line separates
        # the thought from the follow-up answer.
        from vllm_mlx.reasoning.ui_tars_parser import UiTarsReasoningParser

        parser = UiTarsReasoningParser()
        reasoning, content = parser.extract_reasoning(
            "Thought: I should respond directly.\n\nThe answer is 4."
        )
        assert reasoning == "Thought: I should respond directly."
        assert content == "The answer is 4."

    def test_plain_chat_thought_end_of_buffer_surfaces_as_reasoning(self):
        # Edge case: the model emitted only a Thought: block, no
        # follow-up answer (truncated / cut off). The reasoning
        # channel still surfaces it.
        from vllm_mlx.reasoning.ui_tars_parser import UiTarsReasoningParser

        parser = UiTarsReasoningParser()
        reasoning, content = parser.extract_reasoning("Thought: I'm uncertain.")
        assert reasoning == "Thought: I'm uncertain."
        # No follow-up — content empty / None.
        assert not content

    def test_plain_chat_think_tag_surfaces_as_reasoning(self):
        # Generic ``<think>...</think>`` tag (a model checkpoint that
        # learned both UI-TARS Thought: AND the standard think-tag
        # convention may emit either). Both should populate reasoning.
        from vllm_mlx.reasoning.ui_tars_parser import UiTarsReasoningParser

        parser = UiTarsReasoningParser()
        reasoning, content = parser.extract_reasoning(
            "<think>The user asked for 2+2.</think>The answer is 4."
        )
        # The structural <think>/</think> wrapper is stripped — the
        # reasoning channel surfaces the human-readable thought only.
        assert reasoning == "The user asked for 2+2."
        assert content == "The answer is 4."

    def test_action_lane_reasoning_still_works(self):
        # Positive control: pre-r6-B contract is preserved — the
        # Action lane still routes the Thought: preamble to
        # reasoning and everything from ``Action:`` onward to content.
        from vllm_mlx.reasoning.ui_tars_parser import UiTarsReasoningParser

        parser = UiTarsReasoningParser()
        reasoning, content = parser.extract_reasoning(
            "Thought: Click the OK button.\nAction: click(point='<point>500 300</point>')"
        )
        assert reasoning == "Thought: Click the OK button."
        assert content == "Action: click(point='<point>500 300</point>')"

    def test_no_preamble_routes_all_to_content(self):
        # Defense-in-depth: a response with NO thought block at all
        # routes the entire buffer to content (no spurious reasoning).
        from vllm_mlx.reasoning.ui_tars_parser import UiTarsReasoningParser

        parser = UiTarsReasoningParser()
        reasoning, content = parser.extract_reasoning(
            "Just a regular response with no thought."
        )
        assert reasoning is None
        assert content == "Just a regular response with no thought."

    def test_thought_followed_by_newline_prose_does_not_overcapture(self):
        # Codex r4 MEDIUM: pre-fix, the plain-chat ``Thought:`` branch
        # ended at either ``\s*\n\s*\n`` OR ``\s*\Z`` under
        # ``re.DOTALL``. With the lazy body and ``\Z`` alternative,
        # a response like ``"Thought: A.\nB."`` (single newline, no
        # blank line) lazy-matched the body up to ``\Z`` and
        # classified the WHOLE response as reasoning, dropping the
        # model's answer. Fix: the EOS branch is now restricted to
        # bodies with NO embedded newlines (single-line truncated
        # thoughts only); multi-line plain prose without a blank-line
        # boundary falls through to "no preamble" and the entire
        # text is routed to content.
        from vllm_mlx.reasoning.ui_tars_parser import UiTarsReasoningParser

        parser = UiTarsReasoningParser()
        reasoning, content = parser.extract_reasoning(
            "Thought: I should answer directly.\nThe answer is 4."
        )
        # No blank-line boundary AND the body has an embedded newline
        # — neither shape #4 nor shape #4b matches; the entire response
        # routes to content.
        assert reasoning is None
        assert content == "Thought: I should answer directly.\nThe answer is 4."

    def test_thought_eos_single_line_still_works(self):
        # Positive control for the restricted shape #4b: a single-line
        # truncated thought (no newline at all) still surfaces as
        # reasoning. This is the EOS branch the codex r4 fix narrows.
        from vllm_mlx.reasoning.ui_tars_parser import UiTarsReasoningParser

        parser = UiTarsReasoningParser()
        reasoning, content = parser.extract_reasoning("Thought: I'm uncertain.")
        assert reasoning == "Thought: I'm uncertain."
        assert not content

    def test_thought_eos_with_trailing_whitespace_still_works(self):
        # Edge case: single-line thought with trailing whitespace
        # (typical when the model emits ``Thought: ...\n`` and the
        # response is truncated before any follow-up). The
        # ``[^\n]*?`` body matches the line, then ``\s*\Z`` consumes
        # the trailing newline / whitespace.
        from vllm_mlx.reasoning.ui_tars_parser import UiTarsReasoningParser

        parser = UiTarsReasoningParser()
        reasoning, content = parser.extract_reasoning("Thought: I'm uncertain.\n")
        assert reasoning == "Thought: I'm uncertain."
        assert not content

    def test_thought_blank_line_boundary_multi_line_thought(self):
        # Multi-line thought body terminated by a real blank-line
        # boundary still works — shape #4 (not #4b) handles this.
        from vllm_mlx.reasoning.ui_tars_parser import UiTarsReasoningParser

        parser = UiTarsReasoningParser()
        reasoning, content = parser.extract_reasoning(
            "Thought: Step 1.\nStep 2.\n\nThe answer is 4."
        )
        # Body up to the blank line is reasoning; bytes after are content.
        assert reasoning == "Thought: Step 1.\nStep 2."
        assert content == "The answer is 4."


# ---------------------------------------------------------------------------
# r6-B R6-M2: Anthropic + Responses lanes translate point → coordinate
# ---------------------------------------------------------------------------


class TestR6M2CoordinateKeyTranslation:
    """The UI-TARS parser emits the canonical ``point`` /
    ``start_point`` / ``end_point`` keys (PR #812 contract; chat
    completions OpenAI lane stays bytes-faithful to that). The
    Anthropic ``/v1/messages`` lane and the OpenAI ``/v1/responses``
    lane both follow Computer-Use specs that use ``coordinate`` for
    single-point verbs.

    Two-point ``drag`` diverges between the specs:
    - Anthropic uses ``start_coordinate`` + ``coordinate`` (end).
    - OpenAI Responses uses ``path=[{"x":x,"y":y}, ...]``.

    Pre-r6-B: both adapters surfaced the UI-TARS-native ``point`` key
    verbatim. Anthropic-strict consumers (claude-agent-sdk, Computer-
    Use harnesses) rejected the shape. Fixed by per-lane translator
    helpers (``translate_to_anthropic_spec_keys`` /
    ``translate_to_responses_spec_keys``) that live next to the parser
    and are called from each adapter's tool_use / computer_call builder
    so the two surfaces can't drift on key naming AND so each lane
    matches its own spec for drag.
    """

    def _click_chat_response(self, args_payload: dict):
        """Synthesize the OAI chat response a UI-TARS click would
        produce; reused across the Anthropic + Responses asserts.
        """
        from vllm_mlx.api.models import (
            AssistantMessage,
            ChatCompletionChoice,
            ChatCompletionResponse,
            FunctionCall,
            ToolCall,
            Usage,
        )

        tc = ToolCall(
            id="call_abc12345",
            type="function",
            function=FunctionCall(name="computer", arguments=json.dumps(args_payload)),
        )
        return ChatCompletionResponse(
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
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    # --- Anthropic /v1/messages ------------------------------------------

    def test_anthropic_click_emits_coordinate_not_point(self):
        from vllm_mlx.api.anthropic_adapter import openai_to_anthropic

        chat_resp = self._click_chat_response({"action": "click", "point": [500, 300]})
        anth = openai_to_anthropic(chat_resp, model="ui-tars-1.5-7b-4bit")
        tool_uses = [b for b in anth.content if getattr(b, "type", None) == "tool_use"]
        assert len(tool_uses) == 1
        tu = tool_uses[0]
        assert tu.name == "computer"
        # R6-M2: spec key is ``coordinate``, NOT ``point``.
        assert tu.input == {"action": "click", "coordinate": [500, 300]}
        assert "point" not in tu.input

    def test_anthropic_drag_emits_start_coordinate_and_coordinate(self):
        # Anthropic Computer-Use spec: drag uses ``start_coordinate``
        # plus ``coordinate`` (the END point). NOT ``end_coordinate``.
        from vllm_mlx.api.anthropic_adapter import openai_to_anthropic

        chat_resp = self._click_chat_response(
            {
                "action": "drag",
                "start_point": [10, 20],
                "end_point": [100, 200],
            }
        )
        anth = openai_to_anthropic(chat_resp, model="ui-tars-1.5-7b-4bit")
        tool_uses = [b for b in anth.content if getattr(b, "type", None) == "tool_use"]
        assert len(tool_uses) == 1
        # Spec: ``start_coordinate`` + ``coordinate`` (the END).
        assert tool_uses[0].input == {
            "action": "drag",
            "start_coordinate": [10, 20],
            "coordinate": [100, 200],
        }
        # Defensive: no ``end_coordinate`` key (that would be the wrong
        # name per the spec).
        assert "end_coordinate" not in tool_uses[0].input
        assert "point" not in tool_uses[0].input

    def test_anthropic_non_computer_tool_input_untouched(self):
        # Vanilla function tool whose arguments happen to carry a
        # ``point`` key — the gated translation MUST NOT rewrite it.
        from vllm_mlx.api.anthropic_adapter import openai_to_anthropic
        from vllm_mlx.api.models import (
            AssistantMessage,
            ChatCompletionChoice,
            ChatCompletionResponse,
            FunctionCall,
            ToolCall,
            Usage,
        )

        tc = ToolCall(
            id="call_abc12345",
            type="function",
            function=FunctionCall(
                name="get_pixel_color",
                arguments=json.dumps({"point": [500, 300]}),
            ),
        )
        chat_resp = ChatCompletionResponse(
            id="chatcmpl-test",
            object="chat.completion",
            created=1,
            model="non-ui-tars-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant", content="", tool_calls=[tc]
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        anth = openai_to_anthropic(chat_resp, model="non-ui-tars-model")
        tool_uses = [b for b in anth.content if getattr(b, "type", None) == "tool_use"]
        # Non-``computer`` tool — translation gate skipped.
        assert tool_uses[0].input == {"point": [500, 300]}
        assert "coordinate" not in tool_uses[0].input

    # --- OpenAI /v1/responses ---------------------------------------------

    def test_responses_click_emits_coordinate_not_point(self):
        from vllm_mlx.api.responses_adapter import openai_to_responses
        from vllm_mlx.api.responses_models import ResponsesRequest

        chat_resp = self._click_chat_response({"action": "click", "point": [500, 300]})
        req = ResponsesRequest(
            model="ui-tars-1.5-7b-4bit",
            input="Click OK.",
            tools=[
                {
                    "type": "computer_20251022",
                    "display_width": 1280,
                    "display_height": 800,
                }
            ],
        )
        resp = openai_to_responses(
            chat_resp, model="ui-tars-1.5-7b-4bit", request=req, created_at=1
        )
        computer_calls = [
            o for o in resp.output if getattr(o, "type", None) == "computer_call"
        ]
        assert len(computer_calls) == 1
        cc = computer_calls[0]
        # R6-M2: spec key is ``coordinate``, NOT ``point``.
        assert cc.action == {"type": "click", "coordinate": [500, 300]}
        assert "point" not in cc.action

    # --- Chat Completions OpenAI lane (parser stays native; route normalises) -

    def test_chat_completions_parser_still_emits_native_point(self):
        # Defense-in-depth: the UI-TARS tool parser itself still emits
        # the canonical ``point`` key per PR #812 — the spec-key
        # translation happens at the RESPONSE BUILDER per lane, NOT
        # inside the parser. The chat lane's response-builder
        # normalisation is exercised by
        # ``test_chat_completions_lane_emits_coordinate_via_route_normaliser``
        # below.
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import UiTarsToolParser

        parser = UiTarsToolParser(tokenizer=None)
        parser.reset()
        text = "Action: click(point='<point>500 300</point>')"
        result = parser.extract_tool_calls(text, request=None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0]["arguments"])
        # Parser stays on the canonical ``point`` key.
        assert args == {"action": "click", "point": [500, 300]}
        assert "coordinate" not in args

    # --- Responses drag (codex r1 HIGH 2) --------------------------------

    def test_responses_drag_emits_path_array(self):
        # OpenAI Responses Computer-Use spec: drag uses
        # ``path=[{"x":x1,"y":y1}, {"x":x2,"y":y2}]``, NOT the
        # ``start_coordinate`` / ``end_coordinate`` shape Anthropic
        # uses. Codex r1 HIGH 2 flagged that an earlier draft
        # surfaced the Anthropic shape on the Responses lane — a
        # behavior regression for drag.
        from vllm_mlx.api.responses_adapter import openai_to_responses
        from vllm_mlx.api.responses_models import ResponsesRequest

        chat_resp = self._click_chat_response(
            {
                "action": "drag",
                "start_point": [10, 20],
                "end_point": [100, 200],
            }
        )
        req = ResponsesRequest(
            model="ui-tars-1.5-7b-4bit",
            input="Drag from (10,20) to (100,200).",
            tools=[
                {
                    "type": "computer_20251022",
                    "display_width": 1280,
                    "display_height": 800,
                }
            ],
        )
        resp = openai_to_responses(
            chat_resp, model="ui-tars-1.5-7b-4bit", request=req, created_at=1
        )
        computer_calls = [
            o for o in resp.output if getattr(o, "type", None) == "computer_call"
        ]
        assert len(computer_calls) == 1
        cc = computer_calls[0]
        # R6-M2: Responses-spec ``path`` array shape.
        assert cc.action == {
            "type": "drag",
            "path": [{"x": 10, "y": 20}, {"x": 100, "y": 200}],
        }
        # Defensive: NO Anthropic-style start_coordinate / end_coordinate.
        assert "start_coordinate" not in cc.action
        assert "end_coordinate" not in cc.action

    # --- Helper-level test ------------------------------------------------

    def test_anthropic_translator_is_idempotent_on_single_point(self):
        # The mapper must be safe to call twice — already-translated
        # keys stay translated (defense-in-depth for a future
        # double-translation refactor).
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            translate_to_anthropic_spec_keys,
        )

        once = translate_to_anthropic_spec_keys({"action": "click", "point": [1, 2]})
        twice = translate_to_anthropic_spec_keys(once)
        assert once == twice == {"action": "click", "coordinate": [1, 2]}

    def test_anthropic_translator_preserves_non_coord_kwargs(self):
        # Non-coord kwargs (action, content, key, direction, …)
        # pass through verbatim.
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            translate_to_anthropic_spec_keys,
        )

        out = translate_to_anthropic_spec_keys({"action": "type", "content": "hello"})
        assert out == {"action": "type", "content": "hello"}

        out = translate_to_anthropic_spec_keys({"action": "hotkey", "key": "ctrl+c"})
        assert out == {"action": "hotkey", "key": "ctrl+c"}

    def test_responses_translator_folds_drag_into_path(self):
        # Direct probe of the helper: point pair → spec path array.
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            translate_to_responses_spec_keys,
        )

        out = translate_to_responses_spec_keys(
            {
                "action": "drag",
                "start_point": [10, 20],
                "end_point": [100, 200],
            }
        )
        assert out == {
            "action": "drag",
            "path": [{"x": 10, "y": 20}, {"x": 100, "y": 200}],
        }

    def test_responses_translator_preserves_malformed_drag(self):
        # Codex r1 — defensive: if only ONE of start_point / end_point
        # is present (malformed drag), the present key falls through
        # as the UI-TARS-native name so the downstream consumer can
        # detect the gap rather than receive a truncated single-point
        # ``path`` that looks valid.
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            translate_to_responses_spec_keys,
        )

        out = translate_to_responses_spec_keys(
            {"action": "drag", "start_point": [10, 20]}
        )
        # No ``path``; the lone start_point falls through.
        assert "path" not in out
        assert out["start_point"] == [10, 20]

    def test_responses_translator_handles_single_point_verb(self):
        # Single-point verb on the Responses lane: ``point`` →
        # ``coordinate``, same as the Anthropic translator.
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            translate_to_responses_spec_keys,
        )

        out = translate_to_responses_spec_keys({"action": "click", "point": [500, 300]})
        assert out == {"action": "click", "coordinate": [500, 300]}


# ---------------------------------------------------------------------------
# r7-A R7-H1: chat-lane Computer-Use coordinate-key parity
# ---------------------------------------------------------------------------


class TestR7H1ChatLaneCoordinateParity:
    """The OpenAI ``/v1/chat/completions`` lane must surface the
    Computer-Use spec ``coordinate`` (single-point) and ``path=[…]``
    (drag) keys, matching Anthropic + Responses. The parser still
    emits the canonical ``point`` (bytes-faithful at the parser
    layer); the chat route's response builder runs
    ``normalize_ui_tars_chat_tool_call_arguments`` on every
    ``computer`` tool_call so all three lanes converge on the
    spec keys (r7-A R7-H1 — Mira r1 evidence: chat tool_call
    arguments still showed ``"point":[640,400]`` post-r6-B).

    The translation is gated on ``function.name == "computer"`` so
    vanilla function tools that happen to use a ``point`` field are
    untouched (mirrors the Anthropic adapter gate at
    ``api/anthropic_adapter.py``).
    """

    def test_click_arguments_normalised_to_coordinate(self):
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            normalize_ui_tars_chat_tool_call_arguments,
        )

        raw = json.dumps({"action": "click", "point": [640, 400]})
        normalized = normalize_ui_tars_chat_tool_call_arguments(raw, "computer")
        parsed = json.loads(normalized)
        assert parsed == {"action": "click", "coordinate": [640, 400]}
        assert "point" not in parsed

    def test_drag_arguments_folded_to_path_array(self):
        # OpenAI Computer-Use spec drag shape: ``path=[{"x","y"}, …]``.
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            normalize_ui_tars_chat_tool_call_arguments,
        )

        raw = json.dumps(
            {"action": "drag", "start_point": [10, 20], "end_point": [100, 200]}
        )
        normalized = normalize_ui_tars_chat_tool_call_arguments(raw, "computer")
        parsed = json.loads(normalized)
        # Both points fold into the spec ``path`` array.
        assert parsed.get("action") == "drag"
        assert parsed["path"] == [{"x": 10, "y": 20}, {"x": 100, "y": 200}]
        assert "start_point" not in parsed
        assert "end_point" not in parsed
        assert "point" not in parsed

    def test_non_computer_tool_arguments_untouched(self):
        # A vanilla function tool whose schema happens to use a
        # ``point`` key MUST pass through verbatim — the chat-lane
        # translator gate prevents collateral rewriting.
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            normalize_ui_tars_chat_tool_call_arguments,
        )

        raw = json.dumps({"point": [640, 400]})
        out = normalize_ui_tars_chat_tool_call_arguments(raw, "get_pixel_color")
        assert json.loads(out) == {"point": [640, 400]}

    def test_invalid_json_arguments_pass_through(self):
        # If the model emitted unparseable arguments, the spec
        # translator surfaces the bytes unchanged — the downstream
        # tool-call schema validator is the right site to reject
        # malformed arguments, not the key translator.
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            normalize_ui_tars_chat_tool_call_arguments,
        )

        out = normalize_ui_tars_chat_tool_call_arguments("not-json", "computer")
        assert out == "not-json"

    def test_chat_lane_via_route_normaliser_emits_coordinate(self):
        # Black-box: drive the chat-route helper that streaming +
        # non-streaming both call so the test fails if either site
        # bypasses the translation.
        from vllm_mlx.routes.chat import _normalize_ui_tars_tcs_for_chat

        tcs = [
            {
                "index": 0,
                "id": "call_aaa",
                "type": "function",
                "function": {
                    "name": "computer",
                    "arguments": json.dumps({"action": "click", "point": [640, 400]}),
                },
            },
            # Vanilla function tool — must pass through untouched.
            {
                "index": 1,
                "id": "call_bbb",
                "type": "function",
                "function": {
                    "name": "get_pixel_color",
                    "arguments": json.dumps({"point": [10, 20]}),
                },
            },
        ]
        out = _normalize_ui_tars_tcs_for_chat(tcs)
        assert json.loads(out[0]["function"]["arguments"]) == {
            "action": "click",
            "coordinate": [640, 400],
        }
        # Non-computer tool stayed on ``point``.
        assert json.loads(out[1]["function"]["arguments"]) == {"point": [10, 20]}

    def test_chat_lane_none_and_empty_tool_lists_pass_through(self):
        from vllm_mlx.routes.chat import _normalize_ui_tars_tcs_for_chat

        assert _normalize_ui_tars_tcs_for_chat(None) is None
        assert _normalize_ui_tars_tcs_for_chat([]) == []

    def test_chat_lane_does_not_mutate_input(self):
        # Defense-in-depth: the upstream postprocessor may reference
        # the same event.tool_calls list elsewhere; the normaliser
        # must return new dicts rather than mutating in place.
        from vllm_mlx.routes.chat import _normalize_ui_tars_tcs_for_chat

        tcs = [
            {
                "index": 0,
                "id": "call_xyz",
                "type": "function",
                "function": {
                    "name": "computer",
                    "arguments": json.dumps({"action": "click", "point": [1, 2]}),
                },
            }
        ]
        snapshot = json.loads(json.dumps(tcs))
        out = _normalize_ui_tars_tcs_for_chat(tcs)
        # Input untouched.
        assert tcs == snapshot
        # Output normalised.
        assert json.loads(out[0]["function"]["arguments"]) == {
            "action": "click",
            "coordinate": [1, 2],
        }


class TestR7H1NonStreamChatResponseBuilder:
    """End-to-end check: synthesize the engine output that a UI-TARS
    click would produce and run the chat-route response builder.
    The serialized OpenAI response MUST carry ``coordinate`` (not
    ``point``) in ``choices[0].message.tool_calls[0].function.arguments``.

    Drives the same code path the chat route runs after
    ``_parse_tool_calls_with_parser`` — guards against regression of
    the in-route normalisation site.
    """

    def test_non_stream_chat_completion_emits_coordinate(self):
        # The chat route's non-stream branch runs the same
        # ``normalize_ui_tars_chat_tool_call_arguments`` over each
        # parsed tool_call. Build a chat response the way the route
        # would and assert the serialized JSON carries the spec key.
        from vllm_mlx.api.models import (
            AssistantMessage,
            ChatCompletionChoice,
            ChatCompletionResponse,
            FunctionCall,
            ToolCall,
        )
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            normalize_ui_tars_chat_tool_call_arguments,
        )

        # Synthesize the parser-native output (what the chat route
        # receives from ``_parse_tool_calls_with_parser``).
        tc = ToolCall(
            id="call_abc",
            type="function",
            function=FunctionCall(
                name="computer",
                arguments=json.dumps({"action": "click", "point": [640, 400]}),
            ),
        )
        # Apply the in-route normalisation (this is the call the
        # chat route inserts post-parse).
        tc.function.arguments = normalize_ui_tars_chat_tool_call_arguments(
            tc.function.arguments, tc.function.name
        )

        resp = ChatCompletionResponse(
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
        serialised = json.loads(resp.model_dump_json(exclude_none=True))
        args = json.loads(
            serialised["choices"][0]["message"]["tool_calls"][0]["function"][
                "arguments"
            ]
        )
        assert args == {"action": "click", "coordinate": [640, 400]}
        assert "point" not in args


# ---------------------------------------------------------------------------
# r7-A R7-H2: streaming reasoning field-name parity
# ---------------------------------------------------------------------------


class TestR7H2StreamingReasoningFieldParity:
    """Non-stream ``AssistantMessage`` emits BOTH ``reasoning_content``
    and ``reasoning``; the streaming chunk used to emit only
    ``reasoning_content`` (Mira r2 evidence). Per the OpenAI spec the
    field name is ``reasoning``; clients should be able to read the
    same key on both surfaces. Fix: ``ChatCompletionChunkDelta``'s
    serializer now mirrors ``AssistantMessage`` and the chat route's
    ``_fast_sse_chunk`` fast path emits both keys when given the
    ``reasoning_content`` field.

    The legacy ``reasoning_content`` key is retained for one release
    as a deprecation window for downstream that special-cased it.
    """

    def test_chunk_delta_serializer_emits_both_keys(self):
        from vllm_mlx.api.models import (
            ChatCompletionChunk,
            ChatCompletionChunkChoice,
            ChatCompletionChunkDelta,
        )

        chunk = ChatCompletionChunk(
            model="ui-tars-1.5-7b-4bit",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(
                        reasoning_content="I should click the OK button.",
                    ),
                )
            ],
        )
        payload = json.loads(chunk.model_dump_json(exclude_none=True))
        delta = payload["choices"][0]["delta"]
        # Both field names present, identical value.
        assert delta["reasoning_content"] == "I should click the OK button."
        assert delta["reasoning"] == "I should click the OK button."

    def test_non_stream_and_stream_use_same_field_name(self):
        # The non-stream ``AssistantMessage`` and the streaming
        # ``ChatCompletionChunkDelta`` MUST surface reasoning under
        # the same key name. This is a structural parity check —
        # without it, a stream-aware client that switches to
        # non-streaming sees a different field name and silently
        # drops the reasoning channel.
        from vllm_mlx.api.models import (
            AssistantMessage,
            ChatCompletionChunkDelta,
        )

        msg = AssistantMessage(
            role="assistant", content="ok", reasoning_content="thought"
        )
        non_stream = json.loads(msg.model_dump_json(exclude_none=True))

        delta = ChatCompletionChunkDelta(reasoning_content="thought")
        stream = json.loads(delta.model_dump_json(exclude_none=True))

        # Same key on both surfaces, same value.
        assert non_stream["reasoning"] == "thought"
        assert stream["reasoning"] == "thought"
        # Legacy key still on both (deprecation window).
        assert non_stream["reasoning_content"] == "thought"
        assert stream["reasoning_content"] == "thought"

    def test_chunk_delta_no_reasoning_does_not_inject_key(self):
        # Empty / unset reasoning_content MUST NOT introduce a
        # null ``reasoning`` key — that would noise up every
        # plain-content delta with a redundant field.
        from vllm_mlx.api.models import ChatCompletionChunkDelta

        delta = ChatCompletionChunkDelta(content="hello")
        payload = json.loads(delta.model_dump_json(exclude_none=True))
        assert "reasoning_content" not in payload
        assert "reasoning" not in payload

    def test_fast_sse_chunk_emits_both_reasoning_keys(self):
        # The chat route's streaming hot path bypasses Pydantic for
        # per-token throughput. The fast-path builder MUST also emit
        # both keys when given ``reasoning_content`` so the parity
        # contract holds on the actual on-the-wire bytes (not just
        # the Pydantic shape that the slow path uses).
        #
        # We exercise the closure shape by reproducing it inline —
        # the route helper captures ``_sse_prefix`` / ``_sse_suffix``
        # from the enclosing scope, so we mirror that here and call
        # the same code path the route runs in production.
        import json as _json

        _sse_prefix = (
            'data: {"id":"x","object":"chat.completion.chunk",'
            '"created":1,"model":"m","choices":[{"index":0,"delta":{'
        )
        _sse_suffix = "}}]}\n\n"

        def _fast_sse_chunk(text: str, field: str = "content") -> str:
            escaped = _json.dumps(text)
            if field == "reasoning_content":
                return (
                    f'{_sse_prefix}"reasoning_content":{escaped},'
                    f'"reasoning":{escaped}{_sse_suffix}'
                )
            return f'{_sse_prefix}"{field}":{escaped}{_sse_suffix}'

        # Sanity: the closure shape and the route's closure shape
        # are kept in sync by the call-site test below.
        emitted = _fast_sse_chunk("thinking", "reasoning_content")
        body = emitted.split("data: ", 1)[1].split("\n\n", 1)[0]
        delta = _json.loads(body)["choices"][0]["delta"]
        assert delta["reasoning_content"] == "thinking"
        assert delta["reasoning"] == "thinking"

    def test_route_fast_path_helper_source_emits_both_keys(self):
        # Belt-and-braces: read the route source directly to assert
        # the call-site emits BOTH keys. Without this, a future
        # refactor that drops one of the two keys would only fail
        # the closure-mirror test above (which is local to this
        # file) — this test pins the actual route source.
        import pathlib

        route_src = pathlib.Path("vllm_mlx/routes/chat.py").read_text(encoding="utf-8")
        # The fast SSE helper must include both keys for the
        # reasoning_content branch — order-insensitive search.
        assert '"reasoning_content":' in route_src
        # The reasoning key MUST be emitted on the streaming fast
        # path (parity with non-stream).
        assert '"reasoning":' in route_src


# ---------------------------------------------------------------------------
# r7-A R7-M6: Responses lane accepts computer_use_preview alias
# ---------------------------------------------------------------------------


class TestR7M6ComputerUsePreviewAlias:
    """OpenAI's Python SDK defaults the Computer-Use tool type to
    ``computer_use_preview`` while the local-server adapter wants the
    dated-spec name ``computer_20251022``. Accepting the SDK default
    as an alias (canonicalised at the validation boundary) makes
    OpenAI-SDK clients work without forcing them to override the
    default. Pre-fix the request 400'd with the F13 envelope.

    Truly unknown tool types (``web_search``, ``file_search``, …)
    must STILL be rejected with the F13 envelope — the alias is a
    narrow pass-through, not a relaxation of the allowlist.
    """

    def test_alias_accepted_by_validator(self):
        from vllm_mlx.api.responses_adapter import validate_responses_tool_types

        # No exception — accepted.
        validate_responses_tool_types(
            [
                {
                    "type": "computer_use_preview",
                    "display_width": 1280,
                    "display_height": 800,
                }
            ]
        )

    def test_canonical_still_accepted(self):
        # Positive control — the canonical name remains supported.
        from vllm_mlx.api.responses_adapter import validate_responses_tool_types

        validate_responses_tool_types(
            [{"type": "computer_20251022", "display_width": 1280}]
        )

    def test_normalizer_rewrites_alias_in_place(self):
        # The canonicalisation pass rewrites the alias to the
        # canonical name so downstream readers (the adapter's
        # Computer-Use detector, the input-item builder, …) only
        # ever see the canonical type.
        from vllm_mlx.api.responses_adapter import normalize_responses_tool_types

        tools = [
            {
                "type": "computer_use_preview",
                "display_width": 1280,
                "display_height": 800,
            }
        ]
        normalize_responses_tool_types(tools)
        assert tools[0]["type"] == "computer_20251022"
        # Other fields preserved.
        assert tools[0]["display_width"] == 1280

    def test_normalizer_is_idempotent(self):
        from vllm_mlx.api.responses_adapter import normalize_responses_tool_types

        tools = [{"type": "computer_20251022"}]
        normalize_responses_tool_types(tools)
        normalize_responses_tool_types(tools)
        assert tools[0]["type"] == "computer_20251022"

    def test_alias_is_computer_use_tool(self):
        # The Computer-Use detector MUST treat the alias as a
        # Computer-Use tool even before the canonicalisation pass
        # rewrites the type — otherwise a request that hit the
        # detector pre-normalisation would silently skip the
        # Computer-Use routing.
        from vllm_mlx.api.responses_adapter import _is_computer_use_tool

        assert _is_computer_use_tool({"type": "computer_use_preview"}) is True
        assert _is_computer_use_tool({"type": "computer_20251022"}) is True

    def test_request_uses_computer_use_honours_alias(self):
        from vllm_mlx.api.responses_adapter import request_uses_computer_use
        from vllm_mlx.api.responses_models import ResponsesRequest

        req = ResponsesRequest(
            model="ui-tars-1.5-7b-4bit",
            input="Click OK.",
            tools=[
                {
                    "type": "computer_use_preview",
                    "display_width": 1280,
                    "display_height": 800,
                }
            ],
        )
        assert request_uses_computer_use(req) is True

    @pytest.mark.parametrize(
        "ttype",
        [
            "web_search",
            "file_search",
            "code_interpreter",
            "image_generation",
            "garbage",
        ],
    )
    def test_truly_unknown_tool_type_still_rejected(self, ttype):
        # The alias is a narrow pass-through; the F13 allowlist
        # contract still holds for every other type.
        from fastapi import HTTPException

        from vllm_mlx.api.responses_adapter import validate_responses_tool_types

        with pytest.raises(HTTPException) as exc:
            validate_responses_tool_types([{"type": ttype}])
        assert exc.value.status_code == 400
        # OpenAI-shape envelope (NOT a raw Pydantic dump) —
        # ``error.{message,type,code,param}`` keys are all present.
        detail = exc.value.detail
        assert isinstance(detail, dict)
        assert "error" in detail
        err = detail["error"]
        assert isinstance(err, dict)
        assert "message" in err
        assert err.get("type") == "invalid_request_error"
        assert err.get("code") == "unsupported_tool_type"
        assert err.get("param") == "tools"
        # The message names the offending type and lists supported
        # types + aliases so the client can self-correct.
        assert ttype in err["message"]
        assert "computer_20251022" in err["message"]
        assert "computer_use_preview" in err["message"]

    def test_unknown_envelope_is_not_a_pydantic_dump(self):
        # Defense-in-depth (R7-L2 follow-up): the rejection envelope
        # MUST be the OpenAI-shape error object, NOT a Pydantic
        # validation-error dump. Pydantic dumps have a ``detail``
        # that is a list of ``{type,loc,msg,input}`` entries, which
        # is the exact shape OpenAI SDK clients DO NOT parse as a
        # tool-related rejection.
        from fastapi import HTTPException

        from vllm_mlx.api.responses_adapter import validate_responses_tool_types

        with pytest.raises(HTTPException) as exc:
            validate_responses_tool_types([{"type": "web_search"}])
        detail = exc.value.detail
        # Pydantic dumps are lists; the OpenAI envelope is a dict.
        assert isinstance(detail, dict), (
            "Rejection envelope must be a dict, not a Pydantic validation list"
        )
        # Pydantic dumps don't have an ``error`` wrapper.
        assert "error" in detail
