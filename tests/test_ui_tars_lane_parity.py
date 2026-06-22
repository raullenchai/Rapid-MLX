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


# ---------------------------------------------------------------------------
# r6-B R6-M2: Anthropic + Responses lanes translate point → coordinate
# ---------------------------------------------------------------------------


class TestR6M2CoordinateKeyTranslation:
    """The UI-TARS parser emits the canonical ``point`` /
    ``start_point`` / ``end_point`` keys (PR #812 contract; chat
    completions OpenAI lane stays bytes-faithful to that). The
    Anthropic ``/v1/messages`` lane and the OpenAI ``/v1/responses``
    lane both follow specs that require ``coordinate`` (and
    ``start_coordinate`` / ``end_coordinate`` for two-point verbs).

    Pre-r6-B: both adapters surfaced the UI-TARS-native ``point`` key
    verbatim. Anthropic-strict consumers (claude-agent-sdk, Computer-
    Use harnesses) rejected the shape. Fixed by a centralized
    ``translate_to_spec_coordinate_keys`` helper that lives next to
    the parser and is called from both adapters' tool_use / computer_call
    builders so the two surfaces can't drift on key naming.
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

    def test_anthropic_drag_emits_start_end_coordinate(self):
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
        # Two-point verb → start_coordinate / end_coordinate.
        assert tool_uses[0].input == {
            "action": "drag",
            "start_coordinate": [10, 20],
            "end_coordinate": [100, 200],
        }

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

    # --- Chat Completions OpenAI lane stays on point ----------------------

    def test_chat_completions_lane_still_emits_native_point(self):
        # Defense-in-depth: the OpenAI ``/v1/chat/completions`` lane
        # must stay bytes-faithful to the UI-TARS parser's native
        # ``point`` key per PR #812 — only the Anthropic + Responses
        # lanes do the spec translation. A SDK consumer that round-
        # trips the chat-completions arguments → JSON → back will see
        # ``point`` (the parser's bytes-faithful contract).
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import UiTarsToolParser

        parser = UiTarsToolParser(tokenizer=None)
        parser.reset()
        text = "Action: click(point='<point>500 300</point>')"
        result = parser.extract_tool_calls(text, request=None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0]["arguments"])
        # Chat-completions OpenAI lane: parser-native ``point`` key.
        assert args == {"action": "click", "point": [500, 300]}
        assert "coordinate" not in args

    # --- Helper-level test ------------------------------------------------

    def test_translate_helper_is_idempotent(self):
        # The mapper must be safe to call twice — already-translated
        # keys stay translated (defense-in-depth for a future
        # double-translation refactor).
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            translate_to_spec_coordinate_keys,
        )

        once = translate_to_spec_coordinate_keys({"action": "click", "point": [1, 2]})
        twice = translate_to_spec_coordinate_keys(once)
        assert once == twice == {"action": "click", "coordinate": [1, 2]}

    def test_translate_helper_preserves_non_coord_kwargs(self):
        # Non-coord kwargs (action, content, key, direction, …)
        # pass through verbatim.
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            translate_to_spec_coordinate_keys,
        )

        out = translate_to_spec_coordinate_keys({"action": "type", "content": "hello"})
        assert out == {"action": "type", "content": "hello"}

        out = translate_to_spec_coordinate_keys({"action": "hotkey", "key": "ctrl+c"})
        assert out == {"action": "hotkey", "key": "ctrl+c"}
