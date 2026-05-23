# SPDX-License-Identifier: Apache-2.0
"""Regression test for legacy ``functions`` / ``function_call`` shape silently dropped.

Bug: OpenAI's pre-1.0 SDK and several LangChain compat layers still emit the
deprecated ``functions: [{name, parameters, description}]`` + ``function_call``
fields instead of the modern ``tools`` + ``tool_choice``. ChatCompletionRequest
didn't declare either, so Pydantic dropped them on parse and the request ran
as if no tool exists — the model returned a plain-text refusal, no
tool_calls fired. Same blind-spot family as #355 (logit_bias), #459
(max_completion_tokens), #464 (parallel_tool_calls).

Fix: declare both fields and normalize them to the modern slots via a
``model_validator(mode="after")``. Modern wins when both shapes are
provided, matching OpenAI's documented deprecation behavior.
"""

from vllm_mlx.api.models import (
    ChatCompletionRequest,
    Message,
    ToolDefinition,
)


def _msg() -> list[Message]:
    return [Message(role="user", content="hi")]


class TestLegacyFunctionsNormalization:
    """``functions`` → ``tools`` conversion."""

    def test_functions_converted_to_tools(self):
        req = ChatCompletionRequest(
            model="m",
            messages=_msg(),
            functions=[
                {
                    "name": "get_weather",
                    "description": "fetch weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tools[0].type == "function"
        assert req.tools[0].function["name"] == "get_weather"
        assert req.tools[0].function["description"] == "fetch weather"

    def test_functions_multiple_converted_to_tools(self):
        req = ChatCompletionRequest(
            model="m",
            messages=_msg(),
            functions=[
                {"name": "a", "parameters": {"type": "object"}},
                {"name": "b", "parameters": {"type": "object"}},
            ],
        )
        assert len(req.tools) == 2
        assert [t.function["name"] for t in req.tools] == ["a", "b"]

    def test_modern_tools_take_precedence(self):
        """If both legacy and modern are supplied, modern wins. This is
        OpenAI's documented deprecation behavior — clients in the middle
        of migrating may send both, and we should never silently downgrade."""
        req = ChatCompletionRequest(
            model="m",
            messages=_msg(),
            tools=[
                ToolDefinition(
                    type="function",
                    function={"name": "modern", "parameters": {"type": "object"}},
                )
            ],
            functions=[
                {"name": "legacy", "parameters": {"type": "object"}},
            ],
        )
        assert len(req.tools) == 1
        assert req.tools[0].function["name"] == "modern"

    def test_empty_functions_list_is_no_op(self):
        """[] is truthy-empty; should not synthesize an empty tools list."""
        req = ChatCompletionRequest(
            model="m",
            messages=_msg(),
            functions=[],
        )
        assert req.tools is None  # empty list is falsy → skipped

    def test_functions_omitted_leaves_tools_unchanged(self):
        req = ChatCompletionRequest(model="m", messages=_msg())
        assert req.tools is None
        assert req.functions is None


class TestLegacyFunctionCallNormalization:
    """``function_call`` → ``tool_choice`` conversion."""

    def test_function_call_auto_string(self):
        req = ChatCompletionRequest(
            model="m",
            messages=_msg(),
            function_call="auto",
        )
        assert req.tool_choice == "auto"

    def test_function_call_none_string(self):
        req = ChatCompletionRequest(
            model="m",
            messages=_msg(),
            function_call="none",
        )
        assert req.tool_choice == "none"

    def test_function_call_specific_function_dict(self):
        req = ChatCompletionRequest(
            model="m",
            messages=_msg(),
            function_call={"name": "get_weather"},
        )
        assert req.tool_choice == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_modern_tool_choice_takes_precedence(self):
        req = ChatCompletionRequest(
            model="m",
            messages=_msg(),
            tool_choice="auto",
            function_call={"name": "ignored"},
        )
        assert req.tool_choice == "auto"

    def test_function_call_omitted_leaves_tool_choice_unchanged(self):
        req = ChatCompletionRequest(model="m", messages=_msg())
        assert req.tool_choice is None
        assert req.function_call is None


class TestCombinedLegacy:
    """End-to-end legacy → modern round-trip on a realistic legacy request."""

    def test_full_legacy_request_normalized(self):
        req = ChatCompletionRequest(
            model="m",
            messages=_msg(),
            functions=[
                {
                    "name": "get_weather",
                    "description": "current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            function_call={"name": "get_weather"},
        )
        # Both legacy slots round-trip (we don't blank them — clients may
        # inspect what they sent), but modern slots are populated for the
        # route layer.
        assert req.functions is not None
        assert req.function_call == {"name": "get_weather"}
        assert req.tools is not None and len(req.tools) == 1
        assert req.tools[0].function["name"] == "get_weather"
        assert req.tool_choice == {
            "type": "function",
            "function": {"name": "get_weather"},
        }
