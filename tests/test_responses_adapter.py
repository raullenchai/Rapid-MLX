# SPDX-License-Identifier: Apache-2.0
"""
Tests for Responses-API-to-Chat-Completions adapter.

Pure-logic tests for vllm_mlx/api/responses_adapter.py — no MLX
dependency. Mirrors the test shape of test_anthropic_adapter.py.
"""

import json

import pytest

from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    Message,
    PromptTokensDetails,
    ToolCall,
    Usage,
)
from vllm_mlx.api.responses_adapter import (
    _convert_status,
    _convert_text_format,
    _convert_tool_choice,
    _convert_tools,
    _merge_system_messages,
    openai_to_responses,
    responses_to_openai,
)
from vllm_mlx.api.responses_models import (
    ResponsesContentItem,
    ResponsesInputItem,
    ResponsesRequest,
)

# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------


class TestConvertStatus:
    def test_length_to_incomplete(self):
        assert _convert_status("length") == "incomplete"

    def test_stop_to_completed(self):
        assert _convert_status("stop") == "completed"

    def test_tool_calls_to_completed(self):
        assert _convert_status("tool_calls") == "completed"

    def test_none_to_completed(self):
        assert _convert_status(None) == "completed"


# ---------------------------------------------------------------------------
# Tool conversion (Responses-flat → Chat-nested)
# ---------------------------------------------------------------------------


class TestConvertTools:
    def test_none_returns_none(self):
        assert _convert_tools(None) is None

    def test_empty_list_returns_none(self):
        assert _convert_tools([]) is None

    def test_function_tool_flat_to_nested(self):
        tools = _convert_tools(
            [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ]
        )
        assert tools is not None and len(tools) == 1
        td = tools[0]
        assert td.type == "function"
        assert td.function["name"] == "get_weather"
        assert td.function["description"] == "Get weather"
        assert td.function["parameters"]["properties"] == {"city": {"type": "string"}}

    def test_unsupported_tool_types_raise_400(self):
        """Yuki F13 (0.8.5 dogfood): unsupported tool types now raise a
        clean 400 instead of silently dropping. The chat/anthropic lanes
        already 400; the /v1/responses lane now matches.
        """
        from fastapi import HTTPException

        for unsupported in ("web_search", "code_interpreter", "image_generation"):
            with pytest.raises(HTTPException) as exc_info:
                _convert_tools(
                    [
                        {"type": "function", "name": "real_one"},
                        {"type": unsupported},
                    ]
                )
            assert exc_info.value.status_code == 400
            assert "unsupported_tool_type" in str(exc_info.value.detail)

    def test_drops_function_without_name(self):
        tools = _convert_tools([{"type": "function", "description": "no name"}])
        assert tools is None

    def test_missing_parameters_defaults_to_empty_object_schema(self):
        tools = _convert_tools([{"type": "function", "name": "minimal"}])
        assert tools is not None and len(tools) == 1
        assert tools[0].function["parameters"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Tool-choice
# ---------------------------------------------------------------------------


class TestConvertToolChoice:
    def test_none(self):
        assert _convert_tool_choice(None) is None

    def test_strings_pass_through(self):
        assert _convert_tool_choice("auto") == "auto"
        assert _convert_tool_choice("none") == "none"
        assert _convert_tool_choice("required") == "required"

    def test_function_object_renested(self):
        result = _convert_tool_choice({"type": "function", "name": "do_thing"})
        assert result == {"type": "function", "function": {"name": "do_thing"}}

    def test_unknown_object_returns_none(self):
        assert _convert_tool_choice({"type": "wat"}) is None


# ---------------------------------------------------------------------------
# text.format → response_format
# ---------------------------------------------------------------------------


class TestConvertTextFormat:
    def test_none_input(self):
        assert _convert_text_format(None) is None

    def test_no_format_key(self):
        assert _convert_text_format({"verbosity": "medium"}) is None

    def test_text_type_returns_none(self):
        # We don't need response_format for plain text output.
        assert _convert_text_format({"format": {"type": "text"}}) is None

    def test_json_object(self):
        result = _convert_text_format({"format": {"type": "json_object"}})
        assert result is not None
        assert result.type == "json_object"

    def test_json_schema_plumbed_through(self):
        result = _convert_text_format(
            {
                "format": {
                    "type": "json_schema",
                    "name": "Movie",
                    "description": "A movie",
                    "schema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                    },
                    "strict": True,
                }
            }
        )
        assert result is not None
        assert result.type == "json_schema"
        assert result.json_schema is not None
        assert result.json_schema.name == "Movie"
        assert result.json_schema.description == "A movie"
        assert result.json_schema.schema_["properties"] == {"title": {"type": "string"}}
        assert result.json_schema.strict is True

    def test_json_schema_missing_schema_returns_none(self):
        result = _convert_text_format(
            {"format": {"type": "json_schema", "name": "Bad"}}
        )
        assert result is None


# ---------------------------------------------------------------------------
# responses_to_openai — full request shape
# ---------------------------------------------------------------------------


class TestResponsesToOpenai:
    def test_bare_string_input_becomes_user_message(self):
        req = ResponsesRequest(model="gpt-5", input="Hello world")
        chat = responses_to_openai(req)
        assert len(chat.messages) == 1
        assert chat.messages[0].role == "user"
        assert chat.messages[0].content == "Hello world"

    def test_instructions_prepended_as_system(self):
        req = ResponsesRequest(
            model="gpt-5",
            instructions="You are helpful.",
            input="Hi",
        )
        chat = responses_to_openai(req)
        assert chat.messages[0].role == "system"
        assert chat.messages[0].content == "You are helpful."
        assert chat.messages[1].role == "user"
        assert chat.messages[1].content == "Hi"

    def test_developer_role_maps_to_system(self):
        # Codex CLI 0.136.0 uses Responses-API "developer" role for the
        # system-priority instruction channel. Open-weight chat templates
        # (Qwen, Llama, Gemma) only know system/user/assistant/tool —
        # passing "developer" through verbatim raises
        # `jinja2.TemplateError: Unexpected message role.` mid-stream
        # and Codex sees "stream disconnected".
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="message",
                    role="developer",
                    content="Always reply in JSON.",
                ),
            ],
        )
        chat = responses_to_openai(req)
        assert chat.messages[0].role == "system"
        assert chat.messages[0].content == "Always reply in JSON."

    def test_developer_role_with_structured_content_does_not_raise(self):
        # Defensive: today every system message reaches the merge step
        # with a string content (`_message_item_to_chat` joins parts).
        # codex_review flagged that a mutated path could leave a list in
        # `Message.content` and `"\n\n".join([list, list])` would raise
        # `TypeError: sequence item 0: expected str instance, list found`.
        # The adapter must coerce defensively rather than crash.
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="message",
                    role="developer",
                    content=[
                        ResponsesContentItem(type="input_text", text="part one"),
                        ResponsesContentItem(type="input_text", text="part two"),
                    ],
                ),
                ResponsesInputItem(type="message", role="user", content="hi"),
            ],
        )
        # Must not raise.
        chat = responses_to_openai(req)
        assert chat.messages[0].role == "system"
        assert "part one" in chat.messages[0].content
        assert "part two" in chat.messages[0].content

    def test_input_image_is_preserved_as_chat_multimodal_content(self):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="message",
                    role="user",
                    content=[
                        ResponsesContentItem(type="input_text", text="Describe this"),
                        ResponsesContentItem(
                            type="input_image",
                            image_url="data:image/png;base64,abc",
                        ),
                    ],
                ),
            ],
        )
        chat = responses_to_openai(req)

        assert len(chat.messages) == 1
        content = chat.messages[0].content
        assert isinstance(content, list)
        assert content[0].type == "text"
        assert content[0].text == "Describe this"
        assert content[1].type == "image_url"
        assert content[1].image_url.url == "data:image/png;base64,abc"

    def test_input_image_preserves_image_url_options(self):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem.model_construct(
                    type="message",
                    role="user",
                    content=[
                        {
                            "type": "input_image",
                            "image_url": {
                                "url": "data:image/png;base64,abc",
                                "detail": "high",
                                "unexpected": "ignored",
                            },
                        },
                    ],
                )
            ],
        )

        chat = responses_to_openai(req)

        image_url = chat.messages[0].content[0].image_url
        assert image_url.url == "data:image/png;base64,abc"
        assert image_url.detail == "high"
        assert not hasattr(image_url, "unexpected")

    def test_malformed_responses_content_block_does_not_become_empty_prompt(self):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="message",
                    role="user",
                    content=[ResponsesContentItem(type="input_image")],
                ),
            ],
        )

        with pytest.raises(ValueError, match="input_image.image_url"):
            responses_to_openai(req)

    def test_input_audio_content_block_rejected_on_responses_path(self):
        req = ResponsesRequest.model_construct(
            model="gpt-5",
            input=[
                ResponsesInputItem.model_construct(
                    type="message",
                    role="user",
                    content=[
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "AAAA", "format": "wav"},
                        }
                    ],
                )
            ],
        )

        with pytest.raises(ValueError, match="input_audio content blocks"):
            responses_to_openai(req)

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            (None, "Responses message content is required"),
            ([], "Responses message content must not be empty"),
        ],
    )
    def test_empty_message_content_does_not_become_empty_prompt(self, content, match):
        req = ResponsesRequest.model_construct(
            model="gpt-5",
            input=[
                ResponsesInputItem.model_construct(
                    type="message",
                    role="user",
                    content=content,
                )
            ],
        )

        with pytest.raises(ValueError, match=match):
            responses_to_openai(req)

    def test_empty_string_message_content_is_preserved(self):
        req = ResponsesRequest.model_construct(
            model="gpt-5",
            input=[
                ResponsesInputItem.model_construct(
                    type="message",
                    role="user",
                    content="",
                )
            ],
        )

        chat = responses_to_openai(req)

        assert chat.messages[0].role == "user"
        assert chat.messages[0].content == ""

    @pytest.mark.parametrize(
        ("content_item", "match"),
        [
            (ResponsesContentItem(type="input_text"), "input_text.text is required"),
            (
                ResponsesContentItem(type="input_text", text=""),
                "input_text.text must be a non-empty string",
            ),
            (ResponsesContentItem(type="output_text"), "output_text.text is required"),
        ],
    )
    def test_malformed_text_content_block_does_not_become_empty_prompt(
        self, content_item, match
    ):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="message",
                    role="user",
                    content=[content_item],
                ),
            ],
        )

        with pytest.raises(ValueError, match=match):
            responses_to_openai(req)

    def test_merge_system_messages_defends_list_content(self):
        # Directly exercise the defensive `_to_text(list)` path that the
        # public `responses_to_openai` flow cannot reach today (because
        # `_message_item_to_chat` joins parts to a string before merge).
        # Use `model_construct` to bypass pydantic validation and pass a
        # raw list / dict through — without `_to_text` this would crash
        # with `TypeError: sequence item 0: expected str instance, list
        # found` once a future code path leaves `Message.content` un-
        # coerced. codex_review NIT: cover the path directly.
        msgs = [
            Message.model_construct(
                role="system",
                content=[{"text": "alpha"}, {"text": "beta"}],
            ),
            Message.model_construct(
                role="system",
                content={"text": "gamma"},
            ),
            Message(role="user", content="hi"),
        ]
        merged = _merge_system_messages(msgs)
        assert sum(1 for m in merged if m.role == "system") == 1
        assert merged[0].role == "system"
        assert merged[0].content == "alpha\nbeta\n\ngamma"
        assert merged[1].role == "user"

    def test_merge_system_messages_drops_empty_system_after_user(self):
        # codex_review BLOCKING regression: a `developer` item with
        # empty content reaches the merge step as `Message(role="system",
        # content="")`. Old logic branched on whether the merged text
        # was truthy and returned `messages` unchanged when it wasn't —
        # leaving the empty system message at index 1 to trip Qwen's
        # `System message must be at the beginning.` check.
        msgs = [
            Message(role="user", content="hi"),
            Message(role="system", content=""),
        ]
        merged = _merge_system_messages(msgs)
        # No system message survives — and the user message remains.
        assert all(m.role != "system" for m in merged)
        assert any(m.role == "user" and m.content == "hi" for m in merged)

    def test_merge_system_messages_drops_empty_system_only(self):
        # When the ONLY system messages are empty, drop them entirely
        # rather than emit `Message(role="system", content="")` — some
        # templates also reject that.
        msgs = [
            Message(role="system", content=""),
            Message(role="user", content="hi"),
        ]
        merged = _merge_system_messages(msgs)
        assert merged == [Message(role="user", content="hi")]

    def test_merge_system_messages_unknown_shape_does_not_raise(self):
        # `_to_text` returns "" for anything that isn't str / dict / list,
        # so a lone unknown-shape system message yields empty
        # `system_texts` and the message is dropped (same path as the
        # empty-content case — keeping it would leave a non-leading or
        # empty system message that some templates reject). Defends
        # against future content shapes (e.g. int, custom object)
        # without raising.
        msgs = [
            Message.model_construct(role="system", content=12345),
            Message(role="user", content="hi"),
        ]
        # Must not raise.
        merged = _merge_system_messages(msgs)
        assert all(m.role != "system" for m in merged)
        assert merged[0].role == "user"

    def test_multiple_systems_merge_to_single_at_index_0(self):
        # Codex sends BOTH `instructions` (which becomes system) AND a
        # mid-conversation `developer`-role item (which we map to system).
        # Qwen / Llama / Gemma templates require exactly ONE system message
        # at index 0 — otherwise the template raises
        # `System message must be at the beginning.` mid-stream and Codex
        # sees "stream disconnected".
        req = ResponsesRequest(
            model="gpt-5",
            instructions="You are the base agent.",
            input=[
                ResponsesInputItem(type="message", role="user", content="Hi"),
                ResponsesInputItem(
                    type="message", role="developer", content="Be terse."
                ),
            ],
        )
        chat = responses_to_openai(req)
        # Exactly one system message at index 0, preserving order.
        assert sum(1 for m in chat.messages if m.role == "system") == 1
        assert chat.messages[0].role == "system"
        assert chat.messages[0].content == ("You are the base agent.\n\nBe terse.")
        # All other messages preserved in order.
        assert chat.messages[1].role == "user"
        assert chat.messages[1].content[0].type == "text"
        assert chat.messages[1].content[0].text == "Hi"

    def test_message_input_item(self):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="message",
                    role="user",
                    content=[ResponsesContentItem(type="input_text", text="Hello")],
                ),
            ],
        )
        chat = responses_to_openai(req)
        assert len(chat.messages) == 1
        assert chat.messages[0].role == "user"
        assert chat.messages[0].content[0].type == "text"
        assert chat.messages[0].content[0].text == "Hello"

    def test_message_input_item_string_content_uses_text_part_validation(self):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="message",
                    role="user",
                    content="Hello",
                ),
            ],
        )
        chat = responses_to_openai(req)
        assert len(chat.messages) == 1
        assert chat.messages[0].role == "user"
        assert chat.messages[0].content[0].type == "text"
        assert chat.messages[0].content[0].text == "Hello"

    def test_message_input_joins_multiple_text_parts(self):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="message",
                    role="user",
                    content=[
                        ResponsesContentItem(type="input_text", text="line one"),
                        ResponsesContentItem(type="input_text", text="line two"),
                    ],
                ),
            ],
        )
        chat = responses_to_openai(req)
        assert [part.text for part in chat.messages[0].content] == [
            "line one",
            "line two",
        ]

    def test_output_text_content_replays_assistant(self):
        # Codex echoes prior assistant turns as type=message role=assistant
        # with content=[{type:"output_text", text:"..."}].
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="message",
                    role="assistant",
                    content=[
                        ResponsesContentItem(type="output_text", text="prior reply")
                    ],
                ),
            ],
        )
        chat = responses_to_openai(req)
        assert chat.messages[0].role == "assistant"
        assert chat.messages[0].content[0].type == "text"
        assert chat.messages[0].content[0].text == "prior reply"

    def test_function_call_input_item_becomes_assistant_with_tool_calls(self):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="function_call",
                    call_id="call_42",
                    name="run_query",
                    arguments='{"q":"weather"}',
                ),
            ],
        )
        chat = responses_to_openai(req)
        msg = chat.messages[0]
        assert msg.role == "assistant"
        assert msg.tool_calls is not None and len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc["id"] == "call_42"
        assert tc["function"]["name"] == "run_query"
        assert tc["function"]["arguments"] == '{"q":"weather"}'

    def test_function_call_output_with_string_becomes_tool_message(self):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="function_call_output",
                    call_id="call_42",
                    output="sunny, 72F",
                ),
            ],
        )
        chat = responses_to_openai(req)
        msg = chat.messages[0]
        assert msg.role == "tool"
        assert msg.content == "sunny, 72F"
        assert msg.tool_call_id == "call_42"

    def test_function_call_output_with_dict_serialized_to_json(self):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="function_call_output",
                    call_id="call_99",
                    output={"city": "SF", "temp_f": 64},
                ),
            ],
        )
        chat = responses_to_openai(req)
        # Tool message content must be JSON when the original was structured.
        assert json.loads(chat.messages[0].content) == {"city": "SF", "temp_f": 64}

    def test_reasoning_items_dropped(self):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(
                    type="reasoning",
                    encrypted_content="opaque-blob-from-openai",
                ),
                ResponsesInputItem(
                    type="message",
                    role="user",
                    content=[ResponsesContentItem(type="input_text", text="Hi")],
                ),
            ],
        )
        chat = responses_to_openai(req)
        assert len(chat.messages) == 1
        assert chat.messages[0].content[0].text == "Hi"

    def test_unknown_item_types_silently_dropped(self):
        req = ResponsesRequest(
            model="gpt-5",
            input=[
                ResponsesInputItem(type="local_shell_call"),
                ResponsesInputItem(
                    type="message",
                    role="user",
                    content=[ResponsesContentItem(type="input_text", text="Hi")],
                ),
            ],
        )
        chat = responses_to_openai(req)
        assert len(chat.messages) == 1
        assert chat.messages[0].content[0].text == "Hi"

    def test_sampling_fields_forwarded(self):
        req = ResponsesRequest(
            model="gpt-5",
            input="x",
            temperature=0.42,
            top_p=0.93,
            max_output_tokens=128,
            parallel_tool_calls=False,
            stream=True,
        )
        chat = responses_to_openai(req)
        assert chat.temperature == 0.42
        assert chat.top_p == 0.93
        assert chat.max_tokens == 128
        assert chat.parallel_tool_calls is False
        assert chat.stream is True

    def test_temperature_omitted_passes_none(self):
        # The cascade in service/helpers needs to see None so it can
        # fall through to alias defaults — same contract as Anthropic.
        req = ResponsesRequest(model="gpt-5", input="x")
        chat = responses_to_openai(req)
        assert chat.temperature is None
        assert chat.top_p is None

    def test_tools_forwarded(self):
        req = ResponsesRequest(
            model="gpt-5",
            input="x",
            tools=[
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )
        chat = responses_to_openai(req)
        assert chat.tools is not None and len(chat.tools) == 1
        assert chat.tools[0].function["name"] == "search"

    def test_text_format_to_response_format(self):
        req = ResponsesRequest(
            model="gpt-5",
            input="x",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "P",
                    "schema": {"type": "object"},
                }
            },
        )
        chat = responses_to_openai(req)
        assert chat.response_format is not None
        assert chat.response_format.type == "json_schema"

    # ------------------------------------------------------------------
    # R14 task #293 — chat-style response_format migration shim
    # ------------------------------------------------------------------

    def test_chat_style_response_format_json_schema_strict_plumbed(self):
        """Chat-style ``response_format`` with strict json_schema is
        forwarded onto the materialized ``ChatCompletionRequest`` so the
        engine's guided-decoding gate fires identically to
        /v1/chat/completions. Pre-fix, ResponsesRequest silently dropped
        the field and the engine ran unconstrained.
        """
        req = ResponsesRequest(
            model="gpt-5",
            input="give me a movie",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Movie",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                },
            },
        )
        chat = responses_to_openai(req)
        assert chat.response_format is not None
        assert chat.response_format.type == "json_schema"
        assert chat.response_format.json_schema is not None
        assert chat.response_format.json_schema.name == "Movie"
        assert chat.response_format.json_schema.strict is True
        assert chat.response_format.json_schema.schema_ == {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        }

    def test_chat_style_response_format_json_object_plumbed(self):
        """``response_format={"type":"json_object"}`` (JSON mode, no
        schema) is also forwarded — same migration shim coverage."""
        req = ResponsesRequest(
            model="gpt-5",
            input="x",
            response_format={"type": "json_object"},
        )
        chat = responses_to_openai(req)
        assert chat.response_format is not None
        assert chat.response_format.type == "json_object"

    def test_chat_style_response_format_text_type_is_inert(self):
        """``type=text`` is the no-structure default and should not
        block the engine from sampling normally — it forwards as-is
        and the route's strict gate (is_strict_json_schema) returns
        False, so the unconstrained path runs."""
        req = ResponsesRequest(
            model="gpt-5",
            input="x",
            response_format={"type": "text"},
        )
        chat = responses_to_openai(req)
        # The field is forwarded but ``type=text`` is functionally a
        # no-op — is_strict_json_schema returns False, the route's
        # strict gate skips, and the engine runs unconstrained.
        assert chat.response_format is not None
        assert chat.response_format.type == "text"

    def test_text_format_wins_over_response_format(self):
        """When BOTH ``text.format`` and chat-style ``response_format``
        are set, the canonical Responses-spec ``text.format`` wins.
        This mirrors OpenAI cloud behaviour — clients that mix both
        shapes get the spec-canonical answer."""
        req = ResponsesRequest(
            model="gpt-5",
            input="x",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "FromText",
                    "schema": {"type": "object"},
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "FromResponseFormat",
                    "strict": True,
                    "schema": {"type": "object"},
                },
            },
        )
        chat = responses_to_openai(req)
        assert chat.response_format is not None
        assert chat.response_format.json_schema is not None
        assert chat.response_format.json_schema.name == "FromText"

    def test_response_format_absent_keeps_engine_unconstrained(self):
        """Sanity: no response_format and no text.format → the
        materialized ChatCompletionRequest carries None and the engine
        samples freely. Pinning this in case the shim's None-fallback
        ever flips."""
        req = ResponsesRequest(model="gpt-5", input="x")
        chat = responses_to_openai(req)
        assert chat.response_format is None

    def test_chat_style_response_format_strict_must_be_bool(self):
        """The shared ``_validate_response_format_raw`` rejects
        ``strict:"true"`` (string) on this surface too — parity with
        chat / completions / messages. Pre-fix the field was undeclared
        so the validator never ran and the string truthy-but-not-bool
        silently passed."""
        with pytest.raises(Exception) as exc_info:
            ResponsesRequest(
                model="gpt-5",
                input="x",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "X",
                        "strict": "true",  # string, not bool
                        "schema": {"type": "object"},
                    },
                },
            )
        assert "strict" in str(exc_info.value).lower()

    def test_chat_style_response_format_empty_schema_rejected(self):
        """Empty ``json_schema.schema={}`` is rejected at parse time
        on this surface (same gate as chat / completions / messages).
        Closes the F-103 silent-200 hazard for the Responses surface."""
        with pytest.raises(Exception) as exc_info:
            ResponsesRequest(
                model="gpt-5",
                input="x",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "X",
                        "strict": True,
                        "schema": {},
                    },
                },
            )
        assert "schema" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# openai_to_responses — full response shape
# ---------------------------------------------------------------------------


def _chat_response(
    *,
    text: str | None = "",
    tool_calls: list[ToolCall] | None = None,
    reasoning_content: str | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    cached: int = 0,
) -> ChatCompletionResponse:
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    if cached:
        usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=cached)
    return ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=text,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_content,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _bare_request() -> ResponsesRequest:
    return ResponsesRequest(model="gpt-5", input="x")


class TestOpenaiToResponses:
    def test_text_only_output(self):
        chat_resp = _chat_response(text="Hello!")
        resp = openai_to_responses(
            chat_resp, model="test-model", request=_bare_request(), created_at=0
        )
        assert len(resp.output) == 1
        item = resp.output[0]
        assert item.type == "message"
        assert item.role == "assistant"
        assert item.content is not None and len(item.content) == 1
        assert item.content[0].type == "output_text"
        assert item.content[0].text == "Hello!"
        assert resp.status == "completed"

    def test_empty_text_omits_message_item(self):
        # A pure-tool-call turn should NOT emit a phantom empty message
        # item — that's what the public Responses API does.
        chat_resp = _chat_response(
            text=None,
            tool_calls=[
                ToolCall(
                    id="call_x",
                    function=FunctionCall(name="run", arguments='{"a":1}'),
                )
            ],
            finish_reason="tool_calls",
        )
        resp = openai_to_responses(
            chat_resp, model="test-model", request=_bare_request(), created_at=0
        )
        assert len(resp.output) == 1
        assert resp.output[0].type == "function_call"

    def test_empty_stop_emits_empty_message_item(self):
        """D-MISSING-CONTENT-KEY (r12-7): an empty completion +
        ``finish_reason="stop"`` (granite4-h-micro repro: "Reply with
        only the letter A." + ``max_tokens=3``, model emits nothing
        visible) MUST still surface a well-formed ``message`` item with
        ``content: [{type:"output_text", text:""}]`` so /v1/responses
        callers walking ``output[i].content[0].text`` keep their happy
        path. Pre-fix the ``output`` array was empty and clients
        crashed with IndexError / KeyError."""
        chat_resp = _chat_response(text=None, finish_reason="stop")
        resp = openai_to_responses(
            chat_resp,
            model="granite4-h-micro-4bit",
            request=_bare_request(),
            created_at=0,
        )
        assert len(resp.output) == 1, (
            "D-MISSING-CONTENT-KEY: empty stop must surface an assistant "
            "message item, not an empty output array."
        )
        item = resp.output[0]
        assert item.type == "message"
        assert item.role == "assistant"
        assert item.content is not None and len(item.content) == 1
        assert item.content[0].type == "output_text"
        assert item.content[0].text == ""

    def test_reasoning_only_does_not_emit_empty_message_item(self):
        """D-MISSING-CONTENT-KEY (r12-7): a reasoning-only turn (closed
        thought block, no answer text) keeps the OpenAI-spec shape —
        only the ``reasoning`` item is emitted; no empty message item
        is synthesized. The reasoning item itself represents the
        assistant's structured signal for this turn."""
        chat_resp = _chat_response(
            text=None,
            reasoning_content="Let me think... 17 * 23 =",
            finish_reason="stop",
        )
        resp = openai_to_responses(
            chat_resp, model="reasoning-model", request=_bare_request(), created_at=0
        )
        types = [item.type for item in resp.output]
        # Only reasoning, no synthesized empty message item.
        assert types == ["reasoning"], (
            "D-MISSING-CONTENT-KEY: reasoning-only turn must NOT "
            f"synthesize an empty message item; got output types {types!r}."
        )

    def test_text_then_tool_call_ordering(self):
        chat_resp = _chat_response(
            text="Looking that up...",
            tool_calls=[
                ToolCall(
                    id="call_a",
                    function=FunctionCall(name="search", arguments='{"q":"x"}'),
                )
            ],
            finish_reason="tool_calls",
        )
        resp = openai_to_responses(
            chat_resp, model="test-model", request=_bare_request(), created_at=0
        )
        assert len(resp.output) == 2
        # message must come before any function_call — Codex CLI
        # depends on this ordering when re-rendering turns.
        assert resp.output[0].type == "message"
        assert resp.output[1].type == "function_call"
        assert resp.output[1].name == "search"
        assert resp.output[1].arguments == '{"q":"x"}'
        assert resp.output[1].call_id == "call_a"

    def test_length_finish_reason_marks_incomplete(self):
        chat_resp = _chat_response(text="cut off here", finish_reason="length")
        resp = openai_to_responses(
            chat_resp, model="test-model", request=_bare_request(), created_at=0
        )
        assert resp.status == "incomplete"

    def test_usage_block_populated(self):
        chat_resp = _chat_response(
            text="hi", prompt_tokens=100, completion_tokens=50, cached=30
        )
        resp = openai_to_responses(
            chat_resp, model="test-model", request=_bare_request(), created_at=0
        )
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50
        assert resp.usage.total_tokens == 150
        assert resp.usage.input_tokens_details == {"cached_tokens": 30}

    def test_cached_tokens_clamped_to_prompt(self):
        # Defensive against an over-reported cache count — same clamp
        # the Anthropic adapter does.
        chat_resp = _chat_response(
            text="hi", prompt_tokens=10, completion_tokens=5, cached=999
        )
        resp = openai_to_responses(
            chat_resp, model="test-model", request=_bare_request(), created_at=0
        )
        assert resp.usage.input_tokens_details == {"cached_tokens": 10}

    def test_request_metadata_echoed(self):
        req = ResponsesRequest(
            model="gpt-5",
            input="x",
            metadata={"trace_id": "abc"},
            instructions="be brief",
        )
        resp = openai_to_responses(
            _chat_response(text="hi"), model="m", request=req, created_at=42
        )
        assert resp.created_at == 42
        assert resp.metadata == {"trace_id": "abc"}
        assert resp.instructions == "be brief"


class TestReasoningCutoffSentinelDoesNotMaskIncomplete:
    """Regression for the PR #860 → v0.8.13/v0.8.14 cutoff-sentinel /
    Responses-adapter parity bug.

    PR #860 (closes #858) restored ``REASONING_CUTOFF_SENTINEL`` injection
    into ``message.content`` by default for length-stopped reasoning
    generations. The chat / anthropic / responses non-stream routes call
    ``_apply_reasoning_cutoff_notice`` BEFORE handing the
    ``ChatCompletionResponse`` to ``openai_to_responses``. The adapter
    then computes ``downstream_output_seen`` from ``message.content`` —
    so the sentinel injection was being mis-classified as "the model
    produced real downstream output", flipping
    ``output[0].reasoning.status`` from ``"incomplete"`` to
    ``"completed"`` on the non-stream surface only. The streaming surface
    uses its own ``reasoning_block_closed`` predicate (the parser's own
    content channel signal, never the sentinel) so the streaming side
    still reported ``"incomplete"`` — visible as a cross-path parity
    breakage caught by
    ``test_responses_budget_exhaust_streaming.py::TestStreamingNonStreamingParity::test_same_output_shape_under_cutoff``.

    These tests pin the contract at the adapter boundary so a future
    refactor of either ``_apply_reasoning_cutoff_notice`` or the
    adapter's ``downstream_output_seen`` predicate can't silently
    reintroduce the regression.
    """

    def _chat_response_with_reasoning(
        self,
        *,
        text: str | None,
        reasoning_text: str,
        finish_reason: str,
        tool_calls: list[ToolCall] | None = None,
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(
                        content=text,
                        reasoning_content=reasoning_text,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def test_sentinel_in_content_keeps_reasoning_incomplete_on_length(self):
        from vllm_mlx.api.constants import REASONING_CUTOFF_SENTINEL

        chat_resp = self._chat_response_with_reasoning(
            text=REASONING_CUTOFF_SENTINEL,
            reasoning_text="Hmm, let me think about this carefully...",
            finish_reason="length",
        )
        resp = openai_to_responses(
            chat_resp, model="test-model", request=_bare_request(), created_at=0
        )
        assert len(resp.output) >= 1
        assert resp.output[0].type == "reasoning"
        assert resp.output[0].status == "incomplete", (
            "REASONING_CUTOFF_SENTINEL injected by the chat route helper "
            "must NOT flip reasoning_item_status to 'completed' — it is a "
            "UX fallback, not real downstream output."
        )

    def test_real_downstream_content_still_marks_reasoning_completed(self):
        chat_resp = self._chat_response_with_reasoning(
            text="Here is the actual answer to your question.",
            reasoning_text="Hmm, let me think about this carefully...",
            finish_reason="length",
        )
        resp = openai_to_responses(
            chat_resp, model="test-model", request=_bare_request(), created_at=0
        )
        assert len(resp.output) >= 1
        assert resp.output[0].type == "reasoning"
        assert resp.output[0].status == "completed", (
            "Real model content (not the sentinel) IS downstream output — "
            "reasoning has closed and the cutoff scope shrinks to the "
            "message body, so reasoning_item_status must be 'completed'."
        )

    def test_tool_call_after_thinking_still_marks_reasoning_completed(self):
        chat_resp = self._chat_response_with_reasoning(
            text=None,
            reasoning_text="Hmm, I should call the search tool.",
            finish_reason="length",
            tool_calls=[
                ToolCall(
                    id="call_sentinel_regression",
                    function=FunctionCall(name="search", arguments='{"q":"x"}'),
                )
            ],
        )
        resp = openai_to_responses(
            chat_resp, model="test-model", request=_bare_request(), created_at=0
        )
        assert len(resp.output) >= 1
        assert resp.output[0].type == "reasoning"
        assert resp.output[0].status == "completed", (
            "Closed-</think> + tool_call shape: reasoning IS complete, "
            "only tool args were truncated — same contract as before "
            "the sentinel parity fix."
        )

    def test_sentinel_with_no_finish_length_keeps_reasoning_completed(self):
        from vllm_mlx.api.constants import REASONING_CUTOFF_SENTINEL

        chat_resp = self._chat_response_with_reasoning(
            text=REASONING_CUTOFF_SENTINEL,
            reasoning_text="Hmm, let me think...",
            finish_reason="stop",
        )
        resp = openai_to_responses(
            chat_resp, model="test-model", request=_bare_request(), created_at=0
        )
        assert len(resp.output) >= 1
        assert resp.output[0].type == "reasoning"
        assert resp.output[0].status == "completed", (
            "The 'incomplete' branch requires finish_reason='length'; the "
            "sentinel-exclusion fix must not over-extend to non-length "
            "completions."
        )


# ---------------------------------------------------------------------------
# Round-trip — request → chat → response keeps Codex's invariants
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_full_codex_turn_roundtrip(self):
        """A realistic Codex CLI turn: system instructions + replayed
        user turn + replayed function_call + replayed function_call_output,
        followed by a new user message. The adapter must produce 5 chat
        messages in the right order with the right tool_call_id wiring."""
        req = ResponsesRequest(
            model="gpt-5-codex",
            instructions="You are Codex.",
            input=[
                ResponsesInputItem(
                    type="message",
                    role="user",
                    content=[ResponsesContentItem(type="input_text", text="ls -la")],
                ),
                ResponsesInputItem(
                    type="function_call",
                    call_id="call_1",
                    name="run_shell",
                    arguments='{"cmd":"ls -la"}',
                ),
                ResponsesInputItem(
                    type="function_call_output",
                    call_id="call_1",
                    output="total 8\\ndrwxr-xr-x ...",
                ),
                ResponsesInputItem(
                    type="message",
                    role="user",
                    content=[
                        ResponsesContentItem(
                            type="input_text", text="now show the README"
                        )
                    ],
                ),
            ],
            tools=[
                {
                    "type": "function",
                    "name": "run_shell",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
            stream=True,
        )
        chat = responses_to_openai(req)
        # 1 system + 1 user + 1 assistant (with tool_calls) + 1 tool + 1 user
        assert [m.role for m in chat.messages] == [
            "system",
            "user",
            "assistant",
            "tool",
            "user",
        ]
        # Tool wiring: tool message must reference call_1.
        assert chat.messages[3].tool_call_id == "call_1"
        # Assistant tool_call must carry call_1 + the original args string.
        tc = chat.messages[2].tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["function"]["arguments"] == '{"cmd":"ls -la"}'
        # Tool was forwarded.
        assert chat.tools is not None
        assert chat.tools[0].function["name"] == "run_shell"
