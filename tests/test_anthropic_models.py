# SPDX-License-Identifier: Apache-2.0
"""
Tests for Anthropic Messages API Pydantic models.

Tests all request/response models in vllm_mlx/api/anthropic_models.py.
These are pure Pydantic models with no MLX dependency.
"""

import pytest
from pydantic import ValidationError

from vllm_mlx.api.anthropic_models import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicResponseContentBlock,
    AnthropicToolDef,
    AnthropicUsage,
)


class TestAnthropicContentBlock:
    """Tests for AnthropicContentBlock model."""

    def test_text_block(self):
        block = AnthropicContentBlock(type="text", text="hello")
        assert block.type == "text"
        assert block.text == "hello"

    def test_tool_use_block(self):
        block = AnthropicContentBlock(
            type="tool_use",
            id="call_123",
            name="get_weather",
            input={"city": "Paris"},
        )
        assert block.type == "tool_use"
        assert block.id == "call_123"
        assert block.name == "get_weather"
        assert block.input == {"city": "Paris"}

    def test_tool_result_block(self):
        block = AnthropicContentBlock(
            type="tool_result",
            tool_use_id="call_123",
            content="sunny",
        )
        assert block.type == "tool_result"
        assert block.tool_use_id == "call_123"
        assert block.content == "sunny"

    def test_tool_result_with_error(self):
        block = AnthropicContentBlock(
            type="tool_result",
            tool_use_id="call_123",
            content="not found",
            is_error=True,
        )
        assert block.is_error is True

    def test_image_block(self):
        block = AnthropicContentBlock(
            type="image",
            source={"type": "base64", "media_type": "image/png", "data": "abc"},
        )
        assert block.type == "image"
        assert block.source["type"] == "base64"

    def test_optional_fields_default_to_none(self):
        """D-ANTHRO-VALIDATION F4 update: per-type required fields are
        now enforced at construction (text block REQUIRES text). The
        "all-optional" surface for a text block accepted ``{type:'text'}``
        with no payload and let the model run on empty content. The
        original spirit of this test — that *other* fields default to
        None when only the relevant per-type field is set — is
        preserved below by building a well-formed text block."""
        block = AnthropicContentBlock(type="text", text="")
        assert block.text == ""
        assert block.id is None
        assert block.name is None
        assert block.input is None
        assert block.tool_use_id is None
        assert block.content is None
        assert block.is_error is None
        assert block.source is None


class TestAnthropicMessage:
    """Tests for AnthropicMessage model."""

    def test_string_content(self):
        msg = AnthropicMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_list_content(self):
        blocks = [
            AnthropicContentBlock(type="text", text="look at this"),
            AnthropicContentBlock(
                type="image",
                source={"type": "base64", "media_type": "image/png", "data": "abc"},
            ),
        ]
        msg = AnthropicMessage(role="user", content=blocks)
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "image"

    def test_assistant_role(self):
        msg = AnthropicMessage(role="assistant", content="hi there")
        assert msg.role == "assistant"

    def test_missing_role_raises(self):
        with pytest.raises(ValidationError):
            AnthropicMessage(content="hello")

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            AnthropicMessage(role="user")


class TestAnthropicToolDef:
    """Tests for AnthropicToolDef model."""

    def test_minimal(self):
        tool = AnthropicToolDef(name="get_weather")
        assert tool.name == "get_weather"
        assert tool.description is None
        assert tool.input_schema is None

    def test_full(self):
        tool = AnthropicToolDef(
            name="get_weather",
            description="Get weather for a city",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        assert tool.description == "Get weather for a city"
        assert tool.input_schema["required"] == ["city"]

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            AnthropicToolDef(description="no name")


class TestAnthropicRequest:
    """Tests for AnthropicRequest model."""

    def test_minimal_request(self):
        req = AnthropicRequest(
            model="default",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=100,
        )
        assert req.model == "default"
        assert req.max_tokens == 100
        assert req.stream is False
        assert req.temperature is None
        assert req.top_p is None
        assert req.tools is None
        assert req.system is None

    def test_with_system_string(self):
        req = AnthropicRequest(
            model="default",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=100,
            system="You are helpful.",
        )
        assert req.system == "You are helpful."

    def test_with_system_list(self):
        req = AnthropicRequest(
            model="default",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=100,
            system=[{"type": "text", "text": "Be concise."}],
        )
        assert isinstance(req.system, list)
        assert req.system[0]["text"] == "Be concise."

    def test_with_tools(self):
        req = AnthropicRequest(
            model="default",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=100,
            tools=[AnthropicToolDef(name="search")],
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "search"

    def test_with_tool_choice(self):
        req = AnthropicRequest(
            model="default",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=100,
            tool_choice={"type": "auto"},
        )
        assert req.tool_choice == {"type": "auto"}

    def test_streaming(self):
        req = AnthropicRequest(
            model="default",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=100,
            stream=True,
        )
        assert req.stream is True

    def test_all_optional_params(self):
        req = AnthropicRequest(
            model="default",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=256,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            stop_sequences=["END"],
            metadata={"user_id": "123"},
        )
        assert req.temperature == 0.5
        assert req.top_p == 0.9
        assert req.top_k == 40
        assert req.stop_sequences == ["END"]
        assert req.metadata == {"user_id": "123"}

    def test_missing_model_raises(self):
        with pytest.raises(ValidationError):
            AnthropicRequest(
                messages=[AnthropicMessage(role="user", content="hi")],
                max_tokens=100,
            )

    def test_missing_messages_raises(self):
        with pytest.raises(ValidationError):
            AnthropicRequest(model="default", max_tokens=100)

    def test_missing_max_tokens_raises(self):
        with pytest.raises(ValidationError):
            AnthropicRequest(
                model="default",
                messages=[AnthropicMessage(role="user", content="hi")],
            )


class TestAnthropicUsage:
    """Tests for AnthropicUsage model."""

    def test_defaults(self):
        usage = AnthropicUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    def test_with_values(self):
        usage = AnthropicUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_with_cache_fields(self):
        usage = AnthropicUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=80,
        )
        assert usage.cache_creation_input_tokens == 20
        assert usage.cache_read_input_tokens == 80


class TestAnthropicResponseContentBlock:
    """Tests for AnthropicResponseContentBlock model."""

    def test_text_block(self):
        block = AnthropicResponseContentBlock(type="text", text="hello")
        assert block.type == "text"
        assert block.text == "hello"

    def test_tool_use_block(self):
        block = AnthropicResponseContentBlock(
            type="tool_use",
            id="call_abc",
            name="search",
            input={"query": "test"},
        )
        assert block.type == "tool_use"
        assert block.id == "call_abc"
        assert block.name == "search"
        assert block.input == {"query": "test"}

    def test_optional_fields_default_to_none(self):
        block = AnthropicResponseContentBlock(type="text")
        assert block.text is None
        assert block.id is None
        assert block.name is None
        assert block.input is None


class TestAnthropicResponse:
    """Tests for AnthropicResponse model."""

    def test_minimal_response(self):
        resp = AnthropicResponse(
            model="default",
            content=[AnthropicResponseContentBlock(type="text", text="hi")],
        )
        assert resp.model == "default"
        assert resp.type == "message"
        assert resp.role == "assistant"
        assert resp.id.startswith("msg_")
        assert len(resp.id) == len("msg_") + 24
        assert resp.stop_reason is None
        assert resp.stop_sequence is None
        assert resp.usage.input_tokens == 0
        assert resp.usage.output_tokens == 0

    def test_with_stop_reason(self):
        resp = AnthropicResponse(
            model="default",
            content=[AnthropicResponseContentBlock(type="text", text="done")],
            stop_reason="end_turn",
        )
        assert resp.stop_reason == "end_turn"

    def test_with_usage(self):
        resp = AnthropicResponse(
            model="default",
            content=[AnthropicResponseContentBlock(type="text", text="hi")],
            usage=AnthropicUsage(input_tokens=10, output_tokens=5),
        )
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

    def test_unique_ids(self):
        r1 = AnthropicResponse(
            model="default",
            content=[AnthropicResponseContentBlock(type="text", text="a")],
        )
        r2 = AnthropicResponse(
            model="default",
            content=[AnthropicResponseContentBlock(type="text", text="b")],
        )
        assert r1.id != r2.id

    def test_tool_use_response(self):
        resp = AnthropicResponse(
            model="default",
            content=[
                AnthropicResponseContentBlock(type="text", text="Let me search."),
                AnthropicResponseContentBlock(
                    type="tool_use",
                    id="call_1",
                    name="search",
                    input={"q": "test"},
                ),
            ],
            stop_reason="tool_use",
        )
        assert len(resp.content) == 2
        assert resp.content[0].type == "text"
        assert resp.content[1].type == "tool_use"
        assert resp.stop_reason == "tool_use"
