# SPDX-License-Identifier: Apache-2.0
"""Tests for native tool format support (Issue #29).

Tests the SUPPORTS_NATIVE_TOOL_FORMAT capability flag and the
preserve_native_format parameter in extract_multimodal_content().
"""

import pytest

from vllm_mlx.api.utils import extract_multimodal_content
from vllm_mlx.tool_parsers import (
    AutoToolParser,
    DeepSeekToolParser,
    FunctionaryToolParser,
    GraniteToolParser,
    HarmonyToolParser,
    HermesToolParser,
    KimiToolParser,
    LlamaToolParser,
    MistralToolParser,
    NemotronToolParser,
    QwenToolParser,
    ToolParserManager,
    xLAMToolParser,
)


class TestNativeToolFormatCapability:
    """Test SUPPORTS_NATIVE_TOOL_FORMAT class attribute."""

    def test_parsers_with_native_support(self):
        """Parsers that support native tool format should return True."""
        native_parsers = [
            MistralToolParser,
            LlamaToolParser,
            DeepSeekToolParser,
            GraniteToolParser,
            FunctionaryToolParser,
            KimiToolParser,
            HermesToolParser,
            HarmonyToolParser,
        ]
        for parser_cls in native_parsers:
            assert (
                parser_cls.SUPPORTS_NATIVE_TOOL_FORMAT is True
            ), f"{parser_cls.__name__} should support native format"
            assert (
                parser_cls.supports_native_format() is True
            ), f"{parser_cls.__name__}.supports_native_format() should return True"

    def test_parsers_without_native_support(self):
        """Parsers that don't support native tool format should return False."""
        non_native_parsers = [
            QwenToolParser,
            NemotronToolParser,
            xLAMToolParser,
            AutoToolParser,
        ]
        for parser_cls in non_native_parsers:
            assert (
                parser_cls.SUPPORTS_NATIVE_TOOL_FORMAT is False
            ), f"{parser_cls.__name__} should not support native format"
            assert (
                parser_cls.supports_native_format() is False
            ), f"{parser_cls.__name__}.supports_native_format() should return False"

    def test_via_manager(self):
        """Test native format detection via ToolParserManager."""
        # Native support
        for name in [
            "mistral",
            "llama",
            "deepseek",
            "granite",
            "functionary",
            "kimi",
            "hermes",
            "harmony",
        ]:
            parser_cls = ToolParserManager.get_tool_parser(name)
            assert (
                parser_cls.supports_native_format() is True
            ), f"Parser '{name}' should support native format"

        # No native support
        for name in ["qwen", "nemotron", "xlam", "auto"]:
            parser_cls = ToolParserManager.get_tool_parser(name)
            assert (
                parser_cls.supports_native_format() is False
            ), f"Parser '{name}' should not support native format"


class TestExtractMultimodalContentNativeFormat:
    """Test extract_multimodal_content with preserve_native_format parameter."""

    @pytest.fixture
    def messages_with_tool_calls(self):
        """Sample messages with tool calls and tool results."""
        return [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "72°F and sunny",
            },
            {"role": "user", "content": "Thanks!"},
        ]

    def test_default_converts_to_text(self, messages_with_tool_calls):
        """Default behavior converts tool messages to text format."""
        processed, images, videos = extract_multimodal_content(messages_with_tool_calls)

        assert len(processed) == 4

        # User message unchanged
        assert processed[0]["role"] == "user"
        assert processed[0]["content"] == "What is the weather in Paris?"

        # Assistant tool_calls converted to text
        assert processed[1]["role"] == "assistant"
        assert "[Calling tool: get_weather" in processed[1]["content"]
        assert "tool_calls" not in processed[1]

        # Tool result converted to user role
        assert processed[2]["role"] == "user"
        assert "[Tool Result (call_abc123)]" in processed[2]["content"]
        assert "72°F and sunny" in processed[2]["content"]

        # Final user message unchanged
        assert processed[3]["role"] == "user"
        assert processed[3]["content"] == "Thanks!"

    def test_preserve_native_format_true(self, messages_with_tool_calls):
        """preserve_native_format=True keeps native tool format."""
        processed, images, videos = extract_multimodal_content(
            messages_with_tool_calls, preserve_native_format=True
        )

        assert len(processed) == 4

        # User message unchanged
        assert processed[0]["role"] == "user"
        assert processed[0]["content"] == "What is the weather in Paris?"

        # Assistant message keeps tool_calls field
        assert processed[1]["role"] == "assistant"
        assert "tool_calls" in processed[1]
        assert len(processed[1]["tool_calls"]) == 1
        assert processed[1]["tool_calls"][0]["id"] == "call_abc123"
        assert processed[1]["tool_calls"][0]["function"]["name"] == "get_weather"

        # Tool result keeps role="tool"
        assert processed[2]["role"] == "tool"
        assert processed[2]["tool_call_id"] == "call_abc123"
        assert processed[2]["content"] == "72°F and sunny"

        # Final user message unchanged
        assert processed[3]["role"] == "user"
        assert processed[3]["content"] == "Thanks!"

    def test_empty_tool_call_id(self):
        """Handle empty or missing tool_call_id gracefully."""
        messages = [
            {"role": "tool", "content": "result without id"},
        ]

        # Default mode
        processed, _, _ = extract_multimodal_content(messages)
        assert processed[0]["role"] == "user"
        assert "[Tool Result ()]" in processed[0]["content"]

        # Native mode
        processed, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["role"] == "tool"
        assert processed[0]["tool_call_id"] == ""
        assert processed[0]["content"] == "result without id"

    def test_multiple_tool_calls(self):
        """Handle multiple tool calls in a single assistant message."""
        messages = [
            {"role": "user", "content": "Get weather and time"},
            {
                "role": "assistant",
                "content": "I'll check both.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            {"role": "tool", "tool_call_id": "call_2", "content": "3:00 PM"},
        ]

        # Native mode
        processed, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )

        assert len(processed) == 4
        assert len(processed[1]["tool_calls"]) == 2
        assert processed[2]["role"] == "tool"
        assert processed[2]["tool_call_id"] == "call_1"
        assert processed[3]["role"] == "tool"
        assert processed[3]["tool_call_id"] == "call_2"

    def test_mixed_content_preserved(self):
        """Regular text messages are preserved regardless of mode."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Default mode
        processed_default, _, _ = extract_multimodal_content(messages)

        # Native mode
        processed_native, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )

        # Both should be identical for non-tool messages
        assert processed_default == processed_native
        assert len(processed_default) == 3
        assert processed_default[0]["role"] == "system"
        assert processed_default[1]["role"] == "user"
        assert processed_default[2]["role"] == "assistant"

    def test_assistant_with_content_and_tool_calls(self):
        """Assistant message with both content and tool_calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me check that for you.",
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }
                ],
            }
        ]

        # Default mode - content and tool calls merged
        processed, _, _ = extract_multimodal_content(messages)
        assert "Let me check that for you." in processed[0]["content"]
        assert "[Calling tool: search" in processed[0]["content"]

        # Native mode - content and tool_calls separate
        processed, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["content"] == "Let me check that for you."
        assert "tool_calls" in processed[0]
        assert processed[0]["tool_calls"][0]["function"]["name"] == "search"


class TestEdgeCases:
    """Edge cases for native tool format handling."""

    def test_none_content_in_tool_message(self):
        """Handle None content in tool messages."""
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": None},
        ]

        processed, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["content"] == ""

    def test_pydantic_v2_model_tool_calls(self):
        """Handle Pydantic v2 model tool_calls (with model_dump method)."""

        class MockToolCallV2:
            def model_dump(self):
                return {
                    "id": "call_v2",
                    "type": "function",
                    "function": {"name": "v2_fn", "arguments": "{}"},
                }

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [MockToolCallV2()],
            }
        ]

        processed, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["tool_calls"][0]["id"] == "call_v2"
        assert processed[0]["tool_calls"][0]["function"]["name"] == "v2_fn"

    def test_pydantic_v1_model_tool_calls(self):
        """Handle Pydantic v1 model tool_calls (with dict method)."""

        class MockToolCallV1:
            def dict(self):
                return {
                    "id": "call_v1",
                    "type": "function",
                    "function": {"name": "v1_fn", "arguments": "{}"},
                }

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [MockToolCallV1()],
            }
        ]

        processed, _, _ = extract_multimodal_content(
            messages, preserve_native_format=True
        )
        assert processed[0]["tool_calls"][0]["id"] == "call_v1"
        assert processed[0]["tool_calls"][0]["function"]["name"] == "v1_fn"

    def test_images_and_videos_extracted_with_native_format(self):
        """Image/video extraction works with preserve_native_format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.jpg"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Analysis result"},
        ]

        processed, images, videos = extract_multimodal_content(
            messages, preserve_native_format=True
        )

        assert len(images) == 1
        assert images[0] == "http://example.com/img.jpg"
        assert processed[1]["role"] == "tool"
