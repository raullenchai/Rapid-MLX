# SPDX-License-Identifier: Apache-2.0
"""Regression tests for OpenAI tool-call replay normalization before chat templating.

Cherry-pick coverage of upstream waybarrios/vllm-mlx#470. Many chat templates
iterate ``message.tool_calls[i].function.arguments`` as a mapping; OpenAI's API
contract gives it as a JSON string. Without normalization the template render
raises mid-batch with ``AttributeError: 'str' object has no attribute 'items'``.
"""

from vllm_mlx.engine.batched import _normalize_tool_call_arguments_for_template


class TestToolCallReplayNormalization:
    def test_parses_function_arguments_string_to_mapping(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }
                ],
            }
        ]

        normalized = _normalize_tool_call_arguments_for_template(messages)

        assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Tokyo"
        }
        # Original is not mutated — API surface keeps the JSON-string contract.
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
            '{"city": "Tokyo"}'
        )

    def test_wraps_non_mapping_arguments_for_template_items(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "echo",
                            "arguments": '["not", "object"]',
                        }
                    }
                ],
            }
        ]

        normalized = _normalize_tool_call_arguments_for_template(messages)

        assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
            "value": ["not", "object"]
        }

    def test_wraps_malformed_json_arguments(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bad",
                            "arguments": "{not valid json",
                        }
                    }
                ],
            }
        ]

        normalized = _normalize_tool_call_arguments_for_template(messages)

        assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
            "value": "{not valid json"
        }

    def test_non_assistant_messages_are_untouched(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]

        normalized = _normalize_tool_call_arguments_for_template(messages)

        assert normalized == messages

    def test_already_mapping_arguments_are_left_alone(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Tokyo"},
                        }
                    }
                ],
            }
        ]

        normalized = _normalize_tool_call_arguments_for_template(messages)

        assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Tokyo"
        }

    def test_missing_tool_calls_field_is_safe(self):
        messages = [{"role": "assistant", "content": "no tool calls"}]

        normalized = _normalize_tool_call_arguments_for_template(messages)

        assert normalized == messages
