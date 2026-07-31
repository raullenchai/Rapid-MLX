# SPDX-License-Identifier: Apache-2.0

import json

from vllm_mlx.model_aliases import list_profiles
from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.utils.chat_template import apply_chat_template
from vllm_mlx.utils.deepseek_v4_0731 import ASSISTANT, BOS, THINK_END, THINK_START, USER


class _TokenizerWithoutTemplate:
    chat_template = None

    def apply_chat_template(self, *_args, **_kwargs):
        raise AssertionError("0731 must bypass the generic/Jinja path")


def test_alias_points_at_0731_mxfp4_with_ultra_memory_floor():
    profile = list_profiles()["deepseek-v4-flash-0731-mxfp4"]
    assert profile.hf_path == "Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX"
    assert profile.tool_call_parser == "deepseek_v4_0731"
    assert profile.reasoning_parser == "deepseek_r1"
    assert profile.is_moe is True
    assert profile.min_memory_gb == 192
    assert profile.supports_spec_decode is False


def test_official_prompt_shape_bypasses_missing_jinja_template():
    prompt = apply_chat_template(
        _TokenizerWithoutTemplate(),
        [{"role": "user", "content": "hello"}],
        enable_thinking=True,
        model_name="deepseek-v4-flash-0731-mxfp4",
    )
    assert prompt == f"{BOS}{USER}hello{ASSISTANT}{THINK_START}"


def test_chat_mode_uses_official_think_end_generation_prefix():
    prompt = apply_chat_template(
        _TokenizerWithoutTemplate(),
        [{"role": "system", "content": "brief"}, {"role": "user", "content": "hello"}],
        enable_thinking=False,
        model_name="Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX",
    )
    assert prompt == f"{BOS}brief{USER}hello{ASSISTANT}{THINK_END}"


def test_official_multiturn_thinking_drop_rule():
    prompt = apply_chat_template(
        _TokenizerWithoutTemplate(),
        [
            {"role": "system", "content": "helpful"},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "reasoning_content": "old reasoning",
                "content": "Hi",
            },
            {"role": "user", "content": "Capital of France?"},
            {
                "role": "assistant",
                "reasoning_content": "Paris reasoning",
                "content": "Paris",
            },
        ],
        enable_thinking=True,
        model_name="deepseek-v4-flash-0731-mxfp4",
    )
    assert prompt == (
        f"{BOS}helpful{USER}Hello{ASSISTANT}{THINK_END}Hi"
        f"<｜end▁of▁sentence｜>{USER}Capital of France?{ASSISTANT}"
        f"{THINK_START}Paris reasoning{THINK_END}Paris<｜end▁of▁sentence｜>"
    )


def test_dsml_tool_schema_and_tool_result_are_encoded():
    prompt = apply_chat_template(
        _TokenizerWithoutTemplate(),
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": {"city": "Paris", "days": 2},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "forecast",
                    "parameters": {"type": "object"},
                },
            }
        ],
        enable_thinking=False,
        model_name="deepseek-v4-flash-0731-mxfp4",
    )
    assert "## Tools" in prompt
    assert '<｜DSML｜invoke name="weather">' in prompt
    assert '<｜DSML｜parameter name="city" string="true">Paris' in prompt
    assert '<｜DSML｜parameter name="days" string="false">2' in prompt
    assert "<tool_result>sunny</tool_result>" in prompt


def test_dsml_parser_returns_openai_tool_call():
    parser = ToolParserManager.get_tool_parser("deepseek_v4_0731")(None)
    output = (
        "checking\n<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="weather">\n'
        '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="days" string="false">2</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    )
    result = parser.extract_tool_calls(output)
    assert result.tools_called is True
    assert result.content == "checking"
    assert result.tool_calls[0]["name"] == "weather"
    assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Paris", "days": 2}
