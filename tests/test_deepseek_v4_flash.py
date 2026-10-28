import json
from pathlib import Path

import pytest

from rapid_mlx.tool_parsers import ToolParserManager
from rapid_mlx.utils.deepseek_v4_flash import encode_messages

FIXTURES = Path(__file__).parent / "fixtures" / "deepseek_v4_flash"


@pytest.mark.parametrize(
    "number,mode", [(1, "thinking"), (2, "thinking"), (3, "thinking"), (4, "chat")]
)
def test_upstream_fixtures(number, mode):
    data = json.loads((FIXTURES / f"test_input_{number}.json").read_text())
    if number == 1:
        messages = data["messages"]
        messages[0]["tools"] = data["tools"]
    else:
        messages = data
    assert (
        encode_messages(messages, thinking_mode=mode)
        == (FIXTURES / f"test_output_{number}.txt").read_text()
    )


def test_upstream_dsml_parser():
    text = (FIXTURES / "test_output_1.txt").read_text()
    parser = ToolParserManager.get_tool_parser("deepseek_v4_flash")(None)
    result = parser.extract_tool_calls(text.split("<｜Assistant｜><think>", 1)[1])
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "location": "Beijing",
        "unit": "celsius",
    }


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-v4-flash-2bit",
        "deepseek-v4-flash-4bit",
        "deepseek-v4-flash-8bit",
        "deepseek-v4-flash-tensorfold",
        "mlx-community/DeepSeek-V4-Flash-4bit",
    ],
)
@pytest.mark.parametrize("thinking", [False, True])
def test_template_tool_loop(model, thinking):
    from rapid_mlx.model_auto_config import detect_model_config
    from rapid_mlx.utils.chat_template import apply_chat_template

    tools = [
        {
            "type": "function",
            "function": {"name": "weather", "parameters": {"type": "object"}},
        }
    ]
    user = {"role": "user", "content": "Weather?"}
    call = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "checking",
        "tool_calls": [
            {
                "id": "x",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": {"city": "Paris", "days": 2},
                },
            }
        ],
    }
    result = {"role": "tool", "tool_call_id": "x", "content": "sunny"}
    prompt = apply_chat_template(
        object(), [user], tools=tools, enable_thinking=thinking, model_name=model
    )
    history = apply_chat_template(
        object(),
        [user, call, result],
        tools=tools,
        enable_thinking=thinking,
        model_name=model,
    )
    assert '"name": "weather"' in prompt
    assert prompt.endswith("<｜Assistant｜>" + ("<think>" if thinking else "</think>"))
    assert history.startswith(prompt)
    assert (
        '<｜DSML｜parameter name="days" string="false">2</｜DSML｜parameter>' in history
    )
    assert "<tool_result>sunny</tool_result>" in history
    assert detect_model_config(model).tool_call_parser == "deepseek_v4_flash"


def test_parallel_streaming_and_literal_arguments():
    parser = ToolParserManager.get_tool_parser("deepseek_v4_flash")(None)
    wire = '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="a">\n<｜DSML｜parameter name="x" string="true"><\u200bthink></｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="b">\n<｜DSML｜parameter name="n" string="false">[1, true]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
    calls = parser.extract_tool_calls(wire).tool_calls
    assert [json.loads(c["arguments"]) for c in calls] == [
        {"x": "<\u200bthink>"},
        {"n": [1, True]},
    ]
    previous = ""
    emitted = []
    for char in wire:
        current = previous + char
        delta = parser.extract_tool_calls_streaming(previous, current, char)
        if delta and "tool_calls" in delta:
            emitted.extend(delta["tool_calls"])
        previous = current
    assert [json.loads(c["function"]["arguments"]) for c in emitted] == [
        {"x": "<\u200bthink>"},
        {"n": [1, True]},
    ]


def test_parallel_results_follow_call_order():
    calls = [
        {"id": name, "type": "function", "function": {"name": name, "arguments": "{}"}}
        for name in ("first", "second")
    ]
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "second", "content": "two"},
        {"role": "tool", "tool_call_id": "first", "content": "one"},
        {"role": "user", "content": "continue"},
    ]
    prompt = encode_messages(messages, thinking_mode="chat")
    assert (
        prompt.index("<tool_result>one</tool_result>")
        < prompt.index("<tool_result>two</tool_result>")
        < prompt.index("continue")
    )


def test_generation_prompt_can_be_disabled():
    from rapid_mlx.utils.chat_template import apply_chat_template

    prompt = apply_chat_template(
        object(),
        [{"role": "user", "content": "hello"}],
        model_name="deepseek-v4-flash-2bit",
        add_generation_prompt=False,
    )
    assert prompt == "<｜begin▁of▁sentence｜><｜User｜>hello"
