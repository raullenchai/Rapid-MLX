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


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    backend = Tokenizer(
        models.BPE(vocab={char: i for i, char in enumerate(alphabet)}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.mark.parametrize("arguments", ['{"city": "Paris"}', {"city": "Paris"}])
def test_flattened_tool_call_history(arguments, tokenizer):
    from rapid_mlx.utils.chat_template import apply_chat_template

    call = {"id": "x", "type": "function", "name": "weather", "arguments": arguments}
    history = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "x", "content": "sunny"},
        {"role": "user", "content": "Thanks"},
    ]
    nested = [*history]
    nested[1] = {
        **history[1],
        "tool_calls": [
            {
                "id": "x",
                "type": "function",
                "function": {"name": "weather", "arguments": arguments},
            }
        ],
    }
    prompt = apply_chat_template(
        tokenizer, history, model_name="deepseek-v4-flash-2bit"
    )
    assert "weather" in prompt
    assert "Paris" in prompt
    assert prompt == apply_chat_template(
        tokenizer, nested, model_name="deepseek-v4-flash-2bit"
    )
    prompt_0731 = apply_chat_template(
        tokenizer, history, model_name="deepseek-v4-flash-0731-mxfp4"
    )
    assert "weather" in prompt_0731
    assert "Paris" in prompt_0731


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
def test_template_tool_loop(model, thinking, tokenizer):
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
        tokenizer, [user], tools=tools, enable_thinking=thinking, model_name=model
    )
    history = apply_chat_template(
        tokenizer,
        [user, call, result],
        tools=tools,
        enable_thinking=thinking,
        model_name=model,
    )
    import copy

    expected_call = copy.deepcopy(call)
    expected_call["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"city": "Paris", "days": 2}, ensure_ascii=False
    )
    expected = encode_messages(
        [
            {"role": "system", "content": "", "tools": tools},
            user,
            expected_call,
            result,
        ],
        thinking_mode="thinking" if thinking else "chat",
    )
    assert history == expected
    assert '"name": "weather"' in prompt
    assert prompt.endswith("<｜Assistant｜>" + ("<think>" if thinking else "</think>"))
    assert history.startswith(prompt)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    assert tokenizer.decode(tokens) == prompt
    assert tokenizer.encode(history, add_special_tokens=False)[: len(tokens)] == tokens
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


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-v41-flash-1017",
        "rapid-mlx/DeepSeek-V4.1-Flash-1017-MLX",
        "rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX",
    ],
)
def test_v41_models_keep_their_own_template(model, monkeypatch):
    from rapid_mlx.utils import deepseek_v4_flash
    from rapid_mlx.utils.chat_template import apply_chat_template

    def unexpected(*_args, **_kwargs):
        raise AssertionError("V4.1 must not use the original V4 encoder")

    monkeypatch.setattr(deepseek_v4_flash, "encode_messages", unexpected)

    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return "template:" + messages[-1]["content"]

    prompt = apply_chat_template(
        Tokenizer(), [{"role": "user", "content": "hello"}], model_name=model
    )
    assert prompt.startswith("template:")


@pytest.mark.parametrize(
    "model",
    [
        "/models/DeepSeek-V4.1-Flash-custom",
        "someone/deepseek-v4.1-flash-4bit",
        "deepseek-v41-flash-local",
    ],
)
def test_v41_identities_keep_previous_parser(model):
    from rapid_mlx.model_auto_config import detect_model_config

    config = detect_model_config(model)
    assert config is not None
    assert config.tool_call_parser == "deepseek"
    assert config.reasoning_parser == "deepseek_v4"


@pytest.mark.parametrize("thinking", [False, True])
def test_openai_developer_message_is_kept_as_system(thinking):
    from rapid_mlx.utils.chat_template import apply_chat_template

    prompt = apply_chat_template(
        object(),
        [
            {"role": "developer", "content": "Answer in French."},
            {"role": "user", "content": "hello"},
        ],
        model_name="deepseek-v4-flash-2bit",
        enable_thinking=thinking,
    )
    assert prompt.startswith("<｜begin▁of▁sentence｜>Answer in French.<｜User｜>hello")


def test_final_developer_turn_still_gets_generation_prefix():
    from rapid_mlx.utils.chat_template import apply_chat_template

    prompt = apply_chat_template(
        object(),
        [
            {"role": "user", "content": "hello"},
            {"role": "developer", "content": "Reply briefly."},
        ],
        model_name="deepseek-v4-flash-2bit",
        enable_thinking=False,
    )
    assert prompt.endswith("<｜Assistant｜></think>")


@pytest.mark.parametrize(
    "model", ["/models/DeepSeek-V4-128k-Flash", "someone/deepseek-v4-1024ctx-4bit"]
)
def test_v4_names_with_numeric_suffix_stay_on_original_v4(model):
    from rapid_mlx.model_auto_config import detect_model_config

    assert detect_model_config(model).tool_call_parser == "deepseek_v4_flash"


@pytest.mark.parametrize("raw", ["null", "[]", "1", '"text"'])
def test_non_object_arguments_use_the_string_fallback(raw):
    from rapid_mlx.utils.deepseek_v4_flash import encode_arguments_to_dsml

    rendered = encode_arguments_to_dsml({"name": "f", "arguments": raw})
    assert rendered == (
        f'<｜DSML｜parameter name="arguments" string="true">{raw}</｜DSML｜parameter>'
    )


def test_encoder_additional_reference_features():
    from rapid_mlx.utils.deepseek_v4_flash import (
        REASONING_EFFORT_MAX,
        encode_arguments_to_dsml,
        render_message,
        to_json,
    )

    assert 'string="true">bad json' in encode_arguments_to_dsml(
        {"arguments": "bad json"}
    )
    with pytest.raises(TypeError):
        to_json(object())
    system = [{"role": "system", "content": "x", "response_format": {"type": "object"}}]
    assert REASONING_EFFORT_MAX in encode_messages(
        system, "thinking", reasoning_effort="max"
    )
    assert '"type": "object"' in encode_messages(system, "chat")
    developer = [
        {"role": "developer", "content": "x", "response_format": {"type": "object"}}
    ]
    assert '"type": "object"' in encode_messages(developer, "chat")
    blocks = [
        {
            "role": "user",
            "content_blocks": [
                {
                    "type": "tool_result",
                    "content": [{"type": "text", "text": "ok"}, {"type": "image"}],
                },
                {"type": "other"},
            ],
        }
    ]
    rendered = render_message(0, blocks, "chat")
    assert "<tool_result>ok\n\n[Unsupported image]</tool_result>" in rendered
    assert "[Unsupported other]" in rendered
    assert "plain" in render_message(0, [{"role": "user", "content": "plain"}], "chat")
    with pytest.raises(NotImplementedError):
        render_message(0, [{"role": "tool"}], "chat")
    with pytest.raises(NotImplementedError):
        render_message(0, [{"role": "unknown"}], "chat")
    assert not encode_messages(
        [{"role": "assistant", "content": "partial", "wo_eos": True}], "chat"
    ).endswith("<｜end▁of▁sentence｜>")
    assert "<｜query｜>" in encode_messages(
        [{"role": "user", "content": "x", "task": "query"}], "chat"
    )
    assert "<｜User｜>next" in encode_messages(
        [{"role": "user", "content": "next"}],
        "chat",
        context=[{"role": "user", "content": "before"}],
    )


def test_parser_preserves_plain_text_and_malformed_json_value():
    parser = ToolParserManager.get_tool_parser("deepseek_v4_flash")(None)
    plain = parser.extract_tool_calls("plain response")
    assert not plain.tools_called and plain.content == "plain response"
    wire = (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="f">'
        '<｜DSML｜parameter name="value" string="false">invalid</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜tool_calls>"
    )
    parsed = parser.extract_tool_calls(wire)
    assert json.loads(parsed.tool_calls[0]["arguments"]) == {"value": "invalid"}
