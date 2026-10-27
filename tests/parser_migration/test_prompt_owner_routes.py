"""Actual FastAPI streaming surfaces share rendered prompt ownership."""

import pytest

from rapid_mlx.prompt_parse_context import PromptParseContext

from .harness import route


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize(
    "model,contract,initial",
    [
        ("flint-40b-a2.5b-4bit", "flint", "content"),
        ("lfm2.5-24b-a2b-4bit", "chatml", "reasoning"),
    ],
)
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("split", [1, 7, 1000])
@pytest.mark.parametrize("stream", [False, True])
def test_prompt_channels(surface, model, contract, initial, thinking, split, stream):
    raw = (
        "<think>" if initial == "content" else ""
    ) + "private plan</think>Answer one. Answer two."
    context = PromptParseContext(
        "<think>" if initial == "reasoning" else "", None, initial, thinking, contract
    )
    result = route(
        "qwen3",
        "glm47" if contract == "flint" else "lfm",
        raw,
        surface,
        stream,
        chunks=[raw[i : i + split] for i in range(0, len(raw), split)],
        thinking=thinking,
        model=model,
        tools_mode="absent",
        prompt_context=context,
    )
    assert result["content"] == "Answer one. Answer two.", result
    assert result["reasoning"] == "private plan", result
    if stream and split < 1000:
        assert result["content_deltas"] > 1, result
        assert result["first_content_index"] < result["input_chunks"], result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split", [1, 1000])
def test_tool_looking_reasoning_stays_private(surface, stream, split):
    hidden = "<tool_call>run_shell_command<arg_key>command</arg_key><arg_value>secret</arg_value></tool_call>"
    call = "<tool_call>run_shell_command<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"
    raw = "<think>" + hidden + "</think>" + call
    context = PromptParseContext("", None, "content", False, "flint")
    result = route(
        "qwen3",
        "glm47",
        raw,
        surface,
        stream,
        chunks=[raw[i : i + split] for i in range(0, len(raw), split)],
        thinking=False,
        model="flint-40b-a2.5b-4bit",
        prompt_context=context,
    )
    assert result["calls"] == [["run_shell_command", {"command": "pwd"}]], result
    assert result["reasoning"] == hidden, result
    assert "secret" not in result["content"], result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish", ["length", "stop"])
def test_reasoning_only_never_rescued(surface, stream, finish):
    raw = "<think>private unfinished plan"
    context = PromptParseContext("", None, "content", False, "flint")
    result = route(
        "qwen3",
        "glm47",
        raw,
        surface,
        stream,
        thinking=False,
        finish=finish,
        model="flint-40b-a2.5b-4bit",
        tools_mode="absent",
        prompt_context=context,
    )
    assert "private" not in result.get("content", ""), result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("split", [1, 1000])
def test_plain_flint_streams_immediately(surface, split):
    raw = "Hello world. A second sentence."
    context = PromptParseContext("", None, "content", True, "flint")
    result = route(
        "qwen3",
        "glm47",
        raw,
        surface,
        True,
        thinking=True,
        chunks=[raw[i : i + split] for i in range(0, len(raw), split)],
        model="flint-40b-a2.5b-4bit",
        tools_mode="absent",
        prompt_context=context,
    )
    assert result["content"] == raw, result
    assert result["reasoning"] == "", result
    assert result["first_content_index"] == 1, result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_think_markers_in_arguments_are_tool_data(surface, stream):
    raw = '<think>plan</think><tool_call>run_shell_command<arg_key>command</arg_key><arg_value>echo "<think>literal</think>"</arg_value></tool_call>'
    context = PromptParseContext("", None, "content", False, "flint")
    result = route(
        "qwen3",
        "glm47",
        raw,
        surface,
        stream,
        thinking=False,
        model="flint-40b-a2.5b-4bit",
        prompt_context=context,
    )
    assert result["calls"] == [
        ["run_shell_command", {"command": 'echo "<think>literal</think>"'}]
    ], result
    assert result["reasoning"] == "plan", result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("raw", ["Hello <thi", "Hello </thi", "<think>private<thi"])
def test_partial_marker_drains_once(surface, terminal, raw):
    context = PromptParseContext("", None, "content", False, "flint")
    result = route(
        "qwen3",
        "glm47",
        raw,
        surface,
        True,
        thinking=False,
        model="flint-40b-a2.5b-4bit",
        tools_mode="absent",
        prompt_context=context,
        terminal=terminal,
    )
    if raw.startswith("<think>"):
        assert result["reasoning"] == raw[7:], result
    else:
        assert result["content"] == raw, result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("split", [1, 1000])
def test_lfm_tools_only_from_content(surface, stream, native, split):
    start, end = ("<|tool_call_start|>", "<|tool_call_end|>") if native else ("", "")
    hidden = start + '[run_shell_command(command="secret")]' + end
    raw = hidden + "</think>" + start + '[run_shell_command(command="pwd")]' + end
    context = PromptParseContext("<think>", None, "reasoning", False, "chatml")
    result = route(
        "qwen3",
        "lfm",
        raw,
        surface,
        stream,
        thinking=False,
        model="lfm2.5-24b-a2b-4bit",
        prompt_context=context,
        chunks=[raw[i : i + split] for i in range(0, len(raw), split)],
    )
    assert result["calls"] == [["run_shell_command", {"command": "pwd"}]], result
    assert result["reasoning"] == hidden, result
    assert "secret" not in result["content"], result
    assert result["content"] == "", result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split", [1, 1000])
@pytest.mark.parametrize("tools_mode", ["absent", "enabled"])
def test_stray_closer_is_not_sanitized(surface, stream, split, tools_mode):
    raw = "unmarked deliberation</think>answer"
    context = PromptParseContext("", None, "content", False, "flint")
    result = route(
        "qwen3",
        "glm47",
        raw,
        surface,
        stream,
        thinking=False,
        chunks=[raw[i : i + split] for i in range(0, len(raw), split)],
        model="flint-40b-a2.5b-4bit",
        tools_mode=tools_mode,
        prompt_context=context,
    )
    assert result["content"] == raw, result
    assert result["reasoning"] == "", result


def test_owned_content_survives_logprob_serializer():
    raw = "literal</think>answer"
    context = PromptParseContext("", None, "content", False, "flint")
    result = route(
        "qwen3",
        "glm47",
        raw,
        "chat",
        True,
        thinking=False,
        model="flint-40b-a2.5b-4bit",
        tools_mode="absent",
        prompt_context=context,
        logprobs=True,
    )
    assert result["content"] == raw, result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split", [1, 1000])
def test_lfm_native_tool_arguments_ignore_foreign_closer(surface, stream, split):
    command = "echo </tool_call><think>literal</think>"
    raw = (
        'plan</think><|tool_call_start|>[run_shell_command(command="'
        + command
        + '")]<|tool_call_end|>'
    )
    result = route(
        "qwen3",
        "lfm",
        raw,
        surface,
        stream,
        chunks=[raw[i : i + split] for i in range(0, len(raw), split)],
        thinking=False,
        model="lfm2.5",
        prompt_context=PromptParseContext(
            "<think>", None, "reasoning", False, "chatml"
        ),
    )
    assert result["calls"] == [["run_shell_command", {"command": command}]], result
    assert result["reasoning"] == "plan", result
    assert result["content"] == "", result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_existing_control_cleanup_does_not_strip_owned_closer(surface, stream):
    # Preserve the existing non-reasoning cleanup for complete control tokens.
    raw = "Hello<|im_end|>world</think>literal"
    result = route(
        "qwen3",
        "glm47",
        raw,
        surface,
        stream,
        chunks=[raw],
        thinking=False,
        model="flint",
        tools_mode="absent",
        prompt_context=PromptParseContext("", None, "content", False, "flint"),
    )
    assert result["content"] == "Helloworld</think>literal", result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_foreign_marker_spelling_is_literal_flint_content(surface, stream):
    raw = "Literal <|tool_call_start|> and <|tool_call_end|> spellings."
    result = route(
        "qwen3",
        "glm47",
        raw,
        surface,
        stream,
        thinking=False,
        model="flint",
        tools_mode="absent",
        prompt_context=PromptParseContext("", None, "content", False, "flint"),
    )
    assert result["content"] == raw, result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("channel", [None, "content", "tool_call"])
@pytest.mark.parametrize("model", ["flint", "lfm2.5"])
def test_native_tool_event_after_text_ownership(surface, channel, model, monkeypatch):
    from rapid_mlx.engine.base import GenerationOutput

    from .harness import Engine

    context = PromptParseContext(
        "<think>" if model == "lfm2.5" else "",
        None,
        "reasoning" if model == "lfm2.5" else "content",
        False,
        "chatml" if model == "lfm2.5" else "flint",
    )

    async def mixed_stream(self, **kwargs):
        yield GenerationOutput(
            text="",
            finished=False,
            new_text=("private</think>" if model == "lfm2.5" else "") + "Starting. ",
            prompt_parse_context=context,
        )
        yield GenerationOutput(
            text="", finished=False, new_text="Done.", channel="content"
        )
        yield GenerationOutput(
            text="",
            finished=False,
            new_text="Narration. " if channel != "tool_call" else '{"command":"pwd"}',
            channel=channel,
            tool_calls=[
                {"name": "run_shell_command", "arguments": '{"command":"pwd"}'}
            ],
        )
        yield GenerationOutput(text="", finished=True, finish_reason="tool_calls")

    monkeypatch.setattr(Engine, "stream_chat", mixed_stream)
    result = route(
        "qwen3",
        "lfm" if model == "lfm2.5" else "glm47",
        "",
        surface,
        True,
        model=model,
        prompt_context=context,
        thinking=False,
    )
    assert result["calls"] == [["run_shell_command", {"command": "pwd"}]], result
    expected = "Starting. Done." + ("Narration. " if channel != "tool_call" else "")
    assert result["content"] == expected, result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("model", ["flint", "lfm2.5"])
def test_buffered_native_reasoning_sidecar(surface, model, monkeypatch):
    from rapid_mlx.engine.base import GenerationOutput

    from .harness import Engine

    context = PromptParseContext(
        "<think>", None, "reasoning", False, "flint" if model == "flint" else "chatml"
    )

    async def native_result(self, **kwargs):
        return GenerationOutput(
            text="Answer.",
            reasoning_text="Private plan.",
            prompt_parse_context=context,
            finished=True,
            finish_reason="stop",
        )

    monkeypatch.setattr(Engine, "chat", native_result)
    result = route(
        "qwen3",
        "glm47" if model == "flint" else "lfm",
        "",
        surface,
        False,
        model=model,
        prompt_context=context,
        thinking=True,
        tools_mode="absent",
    )
    assert result["content"] == "Answer.", result
    assert result["reasoning"] == "Private plan.", result


@pytest.mark.parametrize("model", ["flint", "lfm2.5"])
def test_buffered_failed_tool_turn_retains_reasoning(model):
    context = PromptParseContext(
        "<think>" if model == "lfm2.5" else "",
        None,
        "reasoning" if model == "lfm2.5" else "content",
        False,
        "chatml" if model == "lfm2.5" else "flint",
    )
    raw = ("<think>" if model == "flint" else "") + "Private unfinished plan."
    result = route(
        "qwen3",
        "glm47" if model == "flint" else "lfm",
        raw,
        "responses",
        False,
        model=model,
        prompt_context=context,
        thinking=False,
        capture=True,
    )
    assert result["content"] == "", result
    assert result["reasoning"] == "Private unfinished plan.", result
    assert result["calls"] == [], result
    assert result["packets"][0]["status"] == "failed"
    assert result["packets"][0]["error"]["code"] == "model_no_final_answer"


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("model", ["flint", "lfm2.5"])
@pytest.mark.parametrize("narration", ["", "Native. "])
def test_native_first_event_retains_prompt_context(
    surface, model, narration, monkeypatch
):
    from rapid_mlx.engine.base import GenerationOutput

    from .harness import Engine

    context = PromptParseContext(
        "<think>" if model == "lfm2.5" else "",
        None,
        "reasoning" if model == "lfm2.5" else "content",
        False,
        "chatml" if model == "lfm2.5" else "flint",
    )

    async def mixed_stream(self, **kwargs):
        yield GenerationOutput(
            text="",
            new_text=narration,
            finished=False,
            channel="content",
            prompt_parse_context=context,
        )
        yield GenerationOutput(
            text="",
            new_text=("<think>" if model == "flint" else "") + "private</think>Answer.",
            finished=False,
        )
        yield GenerationOutput(text="", finished=True, finish_reason="stop")

    monkeypatch.setattr(Engine, "stream_chat", mixed_stream)
    result = route(
        "qwen3",
        "glm47" if model == "flint" else "lfm",
        "",
        surface,
        True,
        model=model,
        prompt_context=context,
        thinking=False,
        tools_mode="absent",
        capture=True,
    )
    assert result["content"] == narration + "Answer.", result
    if surface == "responses" and narration:
        # Existing Responses ordering rejects reasoning after a visible item.
        # Native narration is still preserved and private bytes never leak.
        failure = result["packets"][-1]
        assert failure["type"] == "response.failed"
        assert failure["response"]["error"]["code"] == "invalid_reasoning_event_order"
    else:
        assert result["reasoning"] == "private", result


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "model,contract,suffix,raw",
    [
        ("flint", "flint", "", "<think>\n</think>\nHello"),
        ("lfm2.5", "chatml", "<think>", "\n</think>\nHello"),
    ],
)
def test_messages_omits_blank_owned_thinking(stream, model, contract, suffix, raw):
    context = PromptParseContext(suffix, None, None, False, contract)
    result = route(
        "qwen3",
        "glm47" if contract == "flint" else "lfm",
        raw,
        "messages",
        stream,
        thinking=False,
        model=model,
        tools_mode="absent",
        prompt_context=context,
    )
    assert result["reasoning"] == ""
    assert result["content"] == "\nHello"


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_native_event_releases_held_marker_prefix(surface, monkeypatch):
    from rapid_mlx.engine.base import GenerationOutput

    from .harness import Engine

    context = PromptParseContext("", None, "content", False, "flint")

    async def interrupted_marker(self, **kwargs):
        yield GenerationOutput(
            text="",
            new_text="Hi <thi",
            finished=False,
            prompt_parse_context=context,
        )
        yield GenerationOutput(
            text="", new_text="Native. ", finished=False, channel="content"
        )
        yield GenerationOutput(text="", new_text="nk>secret", finished=False)
        yield GenerationOutput(text="", finished=True, finish_reason="stop")

    monkeypatch.setattr(Engine, "stream_chat", interrupted_marker)
    result = route(
        "qwen3",
        "glm47",
        "",
        surface,
        True,
        model="flint",
        prompt_context=context,
        thinking=False,
        tools_mode="absent",
        capture=True,
    )
    # The native event cannot complete "<thi" + "nk>", so no reasoning opens
    # and the held prefix is emitted before the native narration.
    assert result["reasoning"] == "", result
    assert result["content"] == "Hi <thiNative. nk>secret", result


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_think_start_is_honored_once(surface, stream):
    # A later opener, here inside an unwrapped call's argument, stays literal.
    context = PromptParseContext("<think>", None, None, False, "chatml")
    raw = 'plan</think>[f(a="<think>x</think>")]'
    result = route(
        "qwen3",
        "lfm",
        raw,
        surface,
        stream,
        thinking=True,
        model="lfm2.5",
        tools_mode="absent",
        prompt_context=context,
    )
    assert result["reasoning"] == "plan", result
    assert result["content"] == '[f(a="<think>x</think>")]', result
