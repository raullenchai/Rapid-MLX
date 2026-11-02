import json
from types import SimpleNamespace

import pytest

from rapid_mlx.config import ServerConfig
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.reasoning.deepseek_v4_parser import DeepSeekV4ReasoningParser
from rapid_mlx.service.helpers import (
    _uses_deepseek_v4_reasoning,
)
from rapid_mlx.service.postprocessor import StreamingPostProcessor
from rapid_mlx.tool_parsers.deepseek_v4_0731_tool_parser import (
    DeepSeekV40731ToolParser,
)


def _literal_think_tool_wire(alias: str) -> str:
    return (
        f"<｜DSML｜{alias}tool_calls>"
        f'<｜DSML｜{alias}invoke name="write_file">'
        f'<｜DSML｜{alias}parameter name="content" string="true">'
        "A<think>B</think>C</｜DSML｜parameter>"
        "</｜DSML｜invoke></｜DSML｜tool_calls>"
    )


@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("alias", ["", "r:"])
@pytest.mark.parametrize("chunk_size", [None, 1000, 1, 7])
def test_dsml_literal_think_arguments(enable_thinking, alias, chunk_size) -> None:
    parser = DeepSeekV4ReasoningParser()
    wire = _literal_think_tool_wire(alias)
    text = "<think>plan</think>" + wire + "<think>after</think>answer"
    if chunk_size is None:
        reasoning, content = parser.extract_reasoning(
            text, enable_thinking=enable_thinking
        )
    else:
        parser.configure_request(enable_thinking=enable_thinking)
        deltas = []
        previous = ""
        for offset in range(0, len(text), chunk_size):
            chunk = text[offset : offset + chunk_size]
            delta = parser.extract_reasoning_streaming(
                previous, previous + chunk, chunk
            )
            if delta:
                deltas.append(delta)
            previous += chunk
        tail = parser.finalize_streaming(text)
        if tail:
            deltas.append(tail)
        reasoning = "".join(delta.reasoning or "" for delta in deltas) or None
        content = "".join(delta.content or "" for delta in deltas)

    assert content == wire + "answer"
    assert reasoning == ("planafter" if enable_thinking else None)
    result = DeepSeekV40731ToolParser(None).extract_tool_calls(content)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "write_file"
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "content": "A<think>B</think>C"
    }


@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("alias", ["", "r:"])
@pytest.mark.parametrize("chunk_size", [1000, 1, 7])
def test_postprocessor_dsml_literal_think_arguments(
    enable_thinking, alias, chunk_size
) -> None:
    cfg = ServerConfig()
    cfg.reasoning_parser_name = "deepseek_v4"
    cfg.tool_parser_instance = DeepSeekV40731ToolParser(None)
    processor = StreamingPostProcessor(
        cfg, enable_thinking=enable_thinking, tools_requested=True
    )
    processor.reset()
    text = "<think>plan</think>" + _literal_think_tool_wire(alias)
    events = []
    for offset in range(0, len(text), chunk_size):
        chunk = text[offset : offset + chunk_size]
        events.extend(
            processor.process_chunk(
                GenerationOutput(
                    text=text[: offset + len(chunk)],
                    new_text=chunk,
                    tokens=[1],
                    finished=False,
                )
            )
        )
    events.extend(processor.finalize())

    assert processor.accumulated_reasoning == ("plan" if enable_thinking else "")
    calls = [
        call
        for event in events
        if event.type == "tool_call"
        for call in event.tool_calls
    ]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "content": "A<think>B</think>C"
    }


def test_chat_mode_absorbs_bare_think_close_after_content() -> None:
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=False)

    first = parser.extract_reasoning_streaming("", "ruff passed", "ruff passed")
    close = parser.extract_reasoning_streaming(
        "ruff passed", "ruff passed\n</think>", "\n</think>"
    )

    assert first is not None and first.content == "ruff passed"
    assert close is not None and close.content == "\n"
    assert "</think>" not in close.content


def test_split_bare_close_never_leaks_to_content() -> None:
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=False)

    chunks = ["done<", "/thi", "nk>next"]
    emitted = []
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        result = parser.extract_reasoning_streaming(previous, current, chunk)
        if result and result.content:
            emitted.append(result.content)
        previous = current

    assert "".join(emitted) == "donenext"


def test_disabled_thinking_absorbs_opener_without_creating_reasoning() -> None:
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=False)

    parsed = parser.extract_reasoning_streaming(
        "", "before<think>inside</think>after", "before<think>inside</think>after"
    )

    assert parsed is not None
    assert parsed.reasoning is None
    assert parsed.content == "beforeafter"


def test_thinking_mode_routes_reasoning_then_content() -> None:
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=True)

    thought = parser.extract_reasoning_streaming("", "inspect", "inspect")
    answer = parser.extract_reasoning_streaming(
        "inspect", "inspect</think>fixed", "</think>fixed"
    )

    assert thought is not None and thought.reasoning == "inspect"
    assert answer is not None and answer.content == "fixed"
    assert answer.reasoning is None


def test_thinking_mode_keeps_implicit_reasoning_across_many_chunks() -> None:
    """DeepSeek V4 omits the opener; no heuristic may flip it to content."""
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=True)

    chunks = [
        "We need answer. Need solve classic chickens rabbits. Need show",
        " reasoning. Need be careful: chickens 2 legs, rabbits 4 legs.",
        " c+r=35, 2c+4r=94.",
        "</think>There are 23 chickens and 12 rabbits.",
    ]
    reasoning = []
    content = []
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        parsed = parser.extract_reasoning_streaming(previous, current, chunk)
        if parsed is not None:
            reasoning.append(parsed.reasoning or "")
            content.append(parsed.content or "")
        previous = current

    assert "Need be careful" in "".join(reasoning)
    assert "Need be careful" not in "".join(content)
    assert "".join(content) == "There are 23 chickens and 12 rabbits."


def test_unspecified_thinking_defaults_to_implicit_reasoning_streaming() -> None:
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=None)

    thought = parser.extract_reasoning_streaming(
        "", "private scratch", "private scratch"
    )
    answer = parser.extract_reasoning_streaming(
        "private scratch",
        "private scratch</think>public answer",
        "</think>public answer",
    )

    assert thought is not None and thought.reasoning == "private scratch"
    assert thought.content is None
    assert answer is not None and answer.content == "public answer"


def test_unspecified_thinking_defaults_to_implicit_reasoning_nonstreaming() -> None:
    parser = DeepSeekV4ReasoningParser()

    reasoning, content = parser.extract_reasoning(
        "private scratch</think>public answer", enable_thinking=None
    )

    assert reasoning == "private scratch"
    assert content == "public answer"


def test_dsml_tool_start_implicitly_closes_reasoning() -> None:
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=True)

    parsed = parser.extract_reasoning_streaming(
        "",
        "plan<｜DSML｜tool_calls><｜DSML｜invoke",
        "plan<｜DSML｜tool_calls><｜DSML｜invoke",
    )

    assert parsed is not None
    assert parsed.reasoning == "plan"
    assert parsed.content == "<｜DSML｜tool_calls><｜DSML｜invoke"


def test_dsml_sampled_r_tool_start_implicitly_closes_reasoning() -> None:
    parser = DeepSeekV4ReasoningParser()
    parser.configure_request(enable_thinking=True)

    parsed = parser.extract_reasoning_streaming(
        "",
        "plan<｜DSML｜r:tool_calls><｜DSML｜r:invoke",
        "plan<｜DSML｜r:tool_calls><｜DSML｜r:invoke",
    )

    assert parsed is not None
    assert parsed.reasoning == "plan"
    assert parsed.content == "<｜DSML｜r:tool_calls><｜DSML｜r:invoke"


def test_nonstream_chat_mode_absorbs_control_tokens() -> None:
    parser = DeepSeekV4ReasoningParser()
    reasoning, content = parser.extract_reasoning(
        "first</think>second", enable_thinking=False
    )

    assert reasoning is None
    assert content == "firstsecond"


def test_chat_postprocessor_keeps_sanitizer_active_when_thinking_disabled() -> None:
    cfg = ServerConfig()
    cfg.reasoning_parser_name = "deepseek_v4"
    processor = StreamingPostProcessor(cfg, enable_thinking=False)
    processor.reset()

    visible = []
    for text in ("ruff passed\n<", "/think>", "next"):
        events = processor.process_chunk(
            GenerationOutput(text=text, new_text=text, tokens=[1], finished=False)
        )
        visible.extend(event.content for event in events if event.type == "content")

    assert "".join(visible) == "ruff passed\nnext"
    assert "</think>" not in "".join(visible)


def test_chat_postprocessor_fails_closed_on_request_config_error(monkeypatch) -> None:
    class BrokenParser:
        def configure_request(self, *, enable_thinking=None):
            raise RuntimeError("cannot establish reasoning state")

    cfg = ServerConfig()
    cfg.reasoning_parser_name = "deepseek_v4"
    monkeypatch.setattr(
        StreamingPostProcessor,
        "_create_reasoning_parser",
        lambda self, _cfg: BrokenParser(),
    )

    with pytest.raises(RuntimeError, match="cannot establish reasoning state"):
        StreamingPostProcessor(cfg, enable_thinking=None)


def test_runtime_deepseek_detection_covers_auto_config_forms() -> None:
    empty_cfg = SimpleNamespace(
        reasoning_parser_name=None,
        reasoning_parser=None,
        model_path="/models/DeepSeek-V4-Flash-0731-MXFP4-MLX",
        model_name=None,
    )
    assert _uses_deepseek_v4_reasoning(empty_cfg)

    generic_cfg = SimpleNamespace(
        reasoning_parser_name=None,
        reasoning_parser=None,
        model_path="/models/other",
        model_name=None,
    )
    assert _uses_deepseek_v4_reasoning(generic_cfg, DeepSeekV4ReasoningParser())
