from types import SimpleNamespace

import pytest

from rapid_mlx.api.constants import REASONING_CUTOFF_SENTINEL
from rapid_mlx.config import ServerConfig
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.reasoning.deepseek_v4_parser import DeepSeekV4ReasoningParser
from rapid_mlx.service.helpers import (
    _apply_reasoning_cutoff_notice,
    _uses_deepseek_v4_reasoning,
)
from rapid_mlx.service.postprocessor import StreamingPostProcessor


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


def test_truncated_deepseek_reasoning_notice_never_copies_reasoning_tail() -> None:
    content = _apply_reasoning_cutoff_notice(
        None,
        "private scratch that must remain reasoning-only",
        None,
        "length",
        include_reasoning_tail=False,
    )

    assert content == REASONING_CUTOFF_SENTINEL
    assert "private scratch" not in content


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
