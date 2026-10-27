"""Lifecycle and ownership boundaries for the existing demultiplexer adapter."""

from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.parsing.prompt_owner import (
    ContentDelta,
    PromptOutputOwner,
    ReasoningDelta,
    split_buffered,
)
from rapid_mlx.prompt_parse_context import PromptParseContext


def output(text="", suffix="", contract="flint", finished=False, channel=None):
    return GenerationOutput(
        text=text,
        new_text=text,
        finished=finished,
        channel=channel,
        prompt_parse_context=PromptParseContext(suffix, None, None, False, contract),
    )


def test_last_prompt_marker_wins_and_finish_is_idempotent():
    owner = PromptOutputOwner("lfm2.5")
    events = owner.observe(output("private", "<think>old</think><think>", "chatml"))
    assert [e.text for e in events if isinstance(e, ReasoningDelta)] == ["private"]
    owner.observe(output("</think>answer<thi"))
    events = owner.finish()
    assert [e.text for e in events if isinstance(e, ContentDelta)] == ["<thi"]
    assert "".join(owner.content) == "answer<thi"
    assert owner.finish() == []


def test_unselected_and_native_requests_are_untouched():
    from types import SimpleNamespace

    owner = PromptOutputOwner("qwen3.5")
    assert owner.observe(output("answer", contract="chatml")) is None
    assert owner.finish() == []
    owner = PromptOutputOwner("flint")
    assert owner.observe(output("answer", channel="content")) is None
    assert not owner.active
    owner = PromptOutputOwner("flint")
    assert owner.observe(GenerationOutput(text="answer")) is None
    original = GenerationOutput(text="answer")
    assert split_buffered(original, "qwen3.5") == (original, False)
    duck_output = SimpleNamespace(text="answer")
    assert split_buffered(duck_output, "qwen3.5") == (duck_output, False)


def test_literal_tool_markers_in_reasoning_and_stray_tool_end():
    owner = PromptOutputOwner("flint")
    owner.observe(
        output(
            "<think><tool_call>example</tool_call></think></tool_call>", finished=True
        )
    )
    assert "".join(owner.reasoning) == "<tool_call>example</tool_call>"
    assert "".join(owner.content) == "</tool_call>"
    assert owner.finish() == []


def test_request_instances_do_not_share_state():
    first, second = PromptOutputOwner("flint"), PromptOutputOwner("flint")
    first.observe(output("<think>private"))
    second.observe(output("public", finished=True))
    assert second.reasoning == []
    assert second.content == ["public"]
    assert first.content == []


def test_postprocessor_prefix_replay_and_reset():
    from rapid_mlx.config import reset_config
    from rapid_mlx.reasoning import get_parser
    from rapid_mlx.service.postprocessor import StreamingPostProcessor

    cfg = reset_config()
    cfg.model_name = "flint"
    cfg.tool_call_parser = "glm47"
    cfg.enable_auto_tool_choice = True
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = get_parser("qwen3")()
    processor = StreamingPostProcessor(
        cfg,
        tools_requested=True,
        request={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_shell_command",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        },
    )
    prefix = "<tool_call>run_shell_command"
    legacy_reasoning = processor.reasoning_parser
    processor.seed_forced_assistant_prefix(prefix)
    processor.process_chunk(output(prefix))
    events = processor.process_chunk(
        output(
            "<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>",
            finished=True,
        )
    )
    assert sum(e.type == "tool_call" for e in events) == 1
    assert processor._should_route_through_reasoning()
    processor.finalize()
    processor.reset()
    assert not processor.channels_owned
    assert processor.reasoning_parser is legacy_reasoning
    assert not processor.tool_parser.skip_reasoning_parsing
    assert processor._prompt_owner.finish() == []


def test_structured_buffered_output_is_already_owned():
    original = output("Narration.")
    original.tool_calls = [{"name": "run_shell_command", "arguments": "{}"}]
    assert split_buffered(original, "flint") == (original, False)
    owner = PromptOutputOwner("flint")
    assert owner.observe(original) is None
    assert not owner.active


def test_initial_structured_event_retains_prompt_context():
    owner = PromptOutputOwner("lfm2.5")
    native = output("{}", "<think>", "chatml", channel="tool_call")
    native.tool_calls = [{"name": "run_shell_command", "arguments": "{}"}]
    assert owner.observe(native) is None
    assert not owner.active
    owner.observe(
        GenerationOutput(
            text="",
            new_text="private</think>Answer.",
            finished=True,
            finish_reason="stop",
        )
    )
    assert "".join(owner.reasoning) == "private"
    assert "".join(owner.content) == "Answer."
