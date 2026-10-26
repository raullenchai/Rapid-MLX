"""Retained native adapters keep their EOF and failure contracts after selection."""

from types import SimpleNamespace

import pytest

from rapid_mlx import reasoning
from rapid_mlx.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser
from tests.parser_migration import harness


def parser_fixture(monkeypatch, parser):
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "0")
    original = reasoning.get_parser

    def resolve(name):
        return parser if name == "deepseek_r1" else original(name)

    monkeypatch.setattr(reasoning, "get_parser", resolve)
    monkeypatch.setattr(harness, "get_parser", resolve)


@pytest.mark.parametrize("surface", ["responses", "messages"])
def test_retained_parser_terminal_exception_never_fabricates_answer(
    monkeypatch, surface
):
    class Parser(DeepSeekR1ReasoningParser):
        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            if delta_text == "</think>":
                raise RuntimeError("terminal parser failure")
            return SimpleNamespace(reasoning=delta_text, content=None)

    parser_fixture(monkeypatch, Parser)
    fields = (
        {"thinking": {"type": "enabled", "budget_tokens": 1}}
        if surface == "messages"
        else {"reasoning_max_tokens": 1}
    )
    result = harness.route(
        "deepseek_r1",
        None,
        "xxxx",
        surface,
        True,
        chunks=["xxxx"],
        finish="length",
        template="{% if add_generation_prompt %}<think>{% endif %}",
        tools_mode="absent",
        request_fields=fields,
        capture=True,
    )
    assert not result["content"] and "terminal parser failure" not in str(result)
    assert result["reasoning"] == "xxxx"


def test_retained_parser_finalization_cannot_duplicate_private_tail(monkeypatch):
    class Parser(DeepSeekR1ReasoningParser):
        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            return SimpleNamespace(reasoning=delta_text, content=None)

        def finalize_streaming(self, text, **kwargs):
            return SimpleNamespace(reasoning=None, content=text)

    parser_fixture(monkeypatch, Parser)
    result = harness.route(
        "deepseek_r1",
        None,
        "private",
        "messages",
        True,
        chunks=["private"],
        finish="length",
        template="{% if add_generation_prompt %}<think>{% endif %}",
        tools_mode="absent",
    )
    assert result["reasoning"] == "private" and not result["content"]


def test_retained_parser_final_reasoning_survives_control_only_stream(monkeypatch):
    class Parser(DeepSeekR1ReasoningParser):
        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            return SimpleNamespace(reasoning="<|im_start|>", content=None)

        def finalize_streaming(self, text, **kwargs):
            return SimpleNamespace(reasoning="private", content=None)

    parser_fixture(monkeypatch, Parser)
    result = harness.route(
        "deepseek_r1",
        None,
        "raw",
        "responses",
        True,
        chunks=["raw"],
        finish="length",
        tools_mode="absent",
    )
    assert result["reasoning"] == "private" and not result["content"]


@pytest.mark.parametrize(
    "surface,parser_name",
    [("responses", None), ("responses", "deepseek_r1"), ("messages", "deepseek_r1")],
)
def test_retained_filter_terminal_text_uses_original_channel(
    monkeypatch, surface, parser_name
):
    class Filter:
        def __init__(self):
            self.pending = ""

        def process(self, text):
            self.pending += text
            return ""

        def close_after_stream(self):
            pass

        def flush(self):
            text, self.pending = self.pending, ""
            return text

    from rapid_mlx.tool_parsers import streaming as module

    monkeypatch.setattr(module, "streaming_output_filter", lambda *a, **k: Filter())
    raw = "<think>private</think>Answer"
    result = harness.route(
        parser_name, None, raw, surface, True, chunks=[raw], tools_mode="absent"
    )
    assert result["content"] == "Answer" and result["reasoning"] == "private"


def test_retained_buffered_gate_parses_only_post_reasoning_text(monkeypatch):
    chat = harness.chat
    observed = []
    original = chat._parse_tool_calls_with_parser

    def parse(text, *args, **kwargs):
        observed.append(text)
        return original(text, *args, **kwargs)

    async def gate(*args, **kwargs):
        return SimpleNamespace(reasoning_gate_id=123, _line1_seed_prefix="prompt")

    monkeypatch.setattr(chat, "_offload_tool_grammar_build", gate)
    monkeypatch.setattr(chat, "_line1_context_room_ok", lambda *a: True)
    monkeypatch.setattr(
        chat, "_build_reasoning_budget_processor", lambda *a, **k: object()
    )
    monkeypatch.setattr(chat, "_parse_tool_calls_with_parser", parse)
    result = harness.route(
        "deepseek_r1", "qwen3_coder_xml", "<think>private</think>Answer", "chat", False
    )
    assert observed == ["Answer"]
    assert result["content"] == "Answer" and result["reasoning"] == "private"


def test_retained_buffered_native_finalizer_wire_is_scrubbed(monkeypatch):
    from rapid_mlx.engine.base import GenerationOutput

    async def native(self, **kwargs):
        return GenerationOutput(
            text="Answer",
            raw_text="Answer",
            tool_calls=[
                {"name": "run_shell_command", "arguments": '{"command":"pwd"}'}
            ],
        )

    monkeypatch.setattr(harness.Engine, "chat", native)
    monkeypatch.setattr(
        harness.chat,
        "_finalize_content_and_reasoning",
        lambda *a, **k: (
            '<tool_call>{"name":"unknown","arguments":{}}</tool_call>Answer',
            None,
        ),
    )
    result = harness.route(
        "deepseek_r1",
        "qwen3_coder_xml",
        "Answer",
        "chat",
        False,
        tool_choice="required",
    )
    assert result["content"] == "Answer"
    assert result["calls"] == [["run_shell_command", {"command": "pwd"}]]


@pytest.mark.parametrize("release", ["filter", "finalizer"])
def test_retained_responses_adapter_publishes_sanitizer_released_content(
    monkeypatch, release
):
    """The legacy sanitizer can release an earlier overflow span on a later feed."""

    class Sanitizer:
        def process(self, text, destination):
            return [("content", "Answer")] if text == "Answer" else []

        def flush(self):
            return []

    monkeypatch.setattr(harness.responses, "StreamingReasoningSanitizer", Sanitizer)
    if release == "finalizer":

        class Parser(DeepSeekR1ReasoningParser):
            def extract_reasoning_streaming(
                self, previous_text, current_text, delta_text
            ):
                return SimpleNamespace(reasoning="<|im_start|>", content=None)

            def finalize_streaming(self, text, **kwargs):
                return SimpleNamespace(reasoning="Answer", content=None)

        parser_fixture(monkeypatch, Parser)
        result = harness.route(
            "deepseek_r1",
            None,
            "Answer",
            "responses",
            True,
            chunks=["Answer"],
            finish="length",
            tools_mode="absent",
        )
    else:
        from rapid_mlx.tool_parsers import streaming

        class Filter:
            def __init__(self):
                self.pending = ""

            def process(self, text):
                if text.startswith("<think>"):
                    self.pending = text
                    return ""
                return text

            def flush(self):
                text, self.pending = self.pending, ""
                return text

        monkeypatch.setattr(
            streaming, "streaming_output_filter", lambda *a, **k: Filter()
        )
        result = harness.route(
            None,
            None,
            "<think>Answer</think>",
            "responses",
            True,
            chunks=["<think>Answer</think>"],
            tools_mode="absent",
        )
    assert result["content"] == "Answer" and not result["reasoning"]
