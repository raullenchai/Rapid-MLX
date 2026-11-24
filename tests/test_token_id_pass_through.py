# SPDX-License-Identifier: Apache-2.0
"""Optional token identity crosses only aligned parser handoffs."""

from types import SimpleNamespace

from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.parsing.token_ids import extract_reasoning, extract_tool_calls
from rapid_mlx.reasoning.base import DeltaMessage
from rapid_mlx.service.postprocessor import StreamingPostProcessor


def test_complete_dispatch_uses_ids_only_for_explicit_opt_in():
    class Parser:
        def __init__(self):
            self.calls = []

        def extract_reasoning(self, text):
            self.calls.append(("legacy", text))
            return None, text

        def extract_reasoning_with_tokens(self, text, *, token_ids):
            self.calls.append(("tokens", text, token_ids))
            return None, text

    parser = Parser()
    assert extract_reasoning(parser, "answer", token_ids=[]) == (None, "answer")
    assert extract_reasoning(parser, "answer") == (None, "answer")
    assert parser.calls == [("tokens", "answer", []), ("legacy", "answer")]
    assert DeltaMessage(content="plain").content_token_ids is None


def test_complete_tool_dispatch_preserves_legacy_fallback():
    class Parser:
        def extract_tool_calls(self, text, request):
            return ("legacy", text, request)

        def extract_tool_calls_with_tokens(self, text, request, *, token_ids):
            return ("tokens", text, request, token_ids)

    parser = Parser()
    assert extract_tool_calls(parser, "wire", {}, token_ids=[3]) == (
        "tokens",
        "wire",
        {},
        [3],
    )
    assert extract_tool_calls(parser, "wire", {}) == ("legacy", "wire", {})


def test_postprocessor_buffers_empty_chunk_ids_and_forwards_aligned_content():
    class Reasoning:
        def __init__(self):
            self.seen = []

        def extract_reasoning_streaming_with_tokens(
            self, previous, current, delta, *, delta_token_ids
        ):
            self.seen.append(delta_token_ids)
            return DeltaMessage(content=delta, content_token_ids=(51, 52))

        def extract_reasoning_streaming(self, previous, current, delta):
            self.seen.append(None)
            return DeltaMessage(content=delta)

    class Tool:
        def __init__(self):
            self.seen = []

        def extract_tool_calls_streaming_with_tokens(
            self, previous, current, delta, *, request=None, delta_token_ids
        ):
            self.seen.append(delta_token_ids)
            return None

        def extract_tool_calls_streaming(
            self, previous, current, delta, *, request=None
        ):
            self.seen.append(None)
            return None

    cfg = SimpleNamespace(
        engine=None,
        reasoning_parser=None,
        reasoning_parser_name=None,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        tool_parser_instance=None,
    )
    pp = StreamingPostProcessor(cfg)
    pp.reasoning_parser, pp.tool_parser = Reasoning(), Tool()
    pp.enable_thinking = True
    pp.process_chunk(GenerationOutput(text="", new_text="", finished=False, tokens=[5]))
    pp.process_chunk(
        GenerationOutput(
            text="<|end_message|>",
            new_text="<|end_message|>",
            finished=False,
            tokens=[7],
        )
    )
    assert pp.reasoning_parser.seen == [[5, 7]]
    assert pp.tool_parser.seen == [(51, 52)]


def test_synthetic_reasoning_close_uses_legacy_parser_without_fake_ids():
    class Parser:
        def __init__(self):
            self.seen = []

        def extract_reasoning_streaming(self, previous, current, delta):
            self.seen.append(("legacy", delta))
            return DeltaMessage(content=delta)

        def extract_reasoning_streaming_with_tokens(
            self, previous, current, delta, *, delta_token_ids
        ):
            self.seen.append(("tokens", delta, delta_token_ids))
            return DeltaMessage(content=delta)

    cfg = SimpleNamespace(
        engine=None,
        reasoning_parser=None,
        reasoning_parser_name=None,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        tool_parser_instance=None,
    )
    pp = StreamingPostProcessor(cfg)
    pp.reasoning_parser = Parser()
    pp._reasoning_cap_hit = True
    output = GenerationOutput(
        text="answer", new_text="answer", finished=False, tokens=[8]
    )

    pp._process_with_reasoning("answer", output, token_ids=output.tokens)

    assert pp.reasoning_parser.seen == [("legacy", "</think>answer")]
