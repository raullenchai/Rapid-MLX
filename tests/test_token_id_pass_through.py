# SPDX-License-Identifier: Apache-2.0
"""Optional token identity crosses only aligned parser handoffs."""

from types import SimpleNamespace

import pytest

from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.parsing.token_ids import extract_reasoning, extract_tool_calls
from rapid_mlx.reasoning.base import DeltaMessage
from rapid_mlx.service.postprocessor import StreamingPostProcessor


def _processor(**kwargs):
    return StreamingPostProcessor(
        SimpleNamespace(
            engine=None,
            reasoning_parser=None,
            reasoning_parser_name=None,
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            tool_parser_instance=None,
        ),
        **kwargs,
    )


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
    assert extract_reasoning(parser, "answer", token_ids=[3]) == (None, "answer")
    assert extract_reasoning(parser, "answer", token_ids=[]) == (None, "answer")
    assert extract_reasoning(parser, "answer") == (None, "answer")
    assert parser.calls == [
        ("tokens", "answer", [3]),
        ("legacy", "answer"),
        ("legacy", "answer"),
    ]
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
    assert extract_tool_calls(parser, "wire", {}, token_ids=[]) == (
        "legacy",
        "wire",
        {},
    )


class _Reasoning:
    def __init__(self):
        self.seen = []

    def extract_reasoning_streaming_with_tokens(
        self, previous, current, delta, *, delta_token_ids
    ):
        assert bytes(delta_token_ids).decode() == delta
        self.seen.append((delta, tuple(delta_token_ids)))
        return DeltaMessage(content=delta, content_token_ids=tuple(delta_token_ids))

    def extract_reasoning_streaming(self, previous, current, delta):
        self.seen.append((delta, None))
        return DeltaMessage(content=delta)


def test_postprocessor_buffers_empty_chunk_ids_and_forwards_aligned_content():
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

    pp = _processor()
    pp.reasoning_parser, pp.tool_parser = _Reasoning(), Tool()
    pp.enable_thinking = True
    pp.process_chunk(
        GenerationOutput(text="", new_text="", finished=False, tokens=[195])
    )
    pp.process_chunk(
        GenerationOutput(
            text="é",
            new_text="é",
            finished=False,
            tokens=[169],
        )
    )
    assert pp.reasoning_parser.seen == [("é", (195, 169))]
    assert pp.tool_parser.seen == [(195, 169)]


@pytest.mark.parametrize("case", ["held_prefix", "stop_trimmed", "stop_buffered"])
def test_rewritten_or_stop_buffered_deltas_withhold_ids(case):
    pp = _processor(request={"stop": ["STOP"]} if case != "held_prefix" else None)
    pp.reasoning_parser = _Reasoning()
    pp.enable_thinking = case != "held_prefix"
    if case == "held_prefix":
        pp.process_chunk(
            GenerationOutput(text="<", new_text="<", tokens=list(b"<"), finished=False)
        )
        delta, ids, expected = "think>hello", list(b"think>hello"), "<think>hello"
    else:
        delta, ids, expected = "answer", list(b"answerSTOP"), "answer"
    pp.process_chunk(
        GenerationOutput(
            text=expected,
            new_text=delta,
            tokens=ids,
            finished=case == "stop_trimmed",
            matched_stop="STOP" if case == "stop_trimmed" else None,
        )
    )
    assert pp.reasoning_parser.seen == [(expected, None)]


def test_synthetic_reasoning_close_uses_legacy_parser_without_fake_ids():
    pp = _processor()
    pp.reasoning_parser = _Reasoning()
    pp._reasoning_cap_hit = True
    output = GenerationOutput(
        text="answer", new_text="answer", finished=False, tokens=list(b"answer")
    )

    pp._process_with_reasoning("answer", output, token_ids=output.tokens)

    assert pp.reasoning_parser.seen == [("</think>answer", None)]
