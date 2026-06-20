# SPDX-License-Identifier: Apache-2.0
"""F-100 regression: streaming + ``response_format=json_schema`` on
reasoning models must split ``<think>…</think>`` blocks into
``delta.reasoning_content`` even when the model autonomously re-enters
reasoning by emitting a SECOND ``<think>`` opener AFTER the answer.

Pre-fix repro (phi-4-mini-reasoning-4bit, stream=true, json_schema):

    delta.content = '\n\n{"answer":4}<think>\nOkay let me see…'

The literal ``<think>…</think>`` tags leaked into ``delta.content``
(including SSE-split fragments like ``<th`` / ``ink`` / ``>\n``) instead
of routing the second-block body to ``delta.reasoning_content``. Root
cause: in **implicit mode** (chat template injected ``<think>`` so only
``</think>`` appears in the output), ``_handle_implicit_think``'s
``end_in_prev`` branch returned ``content=delta_text`` unconditionally
— it had no awareness of the multi-block protocol that the explicit
``<think>`` mode (``_handle_explicit_think → multi_block_after_close``)
already implemented.

Fix: in ``extract_reasoning_streaming`` Case 2 (implicit mode), once
``end_in_prev=True`` we delegate to ``_handle_multi_block_after_close``
— the same router that the explicit-mode path uses post-first-close.
The fix also seeds ``_streaming_phase`` in the SSE-boundary recovery
branches of ``_handle_explicit_think`` so a subsequent multi-block call
inherits the correct phase rather than defaulting to "content".

Symmetric with the non-streaming path's ``_sweep_residual_think_tags``
which already handled multi-block output in the non-streaming
``extract_reasoning`` (PR #722). The streaming path now matches that
contract.
"""

from __future__ import annotations

import pytest

from vllm_mlx.reasoning import get_parser


def _stream(parser, chunks: list[str]) -> tuple[str, str]:
    """Replay ``chunks`` through ``parser.extract_reasoning_streaming``.

    Returns ``(reasoning, content)`` as joined strings.
    """
    parser.reset_state()
    accumulated = ""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for delta in chunks:
        previous = accumulated
        accumulated += delta
        msg = parser.extract_reasoning_streaming(previous, accumulated, delta)
        if msg is None:
            continue
        if msg.reasoning:
            reasoning_parts.append(msg.reasoning)
        if msg.content:
            content_parts.append(msg.content)
    return "".join(reasoning_parts), "".join(content_parts)


@pytest.fixture(params=["qwen3", "deepseek_r1"])
def parser(request):
    return get_parser(request.param)()


class TestImplicitModeMultiBlock:
    """Implicit mode (chat template injects ``<think>``) — only
    ``</think>`` appears in the output stream, then the model may emit a
    SECOND autonomous ``<think>…</think>`` block after the answer.
    """

    def test_second_think_after_answer_routes_to_reasoning(self, parser):
        """F-100 core repro: post-first-close, a second ``<think>`` opener
        must route subsequent bytes to reasoning, NOT leak into content.
        """
        # Implicit mode: stream starts inside reasoning, then </think>,
        # then JSON answer, then a SECOND <think> block.
        chunks = [
            "Okay let me think",
            "</think>",
            '{"answer":4}',
            "<think>",
            "double-checking",
            "</think>",
            "tail",
        ]
        reasoning, content = _stream(parser, chunks)
        # First reasoning block + second reasoning block both in reasoning.
        assert "Okay let me think" in reasoning
        assert "double-checking" in reasoning
        # JSON answer + post-second-block tail in content.
        assert '{"answer":4}' in content
        assert "tail" in content
        # No literal tag bytes anywhere on either channel.
        assert "<think>" not in content
        assert "</think>" not in content
        assert "<think>" not in reasoning
        assert "</think>" not in reasoning

    def test_split_second_open_tag_held_and_emitted_as_reasoning(self, parser):
        """SSE-split second opener (``<th`` / ``ink`` / ``>\\n``) must be
        withheld and the body routed to reasoning.
        """
        chunks = [
            "first thought",
            "</think>",
            '{"answer":4}',
            "<th",
            "ink",
            ">\n",
            "second thought",
            "</think>",
            "trailing",
        ]
        reasoning, content = _stream(parser, chunks)
        assert "second thought" in reasoning
        assert "trailing" in content
        # No partial-tag fragments in content.
        assert "<th" not in content
        assert "ink" not in content
        # No structural-tag bytes survive on either side.
        assert "<think>" not in content
        assert "</think>" not in content

    def test_split_second_close_tag_held_until_complete(self, parser):
        """Second-block closer arriving split (``</`` / ``think`` / ``>``)
        must withhold and flip back to content cleanly.
        """
        chunks = [
            "first thought",
            "</think>",
            "answer",
            "<think>",
            "more thought",
            "</",
            "think",
            ">",
            "tail",
        ]
        reasoning, content = _stream(parser, chunks)
        assert "more thought" in reasoning
        assert "tail" in content
        assert "<think>" not in content
        assert "</think>" not in content


class TestExplicitModeMultiBlockRegression:
    """Explicit mode regression guard — keep the pre-F-100 behaviour for
    streams where the model DOES emit the first ``<think>`` opener.
    """

    def test_split_first_open_then_multi_block(self, parser):
        """Tokens: ``<`` ``think`` ``>`` … reasoning … ``</`` ``think`` ``>``
        answer ``<think>`` second ``</think>`` tail.
        """
        chunks = [
            "<",
            "think",
            ">",
            "first",
            "</",
            "think",
            ">",
            "answer",
            "<think>",
            "second",
            "</think>",
            "tail",
        ]
        reasoning, content = _stream(parser, chunks)
        assert "first" in reasoning
        assert "second" in reasoning
        assert "answer" in content
        assert "tail" in content
        assert "<think>" not in content
        assert "</think>" not in content


class TestSingleBlockUnchanged:
    """Sanity check — single-block streams (the common case) keep their
    pre-F-100 behaviour.
    """

    def test_implicit_single_block(self, parser):
        chunks = ["thought ", "more", "</think>", "answer", " tail"]
        reasoning, content = _stream(parser, chunks)
        assert "thought" in reasoning
        assert "more" in reasoning
        assert "answer tail" in content
        assert "</think>" not in content
        assert "</think>" not in reasoning

    def test_explicit_single_block(self, parser):
        chunks = ["<think>", "thinking", "</think>", "answer"]
        reasoning, content = _stream(parser, chunks)
        assert "thinking" in reasoning
        assert "answer" in content
        assert "<think>" not in content
        assert "</think>" not in content
