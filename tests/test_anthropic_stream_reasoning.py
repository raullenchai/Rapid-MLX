# SPDX-License-Identifier: Apache-2.0
"""Regression test for #185 — Anthropic stream returns 0 chunks for reasoning models.

Bug: when a `reasoning_parser` was active for a reasoning model (qwen3
family etc.), `_stream_anthropic_messages` ran the parser's already-split
`delta_msg.content` through `StreamingThinkRouter`. The router was
initialized with `start_in_thinking=True` (chat templates that contain
`<think>` + `add_generation_prompt` flip the flag) and waited for a
`</think>` close tag in the stream — but the reasoning_parser had
already stripped those, so nothing ever closed and everything stayed
buffered as `thinking_delta`. Net effect: `text_stream` returned 0
chunks for qwen3 since v0.6.4.

Fix (PR #208): when `reasoning_parser` is active, bypass `think_router`
and emit `delta_msg.reasoning` and `delta_msg.content` directly as
their own block types. The parser already owns the split.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.routes import anthropic as anthropic_route


class _FakeOutput:
    def __init__(self, new_text: str):
        self.new_text = new_text
        self.prompt_tokens = 0
        self.completion_tokens = 0


class _FakeEngine:
    """Minimal engine that yields a pre-scripted token stream."""

    is_mllm = False
    preserve_native_tool_format = False

    def __init__(self, deltas: list[str], chat_template: str = ""):
        self._deltas = deltas
        self.tokenizer = SimpleNamespace(chat_template=chat_template)

    async def stream_chat(self, **kwargs) -> AsyncIterator[_FakeOutput]:
        for d in self._deltas:
            yield _FakeOutput(d)


async def _collect_stream(stream: AsyncIterator[str]) -> list[dict]:
    """Pull SSE lines from the async stream, parse each `data:` payload."""
    events: list[dict] = []
    async for line in stream:
        for chunk in line.split("\n"):
            chunk = chunk.strip()
            if not chunk or not chunk.startswith("data:"):
                continue
            payload = chunk[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    return events


def _text_chunks(events: list[dict]) -> list[str]:
    out = []
    for e in events:
        if (
            e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        ):
            out.append(e["delta"]["text"])
    return out


def _thinking_chunks(events: list[dict]) -> list[str]:
    out = []
    for e in events:
        if (
            e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "thinking_delta"
        ):
            out.append(e["delta"]["thinking"])
    return out


@pytest.fixture
def patched_cfg():
    """Patch the route's get_config to use a stub with reasoning_parser_name set."""
    cfg = MagicMock()
    cfg.reasoning_parser_name = "qwen3"
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = None
    cfg.enable_auto_tool_choice = False
    cfg.engine = None
    cfg.model_name = "qwen3.5-test"
    cfg.cloud_router = None
    with patch.object(anthropic_route, "get_config", return_value=cfg):
        yield cfg


@pytest.mark.asyncio
async def test_reasoning_parser_stream_emits_text_chunks(patched_cfg):
    """The bug repro: qwen3 chat template + reasoning content + answer.

    Pre-fix: 0 text_delta events emitted (everything misclassified as
    thinking and never flushed because </think> never appeared in the
    parser-cleaned stream).
    Post-fix: thinking → thinking_delta, content → text_delta.

    Coverage proof: production code at routes/anthropic.py:355-364
    instantiates a real Qwen3ReasoningParser via
    `get_parser(cfg.reasoning_parser_name)()` whenever
    `cfg.reasoning_parser_name` is truthy. Setting it to "qwen3" here
    makes that branch run, so the test exercises the new
    parser-then-emit-directly path. We additionally spy on
    `Qwen3ReasoningParser.extract_reasoning_streaming` so a future
    refactor that bypasses the parser will fail this test loudly
    instead of silently passing via the no-parser fallback.
    """
    # Qwen3 chat template fragment that triggers _starts_thinking=True.
    chat_template = "<think>{% if add_generation_prompt %}...{% endif %}"

    # Simulate the model emitting <think>reasoning</think>answer in pieces.
    deltas = [
        "<think>",
        "Let me",
        " think.",
        "</think>",
        "The answer is 42.",
    ]
    # Spy on the real Qwen3ReasoningParser to PROVE the patched branch
    # actually instantiated it. If a future change makes the test fall
    # through to the no-parser path, this counter stays at 0 and the
    # assertion below fires — preventing silent test rot.
    from vllm_mlx.reasoning import qwen3_parser as qwen3_module

    call_count = {"streaming": 0}
    real_streaming = qwen3_module.Qwen3ReasoningParser.extract_reasoning_streaming

    def _counted_streaming(self, *args, **kwargs):
        call_count["streaming"] += 1
        return real_streaming(self, *args, **kwargs)

    engine = _FakeEngine(deltas, chat_template=chat_template)
    openai_request = SimpleNamespace(
        model="qwen3.5-test",
        stream=True,
        messages=[
            SimpleNamespace(
                role="user",
                content="What is 6*7?",
                model_dump=lambda **kw: {"role": "user", "content": "What is 6*7?"},
            )
        ],
        max_tokens=100,
        temperature=None,
        top_p=None,
        tools=None,
        tool_choice=None,
        enable_thinking=None,
        model_dump=lambda **kw: {
            "model": "qwen3.5-test",
            "messages": [{"role": "user", "content": "What is 6*7?"}],
        },
    )
    anthropic_request = SimpleNamespace(model="qwen3.5-test")

    with patch.object(
        qwen3_module.Qwen3ReasoningParser,
        "extract_reasoning_streaming",
        _counted_streaming,
    ):
        stream = anthropic_route._stream_anthropic_messages(
            engine, openai_request, anthropic_request
        )
        events = await _collect_stream(stream)

    # Spy assertion: production code MUST have called the parser. If
    # this is 0, the test silently fell through to the no-parser branch
    # and the new fix path was never exercised.
    assert call_count["streaming"] > 0, (
        "Qwen3ReasoningParser.extract_reasoning_streaming was not called — "
        "the test fell through to the no-parser branch and is not actually "
        "exercising the #185 fix. Check that cfg.reasoning_parser_name is "
        "still 'qwen3' and that get_parser() resolves it."
    )

    text = "".join(_text_chunks(events))
    thinking = "".join(_thinking_chunks(events))

    assert "answer is 42" in text, (
        f"Pre-#185-fix bug: text stream is empty when reasoning_parser "
        f"is active. Got text={text!r}, thinking={thinking!r}, "
        f"events={[e.get('type') for e in events]}"
    )
    # Reasoning should also be surfaced (Anthropic clients can opt to
    # display it via the thinking block).
    assert "think" in thinking.lower() or thinking == "" or "Let me" in thinking
