# SPDX-License-Identifier: Apache-2.0
"""
Regression test for issue #185: Anthropic streaming with reasoning parser.

When a reasoning parser (e.g. qwen3) is active, the streaming adapter must:
1. Emit reasoning content as thinking_delta SSE events
2. Emit final content as text_delta SSE events
3. NOT misclassify all text as thinking when the model outputs no think tags

This test uses a mock engine that outputs raw text; it exercises the full
streaming pipeline in _stream_anthropic_messages.
"""

import asyncio
import json
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Tokenizer stub whose chat_template triggers _starts_thinking=True."""

    chat_template = "<think>\n{% for msg in messages %}{{ msg.content }}\n{% endfor %}{% if add_generation_prompt %}\n<think>\n{% endif %}"


class MockDeltaOutput:
    """Simulates one async chunk from engine.stream_chat()."""

    def __init__(
        self, new_text: str, prompt_tokens: int = 10, completion_tokens: int = 1
    ):
        self.new_text = new_text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockEngine:
    """Minimal engine that yields deltas and exposes a tokenizer."""

    def __init__(self, deltas: list[str]):
        self._deltas = deltas
        self.tokenizer = _FakeTokenizer()
        self.preserve_native_tool_format = False

    async def stream_chat(self, **kwargs) -> Any:
        for d in self._deltas:
            yield MockDeltaOutput(d)


# ---------------------------------------------------------------------------
# Helper to collect SSE events from the async generator
# ---------------------------------------------------------------------------


def _collect_sse_events(gen):
    """Drain an async generator of SSE event strings into a list."""

    async def _collect():
        chunks = []
        async for chunk in gen:
            chunks.append(chunk)
        return chunks

    return asyncio.run(_collect())


def _extract_delta_types(events: list[str]) -> list[tuple[str, str | None]]:
    """Parse SSE event strings, return list of (event_type, delta_type_or_None).

    Returns tuples like ("content_block_start", "thinking"),
    ("content_block_delta", "thinking_delta"), ("content_block_delta", "text_delta"), etc.
    """
    result = []
    for evt in events:
        if not evt.strip():
            continue
        for line in evt.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                t = data.get("type")
                if t == "content_block_start":
                    cb = data.get("content_block", {})
                    result.append(("content_block_start", cb.get("type")))
                elif t == "content_block_delta":
                    d = data.get("delta", {})
                    result.append(("content_block_delta", d.get("type")))
                elif t == "content_block_stop":
                    result.append(("content_block_stop", None))
                elif t in ("message_start", "message_delta", "message_stop"):
                    result.append((t, None))
    return result


def _extract_text_from_deltas(events: list[str]) -> str:
    """Extract all text_delta text content from SSE events."""
    text_parts = []
    for evt in events:
        if not evt.strip():
            continue
        for line in evt.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text_parts.append(delta.get("text", ""))
    return "".join(text_parts)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg_with_reasoning_parser():
    """Set up config singleton with reasoning_parser_name='qwen3'."""
    from vllm_mlx.config import server_config

    saved = {k: v for k, v in server_config._config.__dict__.items()}
    server_config.reset_config()
    server_config._config.model_name = "test-model"
    server_config._config.reasoning_parser_name = "qwen3"
    yield
    # Restore
    server_config._config.__dict__.clear()
    server_config._config.__dict__.update(saved)


@pytest.fixture
def cfg_without_reasoning_parser():
    """Set up config singleton with reasoning_parser_name=None."""
    from vllm_mlx.config import server_config

    saved = {k: v for k, v in server_config._config.__dict__.items()}
    server_config.reset_config()
    server_config._config.model_name = "test-model"
    server_config._config.reasoning_parser_name = None
    yield
    # Restore
    server_config._config.__dict__.clear()
    server_config._config.__dict__.update(saved)


class TestAnthropicStreamingWithReasoningParser:
    """Issue #185: _stream_anthropic_messages with reasoning_parser active."""

    def test_no_think_tags_yields_text_delta(self, cfg_with_reasoning_parser):
        """Model outputs plain text with no think tags → text_delta events.

        The model (e.g. Qwen3.5-4B answering "count 1 to 5") doesn't output
        any <think> or </think> tags. The reasoning parser conservatively
        classifies as reasoning during streaming, but finalize_streaming
        corrects to content. The Anthropic SDK's text_stream must receive
        non-empty content.
        """
        from vllm_mlx.routes.anthropic import (
            AnthropicRequest,
            ChatCompletionRequest,
            _stream_anthropic_messages,
        )

        engine = MockEngine(["1", ", ", "2", ", ", "3", ", ", "4", ", ", "5"])
        openai_req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            max_tokens=80,
        )
        anthropic_req = AnthropicRequest(
            model="test-model",
            max_tokens=80,
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            stream=True,
        )

        gen = _stream_anthropic_messages(engine, openai_req, anthropic_req)
        events = _collect_sse_events(gen)

        delta_types = _extract_delta_types(events)
        text = _extract_text_from_deltas(events)

        # Must have text content
        assert text, f"No text_delta content found in events:\n{delta_types}"
        assert "1" in text and "5" in text, f"Missing expected content in: {text!r}"

        # Must have at least one text_delta event
        text_deltas = [
            t for t in delta_types if t == ("content_block_delta", "text_delta")
        ]
        assert text_deltas, f"No text_delta events in: {delta_types}"

        # Must have at least one text content_block_start
        text_starts = [t for t in delta_types if t == ("content_block_start", "text")]
        assert text_starts, f"No text content_block_start in: {delta_types}"

    def test_both_think_tags_emits_thinking_and_text(self, cfg_with_reasoning_parser):
        """Model outputs <think>...</think> → separated thinking + text blocks."""
        from vllm_mlx.routes.anthropic import (
            AnthropicRequest,
            ChatCompletionRequest,
            _stream_anthropic_messages,
        )

        engine = MockEngine(
            [
                "Let me ",
                "think",
                "</think>",
                "\nThe answer ",
                "is 42.",
            ]
        )
        openai_req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "What is the answer?"}],
            max_tokens=80,
        )
        anthropic_req = AnthropicRequest(
            model="test-model",
            max_tokens=80,
            messages=[{"role": "user", "content": "What is the answer?"}],
            stream=True,
        )

        gen = _stream_anthropic_messages(engine, openai_req, anthropic_req)
        events = _collect_sse_events(gen)

        delta_types = _extract_delta_types(events)
        text = _extract_text_from_deltas(events)

        # Must have reasoning in thinking block
        thinking_deltas = [
            t for t in delta_types if t == ("content_block_delta", "thinking_delta")
        ]
        assert thinking_deltas, f"No thinking_delta events in: {delta_types}"

        # Must have content in text block
        assert "42" in text, f"Missing '42' in text output: {text!r}"

        # thinking block comes before text block (by index order)
        thinking_starts = [
            (i, t)
            for i, t in enumerate(delta_types)
            if t == ("content_block_start", "thinking")
        ]
        text_starts = [
            (i, t)
            for i, t in enumerate(delta_types)
            if t == ("content_block_start", "text")
        ]
        if thinking_starts and text_starts:
            assert thinking_starts[0][0] < text_starts[0][0], (
                f"thinking block must start before text block: {delta_types}"
            )

    def test_only_close_tag_implicit_think(self, cfg_with_reasoning_parser):
        """Only </think> in output (think injected in prompt) → correct split."""
        from vllm_mlx.routes.anthropic import (
            AnthropicRequest,
            ChatCompletionRequest,
            _stream_anthropic_messages,
        )

        engine = MockEngine(
            [
                "reasoning ",
                "here",
                "</think>",
                "final answer",
            ]
        )
        openai_req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Question"}],
            max_tokens=80,
        )
        anthropic_req = AnthropicRequest(
            model="test-model",
            max_tokens=80,
            messages=[{"role": "user", "content": "Question"}],
            stream=True,
        )

        gen = _stream_anthropic_messages(engine, openai_req, anthropic_req)
        events = _collect_sse_events(gen)

        delta_types = _extract_delta_types(events)
        text = _extract_text_from_deltas(events)

        assert "final answer" in text, f"Missing 'final answer' in text: {text!r}"

        # Must have both thinking_delta and text_delta
        thinking_deltas = [
            t for t in delta_types if t == ("content_block_delta", "thinking_delta")
        ]
        text_deltas = [
            t for t in delta_types if t == ("content_block_delta", "text_delta")
        ]
        assert thinking_deltas, f"No thinking_delta in: {delta_types}"
        assert text_deltas, f"No text_delta in: {delta_types}"


class TestAnthropicStreamingWithoutReasoningParser:
    """Fallback path: no reasoning parser → StreamingThinkRouter handles tags."""

    def test_no_parser_fallback_emits_text(self, cfg_without_reasoning_parser):
        """Without reasoning parser, plain text goes through think_router.

        The _starts_thinking heuristic fires (chat template has <think>), so
        the think_router starts in thinking mode. Without a </think> in output
        everything stays as thinking — this is the *current* limitation.
        But the fallback path must still work (no crash, events emitted).
        """
        from vllm_mlx.routes.anthropic import (
            AnthropicRequest,
            ChatCompletionRequest,
            _stream_anthropic_messages,
        )

        engine = MockEngine(["hello", " world"])
        openai_req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=50,
        )
        anthropic_req = AnthropicRequest(
            model="test-model",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
        )

        gen = _stream_anthropic_messages(engine, openai_req, anthropic_req)
        events = _collect_sse_events(gen)

        delta_types = _extract_delta_types(events)

        # Must have at least one content block (even if classified as thinking)
        content_blocks = [t for t in delta_types if t[0] == "content_block_delta"]
        assert content_blocks, f"No content emitted at all: {delta_types}"

        # message_start and message_stop must be present
        has_start = any(t[0] == "message_start" for t in delta_types)
        has_stop = any(t[0] == "message_stop" for t in delta_types)
        assert has_start and has_stop, f"Missing SSE framing: {delta_types}"


class TestAnthropicStreamingChannelRouting:
    """OutputRouter channel-aware branch (harmony/gemma4 models)."""

    def test_unknown_channel_is_suppressed_not_leaked(self, cfg_with_reasoning_parser):
        """Unrecognized ``output.channel`` must NOT leak to user text.

        DeepSeek review on PR #436 flagged that the initial ``else``
        branch caught every non-``reasoning`` channel and emitted it
        as user-facing ``text_delta``. If a future router channel is
        added (e.g. ``"system"``, ``"error"``) without updating this
        route, the implicit-text fallback would silently leak those
        internal tokens. Fix: explicit allowlist
        ``("reasoning", "content", "tool_call")``; unknown channels
        are dropped (logged at WARNING) and the loop ``continue``s,
        so the delta never reaches the client SSE stream.
        """
        from vllm_mlx.routes.anthropic import (
            AnthropicRequest,
            ChatCompletionRequest,
            _stream_anthropic_messages,
        )

        class _ChannelDelta(MockDeltaOutput):
            def __init__(self, text: str, channel: str | None):
                super().__init__(text)
                self.channel = channel

        class _ChannelEngine(MockEngine):
            def __init__(self, deltas):
                self._d = deltas
                self.tokenizer = _FakeTokenizer()
                self.preserve_native_tool_format = False

            async def stream_chat(self, **kwargs):
                for d in self._d:
                    yield d

        # First delta is the unknown channel; second is normal content
        # so the test verifies the unknown one is not emitted as text.
        engine = _ChannelEngine(
            [
                _ChannelDelta("INTERNAL_LEAK_TOKEN", channel="system"),
                _ChannelDelta("safe", channel="content"),
            ]
        )
        openai_req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=20,
        )
        anthropic_req = AnthropicRequest(
            model="test-model",
            max_tokens=20,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )

        gen = _stream_anthropic_messages(engine, openai_req, anthropic_req)
        events = _collect_sse_events(gen)

        joined = "\n".join(events)
        assert "INTERNAL_LEAK_TOKEN" not in joined, (
            f"Unknown channel leaked to client: {joined!r}"
        )
