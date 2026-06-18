# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-request reasoning-token budget (upstream vLLM PRs
#20859 / #42396 / #43402 backport).

Covers:
  * Pydantic validation of ``reasoning_max_tokens`` on
    ``ChatCompletionRequest`` and ``ResponsesRequest``.
  * Anthropic ``output_config.effort`` → ``reasoning_max_tokens``
    mapping (and the legacy ``thinking.budget_tokens`` shape).
  * Streaming postprocessor force-close behavior on both
    channel-routed (gemma4 / harmony) and text-parser (qwen3 /
    deepseek) engine paths.
  * Non-streaming helper truncation in
    ``_finalize_content_and_reasoning``.

These tests intentionally exercise the public Pydantic + helper
surface so they're fast and don't require a loaded MLX engine.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm_mlx.api.anthropic_adapter import (
    _resolve_reasoning_max_tokens,
    anthropic_to_openai,
)
from vllm_mlx.api.anthropic_models import (
    ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS,
    AnthropicOutputConfig,
    AnthropicRequest,
)
from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.service.helpers import _finalize_content_and_reasoning
from vllm_mlx.service.postprocessor import StreamingPostProcessor

# ---------------------------------------------------------------------------
# 1) Request-shape validation
# ---------------------------------------------------------------------------


class TestChatRequestValidation:
    def test_default_is_none(self):
        r = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        assert r.reasoning_max_tokens is None

    def test_positive_value_passes(self):
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            reasoning_max_tokens=128,
        )
        assert r.reasoning_max_tokens == 128

    def test_zero_rejected(self):
        with pytest.raises(Exception) as exc:
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                reasoning_max_tokens=0,
            )
        assert "reasoning_max_tokens" in str(exc.value)

    def test_negative_rejected(self):
        with pytest.raises(Exception) as exc:
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                reasoning_max_tokens=-1,
            )
        assert "reasoning_max_tokens" in str(exc.value)


class TestResponsesRequestValidation:
    def test_default_is_none(self):
        r = ResponsesRequest(model="gpt-5", input="hi")
        assert r.reasoning_max_tokens is None

    def test_positive_value_passes(self):
        r = ResponsesRequest(model="gpt-5", input="hi", reasoning_max_tokens=64)
        assert r.reasoning_max_tokens == 64

    def test_zero_rejected(self):
        with pytest.raises(Exception) as exc:
            ResponsesRequest(model="gpt-5", input="hi", reasoning_max_tokens=0)
        assert "reasoning_max_tokens" in str(exc.value)


# ---------------------------------------------------------------------------
# 2) Anthropic output_config.effort mapping (upstream PR #42396)
# ---------------------------------------------------------------------------


class TestAnthropicEffortMapping:
    def test_canonical_mapping_constants(self):
        # The exact mapping vLLM #42396 / Anthropic SDK v0.22 documents.
        assert ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS == {
            "low": 512,
            "medium": 2048,
            "high": 8192,
            "xhigh": 24000,
            "max": None,
        }

    @pytest.mark.parametrize(
        "effort, expected",
        [
            ("low", 512),
            ("medium", 2048),
            ("high", 8192),
            ("xhigh", 24000),
            ("max", None),
        ],
    )
    def test_effort_translates_to_reasoning_max_tokens(self, effort, expected):
        req = AnthropicRequest(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            output_config=AnthropicOutputConfig(effort=effort),
        )
        assert _resolve_reasoning_max_tokens(req) == expected
        assert anthropic_to_openai(req).reasoning_max_tokens == expected

    def test_absent_output_config_is_none(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )
        assert _resolve_reasoning_max_tokens(req) is None
        assert anthropic_to_openai(req).reasoning_max_tokens is None

    def test_output_config_format_and_effort_coexist(self):
        """Pick 2 (#683) added ``output_config.format`` (json_schema);
        this PR adds ``effort``. Both must be settable on the SAME
        request and processed independently — the format goes through
        the json_schema → response_format path, the effort goes through
        the reasoning-cap path. Without coexistence either Pick would
        have broken the other."""
        req = AnthropicRequest(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            output_config={
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object"},
                },
            },
        )
        assert req.output_config.effort == "low"
        # format went through Pick 2's narrow parser
        assert req.output_config.format is not None
        assert req.output_config.format.type == "json_schema"
        # ``format`` does NOT influence reasoning_max_tokens — the two
        # fields are independent.
        assert _resolve_reasoning_max_tokens(req) == 512

    def test_legacy_thinking_budget_tokens(self):
        """Anthropic v0.20 ``thinking.budget_tokens`` shape (upstream
        vLLM PR #20859) must translate when ``output_config`` is unset."""
        req = AnthropicRequest(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            thinking={"type": "enabled", "budget_tokens": 1234},
        )
        assert _resolve_reasoning_max_tokens(req) == 1234

    def test_output_config_takes_precedence_over_thinking(self):
        """Newer ``output_config.effort`` wins over legacy
        ``thinking.budget_tokens`` per upstream precedence."""
        req = AnthropicRequest(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            output_config=AnthropicOutputConfig(effort="medium"),
            thinking={"budget_tokens": 99},
        )
        assert _resolve_reasoning_max_tokens(req) == 2048

    def test_thinking_budget_tokens_negative_rejected(self):
        with pytest.raises(Exception) as exc:
            AnthropicRequest(
                model="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                thinking={"budget_tokens": -10},
            )
        assert "budget" in str(exc.value).lower()

    @pytest.mark.parametrize(
        "bad_value",
        [
            "100",  # JSON-stringified int — silent-coerce hazard
            "0",
            1.5,  # float
            True,  # bool subclass of int — must be rejected explicitly
            [],
            {},
        ],
    )
    def test_thinking_budget_tokens_wrong_type_rejected(self, bad_value):
        """Codex round-1 BLOCKING #2: an earlier draft only rejected
        non-positive INTS. String wire values like ``"100"`` slipped
        through, were ignored by the adapter helper, and silently
        turned a client-requested cap into no cap. Validate the type
        too so any non-int / non-positive value 422s at parse time."""
        with pytest.raises(Exception):
            AnthropicRequest(
                model="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                thinking={"budget_tokens": bad_value},
            )


# ---------------------------------------------------------------------------
# 3) Streaming postprocessor enforcement
# ---------------------------------------------------------------------------


def _make_cfg(**overrides):
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = False
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = None
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_output(text="", finished=False, channel=None, finish_reason=None):
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = channel
    out.finish_reason = finish_reason or ("stop" if finished else None)
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    out.tool_calls = None
    return out


class TestChannelRoutedReasoningCap:
    """gemma4 / harmony engines emit ``output.channel="reasoning"``
    directly. The cap reroutes the overflow portion to content."""

    def test_no_cap_means_full_passthrough(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, reasoning_max_tokens=None)
        pp.reset()
        # 100 chars ≈ 25 tokens — would blow any cap, but None == no cap.
        events = pp.process_chunk(_make_output("x" * 100, channel="reasoning"))
        reasoning = [e for e in events if e.type == "reasoning"]
        content = [e for e in events if e.type == "content"]
        assert len(reasoning) == 1
        assert len(content) == 0
        assert reasoning[0].reasoning == "x" * 100

    def test_cap_truncates_and_reroutes_overflow(self):
        cfg = _make_cfg()
        # cap of 2 tokens ≈ 8 chars under the chars-÷4 heuristic.
        pp = StreamingPostProcessor(cfg, reasoning_max_tokens=2)
        pp.reset()
        # 40 chars ≈ 10 tokens — overflow becomes content.
        events = pp.process_chunk(_make_output("x" * 40, channel="reasoning"))
        reasoning = [e for e in events if e.type == "reasoning"]
        content = [e for e in events if e.type == "content"]
        # The first 8 chars stayed as reasoning; the rest became content.
        assert len(reasoning) == 1
        assert reasoning[0].reasoning == "x" * 8
        assert len(content) == 1
        assert content[0].content == "x" * 32
        assert pp._reasoning_cap_hit is True

    def test_subsequent_reasoning_chunks_fully_rerouted(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, reasoning_max_tokens=1)
        pp.reset()
        # Fire the cap on chunk 1.
        pp.process_chunk(_make_output("x" * 100, channel="reasoning"))
        assert pp._reasoning_cap_hit is True
        # Chunk 2 — ALL of it should become content now.
        events = pp.process_chunk(_make_output("yyyy", channel="reasoning"))
        reasoning = [e for e in events if e.type == "reasoning"]
        content = [e for e in events if e.type == "content"]
        assert reasoning == []
        assert len(content) == 1
        assert content[0].content == "yyyy"

    def test_thinking_disabled_model_no_op(self):
        """``reasoning_max_tokens`` on a request with no reasoning_parser
        AND no channel-routed reasoning chunks must be a complete no-op."""
        cfg = _make_cfg()  # no reasoning_parser_name
        pp = StreamingPostProcessor(cfg, reasoning_max_tokens=1)
        pp.reset()
        # plain content with no reasoning channel
        events = pp.process_chunk(_make_output("Hello world", channel="content"))
        assert len(events) == 1
        assert events[0].type == "content"
        assert events[0].content == "Hello world"
        assert pp._reasoning_cap_hit is False


class TestTextParserReasoningCap:
    """qwen3 / deepseek / glm47 emit ``<think>...</think>`` text. The
    cap force-closes by injecting ``</think>`` into the next chunk."""

    def _stub_reasoning_parser(self, scripted_deltas):
        """Returns a mock parser whose ``extract_reasoning_streaming`` is
        scripted: each call pops the next ``(reasoning, content)`` pair
        from the list and returns a SimpleNamespace with those fields.

        Codex round-1 NIT: assert the stream-state invariant
        ``current.endswith(delta)`` on every call so a buggy splice
        that leaves ``accumulated_raw`` out of sync with the (previous,
        delta) pair fails this test even though the scripted parser
        doesn't actually use the bytes. The invariant must hold for
        every reasoning-parser contract in the codebase (see how the
        real qwen3 / deepseek parsers read ``current`` for backtracking
        on Case-3 / Case-4 fallbacks).
        """
        from types import SimpleNamespace

        calls = []

        def _extract(previous, current, delta):
            # State-consistency invariants. ``current = previous + delta``
            # is the parser contract — every real parser in
            # ``vllm_mlx/reasoning/`` reads ``current`` for backtracking.
            assert current.endswith(delta), (
                f"current does not end with delta — splice bug? "
                f"previous={previous!r} current={current!r} delta={delta!r}"
            )
            assert current == previous + delta, (
                f"current != previous + delta — accumulated_raw drift! "
                f"previous={previous!r} delta={delta!r} current={current!r}"
            )
            calls.append({"previous": previous, "current": current, "delta": delta})
            if not scripted_deltas:
                return SimpleNamespace(reasoning=None, content=None)
            reasoning, content = scripted_deltas.pop(0)
            return SimpleNamespace(reasoning=reasoning, content=content)

        parser = MagicMock()
        parser.extract_reasoning_streaming = _extract
        parser.reset_state = MagicMock()
        parser.calls = calls
        return parser

    def test_text_parser_force_close_inject(self):
        # Script: chunk 1 produces 40 chars of reasoning (over cap of 2
        # tokens ≈ 8 chars); chunk 2 produces 5 chars of content but the
        # cap has fired so we must see the ``</think>`` injection.
        scripted = [
            ("x" * 40, None),  # chunk 1: parser sees pure reasoning
            (None, "world"),  # chunk 2: parser flips to content
        ]
        parser = self._stub_reasoning_parser(scripted)
        cfg = _make_cfg(
            reasoning_parser=parser,
            reasoning_parser_name=None,  # use injected mock
        )
        pp = StreamingPostProcessor(cfg, enable_thinking=True, reasoning_max_tokens=2)
        pp.reset()
        # Chunk 1: 40-char reasoning, cap fires
        pp.process_chunk(_make_output("xxxxxxxx" * 5))
        assert pp._reasoning_cap_hit is True
        # Chunk 2: the parser-text injection happens here. The fake
        # parser doesn't actually flip state — what we verify is that
        # the postprocessor injected the close marker into delta_text
        # before calling extract_reasoning_streaming.
        pp.process_chunk(_make_output("world"))
        # Call 2's delta should start with the injected </think>.
        assert parser.calls[1]["delta"].startswith("</think>")
        # Idempotent — chunk 3 must NOT re-inject.
        pp.process_chunk(_make_output("more"))
        assert not parser.calls[2]["delta"].startswith("</think>")


# ---------------------------------------------------------------------------
# 4) Non-streaming helper truncation
# ---------------------------------------------------------------------------


class TestFinalizeContentAndReasoningCap:
    def test_no_cap_is_back_compat_no_op(self):
        # No parser, no cap → cleaned_text + reasoning unchanged.
        cleaned, reasoning = _finalize_content_and_reasoning(
            raw_text="<think>abc</think>hello",
            cleaned_text="hello",
            tool_calls=[],
            reasoning_parser=None,
            engine_reasoning_text="abc",
        )
        assert cleaned == "hello"
        assert reasoning == "abc"

    def test_cap_truncates_and_overflows_to_content(self):
        cleaned, reasoning = _finalize_content_and_reasoning(
            raw_text="",
            cleaned_text="final answer",
            tool_calls=[],
            reasoning_parser=None,
            engine_reasoning_text="x" * 40,  # ≈ 10 tokens
            reasoning_max_tokens=2,  # cap ≈ 8 chars
        )
        # First 8 chars stay as reasoning, the rest tacks onto cleaned.
        assert reasoning == "x" * 8
        assert cleaned == "final answer" + "x" * 32

    def test_cap_below_text_is_noop(self):
        # text fits comfortably under cap → no truncation
        cleaned, reasoning = _finalize_content_and_reasoning(
            raw_text="",
            cleaned_text="hi",
            tool_calls=[],
            reasoning_parser=None,
            engine_reasoning_text="abc",
            reasoning_max_tokens=100,
        )
        assert reasoning == "abc"
        assert cleaned == "hi"

    def test_cap_with_no_reasoning_is_noop(self):
        cleaned, reasoning = _finalize_content_and_reasoning(
            raw_text="",
            cleaned_text="just content",
            tool_calls=[],
            reasoning_parser=None,
            engine_reasoning_text="",
            reasoning_max_tokens=5,
        )
        assert reasoning is None or reasoning == ""
        assert cleaned == "just content"


# ---------------------------------------------------------------------------
# 5) Streaming SSE shape after force-close
# ---------------------------------------------------------------------------


class TestStreamingTerminalChunkShape:
    """The terminal chunk after a force-close still needs to carry a
    valid ``finish_reason`` and well-formed StreamEvents — clients that
    consume the SSE byte-for-byte (Codex CLI, Claude Code) parse the
    finish chunk strictly so a mid-stream cap firing must not corrupt
    the wire shape."""

    def test_finish_chunk_after_cap_carries_finish_reason(self):
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, reasoning_max_tokens=1)
        pp.reset()
        # Chunk 1 — fires the cap.
        pp.process_chunk(_make_output("x" * 100, channel="reasoning"))
        assert pp._reasoning_cap_hit is True
        # Final chunk — ``finished=True`` triggers the merged finish path.
        events = pp.process_chunk(
            _make_output(
                "done.", channel="content", finished=True, finish_reason="stop"
            )
        )
        finish = [e for e in events if e.type == "finish"]
        assert len(finish) == 1
        assert finish[0].finish_reason == "stop"
        # content was merged into the finish chunk
        assert finish[0].content == "done."

    def test_cap_does_not_emit_phantom_finish(self):
        """The cap firing mid-stream should NOT itself produce a
        ``finish`` event — only the engine's ``finished=True`` chunk does."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, reasoning_max_tokens=1)
        pp.reset()
        events = pp.process_chunk(_make_output("x" * 100, channel="reasoning"))
        # Cap fires here, but the chunk is NOT finished — should not
        # emit a finish event.
        finish = [e for e in events if e.type == "finish"]
        assert finish == []
