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
from pydantic import ValidationError

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
        # Codex round-8 NIT #4: ``pytest.raises(Exception)`` is too
        # broad — any construction failure (e.g. an unrelated field
        # added in the future) would pass. Anchor to
        # ``ValidationError`` AND check the error message names
        # ``reasoning_max_tokens`` so a regression in the validator
        # itself can't pass silently.
        with pytest.raises(ValidationError) as exc:
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                reasoning_max_tokens=0,
            )
        assert "reasoning_max_tokens" in str(exc.value)

    def test_negative_rejected(self):
        with pytest.raises(ValidationError) as exc:
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
        with pytest.raises(ValidationError) as exc:
            ResponsesRequest(model="gpt-5", input="hi", reasoning_max_tokens=0)
        assert "reasoning_max_tokens" in str(exc.value)


@pytest.mark.parametrize(
    "bad_value",
    [
        "100",  # JSON-stringified int — silent-coerce hazard
        "0",
        1.5,  # float
        True,  # bool subclass of int — must be rejected explicitly
        False,
        [],
        {},
    ],
)
class TestStrictReasoningMaxTokensValidation:
    """Codex round-3 NIT #4-#5: tighten the OpenAI-surface validators
    so they reject the same wire-value classes the Anthropic-surface
    ``thinking.budget_tokens`` validator already rejects. Without
    ``mode="before"`` + an explicit type guard, Pydantic v2 silently
    coerces ``"100"`` to 100 and ``True`` to 1 — turning client-side
    type bugs into silently-accepted caps. Both the
    ``/v1/chat/completions`` and ``/v1/responses`` validators must
    raise; the contract is symmetrical across the three streaming
    surfaces.
    """

    def test_chat_completion_request_rejects(self, bad_value):
        with pytest.raises(ValidationError) as exc:
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                reasoning_max_tokens=bad_value,
            )
        # Codex round-8 NIT #4: anchor on the field name so a
        # regression that drops the validator (and the model
        # construction fails on something unrelated) can't pass.
        assert "reasoning_max_tokens" in str(exc.value)

    def test_responses_request_rejects(self, bad_value):
        with pytest.raises(ValidationError) as exc:
            ResponsesRequest(
                model="gpt-5",
                input="hi",
                reasoning_max_tokens=bad_value,
            )
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
        # Codex round-8 NIT #5: anchor on ``ValidationError`` AND on
        # the field path so a regression that drops the validator
        # itself (and the model construction fails on something
        # unrelated) can't pass silently.
        with pytest.raises(ValidationError) as exc:
            AnthropicRequest(
                model="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                thinking={"budget_tokens": -10},
            )
        # ``budget_tokens`` is in the validator error message.
        assert "budget_tokens" in str(exc.value).lower()

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
        with pytest.raises(ValidationError) as exc:
            AnthropicRequest(
                model="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                thinking={"budget_tokens": bad_value},
            )
        # Anchor the assertion to the validator's error so a
        # regression that drops the type guard can't pass silently
        # via an unrelated construction error.
        assert "budget_tokens" in str(exc.value).lower()


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

    def test_finalize_injects_close_marker_after_terminal_cap_hit(self):
        """Codex round-3 BLOCKING #1: if the reasoning cap latches on
        the LAST chunk of the stream (model stops immediately at the
        budget, or stops within the cap-firing chunk), no subsequent
        ``process_chunk`` call ever runs the ``</think>`` injection.
        The parser is left mid-think, any buffered content past the cap
        is dropped, and the client sees a reasoning-only response with
        no visible answer.

        After the fix, ``finalize()`` must inject ``</think>`` once
        when ``_reasoning_cap_hit and not _reasoning_close_injected``,
        and flush whatever content the parser releases. This test
        scripts a parser that holds back content until it sees the
        close marker and asserts that content is emitted by finalize.
        """
        # Chunk 1 emits reasoning EXACTLY at the boundary (cap*4 = 4
        # chars under cap=1) — latches the cap but produces no
        # overflow, so the in-chunk flip does NOT fire (no overflow ⇒
        # no need to force a transition mid-chunk). The parser is then
        # sitting on "answer-bytes" that it would only release after
        # seeing ``</think>``. Without the finalize-time injection
        # those bytes would be lost when no further chunk arrives.
        scripted_chunks = [("xxxx", None)]  # chunk 1: exact-boundary reasoning

        # finalize_inject is what the parser returns when we splice
        # ``</think>`` into it from finalize().
        finalize_content = "buffered-answer"

        from types import SimpleNamespace

        finalize_calls = []

        def _extract(previous, current, delta):
            # Maintain the stream invariant on every call so a buggy
            # accumulated_text update fails this test.
            assert current == previous + delta
            if scripted_chunks:
                reasoning, content = scripted_chunks.pop(0)
                return SimpleNamespace(reasoning=reasoning, content=content)
            # Terminal injection: parser sees ``</think>`` and releases
            # its buffered content.
            finalize_calls.append({"previous": previous, "delta": delta})
            if delta == "</think>":
                return SimpleNamespace(reasoning=None, content=finalize_content)
            return SimpleNamespace(reasoning=None, content=None)

        parser = MagicMock()
        parser.extract_reasoning_streaming = _extract
        parser.reset_state = MagicMock()

        cfg = _make_cfg(
            reasoning_parser=parser,
            reasoning_parser_name=None,
        )
        pp = StreamingPostProcessor(cfg, enable_thinking=True, reasoning_max_tokens=1)
        pp.reset()
        # Chunk 1 fires the cap at the exact boundary (4 chars = 1
        # token under ceiling-÷4) — no overflow ⇒ no in-chunk flip.
        pp.process_chunk(_make_output("xxxx"))
        assert pp._reasoning_cap_hit is True
        # Critical: close-marker injection has NOT fired yet (no
        # overflow to trigger the in-chunk flip, no subsequent chunk
        # to trigger the next-chunk injection).
        assert pp._reasoning_close_injected is False

        # Stream ends — finalize must trigger the injection.
        events = pp.finalize()
        assert pp._reasoning_close_injected is True, (
            "finalize() must inject </think> when terminal cap-hit "
            "left the parser mid-think"
        )
        # The parser saw exactly one finalize-time call carrying </think>.
        assert len(finalize_calls) == 1
        assert finalize_calls[0]["delta"] == "</think>"
        # The buffered content was released as a content event.
        content_events = [e for e in events if e.type == "content"]
        assert len(content_events) == 1
        assert content_events[0].content == finalize_content

    def test_finalize_does_not_emit_close_marker_through_parser_twice(self):
        """Codex round-5 NIT #4: a real reasoning parser's
        ``finalize_streaming()`` (qwen3 / deepseek non-stream
        re-parse path) can re-emit the SAME buffered answer the
        in-stream forced-close extraction just released, leading to
        duplicated content events on the wire.

        The fix is two-layer: (a) the routes
        (responses._stream_responses, anthropic._stream_anthropic_messages)
        skip ``finalize_streaming`` when terminal injection already
        emitted content, and (b) the postprocessor builds its
        ``</think>`` view LOCALLY rather than mutating the shared
        ``accumulated_text`` buffer, so any future code path that
        re-parses the buffer doesn't see the forged marker. This test
        scripts a parser double whose ``extract_reasoning_streaming``
        returns content on the injection AND a separate
        ``finalize_streaming`` that would re-release the same bytes
        if called against the mutated buffer — asserts the buffered
        bytes appear exactly once on the wire.
        """
        from types import SimpleNamespace

        # The "real answer" the model produced past the cap. Both the
        # forced-close streaming extraction AND a hypothetical
        # non-stream finalize re-parse would yield this same content.
        real_answer = "the-buffered-answer"

        finalize_streaming_calls = []

        def _extract(previous, current, delta):
            assert current == previous + delta
            if delta == "</think>":
                # Forced close: release the held content.
                return SimpleNamespace(reasoning=None, content=real_answer)
            # Initial reasoning chunk.
            return SimpleNamespace(reasoning="x" * 40, content=None)

        def _finalize_streaming(text):
            """Real-parser-style non-stream re-parse: when called on a
            buffer that ends with ``</think>...<real-answer>``, it
            would naturally return the same content the streaming
            extract just released. Track every call so the test can
            assert the gating skipped it."""
            finalize_streaming_calls.append(text)
            return SimpleNamespace(reasoning=None, content=real_answer)

        parser = MagicMock()
        parser.extract_reasoning_streaming = _extract
        parser.finalize_streaming = _finalize_streaming
        parser.reset_state = MagicMock()

        cfg = _make_cfg(reasoning_parser=parser, reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, enable_thinking=True, reasoning_max_tokens=1)
        pp.reset()

        # Cap fires WITH overflow on chunk 1, so the round-9 in-chunk
        # flip releases ``real_answer`` here (the parser's content
        # response to the in-chunk forced ``</think>``).
        chunk1_events = pp.process_chunk(_make_output("x" * 40))
        assert pp._reasoning_cap_hit is True
        finalize_events = pp.finalize()

        all_events = list(chunk1_events) + list(finalize_events)
        content_events = [e for e in all_events if e.type == "content"]
        # Critical: the buffered answer must appear EXACTLY ONCE on the
        # wire — not zero (silent drop) and not two (duplicate-flush
        # hazard codex round-4/5/9 BLOCKING).
        matching = [e for e in content_events if real_answer in e.content]
        assert len(matching) == 1, (
            f"buffered answer must be emitted exactly once after cap-hit; "
            f"got {len(matching)} matching event(s) "
            f"(all content events: {[e.content for e in content_events]})"
        )
        # The postprocessor's ``finalize()`` does NOT call
        # ``finalize_streaming`` (the parser-finalize re-parse only
        # exists in the route-level paths). Document the contract so a
        # future refactor adding that call gets caught here.
        assert finalize_streaming_calls == [], (
            "StreamingPostProcessor.finalize() must NOT call "
            "parser.finalize_streaming — the duplicate-emission "
            "hazard is gated at the route level (responses + anthropic)"
        )
        # The forged ``</think>`` was kept OFF the shared accumulated
        # buffer (codex round-5 BLOCKING #1 — downstream usage chars-÷4
        # would otherwise drift by 8 chars).
        assert "</think>" not in pp.accumulated_text

    def test_finalize_swallows_parser_exception_without_fabricating_content(self):
        """Codex round-5 BLOCKING #2-#3: an earlier draft emitted a
        diagnostic string (``"[reasoning cap hit — parser flush
        failed]"``) directly into ``content`` when the parser raised
        on the forced close-marker call. That fabricated assistant
        text from an INTERNAL server failure — the client would see an
        "answer" that the model never produced.

        After the fix, the exception is logged and NO content event is
        emitted from the failure path (the route's existing 5xx /
        disconnect-guard semantics handle catastrophic failures
        upstream). The latch still flips so a subsequent ``finalize()``
        call is idempotent.
        """
        parser = MagicMock()
        parser.extract_reasoning_streaming = MagicMock(
            side_effect=[
                # Chunk 1 — emits reasoning over the cap.
                MagicMock(reasoning="x" * 40, content=None),
                # Finalize-time injection — parser bug raises.
                RuntimeError("parser blew up on injected </think>"),
            ]
        )
        parser.reset_state = MagicMock()

        cfg = _make_cfg(reasoning_parser=parser, reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, enable_thinking=True, reasoning_max_tokens=1)
        pp.reset()
        pp.process_chunk(_make_output("x" * 40))
        assert pp._reasoning_cap_hit is True
        events = pp.finalize()
        # No content event was fabricated from the parser exception.
        content_events = [e for e in events if e.type == "content"]
        assert content_events == [], (
            "finalize() must NOT fabricate assistant content when the "
            "parser raises on the forced close-marker call — that leaks "
            "server implementation details into the model response"
        )
        # Latch still flipped — second finalize is a no-op.
        assert pp._reasoning_close_injected is True

    def test_finalize_no_op_when_close_injected_in_stream(self):
        """Idempotency: when the close-marker was already spliced
        in-stream (in-chunk flip OR mid-stream injection path),
        ``finalize()`` must NOT re-inject. Otherwise the parser sees
        a second forged ``</think>`` and emits trailing content twice.

        Codex round-9: the in-chunk flip path (cap crosses with
        overflow, postprocessor runs the parser a second time with
        ``</think>`` in the same chunk) sets
        ``_reasoning_close_injected = True`` immediately. The
        ``finalize()`` injection then short-circuits and emits no
        additional parser calls / content events.
        """
        from types import SimpleNamespace

        injection_count = 0

        def _extract(previous, current, delta):
            nonlocal injection_count
            assert current == previous + delta
            if delta.startswith("</think>"):
                injection_count += 1
                return SimpleNamespace(reasoning=None, content="real-answer")
            if not previous:
                # Chunk 1 — overflow reasoning (the cap crosses here).
                return SimpleNamespace(reasoning="x" * 40, content=None)
            # Chunk 2 — ordinary content, no marker.
            return SimpleNamespace(reasoning=None, content="more")

        parser = MagicMock()
        parser.extract_reasoning_streaming = _extract
        parser.reset_state = MagicMock()

        cfg = _make_cfg(
            reasoning_parser=parser,
            reasoning_parser_name=None,
        )
        pp = StreamingPostProcessor(cfg, enable_thinking=True, reasoning_max_tokens=1)
        pp.reset()
        # Chunk 1: cap fires AND in-chunk flip injects ``</think>``.
        pp.process_chunk(_make_output("x" * 40))
        assert pp._reasoning_close_injected is True
        assert injection_count == 1, (
            "in-chunk flip must inject </think> exactly once on cap-crossing"
        )
        # Chunk 2: ordinary content delta, parser sees no marker.
        pp.process_chunk(_make_output("more"))
        assert injection_count == 1, "second chunk must not re-inject"
        # finalize() must be a no-op for the injection path.
        events = pp.finalize()
        assert injection_count == 1, (
            "finalize() must NOT re-inject </think> — the in-chunk flip "
            "already flipped the parser; a second injection would re-emit "
            "the buffered content"
        )
        # No content event from finalize (mid-stream + chunk-2 already
        # emitted whatever was real model output).
        content_from_finalize = [e for e in events if e.type == "content"]
        assert content_from_finalize == []

    def test_failed_in_chunk_flip_does_not_latch_close_marker(self):
        """Codex round-10 BLOCKING #1: if the parser raises on the
        in-chunk forced ``</think>`` flip, the close-injected latch
        must STAY CLEAR so the NEXT chunk retries the forced
        transition. The earlier draft flipped the latch BEFORE the
        parser call — a transient parser bug then left the parser
        permanently mid-think because subsequent chunks all saw the
        latch set and skipped injection.
        """
        from types import SimpleNamespace

        call_count = 0

        def _extract(previous, current, delta):
            nonlocal call_count
            call_count += 1
            assert current == previous + delta
            # Chunk 1 (initial reasoning extraction) — no marker yet.
            if call_count == 1:
                return SimpleNamespace(reasoning="x" * 40, content=None)
            # Chunk 1 in-chunk flip — RAISE so the latch stays clear.
            if call_count == 2:
                assert delta == "</think>"
                raise RuntimeError("transient parser bug on flip")
            # Chunk 2 retries the forced transition — succeeds this time.
            if call_count == 3:
                assert delta.startswith("</think>"), (
                    f"chunk 2 must retry the forced ``</think>`` "
                    f"injection (latch should have stayed clear after "
                    f"chunk 1 flip failure); got delta={delta!r}"
                )
                return SimpleNamespace(reasoning=None, content="recovered")
            return SimpleNamespace(reasoning=None, content=None)

        parser = MagicMock()
        parser.extract_reasoning_streaming = _extract
        parser.reset_state = MagicMock()

        cfg = _make_cfg(reasoning_parser=parser, reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, enable_thinking=True, reasoning_max_tokens=1)
        pp.reset()
        # Chunk 1 — cap fires, in-chunk flip raises.
        pp.process_chunk(_make_output("x" * 40))
        # The latch MUST be clear so chunk 2 retries.
        assert pp._reasoning_close_injected is False, (
            "_reasoning_close_injected must stay False after a failed "
            "flip — otherwise chunk 2 skips injection and the parser "
            "stays permanently mid-think"
        )
        # Chunk 2 — retries injection, succeeds.
        events = pp.process_chunk(_make_output("more"))
        assert pp._reasoning_close_injected is True, (
            "latch must flip after the retry succeeds"
        )
        # The retry released the recovered content.
        content_events = [e for e in events if e.type == "content"]
        recovered = [e for e in content_events if "recovered" in e.content]
        assert len(recovered) == 1, (
            f"chunk 2 retry must release recovered content; "
            f"got events={content_events!r}"
        )

    def test_postprocessor_does_not_pollute_accumulated_text_with_close_marker(self):
        """Codex round-8 BLOCKING #1: ``_process_with_reasoning``
        previously mutated ``self.accumulated_text`` with the synthetic
        ``</think>`` marker after the cap fired, poisoning the shared
        buffer that downstream usage accounting + finalize tool-call
        fallback read. After the fix, the marker only enters the
        parser's LOCAL ``current`` argument — the shared buffer
        holds real model output only.
        """
        from types import SimpleNamespace

        # Cap fires on chunk 1 (40 chars over 1-token cap).
        scripted = [
            ("x" * 40, None),  # chunk 1 reasoning
            (None, "answer"),  # chunk 2 content (after marker injected)
        ]

        recorded = []

        def _extract(previous, current, delta):
            recorded.append({"previous": previous, "current": current, "delta": delta})
            assert current == previous + delta
            if not scripted:
                return SimpleNamespace(reasoning=None, content=None)
            reasoning, content = scripted.pop(0)
            return SimpleNamespace(reasoning=reasoning, content=content)

        parser = MagicMock()
        parser.extract_reasoning_streaming = _extract
        parser.reset_state = MagicMock()

        cfg = _make_cfg(reasoning_parser=parser, reasoning_parser_name=None)
        pp = StreamingPostProcessor(cfg, enable_thinking=True, reasoning_max_tokens=1)
        pp.reset()
        pp.process_chunk(_make_output("x" * 40))  # cap fires
        pp.process_chunk(_make_output("more"))  # close marker injected
        # The PARSER saw ``</think>`` in its delta on chunk 2.
        assert recorded[1]["delta"].startswith("</think>")
        # But the SHARED accumulated_text does NOT contain the
        # synthetic marker — only the raw model deltas.
        assert "</think>" not in pp.accumulated_text
        # Sanity: it contains both raw chunks.
        assert "x" * 40 in pp.accumulated_text
        assert "more" in pp.accumulated_text

    def test_approx_token_count_uses_ceiling_division(self):
        """Codex round-7 NIT #3: the streaming cap heuristic must use
        CEILING division so a 5-char chunk over a 1-token cap overflows
        (matches the non-stream ``helpers._apply_reasoning_cap``
        ``cap * 4`` ceiling). Floor division would compute
        ``5 // 4 == 1 token``, count as exact-boundary, and keep ALL 5
        chars as reasoning — non-stream would have clipped at 4 chars
        with 1 char overflow. Fix the streaming heuristic to ceiling
        so streaming and non-streaming agree.
        """
        # 5 chars: floor=1, ceiling=2. With cap=1: ceiling overflows.
        assert StreamingPostProcessor._approx_token_count("xxxxx") == 2
        # 4 chars: both floor and ceiling = 1.
        assert StreamingPostProcessor._approx_token_count("xxxx") == 1
        # 1 char: floor and ceiling both clamp up to 1 (the ``max(1,
        # ...)`` defense-in-depth floor).
        assert StreamingPostProcessor._approx_token_count("x") == 1
        # Empty: 0 (no advance).
        assert StreamingPostProcessor._approx_token_count("") == 0
        # 8 chars: both = 2.
        assert StreamingPostProcessor._approx_token_count("xxxxxxxx") == 2

    def test_streaming_and_non_streaming_cap_agree_on_5_chars(self):
        """End-to-end agreement: a 5-char reasoning chunk over a
        1-token cap should produce the SAME (kept-reasoning,
        content-overflow) split in both the streaming postprocessor
        AND the non-streaming ``_apply_reasoning_cap`` helper.
        Codex round-7 NIT #3 — single source of truth for the
        chars-÷4 contract.
        """
        from vllm_mlx.service.helpers import _apply_reasoning_cap

        # Non-streaming: 5 chars > 4 (cap*4) → 4 reasoning + 1 content.
        ns_cleaned, ns_reasoning = _apply_reasoning_cap(
            cleaned_text="",
            reasoning_text="xxxxx",
            reasoning_max_tokens=1,
        )
        assert ns_reasoning == "xxxx"
        assert ns_cleaned == "x"

        # Streaming on a channel-routed engine — should agree.
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, reasoning_max_tokens=1)
        pp.reset()
        events = pp.process_chunk(_make_output("xxxxx", channel="reasoning"))
        reasoning_events = [e for e in events if e.type == "reasoning"]
        content_events = [e for e in events if e.type == "content"]
        # Streaming must produce the same split: 4 reasoning + 1 content.
        assert len(reasoning_events) == 1
        assert reasoning_events[0].reasoning == "xxxx"
        assert len(content_events) == 1
        assert content_events[0].content == "x"

    def test_text_parser_force_close_exact_boundary(self):
        """Codex round-2 BLOCKING #4: the cap-fired tests above use
        chunks that EXCEED the cap (40 chars vs 8-char budget), which
        masks the exact-boundary case — when the first reasoning delta
        is exactly ``reasoning_max_tokens * 4`` chars the cumulative
        token count hits the cap exactly. Early drafts used
        ``new_total <= cap`` and left ``_reasoning_cap_hit`` clear on
        exact fit, so the NEXT chunk would slip past the budget before
        ``</think>`` was injected. After the fix, the cap must latch on
        exact fit and the next parser call's delta must start with
        ``</think>``.
        """
        # cap = 2 tokens → exactly 8 chars under the chars-÷4 heuristic.
        # Chunk 1 emits 8 chars of pure reasoning — exact-boundary hit.
        # Chunk 2 emits content; we assert ``</think>`` was prepended.
        scripted = [
            ("x" * 8, None),  # chunk 1: exact-boundary cumulative = cap
            (None, "y"),  # chunk 2: should see injected close marker
        ]
        parser = self._stub_reasoning_parser(scripted)
        cfg = _make_cfg(
            reasoning_parser=parser,
            reasoning_parser_name=None,
        )
        pp = StreamingPostProcessor(cfg, enable_thinking=True, reasoning_max_tokens=2)
        pp.reset()
        pp.process_chunk(_make_output("x" * 8))
        # Exact-boundary latch — must fire even though we did NOT exceed.
        assert pp._reasoning_cap_hit is True, (
            "exact-boundary case (new_total == cap) failed to latch — "
            "force-close marker will leak past budget on next chunk"
        )
        pp.process_chunk(_make_output("y"))
        # Chunk 2's delta must start with the injected </think>.
        assert parser.calls[1]["delta"].startswith("</think>"), (
            f"expected </think> injection on chunk 2 after exact-boundary "
            f"cap hit, got delta={parser.calls[1]['delta']!r}"
        )


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
