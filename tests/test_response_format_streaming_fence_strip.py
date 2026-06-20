# SPDX-License-Identifier: Apache-2.0
"""H-07: streaming + json_mode must NOT leak ```json ... ``` markdown fence.

The non-streaming chat response builder peels a markdown ``` ```json ```
wrapper via ``extract_json_from_response`` (vllm_mlx/api/utils.py) AFTER
assembling the full text. The streaming path concatenated raw model
tokens WITHOUT the same scrub — joined SSE deltas decoded as
``` ```json\\n{...}\\n``` ``` and ``json.loads`` failed for any SDK
consumer assembling ``delta.content`` into a string.

Marisol (0.8TODO r2 H-07) caught the regression: same prompt + same
model + ``response_format={"type":"json_object"}`` + ``stream=True``
produced fenced output while the non-stream path produced bare JSON.

This test file pins the fence-strip state machine in
``StreamingPostProcessor`` (see ``_apply_json_fence_strip``). Design
rationale (state machine in delta builder, not post-join regex):

  * Fence tokens are split across delta chunks. Tokenizers fragment
    ``\\n``` `` arbitrarily ("``", "`json", "\\n"); a post-emission
    regex would not help because we need to SUPPRESS bytes BEFORE
    they reach the wire.
  * The bare-JSON path (model returns ``{...}`` with no fence at all)
    must pass through unchanged — we can't unconditionally buffer.
  * Non-stream regression: the non-stream
    ``extract_json_from_response`` path must keep peeling the fence
    via its existing logic. We verify it still does.

The tests below exercise the postprocessor directly with mocked
``GenerationOutput`` chunks — the same surface area the live SSE route
goes through (``stream_chat_completion`` -> ``processor.process_chunk``
-> ``_filter_events_for_json_fence``).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from vllm_mlx.api.utils import extract_json_from_response
from vllm_mlx.service.postprocessor import StreamingPostProcessor


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


def _make_output(
    text="", finished=False, channel=None, finish_reason=None, tool_calls=None
):
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = channel
    out.finish_reason = finish_reason or ("stop" if finished else None)
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    out.tool_calls = tool_calls
    return out


def _stream_chunks(pp: StreamingPostProcessor, chunks: list[str]) -> str:
    """Feed chunks one-by-one to the postprocessor, joining emitted content.

    Mirrors what the SSE route does: every ``type="content"`` event
    contributes to the joined ``delta.content`` string a client would
    reassemble. Tool-call / reasoning / finish events are ignored for
    fence-strip assertions — H-07 is strictly about the content channel.
    """
    joined = ""
    for chunk in chunks:
        for ev in pp.process_chunk(_make_output(chunk)):
            if ev.type in ("content", "finish") and ev.content:
                joined += ev.content
    for ev in pp.finalize():
        if ev.type == "content" and ev.content:
            joined += ev.content
    return joined


class TestJsonObjectFenceStripping:
    """``response_format={"type":"json_object"}`` + stream=True."""

    def test_full_fence_single_chunk(self):
        """Whole ``` ```json\\n{...}\\n``` `` arrives in one delta."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        joined = _stream_chunks(
            pp, ['```json\n{"name": "iPhone 15", "price": 799.99}\n```']
        )
        # Byte-exact: matches what the non-stream path produces.
        assert joined == '{"name": "iPhone 15", "price": 799.99}'

    def test_fence_split_across_token_boundaries(self):
        """Fence emitted token-by-token (realistic SSE granularity).

        The state machine MUST handle ``\\n```\\n`` being fragmented
        across deltas — the tokenizer routinely splits the closing
        fence into ``\\n``, ``` `` ``, ``json`` style pieces.
        """
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        # Realistic token-level fragmentation captured against
        # Qwen3-0.6B-8bit during the H-07 repro.
        chunks = [
            "```",
            "json",
            "\n",
            "{",
            '"name"',
            ": ",
            '"iPhone 15"',
            ", ",
            '"price"',
            ": ",
            "799.99",
            "}",
            "\n",
            "```",
        ]
        joined = _stream_chunks(pp, chunks)
        assert joined == '{"name": "iPhone 15", "price": 799.99}'

    def test_opening_fence_split_two_chunks(self):
        """Opening ``` ```json `` straddles chunks 1 and 2."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        # Backtick run lands in chunk 1; ``json\n`` + body land in chunk 2.
        joined = _stream_chunks(
            pp,
            [
                "``",
                '`json\n{"k": 1}\n```',
            ],
        )
        assert joined == '{"k": 1}'

    def test_closing_fence_split_two_chunks(self):
        """Closing ``` ``` `` straddles the last two chunks.

        Codex r2 BLOCKING: assert byte-identical equality with the
        non-stream output (bare ``{"k": 1}``). The earlier draft only
        checked ``json.loads`` + no-backticks, which silently passed
        even when a trailing ``\\n`` slipped onto the wire because
        the suffix-hold released the newline before the next chunk's
        backtick completed the fence.
        """
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        joined = _stream_chunks(
            pp,
            [
                '```json\n{"k": 1}\n``',
                "`",
            ],
        )
        # Exact byte equality with the non-stream shape — no leaked
        # trailing newline, no leaked backticks.
        assert joined == '{"k": 1}'

    def test_closing_fence_newline_split_from_backticks(self):
        """Codex r2 BLOCKING: closing fence split as ``\\n`` then
        ``` ``` ``. The hold MUST cover the newline, otherwise
        chunk 1's ``\\n`` lands on the wire before chunk 2's
        backticks transition the state machine to ``done``."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        joined = _stream_chunks(
            pp,
            [
                '```json\n{"k": 1}',
                "\n",
                "```",
            ],
        )
        assert joined == '{"k": 1}'

    def test_closing_fence_newline_plus_two_backticks_split(self):
        """Closing fence split as ``\\n```` `` then ``` ` ``.

        Tests the case codex r2 specifically called out: chunk N ends
        ``...}\\n``` ``; hold must include the ``\\n`` so the next
        chunk's third backtick can close cleanly."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        joined = _stream_chunks(
            pp,
            [
                '```json\n{"k": 1}\n``',
                "`\n",
            ],
        )
        assert joined == '{"k": 1}'

    def test_bare_json_no_fence_passes_through(self):
        """Model returns bare ``{...}`` — NO fence anywhere.

        The state machine must NOT introduce hold-back delays or
        partial deltas on this path; bytes flow through end-to-end.
        """
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        joined = _stream_chunks(
            pp,
            [
                '{"name": ',
                '"iPhone 15", ',
                '"price": 799.99}',
            ],
        )
        assert json.loads(joined) == {"name": "iPhone 15", "price": 799.99}
        # No fence, no leak.
        assert "```" not in joined

    def test_fence_preceded_by_whitespace(self):
        """Model emits whitespace before the opening fence."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        joined = _stream_chunks(pp, ['\n\n```json\n{"k": 1}\n```\n'])
        assert json.loads(joined) == {"k": 1}

    def test_array_payload(self):
        """JSON arrays must work too — the spec allows ``[...]`` root."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        joined = _stream_chunks(
            pp,
            ['```json\n[{"item": 1}, {"item": 2}]\n```'],
        )
        assert json.loads(joined) == [{"item": 1}, {"item": 2}]

    def test_trailing_newline_after_fence_suppressed(self):
        """``\\n``` `` plus a trailing ``\\n`` — the trailing nl is dropped."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        joined = _stream_chunks(pp, ['```json\n{"k": 1}\n```\n\n'])
        assert json.loads(joined) == {"k": 1}

    def test_triple_backticks_inside_json_string_preserved(self):
        """Codex r1 BLOCKING: a JSON STRING VALUE containing literal
        triple-backticks must NOT be truncated by the closing-fence
        scanner. The state machine tracks JSON-string state via a
        cheap ``"`` toggle, skipping fence detection inside string
        literals. Mirrors a real-world response_format payload where
        the model returns a code snippet as a string value."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        # The JSON value carries the literal characters ``` ```python ```,
        # a newline (encoded as ``\\n``), ``x``, another newline, and
        # ``` ``` ``. All of this is INSIDE the string literal.
        joined = _stream_chunks(
            pp,
            [
                '```json\n{"markdown": "',
                "```python\\nx\\n```",
                '"}\n```',
            ],
        )
        # The fence-strip must have peeled the OUTER fence and left
        # the inner triple-backticks alone.
        assert json.loads(joined) == {"markdown": "```python\nx\n```"}

    def test_bare_json_string_ends_with_backticks(self):
        """Codex r1 BLOCKING #2: bare JSON whose final string value
        legitimately ends with backticks must survive the finalize
        flush. The earlier draft rstripped trailing backticks in
        ``_flush_json_fence_tail`` at EOS, corrupting these payloads.

        Stream the closing ``"}`` after the trailing backtick lands
        in its own delta — exactly the chunk-boundary that fooled
        the old flush."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        joined = _stream_chunks(
            pp,
            [
                '{"text": "look: `',
                "`",
                "`",
                '"}',
            ],
        )
        assert json.loads(joined) == {"text": "look: ```"}


class TestJsonSchemaFenceStripping:
    """``response_format={"type":"json_schema",...}`` + stream=True."""

    def test_json_schema_fence_stripped(self):
        """The strip applies to ``json_schema`` requests, not just
        ``json_object`` — the route layer passes the same
        ``json_mode=True`` flag for both ``json_object`` and
        ``json_schema`` (see ``stream_chat_completion`` in
        vllm_mlx/routes/chat.py around the ``json_mode=`` kwarg)."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()
        joined = _stream_chunks(pp, ['```json\n{"answer": 42, "valid": true}\n```'])
        assert json.loads(joined) == {"answer": 42, "valid": True}


class TestNoResponseFormatPassThrough:
    """Streaming WITHOUT ``response_format`` MUST pass any ``` through."""

    def test_fence_in_content_passes_through_when_json_mode_off(self):
        """When the client did NOT request structured output, a model
        that happens to emit ``` should reach the wire — it's plain
        markdown content. The strip is gated on ``json_mode``."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=False)
        pp.reset()
        joined = _stream_chunks(pp, ["Here is some code:\n```python\nx = 1\n```"])
        # Fence must survive — no strip happened.
        assert "```python" in joined
        assert "x = 1" in joined

    def test_fence_in_content_passes_through_no_json_mode_split(self):
        """Same as above, but with the fence split across chunks."""
        cfg = _make_cfg()
        pp = StreamingPostProcessor(cfg, json_mode=False)
        pp.reset()
        joined = _stream_chunks(pp, ["``", "`json\n{}\n``", "`"])
        assert joined.count("```") == 2


class TestNonStreamRegression:
    """Non-stream fence-strip MUST keep working — H-07 is stream-only.

    The non-stream chat response builder calls
    ``extract_json_from_response`` to peel ``` ```json\\n{...}\\n``` ```.
    These tests pin that the helper still peels the wrapper after the
    streaming-side state machine landed.
    """

    def test_non_stream_helper_strips_fence(self):
        wrapped = '```json\n{"name": "iPhone 15", "price": 799.99}\n```'
        peeled = extract_json_from_response(wrapped)
        assert json.loads(peeled) == {"name": "iPhone 15", "price": 799.99}

    def test_non_stream_helper_passes_bare_json(self):
        bare = '{"k": 1}'
        peeled = extract_json_from_response(bare)
        assert peeled == bare
        assert json.loads(peeled) == {"k": 1}

    def test_non_stream_helper_strips_bare_fence_no_json_lang_tag(self):
        wrapped = '```\n{"k": 1}\n```'
        peeled = extract_json_from_response(wrapped)
        assert json.loads(peeled) == {"k": 1}


class TestReasoningParserPath:
    """When a reasoning parser is active, the fence-strip still applies.

    The existing ``_json_preamble_buffer`` path is SKIPPED when a
    reasoning parser is wired (vllm_mlx/service/postprocessor.py:
    ``_process_standard`` gate ``not self.reasoning_parser``), so a
    reasoning model emitting ``` ```json ``` after ``</think>`` would
    previously leak the fence into ``delta.content``. The new state
    machine in ``_filter_events_for_json_fence`` runs in the OUTER
    ``process_chunk`` dispatcher and therefore covers the
    reasoning-parser path too.
    """

    def test_reasoning_parser_path_strips_fence(self):
        """Mock reasoning parser that routes everything to content
        AFTER it sees ``</think>`` — same as the production
        ``Qwen3ReasoningParser`` etc."""
        parser = MagicMock()
        emitted_content = []

        def _fake_extract(prev, curr, delta):
            # Strip ``<think>...</think>`` if present, route the rest
            # to content. Mirrors the behaviour of the live reasoning
            # parsers' streaming path.
            msg = MagicMock()
            msg.content = delta
            msg.reasoning = None
            emitted_content.append(delta)
            return msg

        parser.extract_reasoning_streaming.side_effect = _fake_extract

        cfg = _make_cfg(reasoning_parser=parser)
        pp = StreamingPostProcessor(cfg, json_mode=True)
        pp.reset()

        joined = _stream_chunks(pp, ['```json\n{"k": 1}\n```'])
        assert json.loads(joined) == {"k": 1}
        assert "```" not in joined
