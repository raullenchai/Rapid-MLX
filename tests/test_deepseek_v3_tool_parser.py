# SPDX-License-Identifier: Apache-2.0
"""
D-DSV31 regression: DeepSeek V3 wire-format support in the
``deepseek_v31`` parser.

DeepSeek-R1-0528-Qwen3-8B's chat_template.jinja was inherited from
DeepSeek-V3 and emits the V3 "function-typed, JSON-fenced" tool-call
body shape:

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
    ```json
    {ARGS}
    ```<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

The original ``deepseek_v31`` parser only recognised the V3.1 thinking
channel shape (``<call_begin>NAME<sep>ARGS<call_end>``) and
mis-parsed the V3 shape as ``name="function"`` /
``arguments="NAME\\n```json\\n{...}\\n```"``. This file pins the
auto-detect behaviour.

Why these tests use the DeepSeekV31ToolParser directly and not the
``deepseek`` alias: ``deepseek-r1-8b-4bit`` in ``aliases.json`` is
configured with ``tool_call_parser: deepseek_v31``, so this is the
exact parser instance the server uses for that checkpoint.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from vllm_mlx.service.postprocessor import StreamingPostProcessor
from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.deepseekv31_tool_parser import DeepSeekV31ToolParser

# Wire-format building blocks — all fullwidth pipes (U+FF5C), exactly as
# the DeepSeek chat template emits them. We construct payloads from
# these primitives so the tests double as documentation of the wire
# format the parser is contracted against.
TC_OPEN = "<｜tool▁calls▁begin｜>"
TC_CLOSE = "<｜tool▁calls▁end｜>"
C_OPEN = "<｜tool▁call▁begin｜>"
C_CLOSE = "<｜tool▁call▁end｜>"
SEP = "<｜tool▁sep｜>"


def _v3_block(name: str, args_json: str) -> str:
    """Construct a single V3-format tool-call block body."""
    return f"{C_OPEN}function{SEP}{name}\n```json\n{args_json}\n```{C_CLOSE}"


def _v31_block(name: str, args_body: str) -> str:
    """Construct a single V3.1-format tool-call block body."""
    return f"{C_OPEN}{name}{SEP}{args_body}{C_CLOSE}"


def _envelope(*blocks: str, prefix: str = "") -> str:
    """Wrap one-or-more blocks in the outer ``<calls_begin>`` envelope."""
    return f"{prefix}{TC_OPEN}{''.join(blocks)}{TC_CLOSE}"


@pytest.fixture
def parser() -> DeepSeekV31ToolParser:
    return DeepSeekV31ToolParser()


# --------------------------------------------------------------------
# Verify wire-format primitives are constructed with the *correct*
# fullwidth-pipe Unicode codepoint (U+FF5C) — not the ASCII pipe — and
# match what the real chat_template.jinja emits. Catches accidental
# normalisation of the test data itself.
# --------------------------------------------------------------------
def test_wire_format_primitives_use_fullwidth_pipe() -> None:
    for s in (TC_OPEN, TC_CLOSE, C_OPEN, C_CLOSE, SEP):
        assert "｜" in s, f"{s!r} missing fullwidth pipe"
        assert "|" not in s, f"{s!r} leaked ASCII pipe"


# --------------------------------------------------------------------
# Routing: aliases.json points ``deepseek-r1-8b-4bit`` at the
# ``deepseek_v31`` parser. Confirm the registry lookup the server
# performs (``ToolParserManager.get_tool_parser("deepseek_v31")``)
# returns this fixed class — i.e. the bug fix actually reaches the
# request path the server uses for that alias.
# --------------------------------------------------------------------
@pytest.mark.parametrize("name", ["deepseek_v31", "deepseek_r1_0528"])
def test_registry_lookup_returns_fixed_parser(name: str) -> None:
    cls = ToolParserManager.get_tool_parser(name)
    assert cls is DeepSeekV31ToolParser


# --------------------------------------------------------------------
# V3 wire format — the D-DSV31 P0 case.
# --------------------------------------------------------------------
class TestV3WireFormat:
    """Direct V3-shaped payloads: ``function<sep>NAME\\n```json\\n{...}\\n```\\n``."""

    def test_single_v3_tool_call(self, parser: DeepSeekV31ToolParser) -> None:
        payload = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))

        result = parser.extract_tool_calls(payload)

        assert result.tools_called, "V3-shaped payload must trigger tools_called"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        # The bug: name == "function" and arguments started with "get_weather\n```json…"
        assert tc["name"] == "get_weather", (
            f"V3 type-tag leak: parser returned name={tc['name']!r}"
        )
        assert json.loads(tc["arguments"]) == {"city": "Tokyo"}

    def test_parallel_v3_tool_calls(self, parser: DeepSeekV31ToolParser) -> None:
        """N V3 blocks in one envelope must produce N tool calls."""
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            _v3_block("get_time", '{"tz": "UTC"}'),
            _v3_block("search", '{"q": "deepseek"}'),
        )

        result = parser.extract_tool_calls(payload)

        assert result.tools_called
        assert len(result.tool_calls) == 3
        names = [c["name"] for c in result.tool_calls]
        assert names == ["get_weather", "get_time", "search"]
        # Pin all three argument bodies — protects against any "one
        # block swallows the next" greedy-regex regression.
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Tokyo"}
        assert json.loads(result.tool_calls[1]["arguments"]) == {"tz": "UTC"}
        assert json.loads(result.tool_calls[2]["arguments"]) == {"q": "deepseek"}

    def test_v3_with_leading_content(self, parser: DeepSeekV31ToolParser) -> None:
        """Reasoning text before the envelope must be preserved as content."""
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            prefix="Let me check the weather. ",
        )

        result = parser.extract_tool_calls(payload)

        assert result.tools_called
        assert result.content == "Let me check the weather. "
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_v3_arguments_with_nested_braces(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """Nested JSON inside the fenced body must not confuse the fence detector."""
        args = '{"filter": {"city": "Tokyo", "tags": ["a", "b"]}, "limit": 10}'
        payload = _envelope(_v3_block("search", args))

        result = parser.extract_tool_calls(payload)

        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == json.loads(args)


# --------------------------------------------------------------------
# V3.1 wire format — regression: pre-existing behaviour must hold.
# --------------------------------------------------------------------
class TestV31WireFormat:
    """Plain ``NAME<sep>ARGS`` body — the original V3.1 thinking-channel shape."""

    def test_single_v31_tool_call(self, parser: DeepSeekV31ToolParser) -> None:
        payload = _envelope(_v31_block("get_weather", '{"city": "Paris"}'))

        result = parser.extract_tool_calls(payload)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Paris"}

    def test_parallel_v31_tool_calls(self, parser: DeepSeekV31ToolParser) -> None:
        payload = _envelope(
            _v31_block("f1", '{"a": 1}'),
            _v31_block("f2", '{"b": 2}'),
        )

        result = parser.extract_tool_calls(payload)

        assert len(result.tool_calls) == 2
        assert [c["name"] for c in result.tool_calls] == ["f1", "f2"]

    def test_v31_with_tool_named_function_is_not_misclassified(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """A V3.1 tool literally named ``function_lookup`` (or anything
        starting with ``function``) must NOT be sniffed as V3.

        The V3 sniffer anchors on ``function<sep>`` — the separator
        immediately after the literal type tag — so this only fires
        when the V3 type tag is present.
        """
        payload = _envelope(_v31_block("function_lookup", '{"q": "x"}'))

        result = parser.extract_tool_calls(payload)

        assert result.tool_calls[0]["name"] == "function_lookup"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"q": "x"}


# --------------------------------------------------------------------
# Mixed-shape envelope — V3 and V3.1 blocks coexisting within one
# outer ``<calls_begin>`` envelope. Not seen in any single checkpoint
# today but the block-wise sniffer is supposed to handle them per-block.
# --------------------------------------------------------------------
def test_mixed_v3_and_v31_blocks(parser: DeepSeekV31ToolParser) -> None:
    payload = _envelope(
        _v3_block("get_weather", '{"city": "Tokyo"}'),
        _v31_block("get_time", '{"tz": "UTC"}'),
    )

    result = parser.extract_tool_calls(payload)

    assert result.tools_called
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0]["name"] == "get_weather"
    assert result.tool_calls[1]["name"] == "get_time"


# --------------------------------------------------------------------
# Malformed payloads — must not raise, must not silently consume content.
# --------------------------------------------------------------------
class TestMalformedGraceful:
    def test_no_envelope_passes_through(self, parser: DeepSeekV31ToolParser) -> None:
        text = "Just plain reasoning, no tools here."
        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == text

    def test_truncated_block_falls_back_to_content(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """Outer ``<calls_begin>`` present but the block never closes."""
        payload = f"prefix {TC_OPEN}{C_OPEN}function{SEP}get_weather\n```json\n{{"

        result = parser.extract_tool_calls(payload)

        assert not result.tools_called
        assert result.tool_calls == []
        # Full text passes through — caller can decide what to do with
        # the unfinished generation rather than the parser silently
        # dropping everything after the begin marker.
        assert result.content == payload

    def test_envelope_with_no_blocks(self, parser: DeepSeekV31ToolParser) -> None:
        """``<calls_begin>...<calls_end>`` with nothing inside."""
        payload = f"{TC_OPEN}{TC_CLOSE}"

        result = parser.extract_tool_calls(payload)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == payload

    def test_block_missing_separator(self, parser: DeepSeekV31ToolParser) -> None:
        """Block envelope but no ``<sep>`` — un-parseable body."""
        payload = f"{TC_OPEN}{C_OPEN}garbage_no_sep_here{C_CLOSE}{TC_CLOSE}"

        result = parser.extract_tool_calls(payload)

        assert not result.tools_called
        assert result.tool_calls == []
        # No usable calls extracted; envelope-present payload passes
        # through so the caller can surface or strip it as needed.
        assert result.content == payload

    def test_one_good_one_bad_block(self, parser: DeepSeekV31ToolParser) -> None:
        """A malformed block must not invalidate sibling good blocks."""
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            f"{C_OPEN}no_sep_here{C_CLOSE}",
            _v3_block("get_time", '{"tz": "UTC"}'),
        )

        result = parser.extract_tool_calls(payload)

        assert result.tools_called
        # The bad block in the middle is dropped; good blocks survive.
        assert len(result.tool_calls) == 2
        assert [c["name"] for c in result.tool_calls] == ["get_weather", "get_time"]

    def test_v3_anchored_body_with_broken_fence_does_not_mis_emit_function(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """codex r2 BLOCKING: a body that anchors as V3 (starts with
        ``function<sep>``) but doesn't contain a parseable
        ``\\n``\\`\\`\\`json…`` fence MUST NOT fall through to the V3.1
        ``NAME<sep>ARGS`` split.

        That fallthrough would emit ``name="function"`` with the real
        tool name embedded in ``arguments`` — i.e. recreate the exact
        production failure mode D-DSV31 was filed for.
        """
        # V3 anchor, no fence at all (just ``function<sep>get_weather``
        # then nothing JSON-ish). Must NOT emit ``name="function"``.
        payload = (
            f"{TC_OPEN}{C_OPEN}function{SEP}get_weather is the name{C_CLOSE}{TC_CLOSE}"
        )

        result = parser.extract_tool_calls(payload)

        for tc in result.tool_calls:
            assert tc["name"] != "function", (
                f"V3-anchored malformed body leaked into V3.1 split: {tc!r}"
            )

    def test_literal_marker_text_before_envelope_does_not_misparse(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """codex r8 BLOCKING-1: a response that mentions
        ``<｜tool▁call▁begin｜>`` as literal content (e.g. in a docs
        explanation) and later happens to contain
        ``<｜tool▁calls▁begin｜>`` MUST NOT have the literal text
        treated as a tool call. The block scanner is bounded to the
        outer envelope.
        """
        # Literal mention in plain content, then a real envelope with
        # a real tool call.
        prose = (
            "Here's how DeepSeek tool calls work: "
            f"the format uses {C_OPEN}NAME{SEP}ARGS{C_CLOSE}. For example:\n\n"
        )
        real_call = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))
        payload = prose + real_call

        result = parser.extract_tool_calls(payload)

        # Exactly ONE tool call (the real one inside the envelope),
        # not multiple from the literal markers in prose.
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_envelope_with_truncated_trailing_block_preserves_text(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """codex r8 BLOCKING-2: a response with [valid V3 block,
        truncated trailing block] must surface the truncated text in
        ``content`` rather than silently dropping it. Without
        preservation the user loses the model's partial output.
        """
        good_block = _v3_block("get_weather", '{"city": "Tokyo"}')
        truncated_tail = f"{C_OPEN}function{SEP}get_time\n```json\n{{"  # no close
        payload = f"{TC_OPEN}{good_block}{truncated_tail}"

        result = parser.extract_tool_calls(payload)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        # The truncated trailing text MUST appear somewhere in content
        # so the caller has access to it.
        assert result.content is not None
        assert "get_time" in result.content, (
            f"Truncated trailing block text dropped from content. "
            f"content={result.content!r}"
        )

    def test_v3_anchored_body_with_recoverable_partial_fence(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """V3 anchor + name + opening JSON fence + body but no closing
        fence (truncated mid-args): the bounded recovery should pick up
        the name and pass the body through as args, rather than
        emitting ``name="function"``.
        """
        payload = (
            f"{TC_OPEN}{C_OPEN}function{SEP}get_weather\n"
            f'```json\n{{"city": "Tokyo"'
            f"{C_CLOSE}{TC_CLOSE}"
        )

        result = parser.extract_tool_calls(payload)

        # If we extracted anything, the name MUST be the real tool
        # name — never the literal ``function`` type tag.
        for tc in result.tool_calls:
            assert tc["name"] == "get_weather", (
                f"V3 partial-fence recovery wrong name: {tc!r}"
            )


# --------------------------------------------------------------------
# Streaming — V3 streams suppress mid-stream emission and defer the
# tool-call emission to the postprocessor's end-of-stream finalize
# path (``service/postprocessor.py``: "Fallback tool call detection"),
# which calls ``extract_tool_calls`` on the cumulative buffer.
#
# Why: the legacy delta machine would otherwise leak ``name="function"``
# mid-stream for V3 bodies. After several rounds of attempted
# delta-by-delta short-circuits each composed badly with malformed
# blocks / mixed V3+V3.1 envelopes / delta boundaries, the conservative
# contract is: do not attempt to stream V3 — let the closed-buffer
# extract handle it as a one-shot event. This matches how other
# wrapped-format parsers (GLM-4.7, Seed-OSS) already behave.
#
# V3.1 streaming is unchanged and covered by
# ``test_upstream_regression.TestDeepSeekV31UpstreamStreaming``.
# --------------------------------------------------------------------
class TestV3Streaming:
    def _feed(
        self, parser: DeepSeekV31ToolParser, payload: str, chunk_size: int = 8
    ) -> list[dict | None]:
        """Replay ``payload`` through ``extract_tool_calls_streaming``
        in fixed-size chunks. Returns the per-delta return values so
        tests can assert what was emitted at each step.
        """
        results: list[dict | None] = []
        prev = ""
        for i in range(0, len(payload), chunk_size):
            delta = payload[i : i + chunk_size]
            cur = prev + delta
            r = parser.extract_tool_calls_streaming(
                previous_text=prev,
                current_text=cur,
                delta_text=delta,
            )
            results.append(r)
            prev = cur
        return results

    def _no_streamed_tool_calls(self, events: list[dict | None]) -> None:
        """Assert no mid-stream ``tool_calls`` event leaked from the
        streaming path for a V3 payload."""
        for ev in events:
            if not ev:
                continue
            assert "tool_calls" not in ev, (
                f"V3 stream leaked a mid-stream tool_calls event: {ev!r}. "
                f"V3 streams must defer emission to the postprocessor "
                f"end-of-stream finalize path."
            )

    def test_v3_stream_emits_no_mid_stream_tool_calls(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """Single V3 block: streaming path returns no ``tool_calls``
        events (in particular: NO ``name="function"`` deltas).
        Emission is deferred to the postprocessor's end-of-stream
        finalize, which calls ``extract_tool_calls`` and gets the
        correct ``name="get_weather"``.
        """
        payload = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))

        events = self._feed(parser, payload, chunk_size=12)
        self._no_streamed_tool_calls(events)

        # And the non-streaming extract on the cumulative text emits
        # the correct call — confirming the postprocessor's finalize
        # path will produce the right tool_call.
        result = parser.extract_tool_calls(payload)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Tokyo"}

    def test_v3_parallel_stream_emits_no_mid_stream_tool_calls(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """Multiple V3 blocks: still no mid-stream emission. Finalize
        path produces all the calls in one shot."""
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            _v3_block("get_time", '{"tz": "UTC"}'),
        )

        events = self._feed(parser, payload, chunk_size=18)
        self._no_streamed_tool_calls(events)

        result = parser.extract_tool_calls(payload)
        assert len(result.tool_calls) == 2
        assert [c["name"] for c in result.tool_calls] == ["get_weather", "get_time"]

    def test_mixed_v31_then_v3_stream_no_function_leak(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """Mixed envelope [V3.1, V3]: documented known limitation
        (parser docstring, codex r7) — the postprocessor's finalize
        fallback gates on ``not tool_calls_detected``, so a V3.1 block
        streamed first would prevent finalize from reconciling the
        suppressed V3 block. Mixed envelopes don't occur in practice
        on released DeepSeek checkpoints (each chat template emits a
        single body shape per envelope).

        What we DO require: even in a mixed-envelope stream, the V3
        block must not leak ``name="function"`` mid-stream. The
        suppression engages the moment the V3 anchor arrives, so any
        legacy-machine emission for the V3 block is short-circuited
        before the type tag becomes the streamed name.
        """
        payload = _envelope(
            _v31_block("first_call", '{"a": 1}'),
            _v3_block("second_call", '{"b": 2}'),
        )

        events = self._feed(parser, payload, chunk_size=12)

        for ev in events:
            if not ev or "tool_calls" not in ev:
                continue
            for call in ev["tool_calls"]:
                assert call.get("function", {}).get("name") != "function", (
                    f"V3 type-tag leaked into a streaming name: {call!r}"
                )

        # Non-streaming extract on the same payload still produces
        # both calls correctly — the parser itself isn't lossy on
        # mixed envelopes; the limitation lives in the streaming +
        # postprocessor contract.
        result = parser.extract_tool_calls(payload)
        assert len(result.tool_calls) == 2
        assert [c["name"] for c in result.tool_calls] == [
            "first_call",
            "second_call",
        ]

    def test_v31_tool_named_with_function_prefix_streams_normally(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """A V3.1 tool named ``function_lookup`` (whose name is a
        prefix of the V3 type-tag-plus-sep marker) must NOT trip the
        V3 suppression. ``_cumulative_has_v3_block`` anchors on the
        literal ``function<sep>`` byte sequence — ``function_lookup``
        never reaches that exact prefix.
        """
        payload = _envelope(_v31_block("function_lookup", '{"q": "x"}'))

        events = self._feed(parser, payload, chunk_size=1)

        # The legacy V3.1 path must have emitted the real name.
        names = [
            call.get("function", {}).get("name")
            for ev in events
            if ev and "tool_calls" in ev
            for call in ev["tool_calls"]
            if call.get("function", {}).get("name")
        ]
        assert "function_lookup" in names, (
            f"V3.1 tool 'function_lookup' was hijacked by the V3 "
            f"suppression. Events: {events!r}"
        )

    def test_v31_stream_empty_block_body_does_not_trigger_v3_suppression(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """``"".startswith(...)`` is True for every string — a V3.1
        stream that ended right after ``<call_begin>`` (no body byte
        yet) must NOT trip V3 suppression and silently swallow the
        legacy delta machine's emissions.
        """
        payload = _envelope(_v31_block("get_weather", '{"city": "Paris"}'))

        events = self._feed(parser, payload, chunk_size=1)

        names_emitted = [
            call.get("function", {}).get("name")
            for ev in events
            if ev and "tool_calls" in ev
            for call in ev["tool_calls"]
            if call.get("function", {}).get("name")
        ]
        assert "get_weather" in names_emitted, (
            f"V3.1 stream lost its name event — V3 suppression may "
            f"have hijacked the empty-body intermediate state. Events: "
            f"{events!r}"
        )


# --------------------------------------------------------------------
# Integration: V3 streaming through the real ``StreamingPostProcessor``
# finalize path. Confirms the suppression + finalize composition
# actually delivers a ``tool_call`` event to clients — i.e. it would
# fail if the parser-level suppression worked but the postprocessor's
# fallback gate didn't pick the V3 calls up at end-of-stream.
# (codex r9 BLOCKING-3: parser-only tests can't prove this.)
# --------------------------------------------------------------------
def _make_postprocessor_cfg() -> MagicMock:
    """Mock ServerConfig minimally configured to route through the
    ``deepseek_v31`` tool parser path."""
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "deepseek_v31"
    cfg.tool_parser_instance = None
    return cfg


def _make_generation_output(
    text: str, finished: bool = False, finish_reason: str | None = None
) -> MagicMock:
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = None
    out.finish_reason = finish_reason or ("stop" if finished else None)
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    out.tool_calls = None
    return out


class TestV3StreamingIntegration:
    """End-to-end through ``StreamingPostProcessor`` (codex r9)."""

    def test_v3_payload_emits_tool_call_via_finalize(self) -> None:
        """Drive a V3-shaped payload through the real postprocessor
        chunk-by-chunk and confirm ``finalize()`` emits a
        ``tool_call`` event with ``name="get_weather"``. Pins the
        suppression + fallback composition end-to-end.
        """
        cfg = _make_postprocessor_cfg()
        pp = StreamingPostProcessor(cfg, tools_requested=True)
        pp.reset()

        payload = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))

        # Stream in mid-sized chunks. None of these should emit a
        # ``tool_call`` event — the parser-level suppression returns
        # ``None`` for every V3 delta.
        all_events = []
        for i in range(0, len(payload), 16):
            chunk = payload[i : i + 16]
            is_last = i + 16 >= len(payload)
            output = _make_generation_output(
                chunk,
                finished=is_last,
                finish_reason="tool_calls" if is_last else None,
            )
            all_events.extend(pp.process_chunk(output))

        all_events.extend(pp.finalize())

        # Collect tool_call events from the full event stream.
        tool_call_events = [e for e in all_events if e.type == "tool_call"]
        assert tool_call_events, (
            f"No tool_call event emitted end-to-end. Events: "
            f"{[(e.type, getattr(e, 'tool_calls', None)) for e in all_events]!r}"
        )
        # Aggregate names across emitted events (finalize may emit
        # one event with multiple calls).
        names = []
        for ev in tool_call_events:
            for tc in ev.tool_calls or []:
                names.append(tc.get("function", {}).get("name") or tc.get("name"))
        assert "get_weather" in names, (
            f"V3 finalize path lost the real tool name. Names: {names!r}"
        )
        # And the leaked V3 type tag must NEVER appear.
        assert "function" not in names, (
            f"V3 type-tag leaked into the final tool_call event. Names: {names!r}"
        )


# --------------------------------------------------------------------
# Argument-body passthrough contract.
#
# Upstream vLLM emits the model's argument bytes verbatim (modulo
# leading/trailing whitespace) — see
# ``tests/test_upstream_regression.py::TestDeepSeekV31UpstreamNonStreaming``.
# JSON-canonicalisation is the caller's job; doing it here would break
# bytes-equal regression assertions.
# --------------------------------------------------------------------
class TestArgumentBytesPassthrough:
    def test_valid_json_args_passed_through_verbatim(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """Args body bytes survive end-to-end (no canonicalisation)."""
        args = '{   "k"  :  1  }'
        payload = _envelope(_v3_block("f", args))

        result = parser.extract_tool_calls(payload)

        # Bytes match modulo the leading/trailing whitespace strip that
        # ``_parse_block_body`` applies; the inner spaces survive.
        assert result.tool_calls[0]["arguments"] == args
        # And the body is still loadable when the caller wants.
        assert json.loads(result.tool_calls[0]["arguments"]) == {"k": 1}

    def test_non_json_args_passed_through(self, parser: DeepSeekV31ToolParser) -> None:
        """Non-JSON args body (a quirky V3.1 free-form payload) is
        passed through verbatim — we don't try to manufacture JSON we
        didn't receive."""
        payload = _envelope(_v31_block("explain", "free-form text body"))

        result = parser.extract_tool_calls(payload)

        assert result.tool_calls[0]["arguments"] == "free-form text body"
