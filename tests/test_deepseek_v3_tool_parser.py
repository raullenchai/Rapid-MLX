# SPDX-License-Identifier: Apache-2.0
"""
D-DSV31 / R12-5 regression: DeepSeek V3 wire-format support.

DeepSeek-R1-0528-Qwen3-8B's chat_template.jinja was inherited from
DeepSeek-V3 and emits the V3 "function-typed, JSON-fenced" tool-call
body shape:

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
    ```json
    {ARGS}
    ```<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

History
-------
* D-DSV31 (PR #795 / 0.8.2): extended ``DeepSeekV31ToolParser`` with
  auto-detection of the V3 wire shape so R1-0528 stopped emitting
  ``name="function"`` with the real name leaking into ``arguments``.
* R12-5 (0.8.14): split that auto-detect logic into its own
  ``DeepSeekV3ToolParser`` (``deepseek_v3_tool_parser.py``) so each
  parser owns exactly one wire shape. ``aliases.json`` routes
  R1-0528 to ``deepseek_v3``; V3.1 thinking-channel models stay on
  ``deepseek_v31``.

This file covers BOTH the V3 path (against the new parser) and the
V3.1 path (against the unchanged parser) and pins the cross-shape
boundary that the split makes structural rather than runtime.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from vllm_mlx.service.postprocessor import StreamingPostProcessor
from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.deepseek_v3_tool_parser import DeepSeekV3ToolParser
from vllm_mlx.tool_parsers.deepseekv31_tool_parser import DeepSeekV31ToolParser

# Wire-format building blocks — all fullwidth pipes (U+FF5C).
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
def v3_parser() -> DeepSeekV3ToolParser:
    return DeepSeekV3ToolParser()


@pytest.fixture
def v31_parser() -> DeepSeekV31ToolParser:
    return DeepSeekV31ToolParser()


# --------------------------------------------------------------------
# Verify wire-format primitives are constructed with the *correct*
# fullwidth-pipe Unicode codepoint (U+FF5C) — not the ASCII pipe — and
# match what the real chat_template.jinja emits.
# --------------------------------------------------------------------
def test_wire_format_primitives_use_fullwidth_pipe() -> None:
    for s in (TC_OPEN, TC_CLOSE, C_OPEN, C_CLOSE, SEP):
        assert "｜" in s, f"{s!r} missing fullwidth pipe"
        assert "|" not in s, f"{s!r} leaked ASCII pipe"


# --------------------------------------------------------------------
# Routing: R12-5 — ``aliases.json`` now points ``deepseek-r1-8b-4bit``
# at the ``deepseek_v3`` parser (was: ``deepseek_v31``). Both
# ``deepseek_v3`` and ``deepseek_r1_0528`` resolve to the new dedicated
# class.
# --------------------------------------------------------------------
@pytest.mark.parametrize("name", ["deepseek_v3", "deepseek_r1_0528"])
def test_registry_lookup_returns_v3_parser(name: str) -> None:
    cls = ToolParserManager.get_tool_parser(name)
    assert cls is DeepSeekV3ToolParser


def test_registry_lookup_keeps_v31_separate() -> None:
    cls = ToolParserManager.get_tool_parser("deepseek_v31")
    assert cls is DeepSeekV31ToolParser


# --------------------------------------------------------------------
# V3 wire format — the D-DSV31 P0 case (now against the dedicated
# ``DeepSeekV3ToolParser`` post-R12-5).
# --------------------------------------------------------------------
class TestV3WireFormat:
    """Direct V3-shaped payloads: ``function<sep>NAME\\n```json\\n{...}\\n```\\n``."""

    def test_single_v3_tool_call(self, v3_parser: DeepSeekV3ToolParser) -> None:
        payload = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called, "V3-shaped payload must trigger tools_called"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        # The original bug: name == "function" and arguments started
        # with "get_weather\n```json…"
        assert tc["name"] == "get_weather", (
            f"V3 type-tag leak: parser returned name={tc['name']!r}"
        )
        assert json.loads(tc["arguments"]) == {"city": "Tokyo"}

    def test_parallel_v3_tool_calls(self, v3_parser: DeepSeekV3ToolParser) -> None:
        """N V3 blocks in one envelope must produce N tool calls."""
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            _v3_block("get_time", '{"tz": "UTC"}'),
            _v3_block("search", '{"q": "deepseek"}'),
        )

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert len(result.tool_calls) == 3
        names = [c["name"] for c in result.tool_calls]
        assert names == ["get_weather", "get_time", "search"]
        # Pin all three argument bodies — protects against any "one
        # block swallows the next" greedy-regex regression.
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Tokyo"}
        assert json.loads(result.tool_calls[1]["arguments"]) == {"tz": "UTC"}
        assert json.loads(result.tool_calls[2]["arguments"]) == {"q": "deepseek"}

    def test_v3_with_leading_content(self, v3_parser: DeepSeekV3ToolParser) -> None:
        """Reasoning text before the envelope must be preserved as content."""
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            prefix="Let me check the weather. ",
        )

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert result.content == "Let me check the weather. "
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_v3_arguments_with_nested_braces(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """Nested JSON inside the fenced body must not confuse the fence detector."""
        args = '{"filter": {"city": "Tokyo", "tags": ["a", "b"]}, "limit": 10}'
        payload = _envelope(_v3_block("search", args))

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == json.loads(args)


# --------------------------------------------------------------------
# V3.1 wire format — regression: pre-existing behaviour must hold.
# --------------------------------------------------------------------
class TestV31WireFormat:
    """Plain ``NAME<sep>ARGS`` body — the V3.1 thinking-channel shape.

    R12-5: V3.1 parser is now V3.1-only. It does NOT auto-detect V3
    bodies and the previous "V3 sniffer" tests no longer apply (the
    ``_cumulative_has_v3_block`` machinery is gone — see the separate
    ``TestSplitContract`` block below for the boundary tests).
    """

    def test_single_v31_tool_call(self, v31_parser: DeepSeekV31ToolParser) -> None:
        payload = _envelope(_v31_block("get_weather", '{"city": "Paris"}'))

        result = v31_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Paris"}

    def test_parallel_v31_tool_calls(self, v31_parser: DeepSeekV31ToolParser) -> None:
        payload = _envelope(
            _v31_block("f1", '{"a": 1}'),
            _v31_block("f2", '{"b": 2}'),
        )

        result = v31_parser.extract_tool_calls(payload)

        assert len(result.tool_calls) == 2
        assert [c["name"] for c in result.tool_calls] == ["f1", "f2"]

    def test_v31_with_tool_named_function_passes(
        self, v31_parser: DeepSeekV31ToolParser
    ) -> None:
        """A V3.1 tool literally named ``function_lookup`` MUST be parsed
        as ``function_lookup`` (not mis-sniffed as a V3 type tag — there
        is no V3 sniffer in the V3.1 parser anymore)."""
        payload = _envelope(_v31_block("function_lookup", '{"q": "x"}'))

        result = v31_parser.extract_tool_calls(payload)

        assert result.tool_calls[0]["name"] == "function_lookup"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"q": "x"}


# --------------------------------------------------------------------
# R12-5 split contract: cross-shape boundary tests.
# --------------------------------------------------------------------
class TestSplitContract:
    """The split makes the V3/V3.1 boundary structural (parser-class)
    rather than runtime (per-block auto-detect). These tests pin the
    boundary so any future "let's unify them again" PR is caught."""

    def test_v3_parser_drops_v31_shaped_body(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """V3 parser MUST NOT accept a V3.1-shape body. The aliases
        route V3.1 models to the V3.1 parser — silently parsing it
        here would hide misconfigurations."""
        payload = _envelope(_v31_block("get_weather", '{"city": "Paris"}'))
        result = v3_parser.extract_tool_calls(payload)
        assert result.tools_called is False
        assert result.tool_calls == []

    def test_v31_parser_misparses_v3_body_as_function_named(
        self, v31_parser: DeepSeekV31ToolParser
    ) -> None:
        """V3.1 parser MUST emit ``name='function'`` on V3 bodies (the
        historical D-DSV31 bug). This is the failure the aliases.json
        routing MUST steer around: route R1-0528 to ``deepseek_v3``,
        not ``deepseek_v31``. If a future PR adds V3 auto-detect to
        the V3.1 parser, this test will fail and that's a signal to
        keep the split or update both parsers + their tests in lockstep."""
        payload = _envelope(_v3_block("get_weather", '{"city": "Paris"}'))
        result = v31_parser.extract_tool_calls(payload)
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "function"


# --------------------------------------------------------------------
# Malformed V3 payloads — graceful handling.
# --------------------------------------------------------------------
class TestMalformedGraceful:
    def test_no_envelope_passes_through(self, v3_parser: DeepSeekV3ToolParser) -> None:
        text = "Just plain reasoning, no tools here."
        result = v3_parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == text

    def test_truncated_block_falls_back_to_content(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """Outer ``<calls_begin>`` present but the block never closes."""
        payload = f"prefix {TC_OPEN}{C_OPEN}function{SEP}get_weather\n```json\n{{"

        result = v3_parser.extract_tool_calls(payload)

        assert not result.tools_called
        assert result.tool_calls == []
        # Full text passes through.
        assert result.content == payload

    def test_envelope_with_no_blocks(self, v3_parser: DeepSeekV3ToolParser) -> None:
        payload = f"{TC_OPEN}{TC_CLOSE}"

        result = v3_parser.extract_tool_calls(payload)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == payload

    def test_block_missing_separator(self, v3_parser: DeepSeekV3ToolParser) -> None:
        """Block envelope but no ``<sep>`` — un-parseable body."""
        payload = f"{TC_OPEN}{C_OPEN}garbage_no_sep_here{C_CLOSE}{TC_CLOSE}"

        result = v3_parser.extract_tool_calls(payload)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == payload

    def test_one_good_one_bad_block(self, v3_parser: DeepSeekV3ToolParser) -> None:
        """A malformed block must not invalidate sibling good blocks."""
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            f"{C_OPEN}no_sep_here{C_CLOSE}",
            _v3_block("get_time", '{"tz": "UTC"}'),
        )

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        # The bad block in the middle is dropped; good blocks survive.
        assert len(result.tool_calls) == 2
        assert [c["name"] for c in result.tool_calls] == ["get_weather", "get_time"]

    def test_literal_marker_text_before_envelope_does_not_misparse(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """codex r8 BLOCKING-1 (D-DSV31, preserved): a response that
        mentions ``<｜tool▁call▁begin｜>`` as literal content (e.g. in
        docs prose) and later happens to contain
        ``<｜tool▁calls▁begin｜>`` MUST NOT have the literal text
        treated as a tool call. The block scanner is bounded to the
        outer envelope.
        """
        prose = (
            "Here's how DeepSeek tool calls work: "
            f"the format uses {C_OPEN}NAME{SEP}ARGS{C_CLOSE}. For example:\n\n"
        )
        real_call = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))
        payload = prose + real_call

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_envelope_with_truncated_trailing_block_preserves_text(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """codex r8 BLOCKING-2 (preserved): a response with [valid V3
        block, truncated trailing block] surfaces the truncated text
        in ``content`` rather than silently dropping it.
        """
        good_block = _v3_block("get_weather", '{"city": "Tokyo"}')
        truncated_tail = f"{C_OPEN}function{SEP}get_time\n```json\n{{"  # no close
        payload = f"{TC_OPEN}{good_block}{truncated_tail}"

        result = v3_parser.extract_tool_calls(payload)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.content is not None
        assert "get_time" in result.content, (
            f"Truncated trailing block text dropped from content. "
            f"content={result.content!r}"
        )

    def test_v3_anchored_body_with_partial_fence_drops_block(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """codex r10 BLOCKING (preserved): V3-anchored bodies with an
        incomplete fenced-JSON body are DROPPED entirely (no tool call
        emitted). A bounded recovery could emit a tool call with
        truncated / non-JSON arguments which would then reach downstream
        tool execution.
        """
        payload = (
            f"{TC_OPEN}{C_OPEN}function{SEP}get_weather\n"
            f'```json\n{{"city": "Tokyo"'  # no closing fence, no `}`
            f"{C_CLOSE}{TC_CLOSE}"
        )

        result = v3_parser.extract_tool_calls(payload)

        assert not result.tools_called, (
            f"Partial-fence V3 body must NOT emit a tool call. "
            f"Got: {result.tool_calls!r}"
        )
        assert result.tool_calls == []
        assert result.content is not None
        assert "get_weather" in result.content


# --------------------------------------------------------------------
# Streaming contract — V3 defers entirely to end-of-stream finalize.
# --------------------------------------------------------------------
class TestV3Streaming:
    def _feed(
        self, parser: DeepSeekV3ToolParser, payload: str, chunk_size: int = 8
    ) -> list[dict | None]:
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

    def test_plain_content_before_envelope_streams_normally(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """Pre-envelope tokens emit as content. The model's narration
        before the first tool call is visible to streaming clients."""
        events = self._feed(v3_parser, "Let me check.", chunk_size=4)
        # At least one event should carry plain content.
        content_seen = [ev for ev in events if ev and ev.get("content")]
        assert content_seen, (
            f"No content emitted for pre-envelope tokens. Events: {events!r}"
        )

    def test_pre_envelope_prose_in_same_delta_as_marker_is_emitted(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """codex round-2 P2 regression: a delta that carries ordinary
        prose AND the first envelope marker (``Let me check.<｜tool▁calls▁begin｜>...``
        in one chunk) MUST surface the prose as ``content``. The
        finalize recovery replays only the tool calls, not the model's
        narration leading up to them — without this branch the prose
        was dropped on the floor.
        """
        delta = "Let me check the weather. " + TC_OPEN
        result = v3_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=delta,
            delta_text=delta,
        )
        assert result is not None
        assert result.get("content") == "Let me check the weather. "

    def test_subsequent_deltas_after_marker_return_none(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """Once the marker has been seen in ``previous_text`` we're
        fully inside the envelope; finalize handles emission and the
        streaming path returns ``None``."""
        prev = "Let me check. " + TC_OPEN
        delta = C_OPEN + "function" + SEP + "get_weather"
        result = v3_parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=prev + delta,
            delta_text=delta,
        )
        assert result is None

    def test_v3_stream_emits_no_mid_stream_tool_calls(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """Once the envelope arrives, streaming returns ``None`` until
        finalize. Confirms no mid-stream ``tool_calls`` event leaks
        (in particular: NO ``name="function"`` deltas).
        """
        payload = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))

        events = self._feed(v3_parser, payload, chunk_size=12)
        for ev in events:
            if not ev:
                continue
            assert "tool_calls" not in ev, (
                f"V3 stream leaked mid-stream tool_calls: {ev!r}"
            )

        # Non-streaming extract on the cumulative text emits the
        # correct call — confirming the postprocessor's finalize path
        # will produce the right tool_call.
        result = v3_parser.extract_tool_calls(payload)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Tokyo"}

    def test_v3_parallel_stream_finalize_yields_all_calls(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            _v3_block("get_time", '{"tz": "UTC"}'),
        )

        result = v3_parser.extract_tool_calls(payload)
        assert len(result.tool_calls) == 2
        assert [c["name"] for c in result.tool_calls] == ["get_weather", "get_time"]


# --------------------------------------------------------------------
# Integration: V3 streaming through the real ``StreamingPostProcessor``
# finalize path. Confirms the suppression + finalize composition
# actually delivers a ``tool_call`` event to clients end-to-end.
# --------------------------------------------------------------------
def _make_postprocessor_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "deepseek_v3"
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
    """End-to-end through ``StreamingPostProcessor``."""

    def test_v3_payload_emits_tool_call_via_finalize(self) -> None:
        """Drive a V3-shaped payload through the real postprocessor
        chunk-by-chunk and confirm ``finalize()`` emits a
        ``tool_call`` event with ``name="get_weather"``.
        """
        cfg = _make_postprocessor_cfg()
        pp = StreamingPostProcessor(cfg, tools_requested=True)
        pp.reset()

        payload = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))

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

        tool_call_events = [e for e in all_events if e.type == "tool_call"]
        assert tool_call_events, (
            f"No tool_call event emitted end-to-end. Events: "
            f"{[(e.type, getattr(e, 'tool_calls', None)) for e in all_events]!r}"
        )
        names = []
        for ev in tool_call_events:
            for tc in ev.tool_calls or []:
                names.append(tc.get("function", {}).get("name") or tc.get("name"))
        assert "get_weather" in names, (
            f"V3 finalize path lost the real tool name. Names: {names!r}"
        )
        assert "function" not in names, (
            f"V3 type-tag leaked into the final tool_call event. Names: {names!r}"
        )


# --------------------------------------------------------------------
# Argument-body passthrough contract — bytes-equal preservation.
# --------------------------------------------------------------------
class TestArgumentBytesPassthrough:
    def test_valid_json_args_passed_through_verbatim(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """Args body bytes survive end-to-end (no canonicalisation)."""
        args = '{   "k"  :  1  }'
        payload = _envelope(_v3_block("f", args))

        result = v3_parser.extract_tool_calls(payload)

        assert result.tool_calls[0]["arguments"] == args
        assert json.loads(result.tool_calls[0]["arguments"]) == {"k": 1}

    def test_v31_non_json_args_passed_through(
        self, v31_parser: DeepSeekV31ToolParser
    ) -> None:
        """V3.1 free-form args body is passed through verbatim."""
        payload = _envelope(_v31_block("explain", "free-form text body"))

        result = v31_parser.extract_tool_calls(payload)

        assert result.tool_calls[0]["arguments"] == "free-form text body"
