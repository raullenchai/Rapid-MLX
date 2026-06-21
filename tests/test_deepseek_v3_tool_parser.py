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

import pytest

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


# --------------------------------------------------------------------
# Streaming — V3 short-circuit emits one well-formed tool_call event
# per closed block instead of leaking mid-stream ``name="function"``
# deltas. The legacy V3.1 streaming path remains delta-based and is
# covered by ``test_upstream_regression.TestDeepSeekV31UpstreamStreaming``;
# here we pin only the V3 short-circuit behaviour.
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

    def test_v3_block_emits_single_complete_tool_call_on_close(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """No ``name="function"`` deltas — only one fully-resolved
        ``tool_calls`` event when the block's ``<call_end>`` arrives."""
        payload = _envelope(_v3_block("get_weather", '{"city": "Tokyo"}'))

        events = self._feed(parser, payload, chunk_size=12)

        tool_events = [e for e in events if e and "tool_calls" in e]
        assert len(tool_events) == 1, (
            f"expected exactly one tool_calls event, got {len(tool_events)}: "
            f"{tool_events!r}"
        )
        tc_list = tool_events[0]["tool_calls"]
        assert len(tc_list) == 1
        tc = tc_list[0]
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Tokyo"}
        # Index sequence starts at 0 for the first tool call.
        assert tc["index"] == 0

        # And no event ever leaked ``name="function"`` (the bug shape).
        for ev in events:
            if not ev or "tool_calls" not in ev:
                continue
            for call in ev["tool_calls"]:
                assert call.get("function", {}).get("name") != "function", (
                    f"V3 type-tag leaked into streaming name: {ev!r}"
                )

    def test_parallel_v3_blocks_emit_per_block(
        self, parser: DeepSeekV31ToolParser
    ) -> None:
        """Two V3 blocks → two distinct tool_calls deltas with
        ascending indices."""
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            _v3_block("get_time", '{"tz": "UTC"}'),
        )

        events = self._feed(parser, payload, chunk_size=24)

        emitted_names = []
        emitted_indices = []
        for ev in events:
            if not ev or "tool_calls" not in ev:
                continue
            for call in ev["tool_calls"]:
                emitted_names.append(call["function"]["name"])
                emitted_indices.append(call["index"])

        assert emitted_names == ["get_weather", "get_time"]
        assert emitted_indices == [0, 1]


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
