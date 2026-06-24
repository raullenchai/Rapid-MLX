# SPDX-License-Identifier: Apache-2.0
"""
R12-5: ``DeepSeekV3ToolParser`` regression tests.

Verifies the V3-dedicated parser (split off from the unified
``DeepSeekV31ToolParser`` in PR #795) handles the V3 fenced-JSON wire
shape end-to-end and that the V3.1 parser no longer auto-detects V3
bodies after the split.

Wire shape (D-DSV31):

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
    ```json
    {ARGS}
    ```<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

All pipes are fullwidth (U+FF5C), not ASCII.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.deepseek_v3_tool_parser import DeepSeekV3ToolParser
from vllm_mlx.tool_parsers.deepseekv31_tool_parser import DeepSeekV31ToolParser

# Wire primitives.
TC_OPEN = "<｜tool▁calls▁begin｜>"
TC_CLOSE = "<｜tool▁calls▁end｜>"
C_OPEN = "<｜tool▁call▁begin｜>"
C_CLOSE = "<｜tool▁call▁end｜>"
SEP = "<｜tool▁sep｜>"


def _v3_block(name: str, args_json: str) -> str:
    return f"{C_OPEN}function{SEP}{name}\n```json\n{args_json}\n```{C_CLOSE}"


def _v31_block(name: str, args_body: str) -> str:
    return f"{C_OPEN}{name}{SEP}{args_body}{C_CLOSE}"


def _envelope(*blocks: str, prefix: str = "") -> str:
    return f"{prefix}{TC_OPEN}{''.join(blocks)}{TC_CLOSE}"


@pytest.fixture
def v3_parser() -> DeepSeekV3ToolParser:
    return DeepSeekV3ToolParser()


@pytest.fixture
def v31_parser() -> DeepSeekV31ToolParser:
    return DeepSeekV31ToolParser()


# --------------------------------------------------------------------
# Wire-format sanity — make sure the test data itself uses U+FF5C and
# not the ASCII pipe (catches accidental editor normalisation).
# --------------------------------------------------------------------
def test_wire_primitives_use_fullwidth_pipe() -> None:
    for s in (TC_OPEN, TC_CLOSE, C_OPEN, C_CLOSE, SEP):
        assert "｜" in s, f"{s!r} missing fullwidth pipe (U+FF5C)"
        assert "|" not in s, f"{s!r} leaked ASCII pipe"


# --------------------------------------------------------------------
# Registry routing — the alias entries in ``aliases.json`` and the
# ``model_auto_config.py`` regex both point at the ``deepseek_v3``
# registry name. Confirm that name resolves to the new parser, and
# that the legacy ``DeepSeekToolParser`` no longer owns it.
# --------------------------------------------------------------------
@pytest.mark.parametrize("name", ["deepseek_v3", "deepseek_r1_0528"])
def test_registry_routes_v3_aliases_to_dedicated_parser(name: str) -> None:
    assert ToolParserManager.get_tool_parser(name) is DeepSeekV3ToolParser


def test_deepseek_v31_alias_still_routes_to_v31_parser() -> None:
    """``deepseek_v31`` MUST NOT be hijacked by the V3 split — it stays
    on the V3.1 parser so existing V3.1-shape configurations keep
    working."""
    assert ToolParserManager.get_tool_parser("deepseek_v31") is DeepSeekV31ToolParser


# --------------------------------------------------------------------
# The P0 case — 3 V3-shaped tool calls.
# --------------------------------------------------------------------
class TestV3Extraction:
    def test_single_call_extracts(self, v3_parser: DeepSeekV3ToolParser) -> None:
        payload = _envelope(_v3_block("get_weather", '{"city": "Paris"}'))
        result = v3_parser.extract_tool_calls(payload)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["name"] == "get_weather"
        assert json.loads(tc["arguments"]) == {"city": "Paris"}
        assert result.content is None

    def test_parallel_calls_extract_as_distinct_entries(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """Block-wise scanner must NOT collapse N parallel calls into
        one over-greedy match. This was the original failure mode the
        D-DSV31 block-wise rewrite fixed (codex r8 BLOCKING-1)."""
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Paris"}'),
            _v3_block("get_weather", '{"city": "Tokyo"}'),
            _v3_block("lookup_population", '{"country": "France", "year": 2024}'),
        )
        result = v3_parser.extract_tool_calls(payload)
        assert result.tools_called is True
        assert len(result.tool_calls) == 3
        assert [tc["name"] for tc in result.tool_calls] == [
            "get_weather",
            "get_weather",
            "lookup_population",
        ]
        assert json.loads(result.tool_calls[2]["arguments"]) == {
            "country": "France",
            "year": 2024,
        }

    def test_prefix_content_is_preserved(self, v3_parser: DeepSeekV3ToolParser) -> None:
        """Plain narration before the envelope must survive in
        ``content`` (V3.1 semantics — clean envelopes keep the prefix
        only)."""
        payload = _envelope(
            _v3_block("get_weather", '{"city": "Paris"}'),
            prefix="Let me check the weather. ",
        )
        result = v3_parser.extract_tool_calls(payload)
        assert result.tools_called is True
        assert result.content == "Let me check the weather. "

    def test_tolerant_fence_no_trailing_newline(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """Some checkpoints omit the newline before the closing
        fence (``...}```<call_end>``). The tolerant regex must
        accept it."""
        body = (
            f'{C_OPEN}function{SEP}get_weather\n'
            f'```json\n{{"city": "Paris"}}```{C_CLOSE}'
        )
        payload = _envelope(body)
        result = v3_parser.extract_tool_calls(payload)
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "get_weather"


# --------------------------------------------------------------------
# Malformed-block behaviour. The original V3 path used to emit a
# partial-fence recovery; codex r10 BLOCKING removed it because the
# recovery could emit JSON with no closing brace. The split parser
# must preserve that contract (drop the block, surface raw text).
# --------------------------------------------------------------------
class TestMalformed:
    def test_v3_anchored_no_json_fence_drops_block(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """``function<sep>NAME`` with no fenced JSON body is malformed.
        Must NOT emit a tool call with empty / partial arguments."""
        body = f"{C_OPEN}function{SEP}get_weather{C_CLOSE}"
        payload = _envelope(body)
        result = v3_parser.extract_tool_calls(payload)
        assert result.tools_called is False
        assert result.tool_calls == []
        # Raw text surfaces in ``content`` so the caller can decide.
        assert result.content == payload

    def test_truncated_trailing_block_dropped_but_earlier_calls_kept(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """Open ``<call_begin>`` with no ``<call_end>`` — drop the
        trailing block, keep prior completed calls, surface raw text."""
        payload = (
            TC_OPEN
            + _v3_block("get_weather", '{"city": "Paris"}')
            + f"{C_OPEN}function{SEP}get_weather\n```json\n"  # truncated
            + TC_CLOSE
        )
        result = v3_parser.extract_tool_calls(payload)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        # Truncation present — content carries the raw model output so
        # the caller sees both the parsed call AND the malformed tail.
        assert result.content == payload

    def test_non_v3_body_is_dropped(self, v3_parser: DeepSeekV3ToolParser) -> None:
        """A V3.1-shaped body (``NAME<sep>{json}``, no ``function`` tag)
        is NOT V3 and the V3 parser drops it. Route those to the V3.1
        parser instead — that's the whole point of the R12-5 split."""
        payload = _envelope(_v31_block("get_weather", '{"city": "Paris"}'))
        result = v3_parser.extract_tool_calls(payload)
        assert result.tools_called is False
        assert result.tool_calls == []


# --------------------------------------------------------------------
# Envelope-boundedness. Literal marker text in plain content (e.g.
# documentation) must NOT be misparsed (codex r8 BLOCKING-1).
# --------------------------------------------------------------------
class TestEnvelopeBoundedness:
    def test_no_envelope_returns_content_unchanged(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        plain = "Here is a sentence with no tool calls in it."
        result = v3_parser.extract_tool_calls(plain)
        assert result.tools_called is False
        assert result.content == plain

    def test_marker_in_content_outside_envelope_not_misparsed(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """The fullwidth-pipe markers can appear in user-facing
        documentation or examples. The scanner MUST be bounded to the
        outer ``<tool_calls_begin>`` / ``<tool_calls_end>`` envelope."""
        text = (
            "Here is what a tool call looks like: " + C_OPEN + "example" + C_CLOSE
        )  # markers, but NOT inside the outer envelope
        result = v3_parser.extract_tool_calls(text)
        assert result.tools_called is False
        assert result.content == text


# --------------------------------------------------------------------
# V3.1 parser MUST NOT auto-detect V3 after the split. This is the
# contract the R12-5 refactor codifies: each parser handles its own
# wire shape, no silent fallback.
# --------------------------------------------------------------------
class TestV31DoesNotAcceptV3:
    def test_v31_parser_misparses_v3_as_function_named_call(
        self, v31_parser: DeepSeekV31ToolParser
    ) -> None:
        """Demonstrates WHY ``aliases.json`` must route R1-0528 to
        ``deepseek_v3`` and not ``deepseek_v31``: the V3.1 parser's
        ``NAME<sep>ARGS`` split on a V3 body emits ``name='function'``
        and the real name leaks into ``arguments``. This is exactly
        the D-DSV31 P0 production failure."""
        payload = _envelope(_v3_block("get_weather", '{"city": "Paris"}'))
        result = v31_parser.extract_tool_calls(payload)
        # The V3.1 parser DOES emit a call here, but the call is wrong:
        # ``name='function'`` rather than the real tool name. We pin
        # the wrong shape so any future "let's add V3 fallback to V3.1
        # too" PR is caught by this test failing.
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "function"
        # The real name + fenced JSON leak into arguments — the
        # historic D-DSV31 mojibake-into-content failure.
        assert "get_weather" in result.tool_calls[0]["arguments"]
        assert "```json" in result.tool_calls[0]["arguments"]

    def test_v31_parser_still_handles_v31_correctly(
        self, v31_parser: DeepSeekV31ToolParser
    ) -> None:
        """Sanity: the V3.1 parser still passes its native shape."""
        payload = _envelope(_v31_block("get_weather", '{"city": "Paris"}'))
        result = v31_parser.extract_tool_calls(payload)
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Paris"}


# --------------------------------------------------------------------
# Streaming contract — V3 streaming defers entirely to end-of-stream
# finalize so clients get a well-formed one-shot ``tool_call`` event
# per block rather than mid-fence partial-JSON garbage.
# --------------------------------------------------------------------
class TestStreaming:
    def test_plain_content_before_envelope_streams_normally(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        delta = v3_parser.extract_tool_calls_streaming(
            previous_text="Let me ",
            current_text="Let me check",
            delta_text="check",
        )
        assert delta == {"content": "check"}

    def test_streaming_stops_emitting_once_envelope_seen(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """Once the envelope marker has been seen in a previous delta,
        subsequent deltas return ``None`` — finalize handles
        emission."""
        prev = "Let me check " + TC_OPEN
        # Marker already in previous_text.
        delta = v3_parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=prev + C_OPEN,
            delta_text=C_OPEN,
        )
        assert delta is None

    def test_streaming_emits_pre_marker_prose_in_split_delta(
        self, v3_parser: DeepSeekV3ToolParser
    ) -> None:
        """When the marker first arrives in a delta that also carries
        prior prose, surface the prose as ``content``. Codex round-2
        P2: dropping it loses the model's narration before the tool
        call."""
        delta = "Let me check the weather. " + TC_OPEN
        result = v3_parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=delta,
            delta_text=delta,
        )
        assert result is not None
        assert result["content"] == "Let me check the weather. "


# Forced-tool-choice prefix coverage lives in
# ``tests/test_tool_choice_enforcement.py`` (which already imports
# ``vllm_mlx.routes.chat`` for the prefix helper). Keeping the
# round-trip test there avoids forcing this parser-regression file
# to pull in the full route module — and lets parser-only CI shards
# run without a Metal device (codex round-4 P2).
