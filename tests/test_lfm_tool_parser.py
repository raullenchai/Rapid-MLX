# SPDX-License-Identifier: Apache-2.0
"""Tests for Liquid/LFM tool call parser."""

import json
from unittest.mock import MagicMock

from vllm_mlx.service.postprocessor import StreamingPostProcessor
from vllm_mlx.tool_parsers import AutoToolParser, LfmToolParser, ToolParserManager
from vllm_mlx.tool_parsers.lfm_tool_parser import parse_lfm_tool_calls


class TestLfmRegistration:
    """Test that the LFM parser is registered correctly."""

    def test_registered_as_lfm(self):
        parser_cls = ToolParserManager.get_tool_parser("lfm")
        assert parser_cls is LfmToolParser

    def test_registered_as_liquid(self):
        parser_cls = ToolParserManager.get_tool_parser("liquid")
        assert parser_cls is LfmToolParser


class TestLfmExtractToolCalls:
    """Test non-streaming LFM tool call extraction."""

    def test_single_pythonic_tool_call(self):
        parser = LfmToolParser()
        result = parser.extract_tool_calls(
            'Let me check. [get_current_weather(location="Paris")]'
        )

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_current_weather"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"location": "Paris"}
        assert result.content == "Let me check."

    def test_multiple_pythonic_tool_calls(self):
        parser = LfmToolParser()
        result = parser.extract_tool_calls(
            '[get_current_weather(location="Paris", unit="celsius"), '
            'get_time(timezone="Europe/Paris")]'
        )

        assert result.tools_called
        assert [tc["name"] for tc in result.tool_calls] == [
            "get_current_weather",
            "get_time",
        ]
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "location": "Paris",
            "unit": "celsius",
        }
        assert json.loads(result.tool_calls[1]["arguments"]) == {
            "timezone": "Europe/Paris"
        }

    def test_auto_parser_malformed_bracketed_text_does_not_crash(self):
        """Auto parser should ignore prose brackets that are not LFM calls."""
        parser = AutoToolParser()
        text = "This is prose [not a function call] and should stay content."

        result = parser.extract_tool_calls(text)

        assert not result.tools_called
        assert result.content == text


class TestLfmStreaming:
    """Test streaming LFM tool call extraction."""

    def test_streaming_pythonic_tool_call_emits_when_closing_bracket_arrives(self):
        parser = LfmToolParser()
        previous_text = 'Checking [get_current_weather(location="Paris"'
        current_text = previous_text + ")]"

        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=")]",
        )

        assert result is not None
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        tool_call = result["tool_calls"][0]
        assert tool_call["function"]["name"] == "get_current_weather"
        assert json.loads(tool_call["function"]["arguments"]) == {"location": "Paris"}

    def test_streaming_bracketed_prose_passes_through(self):
        """Non-tool brackets must not be suppressed as pending tool markup."""
        parser = LfmToolParser()
        previous_text = "Here are "
        delta_text = "[two] options."
        current_text = previous_text + delta_text

        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
        )

        assert result == {"content": delta_text}

    def test_streaming_content_after_completed_call_is_emitted(self):
        """Trailing prose after an emitted call must not be held forever.

        Regression: the partial-start tail-hold matched an already-closed
        ``[f(x=1)]`` block (the ``\\(.*`` swallowed everything), so every
        delta after a completed call returned None — and since a tool call
        had fired, finalize() never flushed it either. Content lost.
        """
        parser = LfmToolParser()
        # Stream up to the completed call; the closing delta emits tools.
        result = parser.extract_tool_calls_streaming(
            previous_text="Hi [f(x=1",
            current_text="Hi [f(x=1)]",
            delta_text=")]",
        )
        assert result is not None and "tool_calls" in result

        # The next content delta must come through as content.
        result = parser.extract_tool_calls_streaming(
            previous_text="Hi [f(x=1)]",
            current_text="Hi [f(x=1)] all done now",
            delta_text=" all done now",
        )
        assert result == {"content": " all done now"}

    def test_streaming_later_bracket_does_not_duplicate_tool_calls(self):
        """A ``]`` in trailing prose must not re-emit the same tool call.

        Regression: every delta containing ``]`` re-ran extract_tool_calls
        over the full text and re-emitted the call with a fresh id at
        index 0 — OpenAI-delta clients concatenate per-index arguments,
        corrupting the JSON.
        """
        parser = LfmToolParser()
        result = parser.extract_tool_calls_streaming(
            previous_text="Hi [f(x=1",
            current_text="Hi [f(x=1)]",
            delta_text=")]",
        )
        assert result is not None and "tool_calls" in result

        result = parser.extract_tool_calls_streaming(
            previous_text="Hi [f(x=1)] see [notes",
            current_text="Hi [f(x=1)] see [notes] ok",
            delta_text="] ok",
        )
        assert result is None or "tool_calls" not in result

    def test_streaming_positional_call_passes_through_as_content(self):
        """A closed pythonic-looking block with positional args is content."""
        parser = LfmToolParser()
        previous_text = "see "
        delta_text = "[index(0)] for details"
        current_text = previous_text + delta_text

        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
        )

        assert result == {"content": delta_text}

    def test_flush_held_content_releases_partial_call_prefix(self):
        """A stream ending mid-``[func(`` must release the held bytes."""
        parser = LfmToolParser()
        assert parser.flush_held_content("see [get_we") == "[get_we"
        # Nothing held when the text contains no partial markup.
        assert parser.flush_held_content("plain text") == ""


class TestLfmArgumentHandling:
    """Regression tests for argument evaluation edge cases."""

    def test_positional_args_reject_the_call(self):
        """Positional args can't map to named parameters — the call must
        NOT be emitted with silently-empty arguments."""
        parser = LfmToolParser()
        result = parser.extract_tool_calls('[get_weather("Paris")]')

        assert not result.tools_called
        assert result.content == '[get_weather("Paris")]'

    def test_positional_args_anywhere_reject_the_whole_block(self):
        tool_calls, cleaned = parse_lfm_tool_calls('[f("x"), g(y=1)]')
        assert tool_calls == []
        assert cleaned == '[f("x"), g(y=1)]'

    def test_non_call_element_rejects_the_whole_block(self):
        tool_calls, cleaned = parse_lfm_tool_calls("[f(x=1), note]")
        assert tool_calls == []
        assert cleaned == "[f(x=1), note]"

    def test_keyword_unpack_rejects_the_whole_block(self):
        tool_calls, cleaned = parse_lfm_tool_calls('[f(**{"x": 1})]')
        assert tool_calls == []
        assert cleaned == '[f(**{"x": 1})]'

    def test_list_dict_and_numeric_args(self):
        """Non-scalar kwarg values must parse.

        Regression: ``eval_node`` touched ``ast.Num``/``ast.Str``/
        ``ast.NameConstant``, which were removed in Python 3.14 — any
        list/dict/bare-name argument raised AttributeError and the whole
        tool call was silently dropped.
        """
        parser = LfmToolParser()
        result = parser.extract_tool_calls(
            '[search(tags=["a", "b"], limit=5, opts={"k": 1}, exact=True)]'
        )

        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "tags": ["a", "b"],
            "limit": 5,
            "opts": {"k": 1},
            "exact": True,
        }

    def test_bare_name_arg_becomes_string(self):
        parser = LfmToolParser()
        result = parser.extract_tool_calls("[get_weather(unit=celsius)]")

        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {"unit": "celsius"}

    def test_multiple_separate_blocks_all_parsed(self):
        parser = LfmToolParser()
        result = parser.extract_tool_calls("[f(x=1)] and then [g(y=2)]")

        assert result.tools_called
        assert [tc["name"] for tc in result.tool_calls] == ["f", "g"]
        assert result.content == "and then"


class TestAutoParserLfmStreaming:
    """Regression tests for the LFM hooks in AutoToolParser streaming."""

    def test_prose_starting_with_bracket_streams_through(self):
        """Responses starting with ``[`` (markdown links, ``[1]`` citations)
        must stream as content.

        Regression: a bare ``current_text.startswith("[")`` gate held every
        delta of such responses, and with no flush override the entire
        response was silently dropped.
        """
        parser = AutoToolParser()
        text = ""
        for chunk in ["[link](https://x.com)", " is the ref.", " Done."]:
            previous = text
            text += chunk
            result = parser.extract_tool_calls_streaming(
                previous_text=previous, current_text=text, delta_text=chunk
            )
            assert result == {"content": chunk}

    def test_flush_releases_held_content_when_call_never_completes(self):
        parser = AutoToolParser()
        result = parser.extract_tool_calls_streaming(
            previous_text="", current_text="calc", delta_text="calc"
        )
        assert result == {"content": "calc"}

        # Pythonic-looking marker appears: content is held...
        result = parser.extract_tool_calls_streaming(
            previous_text="calc", current_text="calc [index(0", delta_text=" [index(0"
        )
        assert result is None

        # ...and released at stream end since no tool call ever completed.
        assert parser.flush_held_content("calc [index(0") == " [index(0"

    def test_close_bracket_split_across_deltas_still_emits(self):
        """``)`` and ``]`` arriving in separate deltas must still emit."""
        parser = AutoToolParser()
        text = ""
        results = []
        for chunk in ['[f(x="hi"', ")", "]"]:
            previous = text
            text += chunk
            results.append(
                parser.extract_tool_calls_streaming(
                    previous_text=previous, current_text=text, delta_text=chunk
                )
            )

        assert results[-1] is not None and "tool_calls" in results[-1]
        assert results[-1]["tool_calls"][0]["function"]["name"] == "f"

    def test_no_duplicate_emission_after_tool_call(self):
        parser = AutoToolParser()
        result = parser.extract_tool_calls_streaming(
            previous_text="[f(x=1",
            current_text="[f(x=1)]",
            delta_text=")]",
        )
        assert result is not None and "tool_calls" in result

        result = parser.extract_tool_calls_streaming(
            previous_text="[f(x=1)] see [notes",
            current_text="[f(x=1)] see [notes] ok",
            delta_text="] ok",
        )
        assert result is None or "tool_calls" not in result


class TestPostprocessorLfmFinalize:
    """End-of-stream recovery must recognize pythonic markup."""

    @staticmethod
    def _make_cfg(parser):
        cfg = MagicMock()
        cfg.engine = None
        cfg.reasoning_parser = None
        cfg.reasoning_parser_name = None
        cfg.enable_auto_tool_choice = True
        cfg.tool_call_parser = None
        cfg.tool_parser_instance = parser
        return cfg

    def test_finalize_recovers_pythonic_call_missed_by_streaming(self):
        """Regression: the plausible-markup pre-check only looked for
        ``<``, ``{``, or ``[Calling`` — ``[f(x="y")]`` contains none of
        them, so the finalize() fallback never ran for LFM output."""
        pp = StreamingPostProcessor(self._make_cfg(LfmToolParser()))
        pp.reset()
        pp.tool_accumulated_text = '[get_current_weather(location="Paris")]'

        events = pp.finalize()

        tool_events = [e for e in events if e.type == "tool_call"]
        assert len(tool_events) == 1
