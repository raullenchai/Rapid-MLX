# SPDX-License-Identifier: Apache-2.0
"""Tests for Liquid/LFM tool call parser."""

import json
import time
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

    def test_streaming_second_separate_block_emits_with_next_index(self):
        """A second bracket block completing later must still be emitted.

        Regression: a one-shot ``_streaming_tools_emitted`` latch dropped
        every tool call after the first block. LFM's non-streaming path
        parses N separate blocks, so streaming must reach parity — the
        second call is emitted with a continuing index (1), not lost.
        """
        parser = LfmToolParser()
        first = parser.extract_tool_calls_streaming(
            previous_text="[f(x=1",
            current_text="[f(x=1)]",
            delta_text=")]",
        )
        assert first is not None and "tool_calls" in first
        assert first["tool_calls"][0]["index"] == 0
        assert first["tool_calls"][0]["function"]["name"] == "f"

        second = parser.extract_tool_calls_streaming(
            previous_text="[f(x=1)] then [g(y=2",
            current_text="[f(x=1)] then [g(y=2)]",
            delta_text=")]",
        )
        assert second is not None and "tool_calls" in second
        # Exactly one NEW call, indexed after the first (not restarting at 0).
        assert len(second["tool_calls"]) == 1
        assert second["tool_calls"][0]["index"] == 1
        assert second["tool_calls"][0]["function"]["name"] == "g"

    def test_streaming_preface_and_call_in_same_delta_keeps_content(self):
        """Leading prose + the first call in ONE delta must not drop prose.

        Regression: when a batched chunk / finalize / single-shot stream
        delivers ``Let me check. [get_weather(...)]`` at once, the tool
        branch returned only ``tool_calls`` and never emitted the preface —
        and once a call fires, the EOF flush path is skipped, so
        ``Let me check.`` was lost (non-streaming keeps it). The parser must
        emit BOTH channels in one return (the llama-parser precedent).
        """
        parser = LfmToolParser()
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text='Let me check. [get_weather(location="Paris")]',
            delta_text='Let me check. [get_weather(location="Paris")]',
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result.get("content") == "Let me check. "

    def test_streaming_preface_not_double_emitted_token_by_token(self):
        """Token-by-token, the preface streams once and is NOT re-emitted
        in the tool-call delta."""
        parser = LfmToolParser()
        full = 'Hello there [get_time(zone="CET")]'
        prev = ""
        content = []
        tool_content = None
        for i in range(1, len(full) + 1):
            cur = full[:i]
            r = parser.extract_tool_calls_streaming(prev, cur, full[i - 1 : i])
            if r:
                if r.get("content"):
                    content.append(r["content"])
                if "tool_calls" in r:
                    tool_content = r.get("content")
            prev = cur
        assert "".join(content) == "Hello there "
        # The tool-call delta carried no duplicate preface.
        assert tool_content is None

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


def _stream(parser, chunks):
    """Feed ``chunks`` through the streaming parser, return (content, calls).

    Mirrors what ``StreamingPostProcessor._detect_tool_calls`` does with
    the parser's return value, minus the postprocessor's own sanitizers —
    so the assertions pin the PARSER's contract rather than the last-mile
    scrubber that happened to mask half of this leak in production.
    """
    previous = ""
    content = ""
    calls = []
    for chunk in chunks:
        current = previous + chunk
        result = parser.extract_tool_calls_streaming(previous, current, chunk)
        if result is not None:
            if result.get("content"):
                content += result["content"]
            for call in result.get("tool_calls") or []:
                calls.append(call)
        previous = current
    # Stream end: held bytes are released only when no call ever fired,
    # matching ``StreamingPostProcessor.finalize``.
    if not calls:
        content += parser.flush_held_content(previous)
    return content, calls


#: The exact wire text behind the reported leak. rapid-mlx itself writes
#: ``[Calling tool: name({args})]`` into the prompt for every model whose
#: parser reports no native tool format (``api/utils.py``) — LFM's does,
#: and LFM2.5's chat template drops ``message.tool_calls`` — so the model
#: is taught this dialect by its own history and reproduces it verbatim.
LEAK_TEXT = (
    '[Calling tool: browse({"url": "https://www.iana.org/help/example-domains", '
    '"refresh": true})]'
)

#: Realistic detokenizer chunking for ``LEAK_TEXT``. The reported leak
#: needs this exact split: ``[Calling tool`` lands in one delta (the
#: global sanitizer eats it whole) and everything from ``:`` onward
#: streams to the client.
LEAK_CHUNKS = [
    "[",
    "Calling",
    " tool",
    ":",
    " browse",
    "({",
    '"url"',
    ": ",
    '"https://www.iana.org/help/example-domains"',
    ", ",
    '"refresh"',
    ": ",
    "true",
    "})",
    "]",
]

LEAK_ARGS = {
    "url": "https://www.iana.org/help/example-domains",
    "refresh": True,
}


class TestLfmTextFormatEnvelope:
    """``[Calling tool: name({...})]`` — the text-format degradation."""

    def test_non_streaming_extracts_envelope_call(self):
        parser = LfmToolParser()
        result = parser.extract_tool_calls(LEAK_TEXT)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "browse"
        assert json.loads(result.tool_calls[0]["arguments"]) == LEAK_ARGS
        assert not result.content

    def test_streaming_envelope_token_by_token_leaks_nothing(self):
        """The reported P1: raw markup reached the visible message."""
        parser = LfmToolParser()
        content, calls = _stream(parser, LEAK_CHUNKS)

        assert content == ""
        assert len(calls) == 1
        assert calls[0]["index"] == 0
        assert calls[0]["function"]["name"] == "browse"
        assert json.loads(calls[0]["function"]["arguments"]) == LEAK_ARGS

    def test_streaming_envelope_char_by_char_leaks_nothing(self):
        """Per-character deltas are the worst case for prefix holding."""
        parser = LfmToolParser()
        content, calls = _stream(parser, list(LEAK_TEXT))

        assert content == ""
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == LEAK_ARGS

    def test_streaming_envelope_single_delta_leaks_nothing(self):
        parser = LfmToolParser()
        content, calls = _stream(parser, [LEAK_TEXT])

        assert content == ""
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "browse"

    def test_streaming_envelope_keeps_prose_preface(self):
        """Prose before the call is content; the markup itself is not."""
        parser = LfmToolParser()
        content, calls = _stream(parser, ["One moment. "] + LEAK_CHUNKS)

        assert content == "One moment. "
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "browse"

    def test_streaming_two_envelope_blocks_continue_indices(self):
        parser = LfmToolParser()
        content, calls = _stream(
            parser,
            [
                '[Calling tool: f({"a": 1})]',
                " then ",
                '[Calling tool: g({"b": 2})]',
            ],
        )

        assert "Calling tool" not in content
        assert [c["index"] for c in calls] == [0, 1]
        assert [c["function"]["name"] for c in calls] == ["f", "g"]
        assert len({c["id"] for c in calls}) == 2

    def test_streaming_json_args_without_envelope_leaks_nothing(self):
        """``[name({...})]`` — same dialect with the envelope dropped.

        A single dict positional is unambiguously the arguments object,
        so it must not be rejected into the visible content the way a
        genuinely positional call (``[index(0)]``) is.
        """
        parser = LfmToolParser()
        content, calls = _stream(parser, ['[browse({"url": ', '"https://x"})]'])

        assert content == ""
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "browse"
        assert json.loads(calls[0]["function"]["arguments"]) == {"url": "https://x"}

    def test_unterminated_envelope_is_flushed_not_swallowed(self):
        """A held prefix must be released when no call ever completes."""
        parser = LfmToolParser()
        content, calls = _stream(parser, ["Wait ", "[Calling", " tool", ": brow"])

        assert calls == []
        assert content == "Wait [Calling tool: brow"

    def test_json_literals_are_not_stringified(self):
        """``true`` / ``false`` / ``null`` are JSON, not Python bare names.

        Parsing the payload with ``ast`` turns them into the strings
        ``"true"`` / ``"false"`` / ``"null"``, which invokes the tool with
        materially wrong arguments.
        """
        parser = LfmToolParser()
        expected = {"refresh": True, "cached": False, "cursor": None, "n": 3}
        for text in (
            '[Calling tool: browse({"refresh": true, "cached": false, '
            '"cursor": null, "n": 3})]',
            '[browse({"refresh": true, "cached": false, "cursor": null, "n": 3})]',
        ):
            result = parser.extract_tool_calls(text)
            assert result.tools_called, text
            assert json.loads(result.tool_calls[0]["arguments"]) == expected, text

    def test_openai_legal_tool_names_are_recognised(self):
        """Names follow the OpenAI charset, not Python identifier rules.

        ``^[a-zA-Z0-9_-]{1,64}$`` is what the request validator accepts, so
        ``my-tool`` and ``2fa`` are names a client can register. Rejecting
        them here would stream the raw span while the finalize recovery
        path (which does accept them) extracted the call anyway.
        """
        for text in (
            '[Calling tool: my-tool({"a": 1})]',
            '[my-tool({"a": 1})]',
            '[Calling tool: 2fa({"a": 1})]',
            '[2fa({"a": 1})]',
        ):
            parser = LfmToolParser()
            content, calls = _stream(parser, list(text))
            assert content == "", text
            assert len(calls) == 1, text
            assert json.loads(calls[0]["function"]["arguments"]) == {"a": 1}, text

    def test_call_inside_a_string_argument_is_never_dispatched(self):
        """Markup inside another call's JSON string is data, not a call.

        Mid-stream the outer block is just unterminated, so hunting past
        it for a block that closes finds the ``[g({})]`` sitting inside
        ``f``'s argument and dispatches it at index 0 — which the
        count-based dedup can never retract once ``f`` actually arrives.
        The walk stops at the unterminated opener instead.
        """
        text = '[Calling tool: f({"x": "[g({})]"})]'
        content, calls = _stream(LfmToolParser(), list(text))

        assert [c["function"]["name"] for c in calls] == ["f"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"x": "[g({})]"}
        assert content == ""

    def test_call_inside_a_rejected_block_is_never_dispatched(self):
        """A rejected block's payload is data too, not a place to search.

        Resuming one character into a balanced-but-invalid block finds the
        markup sitting in its string argument and dispatches it — the same
        irreversible mistake as probing past an unterminated opener, just
        with the outer block closed.
        """
        text = '[Calling tool: f({"x": "[Calling tool: g({})]"} trailing)]'
        content, calls = _stream(LfmToolParser(), list(text))

        assert calls == []
        assert content == text
        assert not LfmToolParser().extract_tool_calls(text).tools_called

    def test_auto_parser_does_not_dispatch_from_a_rejected_block(self):
        parser = AutoToolParser()
        text = '[f(x=0, note="[g({})]", 1)]'

        assert not parser.extract_tool_calls(text).tools_called

    def test_sibling_block_after_a_rejected_one_is_still_found(self):
        """Skipping a rejected block must not skip what follows it."""
        parser = LfmToolParser()
        content, calls = _stream(parser, list('[index(0)] then [f({"a": 1})]'))

        assert [c["function"]["name"] for c in calls] == ["f"]
        assert content == "[index(0)] then "

    def test_unterminated_opener_masks_later_blocks_but_loses_nothing(self):
        """An opener with no ``]`` stops the walk — the safe failure.

        Its payload may contain anything, so a later block that closes is
        not provably a separate call (see the string-argument test above).
        Nothing is dispatched and nothing is swallowed: the whole span is
        released as content at EOF.
        """
        text = 'Before [oops({"a": 1}) [browse({"q": "x"})]'
        content, calls = _stream(LfmToolParser(), list(text))

        assert calls == []
        assert content == text

        result = LfmToolParser().extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_prose_after_call_in_same_delta_is_not_dropped(self):
        """Trailing prose sharing the closing delta must still be emitted.

        Once a tool call fires the EOF flush is skipped, so anything the
        tool-call return does not carry is lost for good.
        """
        parser = LfmToolParser()
        content, calls = _stream(parser, ['[Calling tool: f({"a": 1})] all done'])

        assert [c["function"]["name"] for c in calls] == ["f"]
        assert content == " all done"

    def test_prose_between_two_calls_in_same_delta_is_not_dropped(self):
        parser = LfmToolParser()
        content, calls = _stream(
            parser, ["[Calling tool: f({})] then [Calling tool: g({})] end"]
        )

        assert [c["function"]["name"] for c in calls] == ["f", "g"]
        assert content == " then  end"

    def test_forced_prefix_seeded_by_postprocessor_is_not_re_emitted(self):
        """``previous_text`` may be seeded with bytes never streamed here.

        ``StreamingPostProcessor.seed_forced_assistant_prefix`` primes
        ``tool_accumulated_text``; treating those bytes as unemitted would
        replay the forced prefix into the visible message.
        """
        parser = LfmToolParser()
        seeded = "Already sent. "
        result = parser.extract_tool_calls_streaming(
            previous_text=seeded,
            current_text=seeded + "[f(x=1)]",
            delta_text="[f(x=1)]",
        )

        assert result is not None
        assert result["tool_calls"][0]["function"]["name"] == "f"
        assert not result.get("content")


class TestLfmStreamingProseRegressions:
    """Ordinary prose must keep streaming — the hold is not a censor."""

    def test_bare_bracket_in_prose_streams_through(self):
        parser = LfmToolParser()
        content, calls = _stream(parser, ["Use ", "a[i", " to index", " the array."])

        assert calls == []
        assert content == "Use a[i to index the array."

    def test_bracketed_prose_note_is_not_eaten(self):
        parser = LfmToolParser()
        content, calls = _stream(parser, ["See ", "[note", ": read", " this]", " ok"])

        assert calls == []
        assert content == "See [note: read this] ok"

    def test_calling_prose_that_is_not_the_envelope_streams_through(self):
        parser = LfmToolParser()
        content, calls = _stream(parser, ["[Calling", " the", " doctor]", " now"])

        assert calls == []
        assert content == "[Calling the doctor] now"

    def test_unfinished_prose_bracket_does_not_hide_the_real_opener(self):
        """A quote inside an unclosed prose bracket must not mask markup.

        Carrying quote state across the turn made everything after
        ``[note "`` look like one long string, so the ``[Calling tool:``
        that followed was never treated as an opener and streamed out raw.
        """
        parser = LfmToolParser()
        content, calls = _stream(
            parser,
            [
                'See [note "unfinished ',
                "[",
                "Calling",
                " tool",
                ":",
                " f({})",
                "]",
            ],
        )

        assert [c["function"]["name"] for c in calls] == ["f"]
        assert content == 'See [note "unfinished '

    def test_backslash_in_prose_bracket_does_not_hide_the_real_opener(self):
        parser = LfmToolParser()
        content, calls = _stream(
            parser, ["Path [dir\\ ", '[Calling tool: f({"a": 1})]']
        )

        assert [c["function"]["name"] for c in calls] == ["f"]
        assert content == "Path [dir\\ "

    def test_whitespace_between_call_and_prose_survives_chunk_boundary(self):
        """Output must not depend on where the chunk boundary falls."""
        together = _stream(LfmToolParser(), ["[f(x=1)] done"])
        split = _stream(LfmToolParser(), ["[f(x=1)] ", "done"])

        assert together[0] == " done"
        assert split[0] == " done"

    def test_apostrophe_in_prose_does_not_disable_markup_holding(self):
        """A stray quote at bracket depth 0 is prose, not an open string.

        Tracking quotes globally would make ``don't`` swallow every later
        ``[``, so the markup after it would stream out raw.
        """
        parser = LfmToolParser()
        content, calls = _stream(
            parser, ["I can't do that, but ", '[Calling tool: f({"a": 1})]']
        )

        assert [c["function"]["name"] for c in calls] == ["f"]
        assert content == "I can't do that, but "

    def test_bracketed_identifier_prose_is_not_held_past_its_token(self):
        """The wider name charset must not extend the hold into prose."""
        parser = LfmToolParser()
        content, calls = _stream(
            parser, ["Due ", "[2026-08-06", " and later]", " we ship."]
        )

        assert calls == []
        assert content == "Due [2026-08-06 and later] we ship."

    def test_bracket_scan_stays_linear(self):
        """Guard the O(n^2)-per-delta scan the first fix round introduced.

        ``"[!" * n`` is all unbalanced openers: rescanning the suffix for
        each one took ~0.85s at n=3200 and grew ~4x per doubling.
        """
        parser = LfmToolParser()
        elapsed = []
        for size in (2000, 8000):
            text = "[!" * size
            start = time.perf_counter()
            parser._safe_content_prefix(text)
            elapsed.append(time.perf_counter() - start)

        # 4x the input must not cost anywhere near 16x the time.
        assert elapsed[1] < 0.5
        assert elapsed[1] < max(elapsed[0], 1e-4) * 8

    def test_markdown_link_streams_through(self):
        parser = LfmToolParser()
        content, calls = _stream(parser, ["[docs", "](https://x)", " here"])

        assert calls == []
        assert content == "[docs](https://x) here"


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

    @staticmethod
    def _make_output(text, finished=False):
        out = MagicMock()
        out.new_text = text
        out.finished = finished
        out.channel = None
        out.finish_reason = "stop" if finished else None
        out.prompt_tokens = 10
        out.completion_tokens = 5
        out.tokens = []
        out.logprobs = None
        out.tool_calls = None
        return out

    def test_streaming_envelope_emits_no_visible_content(self):
        """End-to-end guard on the shipped path (issue: LFM leak).

        Pre-fix this streamed ``: browse({"url": ...})]`` into the chat
        bubble: the parser released ``[Calling tool`` as one delta, the
        global sanitizer stripped exactly that span, and every remaining
        byte reached the client as ordinary content.
        """
        pp = StreamingPostProcessor(self._make_cfg(LfmToolParser()))
        pp.reset()
        pp.tools_requested = True

        content = ""
        tool_calls = []
        last = len(LEAK_CHUNKS) - 1
        for i, chunk in enumerate(LEAK_CHUNKS):
            for event in pp.process_chunk(self._make_output(chunk, finished=i == last)):
                if event.content:
                    content += event.content
                if event.type == "tool_call":
                    tool_calls += event.tool_calls
        for event in pp.finalize():
            if event.content:
                content += event.content
            if event.type == "tool_call":
                tool_calls += event.tool_calls

        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "browse"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == LEAK_ARGS

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
