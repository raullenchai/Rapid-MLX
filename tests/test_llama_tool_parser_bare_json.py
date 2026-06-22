# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Llama tool parser's bare-JSON + python-tag
support — fixes issue #700 ("F-008").

The Llama 3.1/3.2 official chat template trains the model to emit
assistant tool calls as bare ``{"name": "X", "parameters": {...}}``
JSON, with an optional ``<|python_tag|>`` prefix when the system header
advertises ``Environment: ipython``. Before the fix, the parser only
matched the Llama-4-style ``<function=name>...</function>`` XML wrapper
and therefore let the entire JSON leak into ``message.content`` — see
the issue body for the full repro.

These tests pin the three shapes (XML / python-tag / bare JSON), the
``parameters`` ↔ ``arguments`` key alias, the streaming contract, and a
small set of negative cases that previously could have been misrouted
as tool calls (prose JSON, partial fragments, etc.).
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.tool_parsers import LlamaToolParser


@pytest.fixture
def parser() -> LlamaToolParser:
    return LlamaToolParser()


# ---------------------------------------------------------------------------
# Shape 1 — XML wrapper (Llama 4 / vLLM style), preserved by the fix.
# ---------------------------------------------------------------------------


class TestXmlWrapperShape:
    def test_single_call(self, parser: LlamaToolParser):
        text = '<function=multiply>{"x": 3, "y": 4}</function>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "multiply"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"x": 3, "y": 4}

    def test_multiple_calls(self, parser: LlamaToolParser):
        text = '<function=add>{"a": 1}</function><function=multiply>{"x": 3}</function>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert [tc["name"] for tc in result.tool_calls] == ["add", "multiply"]

    def test_content_before(self, parser: LlamaToolParser):
        text = 'Computing result<function=calc>{"n": 5}</function>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.content == "Computing result"


# ---------------------------------------------------------------------------
# Shape 2 — <|python_tag|>{...} (Llama 3.1 ipython mode).
# ---------------------------------------------------------------------------


class TestPythonTagShape:
    def test_basic(self, parser: LlamaToolParser):
        text = '<|python_tag|>{"name": "web_search", "parameters": {"query": "你好"}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "web_search"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"query": "你好"}
        assert result.content is None

    def test_tag_with_preceding_prose(self, parser: LlamaToolParser):
        text = (
            "Sure, let me search.<|python_tag|>"
            '{"name": "web_search", "parameters": {"query": "weather"}}'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.content == "Sure, let me search."

    def test_tag_with_arguments_alias(self, parser: LlamaToolParser):
        # Some quantizations drift to OpenAI's "arguments" key under
        # fine-tuning. We accept both so users don't lose tool routing.
        text = '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Tokyo"}


# ---------------------------------------------------------------------------
# Shape 3 — bare JSON (Llama 3.1/3.2 default). This is the F-008 root
# cause: every test here would have failed against the pre-fix parser.
# ---------------------------------------------------------------------------


class TestBareJsonShape:
    def test_f008_repro_chinese_greeting(self, parser: LlamaToolParser):
        """Exact wire string from issue #700 / F-008 screenshot."""
        text = '{"name": "web_search", "parameters": {"query": "你好"}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called, (
            "Bare JSON tool-call leaked as content — F-008 regression"
        )
        assert result.tool_calls[0]["name"] == "web_search"
        assert json.loads(result.tool_calls[0]["arguments"]) == {"query": "你好"}
        assert result.content is None

    def test_bare_json_with_arguments_alias(self, parser: LlamaToolParser):
        text = '{"name": "calc", "arguments": {"n": 5}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {"n": 5}

    def test_bare_json_no_arg_uses_empty_parameters(self, parser: LlamaToolParser):
        # No-arg tool calls render an explicit ``"parameters": {}`` in
        # the Llama 3.1/3.2 chat template (``arguments | tojson`` on an
        # empty dict). Bare ``{"name": "X"}`` with no args key is *not*
        # a tool call — it's prose JSON that happens to have a ``name``
        # field; see TestNoFalsePositives.
        text = '{"name": "get_current_time", "parameters": {}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {}

    def test_bare_json_with_nested_args(self, parser: LlamaToolParser):
        args = {"filter": {"city": "Tokyo", "limit": 10, "tags": ["a", "b"]}}
        text = '{"name": "search", "parameters": ' + json.dumps(args["filter"]) + "}"
        # Re-form full payload
        text = json.dumps({"name": "search", "parameters": args["filter"]})
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == args["filter"]

    def test_bare_json_preserves_prefix_content(self, parser: LlamaToolParser):
        text = (
            "Let me look that up. "
            '{"name": "web_search", "parameters": {"query": "weather"}}'
        )
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.content == "Let me look that up."

    def test_string_with_brace_inside(self, parser: LlamaToolParser):
        # The brace-balancer must be string-aware so a "}" inside a JSON
        # string doesn't terminate the object early.
        text = '{"name": "echo", "parameters": {"msg": "hello } world"}}'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert json.loads(result.tool_calls[0]["arguments"]) == {"msg": "hello } world"}


# ---------------------------------------------------------------------------
# Negative cases — must NOT misroute as tool calls.
# ---------------------------------------------------------------------------


class TestNoFalsePositives:
    def test_plain_prose_is_not_a_tool_call(self, parser: LlamaToolParser):
        text = "Hello! How can I help you today?"
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_prose_json_without_name_key(self, parser: LlamaToolParser):
        # Model handed back an object literal as an example — must stay
        # as content, not be routed to a tool call.
        text = 'Here is an example: {"city": "Tokyo", "temp": 22}'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_prose_json_with_name_but_no_args(self, parser: LlamaToolParser):
        # Regression for codex r1 P2: a JSON object that *happens* to
        # carry a ``name`` field but no ``parameters``/``arguments`` key
        # must NOT be routed as a no-arg tool call — that's almost
        # always prose, not a Llama-template tool call (the template
        # always emits ``parameters``).
        text = 'Here is a user: {"name": "Alice", "age": 30}'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_json_with_empty_name(self, parser: LlamaToolParser):
        text = '{"name": "", "parameters": {}}'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called

    def test_malformed_json_left_as_content(self, parser: LlamaToolParser):
        text = '{"name": "broken", "parameters": {oops'
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_empty_input(self, parser: LlamaToolParser):
        result = parser.extract_tool_calls("")
        assert not result.tools_called

    def test_plain_long_prefix_pending_fast_path_skips_json_scan(self, monkeypatch):
        """Plain cumulative prose must stay on the parser-owned fast path.

        ``has_pending_tool_call`` is invoked for every streaming prefix.
        With no ``{``/``<`` anchors, it should not enter the JSON
        scanner at all; otherwise ordinary long text becomes O(n^2).
        """
        import vllm_mlx.tool_parsers.llama_tool_parser as llama_mod

        parser = LlamaToolParser()

        def fail_json_scan(*_args, **_kwargs):
            raise AssertionError("plain text should not enter JSON scanner")

        monkeypatch.setattr(llama_mod, "_find_top_level_json_object", fail_json_scan)

        text = ""
        for _ in range(128):
            text += "ordinary assistant prose without anchors "
            assert parser.has_pending_tool_call(text) is False


# ---------------------------------------------------------------------------
# Streaming contract — clients see content tokens until the JSON closes,
# then a single tool_calls delta.
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_bare_json_streams_content_until_close(self, parser: LlamaToolParser):
        # Mid-stream fragments must not emit tool_calls yet.
        partial = '{"name": "web_search", "parameters": {"query": "wea'
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=partial,
            delta_text=partial,
        )
        assert result is None or "tool_calls" not in result

    def test_first_brace_token_is_pending_not_content(self, parser: LlamaToolParser):
        """Regression for codex r1 P1: the very first ``{`` token of a
        streamed bare-JSON tool call must be treated as pending — we
        cannot wait for the ``"name"`` key to appear because that would
        leak the opening bytes as assistant content."""
        for delta in ("{", '{"', '{"na'):
            result = parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=delta,
                delta_text=delta,
            )
            # Must NOT pass through as content. Either None (still
            # buffering) or already a structured tool_calls dict — but
            # crucially not ``{"content": "{"}``.
            assert result is None or "content" not in result, (
                f"streaming leaked partial JSON prefix as content: "
                f"delta={delta!r} -> {result!r}"
            )

    def test_streaming_buffered_prefix_flushed_if_not_tool(
        self, parser: LlamaToolParser
    ):
        """Companion to test_first_brace_token_is_pending_not_content:
        if we buffered a ``{``-prefixed stream on the bet that it was a
        tool call, and the eventual close reveals it wasn't (no
        ``name``/``parameters`` pair), the buffered text MUST be
        flushed back to the client as content so we don't drop it."""
        # Mid-stream: still pending (no emit).
        previous = '{"city": "Tokyo", "temp"'
        mid = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=previous,
            delta_text=previous,
        )
        assert mid is None or "content" not in mid

        # Closing brace arrives — not a tool call → flush as content.
        delta = ": 22}"
        current = previous + delta
        final = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert final is not None
        assert "content" in final
        assert final["content"] == current

    def test_bare_json_emits_tool_calls_on_close(self, parser: LlamaToolParser):
        previous = '{"name": "web_search", "parameters": {"query": "weather"'
        delta = "}}"
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_python_tag_emits_on_close(self, parser: LlamaToolParser):
        previous = (
            '<|python_tag|>{"name": "web_search", "parameters": {"query": "weather"'
        )
        delta = "}}"
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_xml_wrapper_still_works_streaming(self, parser: LlamaToolParser):
        previous = '<function=add>{"a": 1}'
        delta = "</function>"
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "add"

    def test_plain_content_streams_as_content(self, parser: LlamaToolParser):
        delta = "Hello there"
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=delta,
            delta_text=delta,
        )
        assert result == {"content": delta}

    def test_prose_prefix_then_bare_json_tool_call(self, parser: LlamaToolParser):
        """Regression for codex r2 P2 (1/2): a streamed response of
        ``Let me check. {"name": "X", "parameters": {}}`` must NOT leak
        the JSON tail as content. Earlier prose chunks pass through; the
        JSON anchor onward is buffered and emitted as ``tool_calls``."""
        # Step 1: prose preface — passes through.
        d1 = "Let me check. "
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        assert r1 == {"content": d1}

        # Step 2: opening brace + name/params arrive in one go.
        d2 = '{"name": "search", "parameters": {}}'
        cur = d1 + d2
        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=cur, delta_text=d2
        )
        # The JSON closes in this delta → tool_calls emitted, no leak.
        assert r2 is not None
        assert "tool_calls" in r2
        assert r2["tool_calls"][0]["function"]["name"] == "search"

    def test_post_json_content_streams_through(self, parser: LlamaToolParser):
        """Regression for codex r2 P2 (2/2): after a (non-tool) JSON
        object is flushed as content, subsequent prose deltas MUST
        continue to stream through. Previously the parser kept seeing
        the buffered ``{`` prefix and returned ``None`` for every
        trailing token, dropping the rest of the reply."""
        # Step 1: closed prose JSON — flushed as content on close.
        d1 = '{"city": "Tokyo"}'
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        assert r1 is not None
        assert "content" in r1
        assert r1["content"] == d1

        # Step 2: trailing prose — must pass through.
        d2 = " is nice."
        cur = d1 + d2
        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=cur, delta_text=d2
        )
        assert r2 == {"content": d2}


# ---------------------------------------------------------------------------
# Codex r3 regressions: re-emit, sentinel split, XML string with closer,
# stream-end flush.
# ---------------------------------------------------------------------------


class TestCodexR3Regressions:
    def test_no_reemit_after_tool_close(self, parser: LlamaToolParser):
        """Codex r3 BLOCKING: once a bare-JSON tool call has been
        streamed as a ``tool_calls`` delta, subsequent prose deltas
        MUST NOT re-emit the same tool call. The old
        ``_buffered_region_start``-based path re-scanned from position
        0 on every call and kept hitting the already-emitted span."""
        prev = '{"name": "a", "parameters": {}}'
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=prev, delta_text=prev
        )
        assert r1 is not None and "tool_calls" in r1

        delta = " some prose"
        r2 = parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=prev + delta,
            delta_text=delta,
        )
        assert r2 == {"content": " some prose"}

    def test_back_to_back_bare_json_tool_calls(self, parser: LlamaToolParser):
        """Codex r3 BLOCKING: two bare-JSON tool objects in a row must
        each be emitted once, with ``index`` incremented on the second.
        Pre-fix path either duplicated the first call or dropped the
        second."""
        d1 = '{"name": "a", "parameters": {}}'
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        assert r1 is not None and "tool_calls" in r1
        assert r1["tool_calls"][0]["function"]["name"] == "a"
        assert r1["tool_calls"][0]["index"] == 0

        d2 = '{"name": "b", "parameters": {}}'
        cur = d1 + d2
        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=cur, delta_text=d2
        )
        assert r2 is not None and "tool_calls" in r2
        assert len(r2["tool_calls"]) == 1
        assert r2["tool_calls"][0]["function"]["name"] == "b"
        # Continuation index — clients identify tool calls by index.
        assert r2["tool_calls"][0]["index"] == 1

    def test_idempotent_double_call_on_same_prefix(self, parser: LlamaToolParser):
        """Codex r3 BLOCKING corollary: feeding the same
        ``(previous_text, current_text, delta_text)`` triple twice must
        not double-emit. The state-machine is purely a diff of
        ``_emitted_state`` so identical input yields identical output."""
        prev = '{"name": "a", "parameters": {}}'
        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=prev, delta_text=prev
        )
        assert r1 is not None and "tool_calls" in r1

        # Re-call with the SAME previous_text — invariant means there
        # is no new content past previous_text, so nothing to emit.
        r1_again = parser.extract_tool_calls_streaming(
            previous_text=prev, current_text=prev, delta_text=""
        )
        assert r1_again is None

    @pytest.mark.parametrize(
        "split_at",
        [1, 2, 4, 6, 8, 10, 12],  # various split points within '<|python_tag|>'
    )
    def test_python_tag_split_across_deltas(
        self, parser: LlamaToolParser, split_at: int
    ):
        """Codex r3 MAJOR: partial ``<|python_tag|>`` prefixes (``<|``,
        ``<|py``, ``<|python_ta``...) MUST be held back so per-char SSE
        doesn't leak them as content before the full tag arrives."""
        full = '<|python_tag|>{"name": "x", "parameters": {}}'
        d1 = full[:split_at]
        d2 = full[split_at:]

        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        # The first chunk is a strict prefix of the sentinel — never
        # emit it as content.
        assert r1 is None or "content" not in r1

        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=full, delta_text=d2
        )
        assert r2 is not None and "tool_calls" in r2
        assert r2["tool_calls"][0]["function"]["name"] == "x"

    @pytest.mark.parametrize("split_at", [1, 3, 5, 8])
    def test_function_open_split_across_deltas(
        self, parser: LlamaToolParser, split_at: int
    ):
        """Codex r3 MAJOR: partial ``<function=`` prefixes (``<f``,
        ``<fun``, ``<function``...) MUST be held back."""
        full = '<function=greet>{"name": "Alice"}</function>'
        d1 = full[:split_at]
        d2 = full[split_at:]

        r1 = parser.extract_tool_calls_streaming(
            previous_text="", current_text=d1, delta_text=d1
        )
        assert r1 is None or "content" not in r1

        r2 = parser.extract_tool_calls_streaming(
            previous_text=d1, current_text=full, delta_text=d2
        )
        assert r2 is not None and "tool_calls" in r2
        assert r2["tool_calls"][0]["function"]["name"] == "greet"

    def test_xml_arg_value_contains_function_closer(self, parser: LlamaToolParser):
        """Codex r3 MAJOR: a JSON string inside the XML-wrapped args
        that *contains* ``}</function>`` must NOT terminate the wrapper
        early. The old ``re.compile(r"<function=([^>]+)>(\\{.*?\\})</function>")``
        was delimiter-unsafe and corrupted both the args and the
        trailing content."""
        text = '<function=echo>{"msg": "close }</function> trick"}</function>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "echo"
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "msg": "close }</function> trick"
        }
        # Nothing trails the genuine ``</function>`` — content is empty.
        assert result.content is None

    def test_flush_held_content_at_stream_end(self, parser: LlamaToolParser):
        """Codex r3 MAJOR: when the stream ends with bytes still being
        held as a possible sentinel prefix AND no tool call ever fired,
        those bytes must be released as ordinary content. Mirror of the
        Hermes / harmony streaming-cluster fix."""
        # Held suffix — ``<|python`` is a strict prefix of the
        # python_tag opener.
        assert parser.flush_held_content("abc<|python") == "<|python"
        # No held bytes — flush returns empty so no spurious event.
        assert parser.flush_held_content("hello world") == ""

    def test_mixed_content_and_tool_in_one_delta_returns_both(
        self, parser: LlamaToolParser
    ):
        """Codex r4 BLOCKING (1/2): when a single delta carries both a
        prose preface AND a closed tool span, the parser MUST return
        both channels in one dict so the postprocessor can emit the
        content event before the tool_call event. Returning only
        tool_calls drops the preface, because the postprocessor sets
        ``tool_calls_detected=True`` and short-circuits the finalize
        cross-format fallback."""
        cur = 'Let me check. {"name": "search", "parameters": {}}'
        r = parser.extract_tool_calls_streaming(
            previous_text="", current_text=cur, delta_text=cur
        )
        assert r is not None
        assert r.get("content") == "Let me check. "
        assert "tool_calls" in r
        assert r["tool_calls"][0]["function"]["name"] == "search"
        assert r["tool_calls"][0]["index"] == 0

    def test_mixed_tool_then_trailing_content_in_one_delta(
        self, parser: LlamaToolParser
    ):
        """Codex r4 BLOCKING (2/2): when a single delta carries a
        closed tool span followed by trailing prose, the trailing
        bytes MUST also be returned in the same dict — they would
        otherwise be lost (postprocessor sees ``tool_calls_detected``
        and stops calling the streaming parser)."""
        cur = '{"name": "a", "parameters": {}} tail'
        r = parser.extract_tool_calls_streaming(
            previous_text="", current_text=cur, delta_text=cur
        )
        assert r is not None
        assert "tool_calls" in r
        assert r["tool_calls"][0]["function"]["name"] == "a"
        assert r.get("content") == " tail"

    def test_has_pending_on_prose_with_unclosed_brace(self, parser: LlamaToolParser):
        """Codex r4 MAJOR: ``has_pending_tool_call`` must recognise a
        prose preface followed by ``{`` (with no ``"name"`` yet) as
        pending. The postprocessor fast-path at postprocessor.py:1490
        skips the full streaming branch when ``has_pending`` returns
        False, so before the fix the lonely ``{`` after the preface
        leaked as content."""
        assert parser.has_pending_tool_call("Let me check. {")
        assert parser.has_pending_tool_call('Let me check. {"na')
        # Closed prose JSON (no ``"name"`` key) stays non-pending so
        # plain-prose streams don't pay the streaming-branch cost.
        assert not parser.has_pending_tool_call('Result: {"x": 1}')
        # Plain text — definitely not pending.
        assert not parser.has_pending_tool_call("Hello world")

    def test_has_pending_on_prose_with_whitespace_before_name(
        self, parser: LlamaToolParser
    ):
        """Codex r5 MAJOR: ``has_pending_tool_call`` must also catch
        closed prose-prefixed tool JSON where the model emitted
        whitespace / newlines between ``{`` and ``"name"`` (real
        formatting drift). The pre-fix literal ``{"name"`` substring
        check missed this and the fast-path leaked the whole
        ``Let me check. { "name": ...}`` payload as content."""
        # Whitespace after ``{``.
        assert parser.has_pending_tool_call(
            'Let me check. { "name": "search", "parameters": {}}'
        )
        # Newline + indent (LLM pretty-print drift).
        assert parser.has_pending_tool_call(
            'Calling tool:\n{\n  "name": "search",\n  "parameters": {}\n}'
        )
        # Extra key before ``"name"`` — still a Llama tool call.
        assert parser.has_pending_tool_call(
            '{"type": "function", "name": "search", "parameters": {}}'
        )

    def test_streaming_postprocessor_invariant_violation(self, parser: LlamaToolParser):
        """Codex r3 NIT: when ``current_text`` does not start with
        ``previous_text`` (a postprocessor bug), the parser must not
        crash. We accept a defensive over-emission (extra bytes flushed
        as content) — losing tokens or raising is worse than briefly
        duplicating content under a contract violation."""
        # Pathological input: postprocessor changed history mid-stream.
        previous = "old prefix "
        delta = "new tail"
        current = "DIFFERENT old prefix " + delta  # doesn't start with previous
        # Must not raise and must produce a content event with a string.
        r = parser.extract_tool_calls_streaming(
            previous_text=previous, current_text=current, delta_text=delta
        )
        # Either None (everything held) or a content event — never raise.
        if r is not None:
            assert "content" in r
            assert isinstance(r["content"], str)


# ---------------------------------------------------------------------------
# Wire-format declaration — keep ``test_tool_parser_wire_formats`` happy
# and document the upgrade explicitly.
# ---------------------------------------------------------------------------


def test_expected_wire_formats_declared():
    fmts = LlamaToolParser.EXPECTED_WIRE_FORMATS
    assert "function_bare" in fmts
    assert "llama_python_tag" in fmts
    assert "raw_json" in fmts
