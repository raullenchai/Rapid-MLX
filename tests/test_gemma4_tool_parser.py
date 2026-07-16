# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for Gemma 4 tool call parser.

Covers:
- Bare numeric/bool/null/float args:        {a:3,b:4}
- Quoted string args:                       {city:<|"|>Paris<|"|>}
- Mixed bare + quoted args:                 {a:3,b:<|"|>hi<|"|>}
- Content stripping (no leakage of markup)
- Multi-tool calls in one output
- Streaming dedup behavior
- Text-format fallback ([Calling tool: ...])

Fix history:
- 2026-04-07: original parser only handled `key:<|"|>value<|"|>` form;
  numeric args like `{a:3,b:4}` parsed as empty dict {}.
"""

import json

import pytest

from vllm_mlx.tool_parsers.gemma4_tool_parser import (
    _GEMMA4_MAX_NESTING_DEPTH,
    Gemma4ToolParser,
    _Gemma4ArgumentParser,
    _parse_gemma4_args,
    _recover_incomplete_gemma4_calls,
    _scan_gemma4_tool_calls,
)

# ---------------------------------------------------------------------------
# _parse_gemma4_args — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args_str,expected",
    [
        # bare numeric — the original bug
        ("a:3,b:4", {"a": 3, "b": 4}),
        # bare bool / null / float
        ("flag:true,n:42", {"flag": True, "n": 42}),
        ("x:null", {"x": None}),
        ("rate:0.5", {"rate": 0.5}),
        ("flag:false", {"flag": False}),
        # quoted string (existing form)
        ('city:<|"|>Paris<|"|>', {"city": "Paris"}),
        # mixed: numeric + quoted
        ('a:3,b:<|"|>hi<|"|>', {"a": 3, "b": "hi"}),
        # multiple quoted strings
        (
            'first:<|"|>Alice<|"|>,last:<|"|>Smith<|"|>',
            {"first": "Alice", "last": "Smith"},
        ),
        # mixed: 3 fields, all types
        (
            'flag:true,n:42,name:<|"|>Bob<|"|>',
            {"flag": True, "n": 42, "name": "Bob"},
        ),
        # quoted string containing punctuation
        ('msg:<|"|>hello, world<|"|>', {"msg": "hello, world"}),
        # negative integer
        ("n:-5", {"n": -5}),
        # nested object used by agent-to-agent tool calls
        (
            'arguments:{message:<|"|>How are {you}?<|"|>,'
            'metadata:{urgent:false,tags:[<|"|>xmpp<|"|>,<|"|>demo<|"|>]}},'
            'endpointId:<|"|>jane@example.org<|"|>,'
            'tool:<|"|>conversation.message<|"|>',
            {
                "arguments": {
                    "message": "How are {you}?",
                    "metadata": {"urgent": False, "tags": ["xmpp", "demo"]},
                },
                "endpointId": "jane@example.org",
                "tool": "conversation.message",
            },
        ),
        # empty arg dict
        ("", {}),
    ],
)
def test_parse_gemma4_args(args_str, expected):
    assert _parse_gemma4_args(args_str) == expected


# ---------------------------------------------------------------------------
# extract_tool_calls — full markup
# ---------------------------------------------------------------------------


def test_extract_bare_numeric_args():
    """Original bug: {a:3,b:4} was returning empty arguments."""
    parser = Gemma4ToolParser()
    out = "<|tool_call>call:add{a:3,b:4}<tool_call|>"
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["name"] == "add"
    args = json.loads(tc["arguments"])
    assert args == {"a": 3, "b": 4}
    # Content must NOT leak any markup
    assert res.content is None


def test_extract_quoted_string_args():
    parser = Gemma4ToolParser()
    out = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {"city": "Paris"}
    assert res.content is None


def test_extract_mixed_args():
    parser = Gemma4ToolParser()
    out = '<|tool_call>call:mix{a:3,b:<|"|>hi<|"|>}<tool_call|>'
    res = parser.extract_tool_calls(out)
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {"a": 3, "b": "hi"}
    assert res.content is None


def test_extract_nested_tool_arguments_without_truncation():
    """Nested argument objects must not close the outer tool call early."""
    parser = Gemma4ToolParser()
    out = (
        "<|tool_call>call:agents_call_tool{"
        'arguments:{message:<|"|>How are {you}?<|"|>},'
        'endpointId:<|"|>jane@example.org<|"|>,'
        'tool:<|"|>conversation.message<|"|>'
        "}<tool_call|>"
    )

    res = parser.extract_tool_calls(out)

    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "agents_call_tool"
    assert json.loads(res.tool_calls[0]["arguments"]) == {
        "arguments": {"message": "How are {you}?"},
        "endpointId": "jane@example.org",
        "tool": "conversation.message",
    }
    assert res.content is None


def test_extract_json_string_containing_closing_brace():
    parser = Gemma4ToolParser()
    out = (
        'call:agents_call_tool{arguments:{"message":"brace } stays"},'
        'endpointId:<|"|>jane@example.org<|"|>,'
        'tool:<|"|>conversation.message<|"|>}'
    )

    res = parser.extract_tool_calls(out)

    assert json.loads(res.tool_calls[0]["arguments"]) == {
        "arguments": {"message": "brace } stays"},
        "endpointId": "jane@example.org",
        "tool": "conversation.message",
    }
    assert res.content is None


def test_no_tool_call_returns_content_unchanged():
    parser = Gemma4ToolParser()
    out = "Hello, the answer is 42."
    res = parser.extract_tool_calls(out)
    assert res.tools_called is False
    assert res.content == out
    assert res.tool_calls == []


def test_multiple_tool_calls():
    parser = Gemma4ToolParser()
    out = (
        "<|tool_call>call:add{a:1,b:2}<tool_call|>"
        "<|tool_call>call:multiply{a:3,b:4}<tool_call|>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0]["name"] == "add"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"a": 1, "b": 2}
    assert res.tool_calls[1]["name"] == "multiply"
    assert json.loads(res.tool_calls[1]["arguments"]) == {"a": 3, "b": 4}
    assert res.content is None


def test_tool_call_with_surrounding_content():
    parser = Gemma4ToolParser()
    out = (
        "Let me check the weather. "
        '<|tool_call>call:get_weather{city:<|"|>NYC<|"|>}<tool_call|>'
        " That should help."
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    # Surrounding text preserved, markup stripped
    assert res.content is not None
    assert "<|tool_call>" not in res.content
    assert "<tool_call|>" not in res.content
    assert "weather" in res.content
    assert "should help" in res.content


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_streaming_emits_completed_tool_call_once():
    parser = Gemma4ToolParser()
    parser.reset()
    full = "<|tool_call>call:add{a:3,b:4}<tool_call|>"

    # Feed token-by-token-ish (split in halves)
    midpoint = len(full) // 2
    delta1 = full[:midpoint]
    delta2 = full[midpoint:]

    r1 = parser.extract_tool_calls_streaming("", delta1, delta1)
    # In the middle of an incomplete tool call — should suppress
    assert r1 is None

    r2 = parser.extract_tool_calls_streaming(delta1, full, delta2)
    # Now complete — should emit
    assert r2 is not None
    assert "tool_calls" in r2
    assert len(r2["tool_calls"]) == 1
    tc = r2["tool_calls"][0]
    assert tc["function"]["name"] == "add"
    assert json.loads(tc["function"]["arguments"]) == {"a": 3, "b": 4}

    # Subsequent calls with no new completed tools — no re-emit
    r3 = parser.extract_tool_calls_streaming(full, full, "")
    assert r3 is None


def test_streaming_waits_for_outer_close_with_nested_arguments():
    parser = Gemma4ToolParser()
    parser.reset()
    partial = (
        "<|tool_call>call:agents_call_tool{"
        'arguments:{message:<|"|>How are {you}?<|"|>},'
        'endpointId:<|"|>jane@example.org<|"|>,'
        'tool:<|"|>conversation.message<|"|>'
    )
    full = partial + "}<tool_call|>"

    assert parser.extract_tool_calls_streaming("", partial, partial) is None
    result = parser.extract_tool_calls_streaming(partial, full, full[len(partial) :])

    assert result is not None
    arguments = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert arguments == {
        "arguments": {"message": "How are {you}?"},
        "endpointId": "jane@example.org",
        "tool": "conversation.message",
    }


def test_streaming_passthrough_when_no_markup():
    parser = Gemma4ToolParser()
    parser.reset()
    r = parser.extract_tool_calls_streaming("", "Hello world", "Hello world")
    assert r == {"content": "Hello world"}


# ---------------------------------------------------------------------------
# Stripped-wire-form regression (PR #558)
# ---------------------------------------------------------------------------
#
# HuggingFace's ``tokenizer.decode(skip_special_tokens=True)`` (the default
# the mlx-vlm streaming detokenizer uses) silently strips the outer
# ``<|tool_call>``/``<tool_call|>`` ids (48/49) at decode time, even when
# rapid-mlx keeps them in ``skip_special_token_ids``. Empirically the
# diffusion-gemma-26b-4bit share probe on 2026-06-11 emitted only the
# stripped body ``call:NAME{...}`` (the inner ``<|"|>`` quote markers
# survive because they're emitted as raw BPE bytes, not as special ids).
#
# Before PR #558 the parser required the outer wrappers and silently
# treated the stripped body as natural-language content — the model's
# tool call leaked into the chat surface as plain text.


def test_extract_stripped_form_bare_numeric():
    """Stripped form without outer wrappers — production reality."""
    parser = Gemma4ToolParser()
    out = "call:add{a:432,b:1}"
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["name"] == "add"
    assert json.loads(tc["arguments"]) == {"a": 432, "b": 1}
    assert res.content is None


def test_extract_stripped_form_quoted_string():
    """Inner quote markers survive HF decode (raw BPE bytes), outer don't."""
    parser = Gemma4ToolParser()
    out = 'call:get_weather{location:<|"|>Palo Alto<|"|>}'
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {"location": "Palo Alto"}
    assert res.content is None


def test_extract_stripped_form_calculator_user_report():
    """Exact failure mode from the vnsh.dev share probe report."""
    parser = Gemma4ToolParser()
    out = "call:calculator{expression:432+1}"
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "calculator"
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {"expression": "432+1"}
    assert res.content is None


def test_streaming_stripped_form_suppresses_then_emits():
    parser = Gemma4ToolParser()
    parser.reset()
    full = "call:add{a:3,b:4}"

    # Split AFTER the opener ``{`` so the body-opener regex fires.
    # Splitting before ``{`` is indistinguishable from natural prose
    # ("I will call you later") and intentionally falls through as
    # content — covered separately by
    # ``test_streaming_stripped_form_natural_text_passes_through``.
    open_idx = full.index("{") + 1
    delta1 = full[:open_idx]
    delta2 = full[open_idx:]

    r1 = parser.extract_tool_calls_streaming("", delta1, delta1)
    # Opener seen, closer not yet → suppress.
    assert r1 is None

    r2 = parser.extract_tool_calls_streaming(delta1, full, delta2)
    assert r2 is not None
    assert "tool_calls" in r2
    assert len(r2["tool_calls"]) == 1
    tc = r2["tool_calls"][0]
    assert tc["function"]["name"] == "add"
    assert json.loads(tc["function"]["arguments"]) == {"a": 3, "b": 4}


def test_streaming_stripped_form_natural_text_passes_through():
    """``call:foo{`` is the only false-positive shape — make sure prose
    that happens to mention ``call`` or ``:`` does not get suppressed."""
    parser = Gemma4ToolParser()
    parser.reset()
    text = "I will call you later: see you then."
    r = parser.extract_tool_calls_streaming("", text, text)
    assert r == {"content": text}


def test_has_pending_recognises_stripped_opener():
    parser = Gemma4ToolParser()
    assert parser.has_pending_tool_call("call:foo{x:1}") is True
    assert parser.has_pending_tool_call("call:foo{") is True
    # No opener — must not trigger
    assert parser.has_pending_tool_call("hello world") is False
    assert parser.has_pending_tool_call("call me later") is False


# ---------------------------------------------------------------------------
# Malformed / unterminated tool-call strings
#
# The balanced-brace scanner tracks Gemma quote (``<|"|>``) and JSON string
# state so nested ``{}`` inside string values do not close the call early.
# If the model emits an UNTERMINATED string, the scanner stays in the
# open-string state forever, swallows the trailing ``}``, and returns zero
# matches. A single, conservative best-effort fallback (the historical
# non-greedy ``call:NAME{(.*?)}`` semantics: first ``}`` closes the body)
# recovers such a call — but ONLY on the NON-STREAMING finalize path and ONLY
# when the balanced scanner found no complete call. We do NOT attempt to
# perfectly reconstruct rare mixed valid+malformed multi-call output.
# STREAMING must still hold an incomplete/malformed call pending (return None)
# exactly as before.
# ---------------------------------------------------------------------------


def test_scanner_swallows_unterminated_gemma_string():
    """Pin the raw scanner behavior: an unterminated ``<|"|>`` string makes
    the balanced scanner miss the call (0 matches), which is precisely why
    the non-streaming recovery below is needed."""
    matches, opener_count = _scan_gemma4_tool_calls('call:f{x:<|"|>unterminated}')
    assert matches == []
    assert opener_count == 1


def test_nonstreaming_recovers_unterminated_gemma_string():
    """codex #1102 BLOCKING-1: malformed ``call:f{x:<|"|>unterminated}``
    (missing closing ``<|"|>``) must still surface a tool call in the
    non-streaming finalize path, matching the historical best-effort parser
    instead of dropping the whole call to raw content."""
    parser = Gemma4ToolParser()
    out = 'call:f{x:<|"|>unterminated}'
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "f"
    # Best-effort: the value keeps the raw (unclosed) quote text.
    assert json.loads(res.tool_calls[0]["arguments"]) == {"x": '<|"|>unterminated'}
    assert res.content is None


def test_recover_helper_no_close_brace_returns_nothing():
    """The recovery helper only fires when a closing ``}`` exists. A
    genuinely truncated call (no ``}`` at all) yields no recovery, which is
    what keeps a still-generating stream from being force-closed."""
    assert _recover_incomplete_gemma4_calls('call:f{x:<|"|>hel') == []
    assert _recover_incomplete_gemma4_calls("call:f{x:") == []
    assert _recover_incomplete_gemma4_calls("call:f{") == []


def test_streaming_malformed_complete_stays_pending():
    """CRITICAL: streaming must NOT adopt the non-streaming recovery. A
    malformed-but-terminated call (``call:f{x:<|"|>unterminated}``) has an
    opener the balanced scanner cannot close, so streaming must keep it
    pending (return None) — it never force-emits a best-effort call
    mid-stream."""
    parser = Gemma4ToolParser()
    parser.reset()
    out = 'call:f{x:<|"|>unterminated}'
    assert parser.extract_tool_calls_streaming("", out, out) is None


def test_streaming_genuinely_incomplete_stays_pending():
    """A call still being generated (no closing ``}`` yet) must stay pending
    on the streaming path — unchanged by the recovery."""
    parser = Gemma4ToolParser()
    parser.reset()
    for text in ('call:f{x:<|"|>hel', "call:f{x:", "call:f{"):
        parser.reset()
        assert parser.extract_tool_calls_streaming("", text, text) is None


def test_valid_nested_unaffected_by_recovery():
    """Guard against the recovery over-firing: a well-formed nested call must
    still be parsed by the balanced scanner (clean nested dict), NOT routed
    through the best-effort first-``}`` recovery."""
    parser = Gemma4ToolParser()
    out = 'call:g{a:{b:<|"|>v<|"|>}}'
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert json.loads(res.tool_calls[0]["arguments"]) == {"a": {"b": "v"}}


def test_many_key_object_parses_after_position_key_match():
    """Regression for the compiled-key / position-match optimisation
    (codex #1102 NIT): a many-key object must still parse every key/value."""
    pairs = {f"k{i}": i for i in range(50)}
    body = ",".join(f"k{i}:{i}" for i in range(50))
    res = _parse_gemma4_args(body)
    assert res == pairs


# ---------------------------------------------------------------------------
# codex #1102 round-2 BLOCKING-2: RecursionError guard / nesting-depth limit
#
# Arbitrarily nested objects/arrays recurse through _Gemma4ArgumentParser.
# Without a bound, deep model output overflows CPython's recursion limit and
# raises an UNCAUGHT RecursionError that crashes request processing. The fix
# adds an explicit depth guard (raises a controlled ValueError) AND catches
# RecursionError alongside the existing parse failures, degrading to the
# historical flat/best-effort parse. Deeply-nested input must NEVER crash.
# ---------------------------------------------------------------------------


def test_deeply_nested_object_does_not_crash():
    """A pathologically deep ``{a:{a:{...}}}`` object must degrade to the flat
    best-effort parse, never raise RecursionError."""
    parser = Gemma4ToolParser()
    depth = 5000  # far beyond CPython's ~1000 recursion limit
    body = "deep:" + "{a:" * depth + "1" + "}" * depth
    out = "call:f{" + body + "}"
    res = parser.extract_tool_calls(out)  # must not raise
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "f"
    # Best-effort: the value falls back to a string rather than crashing.
    args = json.loads(res.tool_calls[0]["arguments"])
    assert "deep" in args


def test_deeply_nested_array_does_not_crash():
    """A pathologically deep ``[[[...]]]`` array must not raise."""
    parser = Gemma4ToolParser()
    depth = 5000
    body = "deep:" + "[" * depth + "]" * depth
    out = "call:f{" + body + "}"
    res = parser.extract_tool_calls(out)  # must not raise
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "f"


def test_deeply_nested_bare_json_value_does_not_crash():
    """A deep bare JSON value recurses inside the stdlib ``json`` decoder,
    which the parser's own depth guard cannot see. The explicit
    ``except RecursionError`` in the fallback path must still keep it
    crash-free."""
    parser = Gemma4ToolParser()
    depth = 5000
    body = "deep:" + "[" * depth + "]" * depth
    res = _parse_gemma4_args(body)  # direct: must not raise
    assert isinstance(res, dict)
    out = "call:f{" + body + "}"
    assert parser.extract_tool_calls(out).tools_called is True  # must not raise


def test_nesting_at_limit_parses_over_limit_degrades():
    """Nesting up to the depth limit parses structurally; going well past the
    limit degrades to best-effort without crashing."""
    # Just under the limit → clean recursive parse.
    ok_depth = _GEMMA4_MAX_NESTING_DEPTH - 1
    ok_body = "a:" + "{a:" * ok_depth + "1" + "}" * ok_depth
    ok = _parse_gemma4_args(ok_body)
    assert isinstance(ok, dict)
    # Walk down every nested ``a`` to confirm the full structure survived the
    # recursive parse (no truncation) and bottoms out at the ``1`` value.
    node = ok
    levels = 0
    while isinstance(node, dict):
        assert "a" in node
        node = node["a"]
        levels += 1
    assert node == 1
    assert levels >= ok_depth  # every nesting level was parsed, none dropped

    # Far past the limit → the depth guard raises internally, caller degrades.
    over_depth = _GEMMA4_MAX_NESTING_DEPTH + 50
    over_body = "a:" + "{a:" * over_depth + "1" + "}" * over_depth
    degraded = _parse_gemma4_args(over_body)  # must not raise
    assert isinstance(degraded, dict)


def test_depth_guard_raises_controlled_value_error():
    """The parser's own depth guard raises a ``ValueError`` (not a bare
    RecursionError) so every existing ``except ValueError`` handler routes it
    to the fallback."""
    over_depth = _GEMMA4_MAX_NESTING_DEPTH + 10
    body = "a:" + "{a:" * over_depth + "1" + "}" * over_depth
    with pytest.raises(ValueError):
        _Gemma4ArgumentParser(body).parse_arguments()


# ---------------------------------------------------------------------------
# codex #1102 round-2 NIT: bounded JSON-string decode (O(n) not O(n^2))
#
# Each JSON-quoted value was decoded from a fresh ``self.text[self.index:]``
# slice → O(n^2) across many quoted fields. The fix decodes in place with
# ``raw_decode(self.text, self.index)``. These tests pin correctness across
# the value shapes stdlib json emits.
# ---------------------------------------------------------------------------


def test_bounded_json_decode_many_string_fields_correct():
    """Many JSON-quoted-string fields must all decode correctly (the O(n)
    rewrite must not drop or mis-slice any value)."""
    n = 200
    body = ",".join(f'k{i}:"val {i}"' for i in range(n))
    res = _parse_gemma4_args(body)
    assert res == {f"k{i}": f"val {i}" for i in range(n)}


def test_bounded_json_decode_escaped_characters():
    """Escapes inside a JSON string value survive the in-place decode."""
    res = _parse_gemma4_args(r'msg:"line1\nline2 \"quote\" \\slash"')
    assert res == {"msg": 'line1\nline2 "quote" \\slash'}


def test_bounded_json_decode_quoted_key():
    """A JSON-quoted KEY (which also routes through the bounded decoder) is
    parsed correctly alongside a following field."""
    res = _parse_gemma4_args('"quoted key":5,plain:6')
    assert res == {"quoted key": 5, "plain": 6}


def test_bounded_json_decode_mixed_value_types():
    """Interleaved JSON strings, numbers, and Gemma strings all keep their
    positions after the in-place decode."""
    res = _parse_gemma4_args('a:"s1",b:3,c:<|"|>g<|"|>,d:"s2",e:true')
    assert res == {"a": "s1", "b": 3, "c": "g", "d": "s2", "e": True}


# ---------------------------------------------------------------------------
# Adversarial malformed-input surface — none of these may crash, and each must
# degrade to a recovered call OR a clean passthrough (never a silent drop of a
# recoverable call).
# ---------------------------------------------------------------------------


def test_unbalanced_extra_open_brace_no_crash():
    """An extra unclosed ``{`` (no matching ``}``) has no closable body — it
    must stay pending/as-content, never crash."""
    parser = Gemma4ToolParser()
    out = "call:f{a:{b:1"  # opener + nested open, never closed
    res = parser.extract_tool_calls(out)  # must not raise
    # No closing brace anywhere → nothing recoverable, leaks as content.
    assert res.tools_called is False
    assert res.content == out


def test_unbalanced_extra_close_brace_no_crash():
    """A stray extra ``}`` after a complete call must not crash; the first
    ``}`` closes the body and the extra is treated as content."""
    parser = Gemma4ToolParser()
    out = "call:f{a:1}}"
    res = parser.extract_tool_calls(out)  # must not raise
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "f"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"a": 1}


def test_empty_args_object():
    """``call:f{}`` (empty argument object) parses to an empty dict."""
    parser = Gemma4ToolParser()
    res = parser.extract_tool_calls("call:f{}")
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "f"
    assert json.loads(res.tool_calls[0]["arguments"]) == {}


def test_huge_many_string_object_no_quadratic_blowup():
    """A large many-string object must parse fully and quickly (guards the
    O(n) bounded-decode path against regressions)."""
    n = 1000
    pairs = {f"k{i}": f"v{i}" for i in range(n)}
    body = ",".join(f'k{i}:"v{i}"' for i in range(n))
    res = _parse_gemma4_args(body)
    assert res == pairs


def test_genuinely_truncated_call_not_force_closed_nonstreaming():
    """A call with an opener but NO closing ``}`` at all is genuinely
    incomplete: even on the non-streaming path there is nothing to recover,
    so it must NOT be force-closed into a bogus call."""
    parser = Gemma4ToolParser()
    for out in ("call:f{", "call:f{x:", 'call:f{x:<|"|>hel'):
        res = parser.extract_tool_calls(out)
        assert res.tools_called is False, out
        assert res.content == out
