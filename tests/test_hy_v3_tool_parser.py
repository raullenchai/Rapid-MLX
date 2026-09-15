# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the Hy3 tool call parser (vLLM HYV3ToolParser port).

Architecture under test (see ``rapid_mlx/tool_parsers/hy_v3_tool_parser.py``):
  * suffix resolved ONCE at ``__init__`` from vocab, pinned as fixed strings
  * token-ID / fixed-string gate on streaming entry (no full-text re-parse)
  * two-phase FSM: SEEKING_NAME → STREAMING_ARGS (withhold trailing ``}``)
  * ``<think>`` lives in the SEPARATE reasoning parser (disjoint stream) — the
    tool parser has zero ``<think>`` code
  * malformed-close salvage on the NON-STREAMING path only

Scenarios that encode REAL model behavior (preserved from the pre-pivot
suite because they were authored from ``pipenetwork/Hy3-REAP50/75-MLX-4bit``
output, 2026-07-09 spike):
  * canonical JSON body with/without the ``:opensource`` suffix
  * malformed close ``<tool_call>NAME</arg_value>`` (4-bit numerical noise)
  * XML-pair argument variant + type coercion
  * multiple tool calls
  * JSON string value containing the literal ``</arg_value>`` substring
  * request ``tools`` allowlist filtering
"""

from __future__ import annotations

import json

from rapid_mlx.tool_parsers import HyV3ToolParser, ToolParserManager


# ---------------------------------------------------------------------------
# Registration / declarative surface
# ---------------------------------------------------------------------------
def test_parser_is_registered():
    """The parser must appear in the registry under both aliases so
    downstream ``tool_call_parser="hy_v3"`` (and the CLI-friendly ``hy3``)
    resolve without a ``KeyError``."""
    assert ToolParserManager.get_tool_parser("hy_v3") is HyV3ToolParser
    assert ToolParserManager.get_tool_parser("hy3") is HyV3ToolParser


def test_expected_wire_formats_declared():
    """Structural test — every parser MUST declare a non-empty
    ``EXPECTED_WIRE_FORMATS`` tuple so the audit matrix stays honest."""
    assert HyV3ToolParser.EXPECTED_WIRE_FORMATS == ("hy3_native",)


def test_supports_native_tool_format_flag():
    """The Hy3 chat template renders assistant ``tool_calls`` back as
    ``<tool_call:opensource>…<end_of_tool_call:opensource>``, so the
    native-format flag MUST be True to prevent the tool-history round-trip
    from being converted to synthetic text."""
    assert HyV3ToolParser.SUPPORTS_NATIVE_TOOL_FORMAT is True


def test_suffix_defaults_to_opensource_without_tokenizer():
    """With no tokenizer the parser MUST fall back to the ``:opensource``
    label every current ``pipenetwork/Hy3-*-MLX-4bit`` checkpoint emits, and
    pin the fixed tag strings accordingly."""
    p = HyV3ToolParser()
    assert p.suffix == ":opensource"
    assert p.tool_call_start_token == "<tool_call:opensource>"
    assert p.tool_sep_token == "<tool_sep:opensource>"
    assert p.tool_call_end_token == "<end_of_tool_call:opensource>"
    assert p.arg_value_end_token == "</arg_value:opensource>"


class _FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def test_suffix_resolved_from_vocab_suffixless():
    """When the tokenizer vocab exposes the bare ``<tool_call>`` token (a
    future revision that drops the label), the resolver MUST pin the
    suffix-less strings."""
    tok = _FakeTokenizer(
        {
            "<tool_call>": 1000,
            "<tool_sep>": 1001,
            "<end_of_tool_call>": 1002,
        }
    )
    p = HyV3ToolParser(tokenizer=tok)
    assert p.suffix == ""
    assert p.tool_call_start_token == "<tool_call>"
    assert p.tool_call_start_token_id == 1000


def test_suffix_resolved_from_vocab_labelled():
    """When the vocab carries the COMPLETE labelled token trio, the resolver
    pins the labelled strings AND the token id."""
    tok = _FakeTokenizer(
        {
            "<tool_call:opensource>": 2000,
            "<tool_sep:opensource>": 2001,
            "<end_of_tool_call:opensource>": 2002,
        }
    )
    p = HyV3ToolParser(tokenizer=tok)
    assert p.suffix == ":opensource"
    assert p.tool_call_start_token_id == 2000


def test_suffix_incomplete_set_falls_back_to_default():
    """codex R4 BLOCKING: a vocab exposing only ``<tool_call:foo>`` but not its
    matching ``<tool_sep:foo>`` / ``<end_of_tool_call:foo>`` MUST NOT pin the
    incomplete ``:foo`` suffix (every downstream ``find`` would look for a
    non-existent separator/close). It falls back to the ``:opensource``
    default instead."""
    tok = _FakeTokenizer({"<tool_call:foo>": 5000})
    p = HyV3ToolParser(tokenizer=tok)
    assert p.suffix == ":opensource"
    assert p.tool_call_start_token == "<tool_call:opensource>"


def test_suffix_prefers_complete_over_incomplete_candidate():
    """When one suffix has the complete trio and another has only
    ``<tool_call>``, the COMPLETE suffix is chosen regardless of dict order."""
    tok = _FakeTokenizer(
        {
            "<tool_call:foo>": 6000,  # incomplete — must be ignored
            "<tool_call:opensource>": 6001,
            "<tool_sep:opensource>": 6002,
            "<end_of_tool_call:opensource>": 6003,
        }
    )
    p = HyV3ToolParser(tokenizer=tok)
    assert p.suffix == ":opensource"


# ---------------------------------------------------------------------------
# Non-streaming extraction — real wire shapes
# ---------------------------------------------------------------------------
def test_canonical_json_body_with_opensource_suffix():
    """The chat-template default emission — every tag carries the
    ``:opensource`` suffix and the body is a JSON object."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>get_weather"
        '<tool_sep:opensource>{"city": "Paris"}'
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["name"] == "get_weather"
    assert json.loads(tc["arguments"]) == {"city": "Paris"}
    assert res.content is None


def test_canonical_json_body_without_suffix():
    """Future-proof: a parser whose vocab pinned the suffix-less strings MUST
    accept the plain variant so upstream can drop the ``:opensource`` label
    in a later revision."""
    tok = _FakeTokenizer({"<tool_call>": 1, "<tool_sep>": 2, "<end_of_tool_call>": 3})
    parser = HyV3ToolParser(tokenizer=tok)
    out = '<tool_call>get_weather<tool_sep>{"city": "Paris"}<end_of_tool_call>'
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "get_weather"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"city": "Paris"}


def test_malformed_close_defensive_strip():
    """4-bit numerical noise workaround: the model skips
    ``<tool_sep>{args}<end_of_tool_call>`` and jumps straight to
    ``</arg_value>``. The NON-STREAMING path MUST still surface the tool name
    (empty arguments) rather than dropping the call silently — otherwise the
    user sees an empty ``tool_calls`` array and thinks the model refused.
    Empirically observed on ``pipenetwork/Hy3-REAP50-MLX-4bit`` +
    ``Hy3-REAP75-MLX-4bit`` (10/10 BFCL simple_python prompts, 2026-07-09
    spike; see bug1_comment.md filed against mlx-lm PR #1211)."""
    parser = HyV3ToolParser()
    out = "<tool_call:opensource>get_weather</arg_value:opensource>"
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["name"] == "get_weather"
    assert tc["arguments"] == "{}"


def test_malformed_close_suffix_less():
    """Same defensive strip works when the suffix is absent (vocab pinned the
    suffix-less strings). The vocab exposes the COMPLETE bare token trio, as a
    real suffix-less checkpoint would (codex R4 BLOCKING: an incomplete set is
    never selected)."""
    tok = _FakeTokenizer({"<tool_call>": 1, "<tool_sep>": 2, "<end_of_tool_call>": 3})
    parser = HyV3ToolParser(tokenizer=tok)
    res = parser.extract_tool_calls("<tool_call>get_weather</arg_value>")
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "get_weather"
    assert res.tool_calls[0]["arguments"] == "{}"


def test_truncated_xml_pair_missing_close_is_not_salvaged():
    """codex R7 BLOCKING: a truncated XML-pair call that carries structural
    tokens (``<tool_sep>`` / ``<arg_key>`` / ``<arg_value>``) but is simply
    MISSING its ``<end_of_tool_call>`` must NOT be promoted to a completed
    executable call by the malformed-close salvage — that salvage is reserved
    for the bare ``NAME</arg_value>`` 4-bit-noise shape. It is treated as
    incomplete output (content), not a call."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>lookup<tool_sep:opensource>"
        "<arg_key:opensource>city</arg_key:opensource>"
        "<arg_value:opensource>Paris</arg_value:opensource>"
        # NOTE: no <end_of_tool_call> — truncated mid-stream.
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is False
    assert res.tool_calls == []
    assert res.content == out


def test_completed_call_with_malformed_json_body_degrades_to_empty_args():
    """codex R9 BLOCKING: a COMPLETED call (has ``<end_of_tool_call>``) whose
    JSON body is malformed junk (``{bad}``) must still emit a tool call — with
    args degraded to ``{}`` — NOT be dropped as no-call. Only a body that is
    still truncated (no close token) waits."""
    parser = HyV3ToolParser()
    out = "<tool_call:opensource>fn<tool_sep:opensource>{bad json}<end_of_tool_call:opensource>"
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "fn"
    assert res.tool_calls[0]["arguments"] == "{}"


def test_streaming_completed_call_malformed_json_degrades_to_empty_args():
    """The streaming mirror of the above — a whole call with a malformed JSON
    body but a real close emits ``fn`` with ``{}`` args."""
    parser = HyV3ToolParser()
    tool_acc, _content = _collect_stream(
        parser,
        list(
            "<tool_call:opensource>fn<tool_sep:opensource>{bad}<end_of_tool_call:opensource>"
        ),
    )
    assert tool_acc[0]["name"] == "fn"
    assert json.loads(tool_acc[0]["args"]) == {}


def test_truncated_json_body_no_close_is_not_a_call():
    """A ``{``-body that is still TRUNCATED (no ``<end_of_tool_call>`` yet) is
    NOT a completed call — it is incomplete output, preserved as content."""
    parser = HyV3ToolParser()
    out = '<tool_call:opensource>fn<tool_sep:opensource>{"a": 1'  # no close
    res = parser.extract_tool_calls(out)
    assert res.tools_called is False
    assert res.content == out


def test_suffix_resolution_is_json_only_complete_xml_tokens_not_required():
    """codex R9 NIT (documented-as-intentional): the suffix completeness check
    uses the JSON-only trio (call/sep/end); ``arg_key`` / ``arg_value`` are an
    optional XML-pair variant and are NOT required. A vocab with a complete trio
    under ``:opensource`` but NO arg tokens still resolves ``:opensource`` and
    parses JSON-body calls."""
    tok = _FakeTokenizer(
        {
            "<tool_call:opensource>": 1,
            "<tool_sep:opensource>": 2,
            "<end_of_tool_call:opensource>": 3,
            # deliberately NO <arg_key:...> / <arg_value:...>
        }
    )
    parser = HyV3ToolParser(tokenizer=tok)
    assert parser.suffix == ":opensource"
    res = parser.extract_tool_calls(
        '<tool_call:opensource>fn<tool_sep:opensource>{"a": 1}<end_of_tool_call:opensource>'
    )
    assert res.tools_called is True
    assert json.loads(res.tool_calls[0]["arguments"]) == {"a": 1}


def test_json_body_then_stray_arg_value_opener_salvages_args():
    """Second real malformed shape from the spike: a well-formed JSON body
    followed by a STRAY ``<arg_value>`` opener before the canonical close
    (``NAME<tool_sep>{"radius":5}<arg_value:opensource><end_of_tool_call>``).
    The JSON prefix ``raw_decode`` MUST recover the real args and ignore the
    trailing stray opener."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>calculate_circle_area"
        '<tool_sep:opensource>{"radius": 5}<arg_value:opensource>'
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "calculate_circle_area"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"radius": 5}


def test_xml_pair_argument_variant():
    """The chat template's second-choice emission — each argument as a
    separate ``<arg_key>K</arg_key><arg_value>V</arg_value>`` pair."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>get_weather"
        "<tool_sep:opensource>"
        "<arg_key:opensource>city</arg_key:opensource>"
        "<arg_value:opensource>Paris</arg_value:opensource>"
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "get_weather"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"city": "Paris"}


def test_xml_arg_value_containing_literal_end_token_not_truncated():
    """An ``<arg_value>`` payload can legitimately carry the literal
    ``<end_of_tool_call>`` string as free-form text (codex R15). A plain
    ``str.find`` on the non-JSON (XML-pair) close search would truncate the
    call at that interior literal and DROP the argument. Both the non-streaming
    (``extract_tool_calls``) and streaming (char-by-char reassembled) paths MUST
    preserve the full value — including the literal end-token substring — and
    only close on the REAL trailing ``<end_of_tool_call>``."""
    parser = HyV3ToolParser()
    oc = parser.tool_call_start_token
    sep = parser.tool_sep_token
    end = parser.tool_call_end_token
    ak, ake = parser.arg_key_start_token, parser.arg_key_end_token
    av, ave = parser.arg_value_start_token, parser.arg_value_end_token
    wire = f"{oc}logit{sep}{ak}msg{ake}{av}contains {end} inside{ave}{end}"
    expected = {"msg": f"contains {end} inside"}

    # Non-streaming.
    res = parser.extract_tool_calls(wire)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "logit"
    assert json.loads(res.tool_calls[0]["arguments"]) == expected
    # The literal end-token substring survives in the extracted value.
    assert end in json.loads(res.tool_calls[0]["arguments"])["msg"]

    # Streaming — char-by-char, reassembled.
    parser.reset()
    tool_acc, content = _collect_stream(parser, list(wire))
    assert tool_acc[0]["name"] == "logit"
    assert json.loads(tool_acc[0]["args"]) == expected
    assert end in json.loads(tool_acc[0]["args"])["msg"]
    assert content == ""


def test_xml_pair_multi_key_with_type_coercion():
    """Multi-key XML variant — ``<arg_value>`` payload MUST be JSON-decoded so
    ``1`` → int, ``"two"`` → str, ``true`` → bool."""
    tok = _FakeTokenizer({"<tool_call>": 1, "<tool_sep>": 2, "<end_of_tool_call>": 3})
    parser = HyV3ToolParser(tokenizer=tok)
    out = (
        "<tool_call>lookup<tool_sep>"
        "<arg_key>a</arg_key><arg_value>1</arg_value>"
        '<arg_key>b</arg_key><arg_value>"two"</arg_value>'
        "<arg_key>flag</arg_key><arg_value>true</arg_value>"
        "<end_of_tool_call>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert json.loads(res.tool_calls[0]["arguments"]) == {
        "a": 1,
        "b": "two",
        "flag": True,
    }


def test_sep_less_xml_pair_body():
    """Some 4-bit checkpoints skip ``<tool_sep>`` but still emit full XML
    pairs. The name is the residue before the first ``<arg_key>`` opener and
    the args are recovered from the pairs."""
    tok = _FakeTokenizer({"<tool_call>": 1, "<tool_sep>": 2, "<end_of_tool_call>": 3})
    parser = HyV3ToolParser(tokenizer=tok)
    out = (
        "<tool_call>do_it"
        "<arg_key>x</arg_key><arg_value>1</arg_value>"
        "<arg_key>y</arg_key><arg_value>2</arg_value>"
        "<end_of_tool_call>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "do_it"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"x": 1, "y": 2}


def test_multiple_tool_calls():
    """Two tool_calls in one assistant turn — both must be extracted in wire
    order."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>a<tool_sep:opensource>{}<end_of_tool_call:opensource>"
        '<tool_call:opensource>b<tool_sep:opensource>{"x": 1}'
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0]["name"] == "a"
    assert res.tool_calls[1]["name"] == "b"
    assert json.loads(res.tool_calls[1]["arguments"]) == {"x": 1}


def test_json_body_containing_literal_arg_value_close_parses_correctly():
    """A JSON body whose STRING VALUE legitimately contains the literal
    substring ``</arg_value>`` MUST round-trip unchanged. ``raw_decode``
    consumes only a well-formed JSON prefix so the literal inside a string is
    preserved rather than mistaken for a close boundary."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>log_message<tool_sep:opensource>"
        '{"snippet": "The tag </arg_value:opensource> is not a close here.",'
        ' "level": "info"}'
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {
        "snippet": "The tag </arg_value:opensource> is not a close here.",
        "level": "info",
    }


def test_json_body_containing_literal_opener_parses_as_one_call():
    """codex R6 BLOCKING: a JSON string value that legitimately contains the
    literal ``<tool_call:opensource>`` opener text MUST NOT be split into a
    phantom second call. The non-streaming block scan is JSON-aware — it finds
    the real close over the whole remainder (``raw_decode`` consumes the opener
    literal inside the string) instead of bounding at the interior substring."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>log<tool_sep:opensource>"
        '{"msg": "prefix <tool_call:opensource> suffix", "n": 1}'
        "<end_of_tool_call:opensource>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "log"
    assert json.loads(res.tool_calls[0]["arguments"]) == {
        "msg": "prefix <tool_call:opensource> suffix",
        "n": 1,
    }


def test_no_tool_call_returns_content_unchanged():
    """A pure text response must pass through as content with
    ``tools_called=False``."""
    parser = HyV3ToolParser()
    res = parser.extract_tool_calls("The answer is Paris.")
    assert res.tools_called is False
    assert res.tool_calls == []
    assert res.content == "The answer is Paris."


def test_truncated_opener_no_close_is_not_a_call():
    """codex R4 BLOCKING: an opener with NEITHER a canonical NOR a malformed
    close (truncated / streaming-incomplete output like
    ``<tool_call:opensource>get_weather``) MUST NOT become a parsed call with
    ``{}`` args — it is pending/plain content. ``tools_called`` is False and the
    raw tail is preserved as content."""
    parser = HyV3ToolParser()
    res = parser.extract_tool_calls("<tool_call:opensource>get_weather")
    assert res.tools_called is False
    assert res.tool_calls == []
    assert res.content == "<tool_call:opensource>get_weather"


def test_completed_call_then_truncated_opener_keeps_only_completed():
    """A completed call followed by a truncated opener returns ONLY the
    completed call — the dangling opener is not fabricated into a second empty
    call."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>a<tool_sep:opensource>{}<end_of_tool_call:opensource>"
        "<tool_call:opensource>b_trunc"  # no close
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "a"


def test_tool_name_filter_via_request_tools():
    """When the request supplies a ``tools`` list, unknown tool names MUST be
    filtered out (defence against name hallucination)."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>bogus_tool<tool_sep:opensource>"
        '{"x": 1}<end_of_tool_call:opensource>'
    )
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    res = parser.extract_tool_calls(out, request=request)
    assert res.tools_called is False
    assert res.tool_calls == []


def test_valid_names_filter_preserves_rejected_span_in_content():
    """When ``valid_names`` is set and every parsed call is filtered out, the
    raw span of the rejected call MUST be preserved in ``content`` — silently
    dropping it makes the output look like a refusal when the model actually
    tried to invoke an off-list tool."""
    parser = HyV3ToolParser()
    out = (
        "<tool_call:opensource>bogus_tool<tool_sep:opensource>"
        '{"x": 1}<end_of_tool_call:opensource>'
    )
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    res = parser.extract_tool_calls(out, request=request)
    assert res.tools_called is False
    assert res.tool_calls == []
    assert res.content is not None
    assert "bogus_tool" in res.content


def test_valid_names_filter_preserves_mixed_valid_and_rejected():
    """When SOME calls are valid, ``tools_called=True`` and ``content=None``
    (exclusive-turn policy). The rejected span is lost in that case by design
    — the OpenAI-compatible contract forbids mixed ``tool_calls`` +
    ``content`` in a single assistant turn."""
    parser = HyV3ToolParser()
    out = (
        '<tool_call:opensource>allowed_tool<tool_sep:opensource>{"y": 2}'
        "<end_of_tool_call:opensource>"
        "\n"
        '<tool_call:opensource>bogus_tool<tool_sep:opensource>{"x": 1}'
        "<end_of_tool_call:opensource>"
    )
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    res = parser.extract_tool_calls(out, request=request)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "allowed_tool"
    assert res.content is None


def test_text_format_fallback_reachable_without_native_opener():
    """When there is NO native ``<tool_call>`` opener but the model degraded
    into the shared ``[Calling tool="X" k="v"]`` text form (low-quant
    degradation), the text-format fallback MUST still fire. codex BLOCKING:
    the early return on missing opener previously made this unreachable."""
    parser = HyV3ToolParser()
    out = '[Calling tool="get_weather" city="Paris"]'
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "get_weather"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"city": "Paris"}


def test_text_format_fallback_respects_request_allowlist():
    """codex R7 BLOCKING: the text-format degradation fallback MUST apply the
    request ``tools`` allowlist too — otherwise a degraded
    ``[Calling tool="bogus"]`` bypasses the filtering that native Hy3 calls
    enforce. An off-list name is dropped (no tool_calls) and preserved as
    content."""
    parser = HyV3ToolParser()
    request = {"tools": [{"function": {"name": "get_weather"}}]}
    out = '[Calling tool="bogus" x="1"]'
    res = parser.extract_tool_calls(out, request=request)
    assert res.tools_called is False
    assert res.tool_calls == []
    assert res.content == out


def test_text_format_fallback_admits_on_list_name_with_allowlist():
    """The mirror of the above: an ON-list text-format call still fires when a
    ``tools`` allowlist is present."""
    parser = HyV3ToolParser()
    request = {"tools": [{"function": {"name": "get_weather"}}]}
    out = '[Calling tool="get_weather" city="Paris"]'
    res = parser.extract_tool_calls(out, request=request)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Streaming — token-gate + 2-phase FSM. Boundary cases are now trivially
# green because the opener gate + fixed-string finds replace the bespoke
# straddle machinery.
# ---------------------------------------------------------------------------
def _stepper(parser, request=None):
    """Return a ``step(delta)`` closure that feeds deltas and returns the
    parser's per-delta result."""
    state = {"prev": ""}

    def step(delta: str):
        cur = state["prev"] + delta
        msg = parser.extract_tool_calls_streaming(
            state["prev"], cur, delta, request=request
        )
        state["prev"] = cur
        return msg

    return step


def _collect_stream(parser, chunks, request=None):
    """Feed ``chunks`` and return ``(tool_acc, content)`` where ``tool_acc``
    maps index → {name, args}."""
    parser.reset()
    step = _stepper(parser, request=request)
    tool_acc: dict[int, dict] = {}
    content = ""
    for d in chunks:
        msg = step(d)
        if not msg:
            continue
        if msg.get("content"):
            content += msg["content"]
        for tc in msg.get("tool_calls", []) or []:
            entry = tool_acc.setdefault(tc["index"], {"name": "", "args": ""})
            fn = tc.get("function", {})
            if fn.get("name"):
                entry["name"] = fn["name"]
            if fn.get("arguments"):
                entry["args"] += fn["arguments"]
    return tool_acc, content


def test_streaming_holds_until_close_then_emits_json_body():
    """The name emits when ``<tool_sep>`` lands; the args stream as a JSON
    diff; the delta carrying ``<end_of_tool_call>`` completes the args with
    the trailing ``}``. Reassembled args MUST equal the wire JSON."""
    parser = HyV3ToolParser()
    tool_acc, content = _collect_stream(
        parser,
        [
            "<tool_call:opensource>",
            "get_weather",
            "<tool_sep:opensource>",
            '{"city": "NYC"}',
            "<end_of_tool_call:opensource>",
        ],
    )
    assert 0 in tool_acc
    assert tool_acc[0]["name"] == "get_weather"
    assert json.loads(tool_acc[0]["args"]) == {"city": "NYC"}
    assert content == ""


def test_streaming_char_by_char_json_body_no_leak():
    """Char-by-char delivery (the harshest boundary case) MUST NOT leak any
    raw markup as content and MUST reassemble the args to valid JSON. The
    partial-opener prefix hold catches the char-split opener; the fixed-string
    finds handle every interior boundary."""
    parser = HyV3ToolParser()
    wire = (
        "<tool_call:opensource>get_weather"
        '<tool_sep:opensource>{"city": "NYC"}'
        "<end_of_tool_call:opensource>"
    )
    tool_acc, content = _collect_stream(parser, list(wire))
    assert tool_acc[0]["name"] == "get_weather"
    assert json.loads(tool_acc[0]["args"]) == {"city": "NYC"}
    assert content == "", f"raw markup leaked as content: {content!r}"


def test_streaming_json_body_with_escapes_and_unicode_reassembles():
    """Char-by-char streaming of a JSON body whose string values contain
    escape sequences (``\\n``, ``\\"``, ``\\\\``) AND a ``\\uXXXX`` unicode
    escape MUST reassemble to the exact wire JSON. The parser streams the RAW
    wire text verbatim (never re-serializing via ``json.dumps``), so the
    open prefixes stay byte-aligned with the closed document even when the
    value contains ``\\uXXXX`` — a re-serialize would decode it to the char
    and make the diff non-monotonic (regression guard for the escape-stream
    snapshot bug)."""
    parser = HyV3ToolParser()
    # ``json.dumps`` default (ensure_ascii=True) puts a ``\\uXXXX`` escape on
    # the wire for the é; the other escapes exercise the dangling-backslash
    # and quote-in-value hold logic.
    args_obj = {"path": "/a/b.txt", "content": 'line1\nline2 "q" \\ café'}
    body = json.dumps(args_obj)
    assert "\\u" in body  # confirm the wire really carries a unicode escape
    wire = (
        "<tool_call:opensource>write_file<tool_sep:opensource>"
        + body
        + "<end_of_tool_call:opensource>"
    )
    tool_acc, content = _collect_stream(parser, list(wire))
    assert tool_acc[0]["name"] == "write_file"
    assert json.loads(tool_acc[0]["args"]) == args_obj
    assert content == "", f"raw markup leaked as content: {content!r}"


def test_streaming_json_body_nested_and_mixed_types_reassembles():
    """A JSON body with nested objects/arrays and mixed scalar types streams
    char-by-char and reassembles exactly."""
    parser = HyV3ToolParser()
    args_obj = {"n": 3, "flag": True, "z": None, "list": [1, 2], "obj": {"k": "v"}}
    wire = (
        "<tool_call:opensource>f<tool_sep:opensource>"
        + json.dumps(args_obj)
        + "<end_of_tool_call:opensource>"
    )
    tool_acc, content = _collect_stream(parser, list(wire))
    assert json.loads(tool_acc[0]["args"]) == args_obj
    assert content == ""


def test_streaming_xml_pair_reassembles_args():
    """The XML-pair variant streams pairs into a growing JSON object; the
    reassembled args MUST equal the wire pairs with type coercion."""
    parser = HyV3ToolParser()
    tool_acc, content = _collect_stream(
        parser,
        [
            "<tool_call:opensource>",
            "multi_arg_fn",
            "<tool_sep:opensource>",
            "<arg_key:opensource>city</arg_key:opensource>",
            "<arg_value:opensource>Paris</arg_value:opensource>",
            "<arg_key:opensource>n</arg_key:opensource>",
            "<arg_value:opensource>3</arg_value:opensource>",
            "<end_of_tool_call:opensource>",
        ],
    )
    assert tool_acc[0]["name"] == "multi_arg_fn"
    assert json.loads(tool_acc[0]["args"]) == {"city": "Paris", "n": 3}
    assert content == ""


def test_streaming_first_arg_value_close_does_not_finish_call():
    """A mid-body ``</arg_value>`` (closing the FIRST argument value) MUST NOT
    finish the call — only ``<end_of_tool_call>`` closes it. Otherwise the
    parser would flush truncated args after the first argument."""
    parser = HyV3ToolParser()
    tool_acc, content = _collect_stream(
        parser,
        [
            "<tool_call:opensource>",
            "multi_arg_fn",
            "<tool_sep:opensource>",
            "<arg_key:opensource>city</arg_key:opensource>",
            "<arg_value:opensource>Paris</arg_value:opensource>",
            "<arg_key:opensource>units</arg_key:opensource>",
            "<arg_value:opensource>metric</arg_value:opensource>",
            "<end_of_tool_call:opensource>",
        ],
    )
    assert json.loads(tool_acc[0]["args"]) == {"city": "Paris", "units": "metric"}
    assert content == ""


def test_streaming_close_split_across_sse_boundary_still_emits():
    """The close tag arrives split across two SSE chunks (``<end_of_tool_c``
    then ``all:opensource>``). Because the parser searches for the fixed close
    string in the accumulated buffer, the split resolves on the chunk that
    completes it — no bespoke transition tracking needed."""
    parser = HyV3ToolParser()
    tool_acc, content = _collect_stream(
        parser,
        [
            "<tool_call:opensource>",
            "my_fn",
            "<tool_sep:opensource>",
            "{}",
            "<end_of_tool_c",
            "all:opensource>",
        ],
    )
    assert tool_acc[0]["name"] == "my_fn"
    assert json.loads(tool_acc[0]["args"]) == {}
    assert content == ""


def test_streaming_multiple_tool_calls_each_get_own_index_and_name():
    """TWO tool calls streamed char-by-char in one turn MUST each surface with
    their OWN index, name, and args — the FSM resets to SEEKING_NAME on each
    ``<end_of_tool_call>`` so the second opener starts a fresh indexed call.
    codex BLOCKING: the pre-fix ``_name_sent`` never reset, so the second
    call's name/args were folded into the first index."""
    parser = HyV3ToolParser()
    wire = (
        '<tool_call:opensource>get_a<tool_sep:opensource>{"x": 1}'
        "<end_of_tool_call:opensource>"
        '<tool_call:opensource>get_b<tool_sep:opensource>{"y": 2}'
        "<end_of_tool_call:opensource>"
    )
    tool_acc, content = _collect_stream(parser, list(wire))
    assert sorted(tool_acc.keys()) == [0, 1]
    assert tool_acc[0]["name"] == "get_a"
    assert json.loads(tool_acc[0]["args"]) == {"x": 1}
    assert tool_acc[1]["name"] == "get_b"
    assert json.loads(tool_acc[1]["args"]) == {"y": 2}
    assert content == ""


def test_streaming_passthrough_content_when_no_tool_call():
    """Plain content deltas — no ``<tool_call>`` opener seen — pass through as
    content."""
    parser = HyV3ToolParser()
    parser.reset()
    msg = parser.extract_tool_calls_streaming("", "Hello ", "Hello ")
    assert msg is not None
    assert msg["content"] == "Hello "


def test_streaming_text_format_flows_as_content_recovered_at_finalize():
    """codex R8 BLOCKING (parity note): the ``[Calling tool="X"]`` text-format
    degradation is NOT streamed incrementally as tool_calls — it has no native
    token boundaries. During streaming its bytes flow as ordinary CONTENT; the
    postprocessor's finalize re-runs the (allowlist-aware) non-streaming
    ``extract_tool_calls`` over the full text to recover the structured call.
    This test pins the documented contract: streaming yields content only, and
    the non-streaming extractor recovers the call from the same full text."""
    parser = HyV3ToolParser()
    wire = '[Calling tool="get_weather" city="Paris"]'
    # Streaming: no tool_calls, bytes surface as content (no native opener).
    tool_acc, content = _collect_stream(parser, list(wire))
    assert tool_acc == {}
    assert content == wire
    # Finalize-equivalent: the non-streaming path recovers the structured call.
    parser.reset()
    res = parser.extract_tool_calls(wire)
    assert res.tools_called is True
    assert res.tool_calls[0]["name"] == "get_weather"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"city": "Paris"}


def test_streaming_content_after_completed_tool_call_is_suppressed():
    """Once an assistant turn is a TOOL-CALL turn (any ``<tool_call>`` opener
    has appeared), post-close plain-content deltas MUST be suppressed —
    OpenAI-compatible clients treat ``tool_calls`` and ``content`` as mutually
    exclusive for a single assistant turn."""
    parser = HyV3ToolParser()
    parser.reset()
    step = _stepper(parser)
    step("<tool_call:opensource>")
    step("do_it")
    step("<tool_sep:opensource>")
    step("{}")
    final = step("<end_of_tool_call:opensource>")
    assert final is not None and "tool_calls" in final
    assert step(" now ") is None
    assert step("what?") is None


def test_streaming_partial_opener_prefix_held_then_released_on_falsify():
    """When a delta ends in a partial-opener prefix that later FALSIFIES into
    ordinary text (``<tool_ca`` then ``rrot recipe`` → ``<tool_carrot``), the
    held bytes MUST surface as content once the tail resolves. No prose is
    lost, and no partial markup leaks before resolution."""
    parser = HyV3ToolParser()
    parser.reset()
    step = _stepper(parser)
    m1 = step("Look at this: <tool_ca")
    # The trailing ``<tool_ca`` is a partial-opener prefix — held back.
    assert (m1 or {}).get("content", "") == "Look at this: "
    m2 = step("rrot recipe")
    assert m2 is not None
    assert m2["content"] == "<tool_carrot recipe"


def test_streaming_partial_opener_prefix_resolves_to_tool_call():
    """When the partial-opener prefix COMPLETES into a real opener, the turn
    becomes a tool-call turn; the held bytes are markup (not content) and the
    call streams normally."""
    parser = HyV3ToolParser()
    tool_acc, content = _collect_stream(
        parser,
        [
            "<tool_ca",
            "ll:opensource>get_weather<tool_sep:opensource>{}",
            "<end_of_tool_call:opensource>",
        ],
    )
    assert tool_acc[0]["name"] == "get_weather"
    # The ``<tool_ca`` prefix was held, then absorbed into the tool-call turn.
    assert content == ""


def _stream_raw(parser, chunks, request=None):
    """Feed ``chunks`` and return the list of every non-None per-delta result
    (so a test can assert on argument-only deltas, not just accumulated
    name/args)."""
    parser.reset()
    step = _stepper(parser, request=request)
    results = []
    for d in chunks:
        msg = step(d)
        if msg is not None:
            results.append(msg)
    return results


def test_streaming_respects_request_tool_allowlist():
    """The streaming path MUST honour the request ``tools`` allowlist so a
    hallucinated off-list name emits NEITHER a header NOR argument deltas
    (codex R4 NIT: a name-only assertion would pass even if args leaked)."""
    parser = HyV3ToolParser()
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    chunks = [
        "<tool_call:opensource>",
        "hallucinated_tool",
        "<tool_sep:opensource>",
        '{"leak": "me"}',
        "<end_of_tool_call:opensource>",
    ]
    tool_acc, content = _collect_stream(parser, chunks, request=request)
    # No tool-call surfaces AT ALL for an off-list name — no header, no args.
    assert tool_acc == {}
    assert content == ""
    # And no result ever carried a tool_calls delta for the suppressed index.
    results = _stream_raw(parser, chunks, request=request)
    assert all("tool_calls" not in msg for msg in results)


def test_streaming_off_list_call_in_one_delta_emits_no_args():
    """The suppressed off-list call arriving WHOLE in one delta (name + args +
    close together) must still emit NO argument delta — the same-delta path is
    the exact scenario codex R4 BLOCKING #3 raised for the args leak."""
    parser = HyV3ToolParser()
    request = {"tools": [{"function": {"name": "allowed_tool"}}]}
    one_delta = (
        "<tool_call:opensource>hallucinated<tool_sep:opensource>"
        '{"leak": "me"}<end_of_tool_call:opensource>'
    )
    results = _stream_raw(parser, [one_delta], request=request)
    for msg in results:
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            assert not fn.get("arguments"), f"args leaked: {tc!r}"
            assert fn.get("name") != "hallucinated"


def test_streaming_flush_held_content_releases_dangling_prefix():
    """A stream ending in a partial-opener prefix that never completed leaves
    those bytes held; ``flush_held_content`` MUST release them so the final
    chars are not dropped."""
    parser = HyV3ToolParser()
    parser.reset()
    step = _stepper(parser)
    step("done <tool_ca")
    flushed = parser.flush_held_content("done <tool_ca")
    assert flushed == "<tool_ca"


def test_flush_held_content_empty_when_tool_call_opened():
    """When a real opener is present, the held tail is markup, not content —
    ``flush_held_content`` returns empty so nothing leaks."""
    parser = HyV3ToolParser()
    full = (
        "<tool_call:opensource>fn<tool_sep:opensource>{}<end_of_tool_call:opensource>"
    )
    assert parser.flush_held_content(full) == ""


# ---------------------------------------------------------------------------
# Pending predicate
# ---------------------------------------------------------------------------
def test_has_pending_tool_call_recognises_suffix_variant():
    """The pending-call predicate MUST recognise the pinned ``:opensource``
    opener so streaming shutdown handlers can flush the buffer, and MUST NOT
    report pending once the call has closed."""
    parser = HyV3ToolParser()
    assert parser.has_pending_tool_call("<tool_call:opensource>fn") is True
    assert (
        parser.has_pending_tool_call(
            "<tool_call:opensource>fn<end_of_tool_call:opensource>"
        )
        is False
    )
    assert parser.has_pending_tool_call("just a plain message") is False


# ---------------------------------------------------------------------------
# codex R3 regressions: same-delta multi-call drain + pre-opener content flush
# ---------------------------------------------------------------------------
def test_streaming_two_complete_calls_in_one_delta_both_emit():
    """TWO complete tool calls arriving in a SINGLE streaming delta MUST both
    emit — the streaming path drains every call processable this tick, not just
    the first opener. codex R3 BLOCKING: the pre-fix processed one opener per
    invocation, dropping the second same-delta call until another delta (which
    may never come) arrived."""
    parser = HyV3ToolParser()
    one_delta = (
        '<tool_call:opensource>get_a<tool_sep:opensource>{"x": 1}'
        "<end_of_tool_call:opensource>"
        '<tool_call:opensource>get_b<tool_sep:opensource>{"y": 2}'
        "<end_of_tool_call:opensource>"
    )
    # A SINGLE chunk carrying both complete calls (the exact codex scenario).
    tool_acc, content = _collect_stream(parser, [one_delta])
    assert sorted(tool_acc.keys()) == [0, 1]
    assert tool_acc[0]["name"] == "get_a"
    assert json.loads(tool_acc[0]["args"]) == {"x": 1}
    assert tool_acc[1]["name"] == "get_b"
    assert json.loads(tool_acc[1]["args"]) == {"y": 2}
    assert content == ""


def test_streaming_literal_opener_in_json_arg_is_not_a_phantom_call():
    """codex R6 BLOCKING: a JSON string argument value containing the literal
    ``<tool_call:opensource>`` opener text MUST NOT be split into a phantom
    second call. ``_opener_positions`` walks call spans JSON-aware, so the
    interior opener substring is opaque inside the (parsed) body — char-by-char
    delivery still yields exactly ONE call with the full argument stream."""
    parser = HyV3ToolParser()
    wire = (
        "<tool_call:opensource>log<tool_sep:opensource>"
        '{"msg": "prefix <tool_call:opensource> suffix", "n": 1}'
        "<end_of_tool_call:opensource>"
    )
    tool_acc, content = _collect_stream(parser, list(wire))
    assert sorted(tool_acc.keys()) == [0]  # no phantom index 1
    assert tool_acc[0]["name"] == "log"
    assert json.loads(tool_acc[0]["args"]) == {
        "msg": "prefix <tool_call:opensource> suffix",
        "n": 1,
    }
    assert content == ""


def test_streaming_content_before_opener_in_same_delta_not_dropped():
    """Content that precedes the FIRST opener in the SAME delta as the tool call
    MUST be emitted, not silently dropped. codex R3 BLOCKING: once any opener
    was present, the tool-call branch suppressed all plain text — losing the
    leading prose when the model emitted content + a complete call in one
    chunk.

    codex R5 BLOCKING: the pre-opener content AND the tool-call deltas are now
    emitted in the SAME return (the postprocessor's mixed-content contract —
    ``content`` key alongside ``tool_calls`` — splits them into a leading
    content event then the tool events). Nothing is deferred to a later
    invocation that might never happen on the FINAL delta.
    """
    parser = HyV3ToolParser()
    parser.reset()
    step = _stepper(parser)
    m1 = step(
        "Let me look that up. "
        "<tool_call:opensource>search<tool_sep:opensource>"
        '{"q": "weather"}<end_of_tool_call:opensource>'
    )
    # ONE result carries BOTH the pre-opener content and the complete call.
    assert m1 is not None
    assert m1.get("content") == "Let me look that up. "
    assert "tool_calls" in m1
    name = ""
    args = ""
    for tc in m1["tool_calls"]:
        fn = tc.get("function", {})
        if fn.get("name"):
            name = fn["name"]
        if fn.get("arguments"):
            args += fn["arguments"]
    assert name == "search"
    assert json.loads(args) == {"q": "weather"}


def test_streaming_content_before_opener_char_by_char_not_dropped():
    """The same pre-opener content, delivered char-by-char, still surfaces
    exactly once and the call still parses — the incremental content path and
    the same-delta flush must agree (no double-emit, no drop)."""
    parser = HyV3ToolParser()
    wire = (
        "Sure! "
        "<tool_call:opensource>search<tool_sep:opensource>"
        '{"q": "x"}<end_of_tool_call:opensource>'
    )
    tool_acc, content = _collect_stream(parser, list(wire))
    assert content == "Sure! "
    assert tool_acc[0]["name"] == "search"
    assert json.loads(tool_acc[0]["args"]) == {"q": "x"}


# ---------------------------------------------------------------------------
# codex R10 regressions: sep-less first call in a multi-call delta + the
# text-format pending predicate
# ---------------------------------------------------------------------------
def test_streaming_sepless_first_call_does_not_swallow_second_call():
    """codex R10 BLOCKING #1: a SEP-LESS first call (XML-pair body, no
    ``<tool_sep>``) followed by a normal call in the SAME delta MUST NOT swallow
    the second opener.

    ``_find_call_close_in_body`` used to locate the FIRST ``<tool_sep>`` in the
    whole segment — which for a sep-less first call is the SECOND call's
    separator — and then searched past the second call's JSON for the close.
    That advanced the span cursor past everything, so ``_opener_positions``
    returned only ``[0]`` and the second call vanished; the streaming FSM also
    stalled because the sep-less block never delimited a name. The fix bounds
    the sep to before the first ``<end_of_tool_call>`` (a later sep belongs to
    the next call) and drains a sep-less CLOSED call via the shared body parser.
    """
    parser = HyV3ToolParser()
    wire = (
        "<tool_call:opensource>foo"
        "<arg_key:opensource>k</arg_key:opensource>"
        "<arg_value:opensource>v</arg_value:opensource>"
        "<end_of_tool_call:opensource>"
        '<tool_call:opensource>bar<tool_sep:opensource>{"b": 2}'
        "<end_of_tool_call:opensource>"
    )
    # Single delta carrying both calls — the exact codex scenario.
    tool_acc, content = _collect_stream(parser, [wire])
    assert sorted(tool_acc.keys()) == [0, 1], "second call was swallowed"
    assert tool_acc[0]["name"] == "foo"
    assert json.loads(tool_acc[0]["args"]) == {"k": "v"}
    assert tool_acc[1]["name"] == "bar"
    assert json.loads(tool_acc[1]["args"]) == {"b": 2}
    assert content == ""


def test_streaming_sepless_first_call_char_by_char_both_emit():
    """The same sep-less-first / normal-second pair delivered char-by-char must
    still yield both calls with correct args (the incremental path and the
    single-delta drain must agree)."""
    parser = HyV3ToolParser()
    wire = (
        "<tool_call:opensource>foo"
        "<arg_key:opensource>k</arg_key:opensource>"
        "<arg_value:opensource>v</arg_value:opensource>"
        "<end_of_tool_call:opensource>"
        '<tool_call:opensource>bar<tool_sep:opensource>{"b": 2}'
        "<end_of_tool_call:opensource>"
    )
    tool_acc, content = _collect_stream(parser, list(wire))
    assert sorted(tool_acc.keys()) == [0, 1]
    assert tool_acc[0]["name"] == "foo"
    assert json.loads(tool_acc[0]["args"]) == {"k": "v"}
    assert tool_acc[1]["name"] == "bar"
    assert json.loads(tool_acc[1]["args"]) == {"b": 2}
    assert content == ""


def test_streaming_sepless_xml_arg_value_containing_literal_end_token():
    """A SEP-LESS XML-pair body whose ``<arg_value>`` carries the literal
    ``<end_of_tool_call>`` string as free-form text MUST stream to the full
    value, not truncate at the interior literal (codex R16 — the streaming
    sep-less path, the THIRD occurrence of the class R15 fixed for the sep-full
    streaming ``_find_call_close`` and non-streaming ``_find_call_close_in_body``
    paths). The premature-close hazard is char-by-char: the interior literal
    end-token completes while its ``<arg_value>`` is still open (no
    ``</arg_value>`` yet), so a plain ``in`` gate would fire the sep-less drain
    early and parse the still-open body to ``{}``. The FSM must keep buffering
    until a close lands OUTSIDE every ``<arg_value>…</arg_value>`` span."""
    parser = HyV3ToolParser()
    end = parser.tool_call_end_token
    oc = parser.tool_call_start_token
    ak, ake = parser.arg_key_start_token, parser.arg_key_end_token
    av, ave = parser.arg_value_start_token, parser.arg_value_end_token
    # No ``<tool_sep>`` — the sep-less XML-pair body. The value carries the
    # literal wire end-token before the REAL trailing close.
    wire = f"{oc}doit{ak}m{ake}{av}has {end} literal{ave}{end}"
    expected = {"m": f"has {end} literal"}
    tool_acc, content = _collect_stream(parser, list(wire))
    assert sorted(tool_acc.keys()) == [0]
    assert tool_acc[0]["name"] == "doit"
    assert json.loads(tool_acc[0]["args"]) == expected
    # The literal end-token substring survives in the streamed value.
    assert end in json.loads(tool_acc[0]["args"])["m"]
    assert content == ""


def test_streaming_bare_name_sepless_call_emits_empty_args():
    """A bare-name sep-less call (``<tool_call>ping<end>`` — no separator, no
    args) MUST emit a call with ``{}`` args, not stall the FSM."""
    parser = HyV3ToolParser()
    tool_acc, content = _collect_stream(
        parser,
        ["<tool_call:opensource>ping<end_of_tool_call:opensource>"],
    )
    assert sorted(tool_acc.keys()) == [0]
    assert tool_acc[0]["name"] == "ping"
    assert json.loads(tool_acc[0]["args"]) == {}
    assert content == ""


def test_streaming_sepless_call_respects_request_allowlist():
    """A sep-less closed call whose name is OFF the request allowlist MUST NOT
    emit a tool_call (the drain path applies the same suppression as the JSON
    path)."""
    parser = HyV3ToolParser()
    request = {"tools": [{"type": "function", "function": {"name": "allowed"}}]}
    tool_acc, _content = _collect_stream(
        parser,
        [
            "<tool_call:opensource>forbidden"
            "<arg_key:opensource>k</arg_key:opensource>"
            "<arg_value:opensource>v</arg_value:opensource>"
            "<end_of_tool_call:opensource>"
        ],
        request=request,
    )
    assert tool_acc == {}, "off-list sep-less call must be suppressed"


def test_has_pending_tool_call_text_format_is_not_pending():
    """codex R10 BLOCKING #2: a COMPLETE ``[Calling tool="X" k="v"]``
    text-format message MUST NOT report pending. It is a self-delimited call
    with no trailing close delimiter to wait for, and it is finalized via the
    non-streaming recovery path (gated on the ``[Calling`` marker, not on this
    predicate). Reporting it pending made streaming shutdown treat a finished
    message as perpetually in-flight."""
    parser = HyV3ToolParser()
    assert (
        parser.has_pending_tool_call('[Calling tool="get_weather" city="SF"]') is False
    )
    # A native opener with no close IS still pending.
    assert parser.has_pending_tool_call("<tool_call:opensource>fn") is True
    # A completed native call followed by a fresh unmatched opener is pending
    # (the LAST opener has no close).
    assert (
        parser.has_pending_tool_call(
            "<tool_call:opensource>a<tool_sep:opensource>{}"
            "<end_of_tool_call:opensource><tool_call:opensource>b"
        )
        is True
    )
    # A completed native call with nothing after it is NOT pending.
    assert (
        parser.has_pending_tool_call(
            "<tool_call:opensource>a<tool_sep:opensource>{}"
            "<end_of_tool_call:opensource>"
        )
        is False
    )


def test_text_format_call_still_recovered_by_non_streaming_extract():
    """Dropping text-format from ``has_pending_tool_call`` MUST NOT break
    recovery — the non-streaming ``extract_tool_calls`` (which the postprocessor
    runs at finalize on any text containing ``[Calling``) still recovers the
    structured call."""
    parser = HyV3ToolParser()
    result = parser.extract_tool_calls(
        '[Calling tool="get_weather" city="SF"]', request=None
    )
    assert result.tools_called is True
    assert [c.get("name") for c in result.tool_calls] == ["get_weather"]


# ---------------------------------------------------------------------------
# codex R11 regressions: literal close-token inside an unterminated JSON
# string, garbled opener before a real call
# ---------------------------------------------------------------------------
def test_streaming_literal_end_token_in_unterminated_json_string_does_not_close():
    """codex R11 BLOCKING #1: a still-streaming JSON string value that contains
    the literal ``<end_of_tool_call>`` MUST NOT prematurely close the call.

    On the ``raw_decode``-failed path (JSON not yet complete) the parser used to
    accept the FIRST ``<end_of_tool_call>`` substring — but an unterminated
    string can legitimately carry that literal, so the call closed early and
    emitted ``{}``. The fix accepts only a close token OUTSIDE a JSON string.
    Delivered char-by-char, the args must reassemble to the full JSON with the
    literal preserved inside the string value, and no ``{}`` must be emitted
    early.
    """
    parser = HyV3ToolParser()
    wire = (
        "<tool_call:opensource>log<tool_sep:opensource>"
        '{"m": "contains <end_of_tool_call:opensource> inside"}'
        "<end_of_tool_call:opensource>"
    )
    tool_acc, content = _collect_stream(parser, list(wire))
    assert sorted(tool_acc.keys()) == [0]
    assert tool_acc[0]["name"] == "log"
    assert json.loads(tool_acc[0]["args"]) == {
        "m": "contains <end_of_tool_call:opensource> inside"
    }
    assert content == ""


def test_find_call_close_distinguishes_incomplete_from_malformed_complete():
    """Unit-level: the ``raw_decode``-failed branch must return -1 for an
    INCOMPLETE JSON whose only close-token occurrence is inside an unterminated
    string, but a NON-NEGATIVE offset for a MALFORMED-but-COMPLETE body
    (``{bad}<end>`` — codex R9) whose close sits outside any string."""
    parser = HyV3ToolParser()
    end = parser.tool_call_end_token
    # Incomplete: unterminated string containing the literal close token.
    assert parser._find_call_close(f'{{"m": "x {end} y') == -1
    # Malformed but complete: close token is outside a string.
    assert parser._find_call_close(f"{{bad}}{end}") >= 0


def test_streaming_garbled_opener_before_real_call_does_not_steal_separator():
    """codex R11 BLOCKING #2: a garbled/truncated opener (no ``<tool_sep>`` and
    no ``<end_of_tool_call>`` of its own) immediately before a valid call MUST
    NOT steal the real call's separator and fabricate a bogus name.

    ``_find_call_close_in_body`` used to scan for ``<tool_sep>`` across the whole
    remaining segment, so the garbled first opener consumed the real call's
    separator — producing a single call named ``gar<tool_call>realtool``. The
    fix bounds each call's body to before the next opener and skips a
    close-less/sep-less residue opener, resuming at the later opener.
    """
    parser = HyV3ToolParser()
    wire = (
        "<tool_call:opensource>gar"  # garbled: no sep, no close
        '<tool_call:opensource>realtool<tool_sep:opensource>{"x": 1}'
        "<end_of_tool_call:opensource>"
    )
    tool_acc, _content = _collect_stream(parser, [wire])
    # Exactly the real call surfaces (at its own index); no fabricated name.
    names = {v["name"] for v in tool_acc.values() if v["name"]}
    assert names == {"realtool"}, f"unexpected names: {names}"
    real = next(v for v in tool_acc.values() if v["name"] == "realtool")
    assert json.loads(real["args"]) == {"x": 1}
    # codex R14 BLOCKING: the CLIENT-VISIBLE emitted index for the first (and
    # only) real call MUST be 0, not 1 — the skipped garbled residue opener
    # consumes a PHYSICAL opener slot but must NOT consume a client-visible
    # index. A first real call emitted at index 1 leaves a null hole at 0 in
    # the OpenAI-SDK reconstruction. ``_collect_stream`` keys ``tool_acc`` by
    # the emitted index, so the real call must live at key 0.
    real_indices = [k for k, v in tool_acc.items() if v["name"] == "realtool"]
    assert real_indices == [0], f"client-visible index for realtool: {real_indices}"


def test_streaming_two_real_calls_emit_dense_client_indices():
    """codex R14: two genuine calls must still emit client-visible indices 0
    then 1 in order — the client-index decoupling must not disturb the normal
    multi-call sequence."""
    parser = HyV3ToolParser()
    wire = (
        '<tool_call:opensource>a<tool_sep:opensource>{"x": 1}'
        "<end_of_tool_call:opensource>"
        '<tool_call:opensource>b<tool_sep:opensource>{"y": 2}'
        "<end_of_tool_call:opensource>"
    )
    tool_acc, _content = _collect_stream(parser, [wire])
    assert sorted(tool_acc.keys()) == [0, 1]
    assert tool_acc[0]["name"] == "a"
    assert json.loads(tool_acc[0]["args"]) == {"x": 1}
    assert tool_acc[1]["name"] == "b"
    assert json.loads(tool_acc[1]["args"]) == {"y": 2}


def test_opener_positions_skips_garbled_opener_keeps_real_call():
    """Unit-level for R11 #2: ``_opener_positions`` records BOTH openers (the
    garbled residue and the real call) as distinct spans rather than merging
    them — the garbled opener's body no longer swallows the real opener."""
    parser = HyV3ToolParser()
    oc = parser.tool_call_start_token
    sep = parser.tool_sep_token
    end = parser.tool_call_end_token
    text = f'{oc}gar{oc}realtool{sep}{{"x": 1}}{end}'
    positions = parser._opener_positions(text)
    assert len(positions) == 2
    assert positions[0] == 0
    assert positions[1] == len(f"{oc}gar")


# ---------------------------------------------------------------------------
# codex R12 regressions: close-token after a still-OPEN object, garbled opener
# on the NON-STREAMING path
# ---------------------------------------------------------------------------
def test_find_call_close_incomplete_open_object_is_still_streaming():
    """codex R12 BLOCKING #1: a close token that is structurally OUTSIDE a JSON
    string but reached while the object is still OPEN (braces not balanced back
    to 0 — the value has not arrived) MUST NOT close the call. Only a close
    after the object's braces balance is real.
    """
    parser = HyV3ToolParser()
    end = parser.tool_call_end_token
    # Value after ``"a":`` not arrived — object still open (depth 1).
    assert parser._find_call_close(f'{{"a": {end}') == -1
    # Nested object still open (depth 2).
    assert parser._find_call_close(f'{{"a": {{"b": {end}') == -1
    # Malformed but brace-balanced (depth back to 0) — closes.
    assert parser._find_call_close(f"{{bad}}{end}") >= 0
    assert parser._find_call_close(f'{{"a": {{bad}}}}{end}') >= 0


def test_find_call_close_resyncs_past_noise_end_token():
    """codex R14 BLOCKING: a mid-stream noise ``<end_of_tool_call>`` that lands
    outside a string while an object is still OPEN (``{"a": <end>``) must NOT
    poison brace-depth forever. The scan RESYNCHRONIZES past the noise token and
    still accepts the REAL later ``{...}<end>`` close, and the serialized args
    must be the real object — not ``{}``."""
    parser = HyV3ToolParser()
    end = parser.tool_call_end_token
    # Noise open-object close-token, then the real object + real close.
    body = f'{{"a": {end}{{"a": 42}}{end}'
    off = parser._find_call_close(body)
    assert off >= 0, "resync failed: real close never accepted"
    # The accepted close is the LAST end-token (the real one).
    assert body[off:] == end
    # The args serialize to the real object, not {} — the noise prefix is
    # resynchronized away.
    args_tail = body[:off]
    assert json.loads(parser._final_args_json(args_tail)) == {"a": 42}
    # A genuinely still-open object with no later balanced object stays -1.
    assert parser._find_call_close(f'{{"a": {end}') == -1


def test_streaming_close_token_after_open_object_waits_for_completion():
    """The same open-object case delivered as a stream: a delta ending in
    ``{"a": <end_of_tool_call>`` (value not yet present) must NOT emit a
    premature ``{}``; the args only emit once the REAL object closes AND must
    then carry the real object (codex R14: a poisoned brace-depth used to reject
    the real close forever, so the args never emitted — this test was
    false-green because it only checked the name, not the args)."""
    parser = HyV3ToolParser()
    parser.reset()
    step = _stepper(parser)
    # Opener + name + sep + an open object whose value slot is empty, followed
    # by a stray close token — still streaming.
    m1 = step(
        "<tool_call:opensource>calc<tool_sep:opensource>"
        '{"a": <end_of_tool_call:opensource>'
    )
    # Whatever surfaces must NOT be a completed args delta with ``{}``.
    if m1 and "tool_calls" in m1:
        for tc in m1["tool_calls"]:
            args = tc.get("function", {}).get("arguments", "")
            assert args in ("", None), f"premature args emitted: {args!r}"
    # Now the real value + real close arrive.
    step('{"a": 42}<end_of_tool_call:opensource>')
    # Reassemble across the whole stream.
    parser2 = HyV3ToolParser()
    tool_acc, _content = _collect_stream(
        parser2,
        [
            "<tool_call:opensource>calc<tool_sep:opensource>",
            '{"a": <end_of_tool_call:opensource>',  # noise mid-stream
            '{"a": 42}<end_of_tool_call:opensource>',
        ],
    )
    # The parser holds until a real (brace-balanced) close; final args parse.
    assert 0 in tool_acc
    assert tool_acc[0]["name"] == "calc"
    # The noise prefix must be resynchronized away and the REAL args emitted —
    # not left empty (the false-green this test used to hide, codex R14).
    assert json.loads(tool_acc[0]["args"]) == {"a": 42}


def test_non_streaming_garbled_opener_before_real_call_recovers_real_call():
    """codex R12 BLOCKING #2: the NON-STREAMING ``_next_block`` had the same
    garbled-opener bug as ``_opener_positions`` — a truncated opener before a
    valid call stole the real call's separator and fabricated a bogus name
    (``gar<tool_call>realtool``). The fix bounds the body at a pre-separator
    opener and resumes at the later opener, so the real call is recovered."""
    parser = HyV3ToolParser()
    oc = parser.tool_call_start_token
    sep = parser.tool_sep_token
    end = parser.tool_call_end_token
    text = f'{oc}gar{oc}realtool{sep}{{"x": 1}}{end}'
    result = parser.extract_tool_calls(text, request=None)
    assert result.tools_called is True
    assert [c.get("name") for c in result.tool_calls] == ["realtool"]
    assert json.loads(result.tool_calls[0]["arguments"]) == {"x": 1}


def test_non_streaming_garbled_opener_then_two_real_calls():
    """A garbled residue opener followed by TWO real calls must recover BOTH,
    not just the first — ``_next_block`` resumes cleanly after skipping the
    residue."""
    parser = HyV3ToolParser()
    oc = parser.tool_call_start_token
    sep = parser.tool_sep_token
    end = parser.tool_call_end_token
    text = f'{oc}gar{oc}a{sep}{{"x": 1}}{end}{oc}b{sep}{{"y": 2}}{end}'
    result = parser.extract_tool_calls(text, request=None)
    assert [c.get("name") for c in result.tool_calls] == ["a", "b"]


def test_non_streaming_literal_opener_in_json_string_stays_one_call():
    """Guard the R6 invariant on the non-streaming path after the R12 bounding
    change: a literal ``<tool_call>`` INSIDE a JSON string value (after the
    separator) must still be opaque — exactly one call, args intact."""
    parser = HyV3ToolParser()
    oc = parser.tool_call_start_token
    sep = parser.tool_sep_token
    end = parser.tool_call_end_token
    text = f'{oc}log{sep}{{"msg": "prefix {oc} suffix", "n": 1}}{end}'
    result = parser.extract_tool_calls(text, request=None)
    assert [c.get("name") for c in result.tool_calls] == ["log"]
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "msg": f"prefix {oc} suffix",
        "n": 1,
    }
