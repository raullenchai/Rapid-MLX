# SPDX-License-Identifier: Apache-2.0
"""Stream ↔ non-stream tool-call extraction parity invariant.

Structural meta-fix for the bug class surfaced by #425 (jpcarranza94)
and #429-meta:

  * The non-stream path at ``service/helpers.py::_parse_tool_calls_with_parser``
    runs the configured ``--tool-call-parser`` first, and falls back to the
    multi-format scanner ``api/tool_calling.parse_tool_calls`` when the
    configured parser returns ``tools_called=False``.

  * The streaming path at ``service/postprocessor.py::StreamingPostProcessor.finalize``
    historically ran ONLY the configured parser. PR #426 added the matching
    fallback so the two paths can't drift on wire-format mismatch.

This file enforces the invariant going forward: for every canonical wire
format the project supports, the streaming ``finalize()`` MUST emit the same
``tool_calls`` the non-stream extractor returns. New parsers / new formats
must add a fixture entry OR be documented in ``_PARITY_COVERAGE_EXEMPT`` —
``test_tool_parser_parity_coverage`` is the forcing function that fails CI
if a registered parser ships without coverage.

Pattern mirrors ``tests/test_batched_engine_output_router.py::
test_router_allowlist_tool_call_routing_declared`` (PR #429 meta-fix for
router-allowlist coverage): declare every registered family, force explicit
categorization, catch the asymmetry class structurally rather than by hope.
"""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock

import pytest

from rapid_mlx.api.tool_calling import parse_tool_calls
from rapid_mlx.api.utils import strip_special_tokens
from rapid_mlx.service.postprocessor import StreamingPostProcessor
from rapid_mlx.tool_parsers import ToolParserManager


def _make_cfg_for_parser(parser_name: str) -> MagicMock:
    """Build a minimal ServerConfig pointing at ``parser_name``.

    Mirrors the helper in ``tests/test_postprocessor.py`` — we only need
    the attributes ``StreamingPostProcessor.__init__`` reads.
    """
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = parser_name
    cfg.tool_parser_instance = None
    return cfg


def _normalize(tool_calls) -> list[tuple[str, dict]]:
    """Normalize tool-call payload into (name, args_dict) tuples.

    Both paths return slightly different shapes (the non-stream path returns
    a list of ``ToolCall`` pydantic models; the streaming path returns a list
    of dicts inside ``StreamEvent.tool_calls``). Strip both down to
    (name, args) tuples — IDs are non-deterministic UUIDs, drop them.
    Arguments may be a JSON string or already a dict; normalize to dict so
    equivalent payloads compare equal regardless of serialization choices.
    """

    def _args_to_dict(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {"__raw__": v}
        return {"__unknown__": repr(v)}

    out = []
    for tc in tool_calls or []:
        # ToolCall pydantic model (non-stream path)
        if hasattr(tc, "function"):
            out.append((tc.function.name, _args_to_dict(tc.function.arguments)))
        # dict (streaming path)
        elif isinstance(tc, dict) and "function" in tc:
            f = tc["function"]
            out.append((f["name"], _args_to_dict(f["arguments"])))
    return out


def _extract_nonstream(parser_name: str, text: str) -> list:
    """Reproduce the contract of ``service/helpers.py::_parse_tool_calls_with_parser``
    without depending on the global ``get_config()``.

    Contract: run the configured parser; on ``tools_called=False``, fall
    through to the multi-format ``parse_tool_calls`` scanner.
    """
    parser_cls = ToolParserManager.get_tool_parser(parser_name)
    parser = parser_cls(None)  # tokenizer not required for tested formats
    parser.reset()
    result = parser.extract_tool_calls(text, request=None)
    if result.tools_called:
        # Mirror helpers.py: wrap configured-parser results as ToolCall-like
        # dicts so the comparison normalizer sees the same shape as the
        # streaming finalize path.
        return [
            {
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }
            }
            for tc in result.tool_calls
        ]
    _, tcs = parse_tool_calls(text, None)
    return tcs or []


def _extract_stream(parser_name: str, text: str) -> list:
    """Run ``StreamingPostProcessor.finalize`` with ``text`` pre-loaded.

    Simulates the end-of-stream state where the accumulated tool-text is
    ready and the only question is what ``finalize()`` emits. This is the
    surface PR #426 fixed.
    """
    cfg = _make_cfg_for_parser(parser_name)
    # This helper intentionally exercises a configured parser without a full
    # request object, so declare the capability explicitly.
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    pp.reset()
    pp.tool_accumulated_text = text
    events = pp.finalize()
    for ev in events:
        if ev.type == "tool_call":
            return ev.tool_calls or []
    return []


# --------------------------------------------------------------------------
# Wire-format fixtures
#
# Each entry: (parser_name, wire_label, text, expected_calls).
# ``expected_calls`` is the canonical (name, args_dict) list both paths
# MUST recover. Coverage is intentionally minimal — one canonical format
# per parser family is enough for the parity invariant; expand when a
# field bug surfaces a new variant (PR #426 → qwen3_xml + xml_body is the
# concrete example).
# --------------------------------------------------------------------------
PARITY_FIXTURES: list = [
    (
        "raon_1",
        "tool_call_xml_body",
        "<tool_call>\n<function=run>\n<parameter=text>\nhello\nworld\n</parameter>\n</function>\n</tool_call>",
        [("run", {"text": "hello\nworld"})],
    ),
    (
        "pythonic",
        "pythonic_bracket",
        '[get_current_weather(location="Paris", unit="celsius")]',
        [("get_current_weather", {"location": "Paris", "unit": "celsius"})],
    ),
    # MiniCPM5 — documented native XML without a <tool_call> wrapper.
    (
        "minicpm",
        "minicpm_native",
        '<function name="read_file"><param name="path">/etc/hostname</param></function>',
        [("read_file", {"path": "/etc/hostname"})],
    ),
    # Muse Glimmer — ATEM function-calls block (Anthropic-style XML on
    # recipient-routed channels). Value is a bare string per the chat
    # template ("String and scalar parameters should be specified as is").
    (
        "muse",
        "muse_atem",
        (
            '<atem:function_calls>\n<atem:invoke name="read_file">\n'
            '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        ),
        [("read_file", {"path": "/etc/hostname"})],
    ),
    # hermes — JSON body inside <tool_call>...</tool_call>
    (
        "hermes",
        "json_body",
        '<tool_call>{"name":"read_file","arguments":{"path":"/etc/hostname"}}</tool_call>',
        [("read_file", {"path": "/etc/hostname"})],
    ),
    # hermes — XML body via Nemotron pattern (Qwen3.6 wire format)
    (
        "hermes",
        "xml_body",
        (
            "<tool_call>\n<function=read_file>\n"
            "<parameter=path>\n/etc/hostname\n</parameter>\n"
            "</function>\n</tool_call>"
        ),
        [("read_file", {"path": "/etc/hostname"})],
    ),
    # qwen3_xml — the #425 bug fixed by #426 (847ea15). Parser name implies
    # XML body; registered to QwenToolParser which expects JSON body. The
    # non-stream path recovered via parse_tool_calls fallback at
    # helpers.py:604; the streaming finalize path now does too (#426).
    # Asserting parity here ensures the asymmetry doesn't regress.
    (
        "qwen3_xml",
        "xml_body",
        (
            "<tool_call>\n<function=read_file>\n"
            "<parameter=path>\n/etc/hostname\n</parameter>\n"
            "</function>\n</tool_call>"
        ),
        [("read_file", {"path": "/etc/hostname"})],
    ),
    # qwen3_coder_xml — canonical Qwen3-Coder XML body. Streaming-side
    # incremental string emission (#479) must still concatenate to the same
    # final JSON the non-stream path returns.
    (
        "qwen3_coder_xml",
        "xml_body",
        (
            "<tool_call>\n<function=read_file>\n"
            "<parameter=path>\n/etc/hostname\n</parameter>\n"
            "</function>\n</tool_call>"
        ),
        [("read_file", {"path": "/etc/hostname"})],
    ),
    # hy_v3 — canonical Tencent Hunyuan 3 wire format. Suffix-tolerant
    # (``:opensource`` optional). Streaming path buffers until the close
    # tag arrives on a delta; non-streaming path uses the same
    # ``extract_tool_calls`` regex walk. Both must recover the tool call
    # identically.
    (
        "hy_v3",
        "hy3_native",
        (
            "<tool_call:opensource>read_file<tool_sep:opensource>"
            '{"path":"/etc/hostname"}<end_of_tool_call:opensource>'
        ),
        [("read_file", {"path": "/etc/hostname"})],
    ),
    # ui_tars — Computer-Use action line. Stream / non-stream paths both
    # MUST recover the canonical (name="computer", args={action, point})
    # tuple. The streaming finalize path runs the same _iter_actions
    # scanner as the non-stream path, so coverage is the same shape —
    # this fixture guards against a future regression where they diverge.
    (
        "ui_tars",
        "ui_tars_action",
        "Thought: Click search.\nAction: click(point='<point>200 300</point>')",
        [("computer", {"action": "click", "point": [200, 300]})],
    ),
    # lfm — pythonic bracket calls (Liquid LFM2.x). Parity here depends on
    # finalize()'s plausible-markup pre-check recognizing ``[name(`` —
    # ``[f(x="y")]`` contains none of the ``<`` / ``{`` / ``[Calling``
    # markers the pre-check originally looked for, so the streaming path
    # silently skipped extraction while non-stream succeeded.
    (
        "lfm",
        "pythonic_bracket",
        'Checking. [get_current_weather(location="Paris", unit="celsius")]',
        [("get_current_weather", {"location": "Paris", "unit": "celsius"})],
    ),
    (
        "deepseek_v4_flash",
        "deepseek_v4_dsml",
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="read_file">\n'
        '<｜DSML｜parameter name="path" string="true">sample.txt</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        [("read_file", {"path": "sample.txt"})],
    ),
    # DeepSeek V4 0731 — DSML invoke/parameter wire format.
    (
        "deepseek_v4_0731",
        "deepseek_v4_dsml",
        (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="read_file">\n'
            '<｜DSML｜parameter name="path" string="true">'
            "/etc/hostname</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        ),
        [("read_file", {"path": "/etc/hostname"})],
    ),
]


# Parsers whose canonical wire format isn't exercised here yet — explicit
# exemption with reason is required so coverage gaps are visible. Each
# entry should reference an issue or note WHY no fixture exists. Adding
# a fixture removes the entry; new parsers added without a fixture or
# exemption fail ``test_tool_parser_parity_coverage``.
_PARITY_COVERAGE_EXEMPT: dict[str, str] = {
    # Multi-channel control-token formats — fixture authoring needs a
    # tokenizer fixture (vocab IDs) rather than raw text, which is a
    # different test surface (see tests/test_batched_engine_output_router.py
    # for the gemma4 token-level coverage already exercised there).
    "gemma4": "tokenizer-level test in test_batched_engine_output_router.py",
    "gemma_4": "alias of gemma4",
    "harmony": "tokenizer-level test in test_batched_engine_output_router.py",
    "gpt-oss": "alias of harmony",
    "gpt_oss": "alias of seed_oss (separate from harmony — needs own fixture)",
    "seed_oss": "TODO: add seed-oss wire-format fixture (no canonical example yet)",
    "seed": "alias of seed_oss",
    # Family-specific control-token formats that need dedicated fixtures.
    # Listed here so coverage gaps are visible; each TODO is a follow-up
    # cleanup ticket, not a blocker for the parity invariant.
    "glm47": "TODO: add GLM-4.7/5 wire-format fixture",
    "glm4": "alias of glm47",
    "granite": "TODO: add Granite4 wire-format fixture",
    "granite3": "alias of granite",
    "llama": "TODO: add Llama 3/4 wire-format fixture",
    "llama3": "alias of llama",
    "llama4": "alias of llama",
    "minimax": "TODO: add MiniMax M2/M2.5 wire-format fixture",
    "minimax_m2": "alias of minimax",
    "mistral": "TODO: add Mistral [TOOL_CALLS] wire-format fixture",
    "nemotron": "TODO: add Nemotron wire-format fixture",
    "nemotron3": "alias of nemotron",
    "kimi": "TODO: add Kimi K2 wire-format fixture",
    "kimi_k2": "alias of kimi",
    "moonshot": "alias of kimi",
    "deepseek": "TODO: add DeepSeek V2 / R1-distill wire-format fixture",
    "deepseek_r1": "alias of deepseek",
    "deepseek_v3": "TODO: add DeepSeek V3 fenced-JSON wire-format fixture (R12-5)",
    "deepseek_r1_0528": "alias of deepseek_v3 (R12-5)",
    "deepseek_v31": "TODO: add DeepSeek V3.1 wire-format fixture",
    "functionary": "TODO: add Functionary wire-format fixture",
    "meetkai": "alias of functionary",
    "xlam": "TODO: add xLAM wire-format fixture",
    "qwen": "JSON-body Qwen variant — covered by hermes json_body fixture (same shape)",
    "qwen3": "alias of qwen",
    "qwen3_coder": "alias of hermes",
    "nous": "alias of hermes",
    "hy3": "alias of hy_v3",
    "ui-tars": "alias of ui_tars (kebab-case spelling)",
    "uitars": "alias of ui_tars (no-separator spelling)",
    "liquid": "alias of lfm",
    "k2_horizon": (
        "IFM extraction requires the request's declared tool schema; dedicated "
        "character-boundary stream/non-stream contracts live in "
        "test_k2_horizon_parsers.py"
    ),
    "auto": "router, not a wire-format parser",
    "generic": "router, not a wire-format parser",
}


@pytest.mark.parametrize("parser_name,wire_label,text,expected", PARITY_FIXTURES)
def test_stream_nonstream_tool_call_parity(parser_name, wire_label, text, expected):
    """Streaming ``finalize()`` MUST extract the same tool calls as the
    non-streaming ``_parse_tool_calls_with_parser`` for every supported
    (parser, wire-format) pair.

    Bug class: when the configured ``--tool-call-parser`` is bound to one
    wire format but the model emits a different one, the non-stream path
    falls back to ``parse_tool_calls`` (multi-format scanner) at
    ``service/helpers.py::_parse_tool_calls_with_parser`` line 604. The
    streaming path's ``finalize()`` historically ran ONLY the configured
    parser — so the same prompt would return a structured ``tool_calls``
    array non-streaming but emit zero deltas streaming. PR #426
    (jpcarranza94, fixes #425) closes this. This parametrized test enforces
    that the two paths can't drift again.
    """
    ns_calls = _extract_nonstream(parser_name, text)
    stream_calls = _extract_stream(parser_name, text)

    ns_norm = _normalize(ns_calls)
    stream_norm = _normalize(stream_calls)

    assert ns_norm == expected, (
        f"{parser_name}/{wire_label}: non-stream did not return expected "
        f"calls. expected={expected!r} got={ns_norm!r}"
    )
    assert stream_norm == ns_norm, (
        f"{parser_name}/{wire_label}: stream/non-stream parity broken. "
        f"non-stream={ns_norm!r} stream={stream_norm!r} — the streaming "
        f"finalize() path is missing a fallback the non-stream path has. "
        f"See service/helpers.py:604 and PR #426."
    )


def test_tool_parser_parity_coverage():
    """Every registered ``ToolParserManager`` parser must be either
    exercised by ``PARITY_FIXTURES`` or documented in
    ``_PARITY_COVERAGE_EXEMPT`` with a reason.

    Forcing function for the bug class surfaced by #425: any future
    parser added without coverage triggers this assertion, requiring
    the author to either add a fixture (preferred) or document why
    no canonical wire-format example exists (TODO entry, with a path
    to closing it later).
    """
    registered = {name for name in ToolParserManager.tool_parsers}

    # ``PARITY_FIXTURES`` may contain bare tuples OR ``pytest.param`` objects
    # (for xfail-marked cases). Normalize to the underlying tuple via the
    # ``ParameterSet.values`` attribute when present.
    covered: set[str] = set()
    _ParamCls = type(pytest.param(0))
    for entry in PARITY_FIXTURES:
        if isinstance(entry, _ParamCls):
            covered.add(entry.values[0])
        else:
            covered.add(entry[0])

    uncategorized = registered - covered - set(_PARITY_COVERAGE_EXEMPT.keys())
    assert not uncategorized, (
        f"Registered tool parsers with no parity coverage and no exemption: "
        f"{sorted(uncategorized)}. Add a fixture to PARITY_FIXTURES "
        f"(preferred — actually exercises the invariant) OR add an entry "
        f"to _PARITY_COVERAGE_EXEMPT with a reason explaining why no "
        f"canonical wire-format example is available yet. See PR #426 / #425 "
        f"for the bug class this gates."
    )


class _StreamedCallsError(AssertionError):
    """Complete calls were not fully emitted before EOS recovery."""


class _ContentParityError(AssertionError):
    """Distinguish content mismatches from tool-call assertion failures."""


_QWEN_CONTENT_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=_ContentParityError,
    reason="pre-existing Qwen3-coder streaming content framing mismatch, tracked as follow-up",
)


def _mark_qwen_content_mismatch(request, parity, affected=True):
    if parity == "content" and affected:
        request.node.add_marker(_QWEN_CONTENT_XFAIL)


def _stream_xml_deltas(
    parser_name,
    text,
    chunks,
    request,
    *,
    raw_arguments=False,
    allow_eos_recovery=False,
    check_content=False,
    require_streamed_calls=True,
    expected_content=None,
    expected_streamed_calls=None,
):
    parser = ToolParserManager.get_tool_parser(parser_name)(None)
    accumulated = ""
    deltas = []
    for chunk in chunks:
        previous = accumulated
        accumulated += chunk
        delta = parser.extract_tool_calls_streaming(
            previous, accumulated, chunk, request=request
        )
        if delta:
            for tc in delta.get("tool_calls", []):
                name = tc.get("function", {}).get("name")
                if name:
                    assert f"<function={name}>" in accumulated
            deltas.append(delta)
    if expected_streamed_calls is not None:
        # Check before either parser or postprocessor EOS recovery can supply
        # missing calls (or complete a partially streamed argument payload).
        try:
            streamed = _merge_xml_deltas(deltas, raw_arguments=raw_arguments)
        except (AssertionError, ValueError, KeyError) as exc:
            raise _StreamedCallsError("incomplete calls before finalize") from exc
        if streamed != expected_streamed_calls:
            raise _StreamedCallsError(
                f"calls before finalize: {streamed!r} != {expected_streamed_calls!r}"
            )
    finalize = getattr(parser, "finalize_legacy_raw_stream", None)
    if finalize:
        delta = finalize(text, request=request)
        if delta:
            deltas.append(delta)
    pp = StreamingPostProcessor(_make_cfg_for_parser(parser_name), request=request)
    pp.tool_parser = parser
    pp.tool_accumulated_text = text
    content = []
    for delta in deltas:
        # Mirror the endpoint's content pipeline, including mixed tool/content
        # deltas. Keep the sanitizer alive through finalize to drain held bytes.
        content.append(
            pp._sanitize_content(strip_special_tokens(delta.get("content") or ""))
        )
        if delta.get("tool_calls"):
            pp.tool_calls_detected = True
            pp._preserve_post_tool_content = bool(
                delta.get("preserve_post_tool_content")
            )
    if require_streamed_calls and not any(delta.get("tool_calls") for delta in deltas):
        assert allow_eos_recovery and (
            parser_name == "qwen3_coder_xml"
            and len(chunks) == 2
            and chunks[0] in ["<tool_call>"[:i] for i in range(2, 11)]
        ), chunks
        # Restrict endpoint EOS recovery to the explicitly allowed split
        # wrapper boundaries; every other chunking must emit streaming calls.
    for event in pp.finalize():
        if event.type in ("content", "finish"):
            content.append(event.content or "")
        if event.type == "tool_call":
            deltas.append({"tool_calls": event.tool_calls})
    result = ToolParserManager.get_tool_parser(parser_name)(None).extract_tool_calls(
        text, request=request
    )
    if expected_content is not None:
        assert "".join(content) == expected_content
    if check_content and "".join(content) != (result.content or ""):
        raise _ContentParityError(
            f"content parity: {chunks!r}: {''.join(content)!r} != {result.content!r}"
        )
    if require_streamed_calls:
        assert any(delta.get("tool_calls") for delta in deltas), chunks
    return _merge_xml_deltas(deltas, raw_arguments=raw_arguments)


def _merge_xml_deltas(deltas, *, raw_arguments=False):
    calls = {}
    name_emissions = []
    for delta in deltas:
        for tc in delta.get("tool_calls", []):
            index = tc["index"]
            function = tc.get("function", {})
            call = calls.setdefault(index, {"name": "", "arguments": ""})
            if function.get("name"):
                name_emissions.append(index)
            for field in ("name", "arguments"):
                call[field] += function.get(field, "")
    assert name_emissions == list(range(len(calls)))
    return [
        (
            calls[i]["name"],
            calls[i]["arguments"]
            if raw_arguments
            else json.loads(calls[i]["arguments"]),
        )
        for i in range(len(calls))
    ]


def _every_split(text):
    return [[text], list(text)] + [[text[:i], text[i:]] for i in range(1, len(text))]


def _xml_request(names):
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            key: {"type": "string"}
                            for key in ("filePath", "new_string", "path")
                        },
                    },
                },
            }
            for name in names
        ]
    }


@pytest.mark.parametrize("parser_name", ["qwen3_coder_xml", "hermes"])
@pytest.mark.parametrize("quoted", [True, False])
@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.parametrize(
    "last_name", ["read", "read|", "read_async", 'read"', "view </parameter"]
)
@pytest.mark.parametrize("parity", ["tool_calls", "content"])
def test_parallel_names_are_complete_wire_names_at_every_split(
    parser_name, last_name, quoted, wrapped, parity, request
):
    """Repeated names never share deltas, even with malformed model headers."""
    _mark_qwen_content_mismatch(
        request, parity, parser_name == "qwen3_coder_xml" and wrapped
    )
    from rapid_mlx.api.tool_logits import create_tool_logits_processor

    request = _xml_request(("write", "read", "view"))
    # These parsers have no decoding processor, including for subsequent calls.
    tokenizer = MagicMock()
    processor = create_tool_logits_processor(
        parser_name, tokenizer, tools=request["tools"]
    )
    assert processor is None
    tokenizer.encode.assert_not_called()
    names = ["write", "read", last_name]
    text = ""
    for index, name in enumerate(names):
        value = f"file{index}.txt"
        if quoted:
            value = json.dumps(value)
        body = f"<function={name}><parameter=filePath>{value}</parameter></function>"
        text += f"<tool_call>{body}</tool_call>" if wrapped else body
    expected = [
        (name, {"filePath": f"file{index}.txt"}) for index, name in enumerate(names)
    ]
    parser_cls = ToolParserManager.get_tool_parser(parser_name)
    result = parser_cls(None).extract_tool_calls(text, request=request)
    assert [
        (tc["name"], json.loads(tc["arguments"])) for tc in result.tool_calls
    ] == expected
    for chunks in _every_split(text):
        assert (
            _stream_xml_deltas(
                parser_name, text, chunks, request, check_content=parity == "content"
            )
            == expected
        )


@pytest.mark.parametrize("parser_name", ["hermes", "qwen3_coder_xml"])
@pytest.mark.parametrize("tail", ["", "\n</parameter>", "\n</parameter>\n</parameter>"])
@pytest.mark.parametrize("sibling", ["", "<parameter=path>nonce.txt</parameter>"])
@pytest.mark.parametrize("parity", ["tool_calls", "content"])
def test_duplicate_parameter_closers_streaming_parity(
    parser_name, tail, sibling, parity, request
):
    """A duplicate closer is markup, never file content (#5830)."""
    _mark_qwen_content_mismatch(request, parity, parser_name == "qwen3_coder_xml")
    text = (
        "<tool_call><function=edit>"
        "<parameter=new_string>\nhi-second\n</parameter>"
        + tail
        + sibling
        + "</function></tool_call>"
    )
    args = {"new_string": "hi-second"}
    if sibling:
        args["path"] = "nonce.txt"
    expected = [("edit", args)]
    assert _normalize(_extract_nonstream(parser_name, text)) == expected
    assert _normalize(_extract_stream(parser_name, text)) == expected
    request = _xml_request(("edit",))
    for chunks in _every_split(text):
        assert (
            _stream_xml_deltas(
                parser_name,
                text,
                chunks,
                request,
                allow_eos_recovery=True,
                check_content=parity == "content",
            )
            == expected
        )


@pytest.mark.parametrize("parser_name", ["hermes", "qwen3_coder_xml"])
@pytest.mark.parametrize(
    "value,expected_value",
    [
        (
            "\n<parameter=example>literal</parameter>\n",
            "<parameter=example>literal</parameter>",
        ),
        ("x\n</parameter>\n</parameter>", "x"),
        # Raw marker text is ambiguous: the value stays exactly as selected.
        (
            "\n<parameter=example>literal</parameter>\n</parameter>\n",
            "<parameter=example>literal</parameter>\n</parameter>",
        ),
    ],
    ids=["balanced-literal", "duplicate-closers", "literal-and-duplicate"],
)
@pytest.mark.parametrize("sibling", ["", "<parameter=path>nonce.txt</parameter>"])
@pytest.mark.parametrize("parity", ["tool_calls", "content"])
def test_balanced_literal_parameter_markup_at_every_byte_split(
    parser_name, value, expected_value, sibling, parity, request
):
    _mark_qwen_content_mismatch(request, parity, parser_name == "qwen3_coder_xml")
    text = (
        "<tool_call><function=edit>"
        f"<parameter=new_string>{value}</parameter>{sibling}"
        "</function></tool_call>"
    )
    arguments = {"new_string": expected_value}
    if sibling:
        arguments["path"] = "nonce.txt"
    expected = [("edit", arguments)]
    request = _xml_request(("edit",))
    parser = ToolParserManager.get_tool_parser(parser_name)(None)
    result = parser.extract_tool_calls(text, request=request)
    assert [
        (call["name"], json.loads(call["arguments"])) for call in result.tool_calls
    ] == expected
    assert _normalize(_extract_stream(parser_name, text)) == expected
    # ASCII text makes every character split a byte boundary too.
    for chunks in _every_split(text):
        assert (
            _stream_xml_deltas(
                parser_name,
                text,
                chunks,
                request,
                allow_eos_recovery=True,
                check_content=parity == "content",
            )
            == expected
        ), chunks


@pytest.mark.parametrize("parser_name", ["qwen3_coder_xml", "hermes"])
@pytest.mark.parametrize("names", [("read", "read", "read"), ("write", "read", "read")])
@pytest.mark.parametrize("quoted", [True, False])
@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.parametrize("prefix", ["", "Working.\n"])
@pytest.mark.parametrize("parity", ["tool_calls", "content"])
def test_three_parallel_calls_keep_arguments_at_every_byte_split(
    parser_name, names, quoted, wrapped, prefix, parity, request
):
    _mark_qwen_content_mismatch(
        request, parity, parser_name == "qwen3_coder_xml" and wrapped
    )
    if parser_name == "hermes" and prefix and parity == "content":
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=_ContentParityError,
                reason="pre-existing Hermes streaming prefix/whitespace mismatch, tracked as follow-up",
            )
        )
    request = _xml_request(set(names))
    text = ""
    for index, name in enumerate(names):
        value = f"file{index}.txt"
        if quoted:
            value = json.dumps(value)
        body = f"<function={name}><parameter=filePath>{value}</parameter></function>"
        text += f"<tool_call>{body}</tool_call>" if wrapped else body
    text = prefix + text
    parser_cls = ToolParserManager.get_tool_parser(parser_name)
    result = parser_cls(None).extract_tool_calls(text, request=request)
    expected = [(call["name"], call["arguments"]) for call in result.tool_calls]
    assert [(name, json.loads(args)) for name, args in expected] == [
        (name, {"filePath": f"file{index}.txt"}) for index, name in enumerate(names)
    ]
    # ASCII wire text makes every character split a byte boundary as well.
    header_end = text.index(">", text.index("<function=")) + 1
    splits = _every_split(text) + [
        [text[:prefix_end], text[prefix_end:i], text[i:]]
        for prefix_end in (1, text.index(">") + 1, header_end)
        for i in range(prefix_end + 1, len(text))
    ]
    if wrapped:
        second_header = text.index("<function=", header_end) + 3
        wrapper_split = len(prefix) + len("<tool_")
        splits += [
            [
                text[:wrapper_split],
                text[wrapper_split:second_header],
                text[second_header:i],
                text[i:],
            ]
            for i in range(second_header + 1, len(text))
        ]
    for chunks in splits:
        assert (
            _stream_xml_deltas(
                parser_name,
                text,
                chunks,
                request,
                raw_arguments=True,
                check_content=parity == "content",
            )
            == expected
        ), chunks


@pytest.mark.parametrize("quoted", [True, False])
@pytest.mark.parametrize("tail", ["</parameter>", "\n</parameter>\n</parameter>"])
def test_literal_marker_and_duplicate_closers_together_at_every_split(quoted, tail):
    """Literal data and duplicate framing must coexist without truncation."""
    value = 'before </parameter> after "quoted" \\ end'
    wire_value = json.dumps(value) if quoted else value
    text = (
        "<tool_call><function=edit><parameter=new_string>"
        + wire_value
        + "</parameter>"
        + tail
        + '<parameter=path>"ok.txt"</parameter></function></tool_call>'
    )
    request = _xml_request(("edit",))
    if quoted:
        expected = [("edit", {"new_string": value, "path": "ok.txt"})]
    else:
        # Raw marker text is ambiguous; streaming must match the complete parse.
        expected = _normalize(_extract_nonstream("qwen3_coder_xml", text))
    for chunks in _every_split(text):
        assert (
            _stream_xml_deltas(
                "qwen3_coder_xml", text, chunks, request, allow_eos_recovery=True
            )
            == expected
        )


def test_later_quoted_marker_payload_is_lossless_at_every_byte_split():
    from unittest.mock import patch

    from tests.qwen3coder_stream_harness import non_stream, stream

    value = "before </parameter><parameter=description>literal</parameter> after"
    text = (
        '<tool_call><function=edit><parameter=path>"a.txt"</parameter>'
        "<parameter=new_string>"
        + json.dumps(value)
        + "</parameter></function></tool_call>"
    )
    request = _xml_request(("edit",))
    expected = [("edit", {"path": "a.txt", "new_string": value})]
    calls, _ = non_stream(text, request)
    assert [(name, json.loads(args)) for name, args in calls] == expected
    assert text.isascii()
    for chunks in _every_split(text):
        with patch("tests.qwen3coder_stream_harness.deltas", return_value=chunks):
            calls, _ = stream(text, request, 0)
        assert [(name, json.loads(args)) for name, args in calls] == expected, chunks


@pytest.mark.parametrize(
    "value,expected_value",
    [
        # Quoted marker fragment followed by a duplicate closer.
        ('"<parameter="</parameter>', "<parameter="),
        # Whitespace inside the literal opener is still marker text.
        (
            "<parameter =example>literal</parameter>\n",
            "<parameter =example>literal</parameter>",
        ),
        # Whitespace before a duplicate closer is payload.
        ("\nhello  \n</parameter>\n", "hello  "),
    ],
    ids=["quoted-opener-fragment", "spaced-literal-opener", "trailing-spaces"],
)
def test_duplicate_closer_edge_cases_match_at_every_split(value, expected_value):
    text = (
        "<tool_call><function=edit><parameter=new_string>"
        + value
        + "</parameter></function></tool_call>"
    )
    request = _xml_request(("edit",))
    expected = [("edit", {"new_string": expected_value})]
    assert _normalize(_extract_nonstream("qwen3_coder_xml", text)) == expected
    for chunks in _every_split(text):
        assert (
            _stream_xml_deltas(
                "qwen3_coder_xml", text, chunks, request, allow_eos_recovery=True
            )
            == expected
        )


@pytest.mark.parametrize("fresh_count", [0, 1])
def test_complete_body_short_batch_keeps_last_emitted_index(monkeypatch, fresh_count):
    """An empty or single-call batch slice must use the current-call path."""
    parser_cls = ToolParserManager.get_tool_parser("qwen3_coder_xml")
    parser = parser_cls(None)
    request = _xml_request(("read",))
    bodies = [
        f'<tool_call><function=read><parameter=filePath>"file{i}.txt"'
        "</parameter></function></tool_call>"
        for i in range(3)
    ]
    text = "".join(bodies)
    expected = parser.extract_tool_calls(text, request=request).tool_calls
    extract = parser.extract_tool_calls
    short_batches = []

    def extract_with_short_batch(current_text, request=None):
        result = extract(current_text, request=request)
        if len(parser.prev_tool_call_arr) == 2:
            # Model a complete-body pass whose recovery list has no new call,
            # or just the current call. Structural parsing still has its body.
            result.tool_calls = result.tool_calls[
                : 2 + (fresh_count if parser.current_tool_index == 2 else 0)
            ]
            if parser.current_tool_index == 2:
                short_batches.append(len(result.tool_calls[2:]))
        return result

    monkeypatch.setattr(parser, "extract_tool_calls", extract_with_short_batch)
    chunks = ["".join(bodies[:2]), bodies[2][: -len("</tool_call>")], "</tool_call>"]
    accumulated = ""
    calls = []
    for chunk in chunks:
        previous = accumulated
        accumulated += chunk
        delta = parser.extract_tool_calls_streaming(
            previous, accumulated, chunk, request=request
        )
        calls.extend((delta or {}).get("tool_calls", []))
    assert short_batches == [fresh_count]
    assert parser.current_tool_index == 2
    assert calls[-1]["id"] == parser._current_tool_id
    assert [call["index"] for call in calls] == [0, 1, 2]
    assert [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in calls
    ] == [(call["name"], json.loads(call["arguments"])) for call in expected]
    # Later framing must not reopen an already emitted call.
    for chunk in ("\n", " ", "\n"):
        previous = accumulated
        accumulated += chunk
        delta = parser.extract_tool_calls_streaming(
            previous, accumulated, chunk, request=request
        )
        assert not (delta or {}).get("tool_calls")


@pytest.mark.parametrize("parity", ["tool_calls", "content"])
def test_qwen3coder_adjacent_wrappers_keep_content_parity_at_every_boundary_split(
    parity,
    request,
):
    """Round 10: closing framing and the next opener share split deltas."""
    _mark_qwen_content_mismatch(request, parity)
    request = _xml_request(("read",))
    first = '<tool_call><function=read><parameter=path>"a"</parameter></function>'
    boundary = "</tool_call><tool_call>"
    second = '<function=read><parameter=path>"b"</parameter></function></tool_call>'
    text = first + boundary + second
    expected = [("read", {"path": "a"}), ("read", {"path": "b"})]
    result = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(text, request=request)
    assert not result.content
    assert [
        (call["name"], json.loads(call["arguments"])) for call in result.tool_calls
    ] == expected

    # Exact reproduction: the middle chunk closes one wrapper and starts the
    # next, leaving '<to' pending until the final chunk supplies 'ol_call>'.
    splits = [[first + "</tool_ca", "ll><to", "ol_call>" + second]]
    splits.extend(
        [first + boundary[:left], boundary[left:right], boundary[right:] + second]
        for left in range(len(boundary) + 1)
        for right in range(left + 1, len(boundary) + 1)
    )
    for chunks in splits:
        assert (
            _stream_xml_deltas(
                "qwen3_coder_xml",
                text,
                chunks,
                request,
                check_content=parity == "content",
            )
            == expected
        ), chunks


@pytest.mark.parametrize("prefix", ["", "The tag </tool_call> precedes it. "])
@pytest.mark.parametrize("parity", ["tool_calls", "content"])
def test_qwen3coder_stray_wrapper_closer_is_content_at_every_prose_split(
    prefix, parity
):
    """Round 11: a split literal closer after an admitted call stays visible."""
    request = _xml_request(("read",))
    call = (
        '<tool_call><function=read><parameter=path>"a"</parameter>'
        "</function></tool_call>"
    )
    prose = "The tag </tool_call> closes it."
    text = prefix + call + prose
    result = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(text, request=request)
    expected = [("read", {"path": "a"})]
    assert result.content == prefix + prose
    assert [
        (tc["name"], json.loads(tc["arguments"])) for tc in result.tool_calls
    ] == expected

    # Keep the call fully closed before the literal closer arrives, including
    # when a chunk boundary falls anywhere within that literal tag.
    before_prose = [prefix, call] if prefix else [call]
    chunks = [[*before_prose, prose]] + [
        [*before_prose, prose[:split], prose[split:]] for split in range(len(prose) + 1)
    ]
    for parts in chunks:
        assert (
            _stream_xml_deltas(
                "qwen3_coder_xml",
                text,
                parts,
                request,
                check_content=parity == "content",
            )
            == expected
        ), parts


def test_xml_stream_helper_requires_streamed_calls(monkeypatch):
    """A silent streaming parser cannot pass parity via complete extraction."""
    parser_cls = ToolParserManager.get_tool_parser("qwen3_coder_xml")
    text = '<tool_call><function=read><parameter=path>"a"</parameter></function></tool_call>'
    monkeypatch.setattr(
        parser_cls, "extract_tool_calls_streaming", lambda *a, **kw: None
    )
    for allow_eos_recovery in (False, True):
        with pytest.raises(AssertionError):
            _stream_xml_deltas(
                "qwen3_coder_xml",
                text,
                [text],
                _xml_request(("read",)),
                allow_eos_recovery=allow_eos_recovery,
            )


@pytest.mark.parametrize("source", ["parser", "finalizer"])
@pytest.mark.parametrize("leak", ["duplicated prose", "<tool_call>", "</tool_call>"])
def test_xml_stream_helper_rejects_content_leaks(monkeypatch, source, leak):
    """Valid tool deltas cannot hide leaked prose or XML framing."""
    from rapid_mlx.domain.events import StreamEvent

    parser_cls = ToolParserManager.get_tool_parser("qwen3_coder_xml")
    text = '<tool_call><function=read><parameter=path>"a"</parameter></function></tool_call>'
    if source == "parser":
        extract = parser_cls.extract_tool_calls_streaming

        def extract_with_leak(self, *args, **kwargs):
            delta = extract(self, *args, **kwargs)
            return {**delta, "content": (delta.get("content") or "") + leak}

        monkeypatch.setattr(
            parser_cls, "extract_tool_calls_streaming", extract_with_leak
        )
    else:
        finalize = StreamingPostProcessor.finalize

        def finalize_with_leak(self):
            return finalize(self) + [StreamEvent(type="content", content=leak)]

        monkeypatch.setattr(StreamingPostProcessor, "finalize", finalize_with_leak)
    with pytest.raises(AssertionError, match="content parity"):
        _stream_xml_deltas(
            "qwen3_coder_xml", text, [text], _xml_request(("read",)), check_content=True
        )


@pytest.mark.parametrize("split", range(2, 11))
def test_xml_stream_helper_checks_eos_recovery_content(monkeypatch, split):
    """Explicit EOS recovery compares content as well as recovered calls."""
    from rapid_mlx.domain.events import StreamEvent

    parser_cls = ToolParserManager.get_tool_parser("qwen3_coder_xml")
    text = '<tool_call><function=read><parameter=path>"a"</parameter></function></tool_call>'
    monkeypatch.setattr(
        parser_cls, "extract_tool_calls_streaming", lambda *a, **kw: None
    )
    assert _stream_xml_deltas(
        "qwen3_coder_xml",
        text,
        [text[:split], text[split:]],
        _xml_request(("read",)),
        allow_eos_recovery=True,
        check_content=True,
    ) == [("read", {"path": "a"})]
    finalize = StreamingPostProcessor.finalize

    def finalize_with_leak(self):
        return finalize(self) + [StreamEvent(type="content", content="<tool_call>")]

    monkeypatch.setattr(StreamingPostProcessor, "finalize", finalize_with_leak)
    with pytest.raises(AssertionError, match="content parity"):
        _stream_xml_deltas(
            "qwen3_coder_xml",
            text,
            [text[:split], text[split:]],
            _xml_request(("read",)),
            allow_eos_recovery=True,
            check_content=True,
        )


def _randomized_qwen_case(seed):
    """Small reproducible documents exercise arbitrary framing boundaries."""
    rng = random.Random(seed)
    bare = '<function=read><parameter=path>"a"</parameter></function>'
    wrapped = "<tool_call>" + bare + "</tool_call>"
    corpus = [
        "Some prose. ",
        " The tag <tool_call> opens it. ",
        " The tag </tool_call> closes it. ",
        wrapped,
        bare,
    ]
    pieces = rng.choices(corpus, k=rng.randint(1, 5))
    family = "closed_calls" if any(p in (wrapped, bare) for p in pieces) else "prose"
    text = "".join(pieces)
    # Include literal framing and incomplete suffixes independently of calls.
    if seed % 4 == 0:
        text += " A literal <function= in prose. "
    if seed % 3 == 0:
        text += rng.choice(["<", "<tool_ca", "<tool_call>", "<function="])
    if seed % 10 == 0:
        chunks = list(text)
    else:
        chunks = []
        cursor = 0
        while cursor < len(text):
            end = cursor + rng.randint(1, 40)
            chunks.append(text[cursor:end])
            cursor = end
    return family, text, chunks


# No current family truncates an actual call: appended partial framing tokens
# do not truncate the complete calls already generated in `pieces`. A future
# intentionally unclosed-call family must be named here to permit EOS recovery.
_QWEN_EOS_RECOVERY_FAMILIES = frozenset()


def _assert_randomized_qwen_parity(seed, *, check_content):
    family, text, chunks = _randomized_qwen_case(seed)
    request = _xml_request(("read",))
    result = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(text, request=request)
    expected = [
        (call["name"], json.loads(call["arguments"])) for call in result.tool_calls
    ]
    require_streamed_calls = (
        family != "prose" and family not in _QWEN_EOS_RECOVERY_FAMILIES
    )
    assert (
        _stream_xml_deltas(
            "qwen3_coder_xml",
            text,
            chunks,
            request,
            require_streamed_calls=require_streamed_calls,
            expected_streamed_calls=expected if require_streamed_calls else None,
            check_content=check_content,
        )
        == expected
    )


# Measured with the parser unchanged at HEAD 1165bb0413d1.
_QWEN_EOS_ONLY_SEEDS = frozenset({201})
_QWEN_EOS_ONLY_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=_StreamedCallsError,
    reason="pre-existing: call only recovered at EOS",
)


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(seed, marks=_QWEN_EOS_ONLY_XFAIL)
        if seed in _QWEN_EOS_ONLY_SEEDS
        else seed
        for seed in range(300)
    ],
)
def test_qwen3coder_randomized_tool_call_parity(seed):
    _assert_randomized_qwen_parity(seed, check_content=False)


def test_randomized_qwen_checker_rejects_silent_streaming(monkeypatch):
    """Every closed-call seed fails even though finalize could recover it."""
    parser_cls = ToolParserManager.get_tool_parser("qwen3_coder_xml")
    monkeypatch.setattr(
        parser_cls, "extract_tool_calls_streaming", lambda *a, **kw: None
    )
    checked = 0
    for seed in range(300):
        family, _, _ = _randomized_qwen_case(seed)
        if family == "closed_calls":
            with pytest.raises(_StreamedCallsError):
                _assert_randomized_qwen_parity(seed, check_content=False)
            checked += 1
    assert checked > 0


def test_randomized_qwen_checker_requires_every_streamed_call(monkeypatch):
    """Emitting the first call cannot mask later calls recovered at EOS."""
    parser_cls = ToolParserManager.get_tool_parser("qwen3_coder_xml")
    extract = parser_cls.extract_tool_calls_streaming

    def extract_first_call_only(self, *args, **kwargs):
        delta = extract(self, *args, **kwargs)
        if delta:
            return {
                **delta,
                "tool_calls": [
                    tc for tc in delta.get("tool_calls", []) if tc["index"] == 0
                ],
            }
        return delta

    monkeypatch.setattr(
        parser_cls, "extract_tool_calls_streaming", extract_first_call_only
    )
    # Seed 5 contains multiple complete calls and passes with the original parser.
    with pytest.raises(_StreamedCallsError):
        _assert_randomized_qwen_parity(5, check_content=False)


# Explicit failures measured against HEAD 1165bb0413d1; no runtime classification.
_QWEN_CONTENT_MISMATCH_SEEDS = frozenset(
    (
        0,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        14,
        15,
        16,
        17,
        19,
        20,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        32,
        33,
        34,
        36,
        37,
        38,
        39,
        40,
        41,
        44,
        45,
        46,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        56,
        58,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        68,
        71,
        72,
        75,
        76,
        77,
        78,
        80,
        81,
        82,
        83,
        84,
        86,
        87,
        88,
        90,
        92,
        93,
        94,
        95,
        96,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        110,
        111,
        112,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        138,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        150,
        151,
        152,
        153,
        155,
        156,
        157,
        158,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        170,
        171,
        172,
        175,
        176,
        177,
        180,
        181,
        182,
        184,
        186,
        187,
        188,
        189,
        191,
        192,
        193,
        194,
        195,
        196,
        197,
        198,
        200,
        201,
        202,
        204,
        205,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        231,
        232,
        234,
        235,
        236,
        237,
        238,
        240,
        241,
        243,
        244,
        245,
        247,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
        255,
        256,
        257,
        258,
        260,
        261,
        262,
        263,
        264,
        265,
        267,
        268,
        271,
        272,
        273,
        276,
        278,
        279,
        280,
        281,
        282,
        284,
        285,
        286,
        287,
        288,
        289,
        290,
        291,
        292,
        293,
        294,
        295,
        296,
        297,
    )
)


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(seed, marks=_QWEN_EOS_ONLY_XFAIL)
        if seed in _QWEN_EOS_ONLY_SEEDS
        else pytest.param(seed, marks=_QWEN_CONTENT_XFAIL)
        if seed in _QWEN_CONTENT_MISMATCH_SEEDS
        else seed
        for seed in range(300)
    ],
)
def test_qwen3coder_randomized_content_parity(seed):
    _assert_randomized_qwen_parity(seed, check_content=True)


@pytest.mark.parametrize("parity", ["tool_calls", "content"])
def test_qwen3coder_round12_literal_split_opener(parity):
    call = '<tool_call><function=read><parameter=path>"a"</parameter></function></tool_call>'
    chunks = [call, " The tag <", "tool_call> opens it."]
    assert _stream_xml_deltas(
        "qwen3_coder_xml",
        "".join(chunks),
        chunks,
        _xml_request(("read",)),
        check_content=parity == "content",
    ) == [("read", {"path": "a"})]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("parity", ["tool_calls", "content"])
def test_qwen3coder_literal_function_before_call_parity(seed, parity, request):
    _mark_qwen_content_mismatch(request, parity)
    _assert_literal_function_before_call_parity(seed, check_content=parity == "content")


def _assert_literal_function_before_call_parity(seed, *, check_content):
    rng = random.Random(seed)
    text = (
        " A literal <function= in prose. <tool_call><function=read>"
        '<parameter=path>"a"</parameter></function></tool_call>'
    )
    chunks = []
    cursor = 0
    while cursor < len(text):
        end = cursor + (1 if seed == 0 else rng.randint(1, 20))
        chunks.append(text[cursor:end])
        cursor = end
    request = _xml_request(("read",))
    result = ToolParserManager.get_tool_parser("qwen3_coder_xml")(
        None
    ).extract_tool_calls(text, request=request)
    assert _stream_xml_deltas(
        "qwen3_coder_xml",
        text,
        chunks,
        request,
        require_streamed_calls=True,
        expected_streamed_calls=[
            (call["name"], json.loads(call["arguments"])) for call in result.tool_calls
        ],
        check_content=check_content,
    ) == [(call["name"], json.loads(call["arguments"])) for call in result.tool_calls]


@pytest.mark.parametrize("seed", [0, 1])
def test_literal_function_checker_rejects_silent_streaming(monkeypatch, seed):
    parser_cls = ToolParserManager.get_tool_parser("qwen3_coder_xml")
    monkeypatch.setattr(
        parser_cls, "extract_tool_calls_streaming", lambda *a, **kw: None
    )
    with pytest.raises(_StreamedCallsError):
        _assert_literal_function_before_call_parity(seed, check_content=False)


@pytest.mark.parametrize("tail", ["<", "<tool_ca", "<tool_call>", "<function="])
@pytest.mark.parametrize("parity", ["tool_calls", "content"])
def test_qwen3coder_held_framing_at_eos(tail, parity, request):
    _mark_qwen_content_mismatch(request, parity, tail != "<tool_ca")
    call = '<tool_call><function=read><parameter=path>"a"</parameter></function></tool_call>'
    chunks = [call, tail]
    assert _stream_xml_deltas(
        "qwen3_coder_xml",
        "".join(chunks),
        chunks,
        _xml_request(("read",)),
        check_content=parity == "content",
    ) == [("read", {"path": "a"})]


def test_qwen3coder_round13_empty_wrapper_then_bare_call_keeps_visible_answer():
    chunks = [
        "<tool_call></tool_call>",
        '<function=read><parameter=path>"a"</parameter></function>',
        "Visible answer.",
    ]
    assert _stream_xml_deltas(
        "qwen3_coder_xml",
        "".join(chunks),
        chunks,
        _xml_request(("read",)),
        check_content=True,
        expected_content="Visible answer.",
    ) == [("read", {"path": "a"})]
