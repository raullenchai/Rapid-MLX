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
from unittest.mock import MagicMock

import pytest

from vllm_mlx.api.tool_calling import parse_tool_calls
from vllm_mlx.service.postprocessor import StreamingPostProcessor
from vllm_mlx.tool_parsers import ToolParserManager


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
    pp = StreamingPostProcessor(cfg)
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
    "deepseek": "TODO: add DeepSeek wire-format fixture",
    "deepseek_v3": "alias of deepseek",
    "deepseek_r1": "alias of deepseek",
    "deepseek_v31": "TODO: add DeepSeek V3.1 wire-format fixture",
    "deepseek_r1_0528": "alias of deepseek_v31",
    "functionary": "TODO: add Functionary wire-format fixture",
    "meetkai": "alias of functionary",
    "xlam": "TODO: add xLAM wire-format fixture",
    "qwen": "JSON-body Qwen variant — covered by hermes json_body fixture (same shape)",
    "qwen3": "alias of qwen",
    "qwen3_coder": "alias of hermes",
    "nous": "alias of hermes",
    "liquid": "alias of lfm",
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
