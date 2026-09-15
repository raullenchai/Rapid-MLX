# SPDX-License-Identifier: Apache-2.0
r"""A parameter value keeps the indentation of its FIRST line (#3401).

Reported by andreiiliuta against 0.14.1: every code edit a coding agent made
through Rapid-MLX lost the leading whitespace of the opening line of
``new_string``/``old_string``, while lines 2..n kept theirs. Four of six
otherwise-correct Python files in a real agent transcript did not compile.

The cause was ``split_marked_parameters`` returning ``segmented[0].strip()``.
On this wire exactly ONE newline per side is markup -- the Qwen3-Coder chat
template renders ``<parameter=NAME>\n`` + value + ``\n</parameter>\n`` -- so
``.strip()`` also ate the payload indentation of line 1. The asymmetry is why
it survived so long: a single-line value looked merely "trimmed", and a
multi-line value looked correct everywhere except its first line.

Both mature engines serving this format drop one newline per side and nothing
else, in streaming and non-streaming alike:

  * vLLM ``vllm/parser/qwen3.py::_trim_wrapping_newlines``
  * SGLang ``srt/function_call/qwen3_coder_detector.py`` ("Remove prefixing
    and trailing \n"), in both ``detect_and_parse`` and the streaming lexer.

``tests/test_tool_call_value_fidelity.py`` guards the same class of defect for
INTERIOR whitespace; this file guards the edges.
"""

from __future__ import annotations

import json

import pytest

from rapid_mlx.tool_call_scan import split_marked_parameters, trim_wrapping_newlines
from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

PARAM_OPENER = r"<parameter=([^>]+)>"
PARAM_CLOSER = "</parameter>"

# The exact edit shape from the report: 8-space indent on line 1, deeper
# indent on line 2. Line 2 always survived; line 1 did not.
INDENTED_BODY = "        if is_literal_job(job):\n            raise ValueError()"

EDIT_REQUEST = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "Edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                },
            },
        }
    ]
}


def _wire(**params: str) -> str:
    """Render params exactly as tool_chat_template_qwen3coder.jinja does."""
    body = "".join(
        f"<parameter={name}>\n{value}\n</parameter>\n" for name, value in params.items()
    )
    return f"<tool_call>\n<function=Edit>\n{body}</function>\n</tool_call>"


def _arguments(text: str, request: dict | None = EDIT_REQUEST) -> dict:
    parser = Qwen3CoderToolParser(tokenizer=None)
    parser.reset()
    result = parser.extract_tool_calls(text, request=request)
    assert result.tools_called, f"no tool call recovered from {text!r}"
    return json.loads(result.tool_calls[0]["arguments"])


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # One newline per side is markup, and only one.
        ("\n        return 2\n", "        return 2"),
        # A value that legitimately opens or closes on a blank line keeps it:
        # only the single wrapping newline belongs to the wire.
        ("\n\nleading blank kept\n\n", "\nleading blank kept\n"),
        # \r\n is NOT a two-byte wrapper: under the template's LF framing a
        # payload ending in \r is indistinguishable from CRLF markup, so the
        # byte is kept rather than guessed away (codex review, round 1).
        ("\n  x  \r\n", "  x  \r"),
        ("\r\ntext\r\n", "\r\ntext\r"),
        # Nothing to trim: an inline value is returned byte-identical, which
        # is what keeps single-line scalars working.
        ("42", "42"),
        (" spaced ", " spaced "),
        # Degenerate inputs must not underflow into the payload.
        ("", ""),
        ("\n", ""),
        # Both newlines are the wire's own: an empty value, not a newline.
        ("\n\n", ""),
    ],
)
def test_trim_wrapping_newlines(raw: str, expected: str) -> None:
    assert trim_wrapping_newlines(raw) == expected


def test_split_marked_parameters_keeps_first_line_indentation() -> None:
    block = f"<parameter=new_string>\n{INDENTED_BODY}\n</parameter>\n"
    assert split_marked_parameters(block, PARAM_OPENER, PARAM_CLOSER) == [
        ("new_string", INDENTED_BODY)
    ]


def test_parameter_names_are_still_stripped() -> None:
    """Names are identifiers, not payload -- that half of the old behaviour
    was correct and must not regress with the value fix."""
    block = "<parameter= spaced_name >\nv\n</parameter>\n"
    parsed = split_marked_parameters(block, PARAM_OPENER, PARAM_CLOSER)
    assert parsed == [("spaced_name", "v")]


# ---------------------------------------------------------------------------
# The reported failure, end to end
# ---------------------------------------------------------------------------


def test_edit_arguments_keep_first_line_indentation() -> None:
    args = _arguments(
        _wire(
            file_path="/tmp/x.py",
            old_string="        return 1",
            new_string=INDENTED_BODY,
        )
    )
    assert args["old_string"] == "        return 1"
    assert args["new_string"] == INDENTED_BODY
    # The report's actual symptom: line 1 de-indented while line 2 kept its
    # indentation. Assert the asymmetry itself, not just the whole value.
    first, second = args["new_string"].split("\n")
    assert first.startswith("        "), "first line lost its indentation"
    assert second.startswith("            "), "second line lost its indentation"


def test_edited_python_still_compiles() -> None:
    """The consequence, stated as the reporter experienced it."""
    source = "def f(job):\n" + _arguments(_wire(new_string=INDENTED_BODY))["new_string"]
    compile(source, "<edit>", "exec")


def test_trailing_newline_in_payload_survives() -> None:
    """A file body ending in a newline must arrive with it: the wire adds one
    newline, the payload's own is the second. Dropping it is what makes git
    report "\\ No newline at end of file" (same rationale as
    ``_decode_json_like``)."""
    assert _arguments(_wire(new_string="body\n"))["new_string"] == "body\n"


def test_streaming_finalize_matches_non_streaming() -> None:
    """Bare (non-JSON-quoted) values are recovered by
    ``finalize_legacy_raw_stream``, which re-enters ``extract_tool_calls``.
    Pin that the two paths agree on the indentation."""
    from unittest.mock import MagicMock

    from rapid_mlx.service.postprocessor import StreamingPostProcessor

    text = _wire(file_path="/tmp/x.py", new_string=INDENTED_BODY)
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    cfg.tool_parser_instance = None
    pp = StreamingPostProcessor(cfg, tools_requested=True)
    pp.reset()
    pp.tool_accumulated_text = text
    streamed = [ev for ev in pp.finalize() if ev.type == "tool_call"]
    assert streamed, "finalize emitted no tool call"
    args = streamed[0].tool_calls[0]["function"]["arguments"]
    assert json.loads(args)["new_string"] == INDENTED_BODY


# ---------------------------------------------------------------------------
# Guards on the fix itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declared_type", "emitted", "expected"),
    [
        ("boolean", " true ", True),
        ("boolean", "\n true \n", True),
        ("integer", " 42 ", 42),
        ("number", " 1.5 ", 1.5),
        # ``null`` is matched WITHOUT trimming in the GLOBAL check (see
        # _convert_param_value): that check also sees string-typed values,
        # where trimming would widen a pre-existing stream/non-stream
        # divergence in the close path. For a string parameter the padding is
        # therefore payload -- the deliberate contract change of this PR.
        ("string", " null ", " null "),
        ("string", "null", None),
        # Past the string branch the padding cannot be payload, so the
        # keyword is recognised there instead.
        ("boolean", " null ", None),
        ("integer", " null ", None),
    ],
)
def test_padded_scalars_still_convert(declared_type, emitted, expected) -> None:
    """Values used to reach ``_convert_param_value`` already ``.strip()``-ed.
    Now that only the wrapping newline is removed, a model that pads a scalar
    must still resolve to the scalar -- otherwise this fix would trade a
    string bug for a boolean bug (``" true "`` silently becoming ``False``)."""
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Edit",
                    "parameters": {
                        "type": "object",
                        "properties": {"v": {"type": declared_type}},
                    },
                },
            }
        ]
    }
    text = f"<tool_call>\n<function=Edit>\n<parameter=v>{emitted}</parameter>\n</function>\n</tool_call>"
    assert _arguments(text, request)["v"] == expected


def test_nemotron_xml_body_keeps_indentation() -> None:
    """``split_marked_parameters`` is shared: the Nemotron XML body carries the
    identical ``<parameter=…>`` markup, so the same rule has to hold there or
    the two wires disagree about the same bytes."""
    from rapid_mlx.api.tool_calling import parse_tool_calls

    text = (
        "<tool_call>\n<function=Edit>\n"
        f"<parameter=new_string>\n{INDENTED_BODY}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    _, calls = parse_tool_calls(text, None)
    assert calls, "nemotron scanner recovered no call"
    assert json.loads(calls[0].function.arguments)["new_string"] == INDENTED_BODY


# ---------------------------------------------------------------------------
# Parity guard for the scalar-keyword trim (codex adversarial review, r2/r4)
# ---------------------------------------------------------------------------


def _stream_arguments(chunks: list[str], request: dict) -> dict:
    """Concatenate streamed ``function.arguments`` fragments and parse them."""
    parser = Qwen3CoderToolParser(tokenizer=None)
    parser.reset()
    previous = ""
    fragments: list[str] = []
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            request=request,
        )
        for tc in (delta or {}).get("tool_calls") or []:
            args = (tc.get("function") or {}).get("arguments")
            if args:
                fragments.append(args)
        previous = current
    return json.loads("".join(fragments))


def _one_param_request(declared_type: str) -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": declared_type}},
                    },
                },
            }
        ]
    }


def _chunks_for(wire: str) -> list[str]:
    """Split so ``</parameter>`` arrives in the same chunk as the value tail —
    the shape that reaches the close path before anything has been emitted."""
    return [
        "<tool_call>\n<function=f>\n",
        "<parameter=x>\n" + wire[:1],
        wire[1:] + "\n</parameter>\n</function>\n</tool_call>",
    ]


def test_padded_bare_boolean_agrees_across_paths() -> None:
    """The scalar-keyword trim must not create a stream/non-stream split, and
    in fact closes one: on v0.14.1 a padded bare ``true`` streamed as ``False``
    while the non-streaming path returned ``True``."""
    request = _one_param_request("boolean")
    chunks = _chunks_for(" true ")
    assert _stream_arguments(chunks, request) == {"x": True}
    assert _arguments("".join(chunks), request) == {"x": True}


@pytest.mark.parametrize("wire", ['" null "', '" true "', '"  padded  "'])
def test_padded_json_quoted_string_keeps_stream_parity(wire: str) -> None:
    """``null`` is matched WITHOUT trimming, unlike the typed scalars.

    ``_convert_param_value`` runs before the type dispatch and is not
    idempotent, and the streaming close path can convert an already-decoded
    value a second time -- a defect in ``_close_string_increment`` that
    predates this PR and is a non-goal here. A trimmed ``null`` match would
    have widened it, turning the string ``" null "`` into ``None`` in the
    streamed arguments only. Pin the parity so that stays true.
    """
    request = _one_param_request("string")
    chunks = _chunks_for(wire)
    streamed = _stream_arguments(chunks, request)
    assert streamed == _arguments("".join(chunks), request)
    assert streamed["x"] == json.loads(wire)


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        # The shape that made this worth a round of review: a nullable boolean
        # whose padded `null` reached the boolean branch and became `False` --
        # "no value" silently turning into "off".
        ({"type": ["boolean", "null"]}, None),
        ({"type": "null"}, None),
        ({"type": "boolean"}, None),
        ({"type": "integer"}, None),
        # Shapes for which ``_schema_type`` returns None, so they never reach
        # the type dispatch at all. A per-branch guard missed every one of
        # these; the rule has to live at the ``_is_string_param`` boundary.
        ({"type": ["null"]}, None),
        ({"anyOf": [{"type": "null"}]}, None),
        ({"oneOf": [{"type": "null"}]}, None),
        ({"description": "no type key"}, None),
        # A string parameter keeps the padding: there it is payload.
        ({"type": "string"}, " null "),
    ],
)
def test_padded_null_matches_v0_14_1_for_typed_parameters(schema, expected) -> None:
    """A padded ``null`` must still read as the keyword for every type that
    cannot hold whitespace as payload.

    Values used to reach the converter already ``.strip()``-ed, so v0.14.1
    resolved ``<parameter=x>\n null \n</parameter>`` to ``None`` for every
    schema. Removing that strip without replacing typed-null recognition
    regressed the non-streaming path; measured against v0.14.1, every row
    below except the string one matches what the old tree returned.
    """
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {"x": schema}},
                },
            }
        ]
    }
    text = (
        "<tool_call>\n<function=f>\n"
        "<parameter=x>\n null \n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert _arguments(text, request)["x"] == expected


def test_padded_null_on_an_undeclared_parameter() -> None:
    """An undeclared parameter has no schema, so it cannot be string-typed and
    the padded keyword still resolves. v0.14.1 returned ``None`` here because
    the value arrived pre-stripped."""
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    }
    text = (
        "<tool_call>\n<function=f>\n"
        "<parameter=x>\n null \n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert _arguments(text, request)["x"] is None
