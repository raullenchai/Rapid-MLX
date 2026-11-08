# SPDX-License-Identifier: Apache-2.0
"""An undeclared XML parameter must not become executable string payload.

Bare XML cannot distinguish a sibling after a close from literal marker-shaped
payload. Reject the ambiguous call instead of choosing a command the model may
not have meant. Complete JSON-quoted payloads remain unambiguous and lossless.

Streaming may already have sent a call header before ambiguity is known. It
must leave that prefix incomplete (a client argument-parse failure), never
fabricate an empty call or attach an accepted sibling's arguments to its name.
Non-streaming returns no call for the rejected span. Existing content-preserving
and retry behavior is unchanged; this is not an automatic repair/retry policy.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from rapid_mlx.api.tool_calling import parse_tool_calls
from rapid_mlx.config import get_config
from rapid_mlx.service import helpers
from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from tests.qwen3coder_stream_harness import non_stream, stream

REQUEST = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "security_risk": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        }
    ],
    "tool_choice": "auto",
}


def wire(command="printf OK", *, extra=True):
    body = f"<parameter=command>\n{command}\n</parameter>\n"
    if extra:
        body += "<parameter=description>\ninspect files\n</parameter>\n"
    body += "<parameter=security_risk>\nLOW\n</parameter>\n"
    return f"<tool_call>\n<function=terminal>\n{body}</function>\n</tool_call>"


def test_nonstream_rejects_undeclared_sibling():
    calls, _ = non_stream(wire(), REQUEST)
    assert calls == []


@pytest.mark.parametrize("size", [0, 1, 2, 7, 13, 10000])
def test_stream_rejects_undeclared_sibling(size):
    calls, _ = stream(wire(), REQUEST, size)
    # Fine deltas may have already emitted a name and opening JSON brace.
    # They cannot be retracted, but must never become executable arguments
    # (including an invented empty object for a rejected parse).
    for _, arguments in calls:
        with pytest.raises(json.JSONDecodeError):
            json.loads(arguments)


def test_helper_does_not_resurrect_rejected_call(monkeypatch):
    class Request:
        tools = REQUEST["tools"]

        def model_dump(self):
            return REQUEST

    cfg = get_config()
    monkeypatch.setattr(cfg, "engine", None)
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", True)
    monkeypatch.setattr(cfg, "tool_call_parser", "qwen3_coder_xml")
    _, calls = helpers._parse_tool_calls_with_parser(wire(), Request())
    assert not calls


def test_generic_fallback_already_rejects_undeclared_sibling():
    _, calls = parse_tool_calls(wire(), REQUEST)
    assert not calls


@pytest.mark.parametrize("size", [None, 0, 1, 13])
def test_valid_multiline_command_is_byte_exact(size):
    command = "python3 - <<'PY'\nprint(\"quoted && literal\")\nPY"
    text = wire(command, extra=False)
    calls, _ = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert len(calls) == 1
    assert json.loads(calls[0][1]) == {"command": command, "security_risk": "LOW"}


def test_complete_quoted_marker_payload_remains_data():
    command = 'printf "</parameter><parameter=description>literal</parameter>"'
    text = wire(json.dumps(command), extra=False)
    result = Qwen3CoderToolParser().extract_tool_calls(text, REQUEST)
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"])["command"] == command


@pytest.mark.parametrize("size", [0, 1, 2, 7, 13, 10000])
def test_stream_does_not_attach_later_arguments_to_rejected_header(size):
    request = deepcopy(REQUEST)
    other_tool = deepcopy(REQUEST["tools"][0])
    other_tool["function"]["name"] = "other_terminal"
    request["tools"].append(other_tool)
    second = wire("printf SECOND", extra=False).replace(
        "<function=terminal>", "<function=other_terminal>"
    )
    calls, _ = stream(wire() + second, request, size)
    for name, arguments in calls:
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            continue
        assert name == "other_terminal"
        assert parsed == {"command": "printf SECOND", "security_risk": "LOW"}


@pytest.mark.parametrize("size", [None, 0, 1, 13])
def test_quoted_literal_markup_survives_both_paths(size):
    command = 'printf "</parameter><parameter=description>literal</parameter>"'
    text = wire(json.dumps(command), extra=False)
    calls, _ = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert len(calls) == 1
    assert json.loads(calls[0][1]) == {"command": command, "security_risk": "LOW"}


@pytest.mark.parametrize(
    "choice",
    ["auto", "required", {"type": "function", "function": {"name": "terminal"}}],
)
def test_rejection_is_not_limited_to_auto_choice(choice):
    request = {**REQUEST, "tool_choice": choice}
    assert non_stream(wire(), request)[0] == []


def test_rejection_log_does_not_include_argument_values(caplog):
    with caplog.at_level("WARNING"):
        non_stream(wire("PRIVATE_COMMAND_VALUE"), REQUEST)
    assert "rejecting an ambiguous undeclared parameter" in caplog.text
    assert "PRIVATE_COMMAND_VALUE" not in caplog.text


def test_parser_reuse_clears_rejection_state():
    parser = Qwen3CoderToolParser()
    assert not parser.extract_tool_calls(wire(), REQUEST).tools_called
    assert parser._ambiguous_parameter_rejected
    assert parser.extract_tool_calls(wire(extra=False), REQUEST).tools_called
    assert not parser._ambiguous_parameter_rejected


@pytest.mark.parametrize("size", [None, 0, 1, 13])
def test_ambiguous_bare_marker_payload_is_rejected_not_rewritten(size):
    # On the bare wire this could equally mean a literal code sample or an
    # undeclared sibling. Neither interpretation can safely be guessed.
    command = 'printf "</parameter><parameter=description>literal</parameter>"'
    text = wire(command, extra=False)
    calls, _ = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    for _, arguments in calls:
        with pytest.raises(json.JSONDecodeError):
            json.loads(arguments)


@pytest.mark.parametrize("size", [None, 0, 1, 13])
def test_literal_opener_without_preceding_close_remains_payload(size):
    command = 'printf "<parameter=description>literal"'
    text = wire(command, extra=False)
    calls, _ = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert len(calls) == 1
    assert json.loads(calls[0][1]) == {"command": command, "security_risk": "LOW"}


@pytest.mark.parametrize("bare", [False, True])
def test_wrapperless_and_truncated_calls_cannot_bypass_rejection(bare):
    text = wire().removesuffix("</tool_call>")
    if bare:
        text = text.removeprefix("<tool_call>\n")
    assert non_stream(text, REQUEST)[0] == []


def test_nonstream_keeps_independent_valid_sibling():
    calls, _ = non_stream(wire() + wire("printf SECOND", extra=False), REQUEST)
    assert len(calls) == 1
    assert json.loads(calls[0][1]) == {
        "command": "printf SECOND",
        "security_risk": "LOW",
    }


@pytest.mark.parametrize("size", [0, 1, 13, 10000])
def test_cross_format_fallback_cannot_execute_text_from_rejected_payload(size):
    text = wire('Calling tool: terminal({"command": "NOT_AN_INVOCATION"})')
    calls, _ = stream(text, REQUEST, size)
    for _, arguments in calls:
        with pytest.raises(json.JSONDecodeError):
            json.loads(arguments)
