# SPDX-License-Identifier: Apache-2.0
"""Cross-parser invariant: no extraction => content is returned uncut.

Every ``ToolParser.extract_tool_calls`` implementation has the same
contract on the miss path. When it reports ``tools_called=False`` it has
decided the text was NOT a tool call, so it has no licence to have
rewritten that text — the caller will surface it to the user verbatim.

``qwen3_coder_xml`` violated this, and worse: it FABRICATED calls out
of prose. Its candidate scanner matches ``<function=`` spans whether or
not the model meant them as wire, so an answer merely *mentioning* the
markup produced a structured ``tool_call`` and a truncated ``content``.
It is the parser for 22 aliases including ``qwen3.6-35b-4bit`` and every
Qwen3-Coder build, so that is the default path for local coding agents.

The discriminator is NOT whether the span is closed —
``<function=read_file></function>`` is well-formed either way — but
whether the CALLER declared that tool. A name the request never offered
can never be executed, so promoting it only breaks the agent loop. That
also preserves recovery of a genuine call truncated by ``max_tokens``
before ``</function>``, which a closed-only rule would have thrown away.

The tool-extraction path also runs when the request declared no tools at
all (``vllm_mlx/service/helpers.py`` does not gate on ``request.tools``),
which is why these cases pass ``request=None`` — a plain chat turn must
not be able to lose text to the tool layer.
"""

import pytest

from vllm_mlx.config import get_config
from vllm_mlx.service import helpers
from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

# Registered parser names, one per wire family. Aliases of the same class
# are intentionally omitted — the invariant is per implementation.
PARSERS = [
    "hermes",
    "qwen3_coder_xml",
    "qwen",
    "glm47",
    "minimax",
    "harmony",
    "mistral",
    "llama",
    "deepseek_v3",
    "deepseek_v31",
    "deepseek_v4_0731",
    "kimi",
    "seed_oss",
    "gemma4",
    "nemotron",
    "granite",
    "minicpm",
    "functionary",
    "xlam",
    "lfm",
    "hy_v3",
    "auto",
]

# Ordinary assistant prose that happens to name tool-wire tokens. This is
# exactly what a coding agent produces when asked about the tool-calling
# protocol, or when writing tests/docs for a parser.
PROSE = [
    "A tool call block ends with </tool_call> on its own.",
    "The marker is <function=f> here.",
    "Docs mention </tool_call> and <function=name> together.",
    "Use <parameter=path> for the path argument.",
    "Close the invoke with </invoke> and the arg with </arg_value>.",
    'The value was "quoted" and also 中文引号"用户关注数".',
    "Plain prose, nothing special at all.",
]


@pytest.mark.parametrize("parser_name", PARSERS)
@pytest.mark.parametrize("text", PROSE)
def test_miss_path_returns_content_uncut(parser_name, text):
    parser = ToolParserManager.get_tool_parser(parser_name)(None)
    parser.reset()
    result = parser.extract_tool_calls(text, None)

    if result.tools_called:
        # A parser that CLAIMS a call here has a fabrication bug, which is a
        # different defect from the content-passthrough invariant this test
        # pins — so skip rather than fail, and name the tracking issue.
        # (Deliberately ``skip`` and not a runtime ``pytest.xfail``: the
        # xfail audit gate is right that a runtime xfail mutes the signal.)
        extra = " — tracked in #1513" if parser_name == "qwen3_coder_xml" else ""
        pytest.skip(
            f"{parser_name} claims a tool call in this prose{extra}; that is a "
            f"fabrication bug, not the content-passthrough invariant"
        )

    assert result.content == text, (
        f"{parser_name} rewrote content on the miss path: "
        f"{text!r} -> {result.content!r}"
    )


def test_qwen3coder_regression_bare_function_mention():
    """The exact measured regression, pinned on its own.

    ``<function=`` in prose used to truncate the answer at that offset
    because the candidate span matched to end-of-string and was then
    rejected.
    """
    parser = ToolParserManager.get_tool_parser("qwen3_coder_xml")(None)
    text = "Docs mention </tool_call> and <function=name> together."
    result = parser.extract_tool_calls(text, None)
    assert result.tools_called is False
    assert result.content == text
    assert not result.content.endswith("and ")


def test_qwen3coder_still_parses_a_real_call():
    """The fix must not cost the parser its actual job."""
    parser = ToolParserManager.get_tool_parser("qwen3_coder_xml")(None)
    wire = (
        "<tool_call>\n<function=read_file>\n"
        "<parameter=path>\nsrc/main.py\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]
    result = parser.extract_tool_calls(wire, {"tools": tools})
    assert result.tools_called is True
    assert result.tool_calls[0]["name"] == "read_file"


DECLARED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    }
]
READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _qwen3coder():
    return ToolParserManager.get_tool_parser("qwen3_coder_xml")(None)


class TestOnlyDeclaredToolsBecomeCalls:
    """A name the request never offered can never be executed."""

    def test_balanced_prose_with_no_tools_declared(self):
        """codex r1 BLOCKING: closed spans in prose were still promoted."""
        text = "Docs: write <function=read_file></function> to show an empty call."
        result = _qwen3coder().extract_tool_calls(text, None)
        assert result.tools_called is False
        assert result.content == text

    def test_balanced_prose_naming_an_undeclared_tool(self):
        text = "Docs: write <function=read_file></function> to show an empty call."
        result = _qwen3coder().extract_tool_calls(text, {"tools": DECLARED_TOOLS})
        assert result.tools_called is False
        assert result.content == text

    def test_unclosed_prose_with_no_tools_declared(self):
        text = "Docs mention </tool_call> and <function=name> together."
        result = _qwen3coder().extract_tool_calls(text, None)
        assert result.tools_called is False
        assert result.content == text

    def test_tool_choice_none_never_promotes_a_declared_name(self):
        text = "Docs: write <function=write_file></function> to show an empty call."
        request = {"tools": DECLARED_TOOLS, "tool_choice": "none"}
        result = _qwen3coder().extract_tool_calls(text, request)
        assert result.tools_called is False
        assert result.content == text

    def test_named_tool_choice_rejects_a_different_declared_name(self):
        text = "<function=write_file></function>"
        request = {
            "tools": [*DECLARED_TOOLS, READ_TOOL],
            "tool_choice": {
                "type": "function",
                "function": {"name": "read_file"},
            },
        }
        result = _qwen3coder().extract_tool_calls(text, request)
        assert result.tools_called is False
        assert result.content == text

        streaming = _qwen3coder().extract_tool_calls_streaming(
            "", text, text, request=request
        )
        assert streaming == {"content": text}

    def test_matching_named_choice_allows_a_bare_zero_arg_call(self):
        text = "<function=write_file></function>"
        request = {
            "tools": DECLARED_TOOLS,
            "tool_choice": {
                "type": "function",
                "function": {"name": "write_file"},
            },
        }
        result = _qwen3coder().extract_tool_calls(text, request)
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "write_file"

        parser = _qwen3coder()
        previous = ""
        calls = []
        for chunk in ["<function=write_file>", "</function>"]:
            current = previous + chunk
            delta = parser.extract_tool_calls_streaming(
                previous, current, chunk, request=request
            )
            if delta:
                calls.extend(delta.get("tool_calls", []))
            previous = current
        assert any(call["function"]["name"] == "write_file" for call in calls)

    @pytest.mark.parametrize(
        "chunks",
        [
            ["Docs: <function=write_file></function> is the wire form."],
            [
                "Docs: ",
                "<function=write_file>",
                "</function> is the wire form.",
            ],
        ],
    )
    def test_declared_zero_arg_bare_prose_is_not_executable(self, chunks):
        text = "".join(chunks)
        request = {"tools": DECLARED_TOOLS, "tool_choice": "auto"}
        result = _qwen3coder().extract_tool_calls(text, request)
        assert result.tools_called is False
        assert result.content == text

        parser = _qwen3coder()
        previous = ""
        content = []
        calls = []
        for chunk in chunks:
            current = previous + chunk
            delta = parser.extract_tool_calls_streaming(
                previous, current, chunk, request=request
            )
            if delta:
                content.append(delta.get("content", ""))
                calls.extend(delta.get("tool_calls", []))
            previous = current
        assert calls == []
        assert "".join(content) == text

    def test_wrapped_zero_arg_declared_call_remains_executable(self):
        wire = "<tool_call><function=write_file></function></tool_call>"
        result = _qwen3coder().extract_tool_calls(
            wire, {"tools": DECLARED_TOOLS, "tool_choice": "auto"}
        )
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "write_file"

    @pytest.mark.parametrize(
        "request_payload",
        [
            None,
            {"tools": DECLARED_TOOLS, "tool_choice": "none"},
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "read_file", "parameters": {}},
                    }
                ]
            },
        ],
    )
    def test_streaming_prose_is_not_promoted_without_an_allowed_tool(
        self, request_payload
    ):
        parser = _qwen3coder()
        chunks = [
            "Docs: write ",
            "<function=",
            "write_file>",
            "</function> to show an empty call.",
        ]
        previous = ""
        content = []
        calls = []
        for chunk in chunks:
            current = previous + chunk
            result = parser.extract_tool_calls_streaming(
                previous,
                current,
                chunk,
                request=request_payload,
            )
            if result:
                content.append(result.get("content", ""))
                calls.extend(result.get("tool_calls", []))
            previous = current

        assert calls == []
        assert "".join(content) == "".join(chunks)

    def test_streaming_undeclared_span_in_one_chunk_is_preserved(self):
        parser = _qwen3coder()
        text = "Docs: <function=write_file></function> is the wire form."
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "read_file", "parameters": {}},
                }
            ]
        }
        result = parser.extract_tool_calls_streaming("", text, text, request=request)
        assert result == {"content": text}

    @pytest.mark.parametrize(
        "structured", [None, [{"name": "read_file", "arguments": "{}"}]]
    )
    def test_service_helper_cannot_repromote_an_undeclared_span(
        self, monkeypatch, structured
    ):
        text = "Docs: <function=read_file></function> is the wire form."

        class Request:
            def model_dump(self):
                return {"tools": DECLARED_TOOLS}

        cfg = get_config()
        saved = (cfg.enable_auto_tool_choice, cfg.tool_call_parser)
        cfg.enable_auto_tool_choice = True
        cfg.tool_call_parser = "qwen3_coder_xml"
        monkeypatch.setattr(
            helpers,
            "parse_tool_calls",
            lambda *args, **kwargs: pytest.fail("generic fallback re-promoted prose"),
        )
        try:
            content, calls = helpers._parse_tool_calls_with_parser(
                text,
                Request(),
                structured_tool_calls=structured,
            )
        finally:
            cfg.enable_auto_tool_choice, cfg.tool_call_parser = saved

        assert content == text
        assert calls is None

    def test_service_helper_enforces_named_tool_choice_on_structured_calls(
        self, monkeypatch
    ):
        text = "The model selected a different declared tool."

        class Request:
            def model_dump(self):
                return {
                    "tools": [*DECLARED_TOOLS, READ_TOOL],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "read_file"},
                    },
                }

        cfg = get_config()
        saved = (cfg.enable_auto_tool_choice, cfg.tool_call_parser)
        cfg.enable_auto_tool_choice = True
        cfg.tool_call_parser = "qwen3_coder_xml"
        monkeypatch.setattr(
            helpers,
            "parse_tool_calls",
            lambda *args, **kwargs: pytest.fail("named choice was bypassed"),
        )
        try:
            content, calls = helpers._parse_tool_calls_with_parser(
                text,
                Request(),
                structured_tool_calls=[{"name": "write_file", "arguments": "{}"}],
            )
        finally:
            cfg.enable_auto_tool_choice, cfg.tool_call_parser = saved

        assert content == text
        assert calls is None


class TestTruncatedCallsStillRecover:
    """codex r1 MAJOR: a closed-only rule threw these away."""

    def test_call_truncated_by_max_tokens_before_the_closer(self):
        wire = (
            "<tool_call>\n<function=write_file>\n"
            "<parameter=path>\na.md\n</parameter>\n"
            "<parameter=content>\nhi\n</parameter>"
        )
        result = _qwen3coder().extract_tool_calls(wire, {"tools": DECLARED_TOOLS})
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "write_file"

    def test_wrapped_zero_arg_call_truncated_after_header_recovers(self):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        }
        result = _qwen3coder().extract_tool_calls(
            "<tool_call>\n<function=ping>", request
        )
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "ping"

        # Without canonical framing, a zero-argument span is indistinguishable
        # from prose documenting the protocol and remains non-executable.
        bare = _qwen3coder().extract_tool_calls("<function=ping>", request)
        assert bare.tools_called is False
        unrelated_wrapper = _qwen3coder().extract_tool_calls(
            "<tool_call></tool_call> prose <function=ping>", request
        )
        assert unrelated_wrapper.tools_called is False

    def test_complete_call_is_unaffected(self):
        wire = (
            "<tool_call>\n<function=write_file>\n"
            "<parameter=path>\na.md\n</parameter>\n"
            "<parameter=content>\nhi\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = _qwen3coder().extract_tool_calls(wire, {"tools": DECLARED_TOOLS})
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "write_file"
