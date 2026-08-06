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


@pytest.mark.xfail(
    strict=True,
    reason="#1513 — qwen3_coder_xml still fabricates calls from prose. "
    "Split out of the </tool_call> fix: the parser gate needs the streaming "
    "path, the generic fallback and tool_choice to be handled together.",
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


def _qwen3coder():
    return ToolParserManager.get_tool_parser("qwen3_coder_xml")(None)


class TestOnlyDeclaredToolsBecomeCalls:
    """A name the request never offered can never be executed."""

    @pytest.mark.xfail(
        strict=True,
        reason="#1513 — qwen3_coder_xml still fabricates calls from prose. "
        "Split out of the </tool_call> fix: the parser gate needs the streaming "
        "path, the generic fallback and tool_choice to be handled together.",
    )
    def test_balanced_prose_with_no_tools_declared(self):
        """codex r1 BLOCKING: closed spans in prose were still promoted."""
        text = "Docs: write <function=read_file></function> to show an empty call."
        result = _qwen3coder().extract_tool_calls(text, None)
        assert result.tools_called is False
        assert result.content == text

    @pytest.mark.xfail(
        strict=True,
        reason="#1513 — qwen3_coder_xml still fabricates calls from prose. "
        "Split out of the </tool_call> fix: the parser gate needs the streaming "
        "path, the generic fallback and tool_choice to be handled together.",
    )
    def test_balanced_prose_naming_an_undeclared_tool(self):
        text = "Docs: write <function=read_file></function> to show an empty call."
        result = _qwen3coder().extract_tool_calls(text, {"tools": DECLARED_TOOLS})
        assert result.tools_called is False
        assert result.content == text

    @pytest.mark.xfail(
        strict=True,
        reason="#1513 — qwen3_coder_xml still fabricates calls from prose. "
        "Split out of the </tool_call> fix: the parser gate needs the streaming "
        "path, the generic fallback and tool_choice to be handled together.",
    )
    def test_unclosed_prose_with_no_tools_declared(self):
        text = "Docs mention </tool_call> and <function=name> together."
        result = _qwen3coder().extract_tool_calls(text, None)
        assert result.tools_called is False
        assert result.content == text


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
