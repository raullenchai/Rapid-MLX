"""#5721: auto + parallel_tool_calls=False keeps the auto grammar shape.

Pure-Python (no MLX): auto never narrows the call quantifier, and a
section-wrapper family stays free-form on auto even with single_call.
"""

from rapid_mlx.api import tool_grammar
from rapid_mlx.api.tool_grammar import StructureInfo, build_tool_lark

TOOLS = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def _hermes_info():
    return StructureInfo(
        begin='<tool_call>\n{"name": "get_weather", "arguments": ',
        end="}\n</tool_call>",
        trigger="<tool_call>",
        sentinels=("<tool_call>", "</tool_call>"),
    )


def test_auto_single_call_keeps_zero_or_more_quantifier():
    auto_single = build_tool_lark(TOOLS, "auto", [_hermes_info()], single_call=True)
    auto_multi = build_tool_lark(TOOLS, "auto", [_hermes_info()], single_call=False)
    assert auto_single == auto_multi


def test_required_single_call_still_narrows():
    single = build_tool_lark(TOOLS, "required", [_hermes_info()], single_call=True)
    multi = build_tool_lark(TOOLS, "required", [_hermes_info()], single_call=False)
    assert single != multi


def test_section_wrapper_auto_single_call_stays_free_form(monkeypatch):
    monkeypatch.setattr(tool_grammar, "HAS_LLGUIDANCE", True)

    class SectionWrapperParser:
        TOOL_GRAMMAR_SECTION_WRAPPER = True

    assert (
        tool_grammar.build_tool_grammar(
            TOOLS, "auto", SectionWrapperParser(), single_call=True
        )
        is None
    )
