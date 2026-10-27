"""Selected Qwen-reasoning envelope owner: one result for every API mode."""

import itertools
from types import SimpleNamespace

import pytest

from rapid_mlx.api.models import ChatCompletionRequest
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.parsing.output_demux import ToolEnd, ToolStart
from rapid_mlx.parsing.prompt_owner import (
    EnvelopeTools,
    PromptOutputOwner,
    requested_thinking,
    selected_envelope_tools,
    split_buffered,
)
from rapid_mlx.prompt_parse_context import PromptParseContext

from .harness import route

SURFACES = ["chat", "responses", "messages"]
MODES = list(itertools.product(SURFACES, [False, True], [1, 7, 1000]))
EXPECTED = [["run_shell_command", {"command": "pwd"}]]
HERMES = (
    '<tool_call>{"name":"run_shell_command","arguments":{"command":"pwd"}}</tool_call>'
)
GLM = "<tool_call>run_shell_command<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"
REQUEST = {
    "tools": [
        {
            "type": "function",
            "function": {"name": "run_shell_command", "parameters": {"type": "object"}},
        }
    ]
}
CASES = {
    # #5272: tool wire is never also reasoning, buffered or streamed.
    "wire_only": ("hermes", HERMES, "", "", EXPECTED),
    "glm_wire_only": ("glm47", GLM, "", "", EXPECTED),
    # #5273: an envelope inside reasoning keeps reasoning, call and answer.
    "promoted": (
        "hermes",
        "<think>Secret." + HERMES + "</think>Answer",
        "Secret.",
        "Answer",
        EXPECTED,
    ),
    "glm_promoted": (
        "glm47",
        "<think>Secret." + GLM + "</think>Answer",
        "Secret.",
        "Answer",
        EXPECTED,
    ),
    # #5274: orphan argument markup is a failed tool span, never visible text.
    "orphan": (
        "glm47",
        "<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>Answer",
        "",
        "Answer",
        [],
    ),
    "implicit": (
        "hermes",
        "Secret.</think>" + HERMES + "Answer",
        "Secret.",
        "Answer",
        EXPECTED,
    ),
}


@pytest.mark.parametrize("surface,stream,split", MODES)
@pytest.mark.parametrize("case", sorted(CASES))
def test_issue_contracts_agree_across_modes(case, surface, stream, split):
    tool, raw, reasoning, content, calls = CASES[case]
    chunks = [raw[i : i + split] for i in range(0, len(raw), split)]
    result = route("qwen3", tool, raw, surface, stream, chunks)
    assert (result["reasoning"], result["content"], result["calls"]) == (
        reasoning,
        content,
        calls,
    )


def cfg(reasoning="qwen3", tool="hermes", auto=True):
    return SimpleNamespace(
        reasoning_parser_name=reasoning,
        tool_call_parser=tool,
        enable_auto_tool_choice=auto,
        model_path="",
        model_name="qwen3-test",
        engine=None,
    )


def test_selection_gates_keep_the_existing_parser():
    assert selected_envelope_tools(cfg(), REQUEST) is not None
    assert selected_envelope_tools(cfg(reasoning="deepseek_r1"), REQUEST) is None
    assert selected_envelope_tools(cfg(tool="seed_oss"), REQUEST) is None
    assert selected_envelope_tools(cfg(auto=False), REQUEST) is None
    assert selected_envelope_tools(cfg(), None) is None
    assert selected_envelope_tools(cfg(), {"tools": []}) is None
    assert selected_envelope_tools(cfg(), {**REQUEST, "tool_choice": "none"}) is None
    assert selected_envelope_tools(cfg(), REQUEST, reasoning_max_tokens=0) is None
    budget = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reasoning_max_tokens=8,
        **REQUEST,
    )
    assert selected_envelope_tools(cfg(), budget) is None
    glm = selected_envelope_tools(cfg(tool="glm47"), REQUEST)
    assert [t.text for t in glm.terminals][-1] == "<arg_key>"


def test_thinking_off_requests_keep_the_existing_parser():
    off = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        chat_template_kwargs={"enable_thinking": False},
        **REQUEST,
    )
    assert requested_thinking(cfg(), off) is False
    assert PromptOutputOwner.for_request(cfg(), off).tools is None
    assert PromptOutputOwner.for_request(cfg(), REQUEST, thinking=False).tools is None
    assert PromptOutputOwner.for_request(cfg(), REQUEST, thinking=True).tools


def owner(tool="hermes"):
    return PromptOutputOwner("qwen3-test", tools=EnvelopeTools(tool, REQUEST))


def run(owner, text, finished=True):
    events = owner.observe(GenerationOutput(text=text, new_text=text))
    events += owner.finish()
    return "".join(owner.reasoning), "".join(owner.content), events


def test_markers_are_honored_once_and_stray_closers_stay_literal():
    reasoning, content, _ = run(
        owner(), "<think>a</think>b<think>c</think></tool_call>"
    )
    assert (reasoning, content) == ("a", "b<think>c</think></tool_call>")


def test_failed_envelope_is_not_published():
    the_owner = owner()
    reasoning, content, _ = run(the_owner, "</think><tool_call>oops</tool_call>Answer")
    assert (reasoning, content, the_owner.calls) == ("", "Answer", [])


def test_open_envelope_at_finish_decodes_or_stays_in_its_channel():
    the_owner = owner()
    _, _, events = run(
        the_owner, '<tool_call>{"name":"run_shell_command","arguments":{}}'
    )
    assert [call["name"] for call in the_owner.calls] == ["run_shell_command"]
    assert any(isinstance(event, ToolEnd) for event in events)
    assert the_owner.finish() == []
    the_owner = owner()
    reasoning, content, _ = run(the_owner, "plan <tool_call>not a call")
    assert (reasoning, content, the_owner.calls) == (
        "plan <tool_call>not a call",
        "",
        [],
    )


def test_every_decoded_call_gets_one_index_and_contiguous_spans():
    the_owner = owner()
    the_owner.tools.decode = lambda text: [
        {"id": "a", "name": "a", "arguments": "{}"},
        {"id": "b", "name": "b", "arguments": "{}"},
    ]
    _, _, events = run(the_owner, "</think><tool_call>x</tool_call>")
    starts = [event for event in events if isinstance(event, ToolStart)]
    assert [(event.index, event.name) for event in starts] == [(0, "a"), (1, "b")]
    assert starts[1].span.start == starts[1].span.end == starts[0].span.end
    assert [call["id"] for call in the_owner.calls] == ["a", "b"]


def test_buffered_split_returns_structured_calls_and_flint_keeps_precedence():
    output = GenerationOutput(text="<think>\nplan\n</think>\n\n" + HERMES + "\nAnswer")
    split, owned = split_buffered(output, "qwen3-test", cfg=cfg(), request=REQUEST)
    assert owned and (split.reasoning_text, split.text) == ("plan", "Answer")
    assert [call["name"] for call in split.tool_calls] == ["run_shell_command"]
    flint = PromptOutputOwner("flint", tools=EnvelopeTools("glm47", REQUEST))
    flint.observe(
        GenerationOutput(
            text=GLM,
            new_text=GLM,
            prompt_parse_context=PromptParseContext("", None, None, False, "flint"),
        )
    )
    assert type(flint.parser.grammar).__name__ == "ContentToolGrammar"
    native = owner()
    assert native.observe(GenerationOutput(text="x", channel="content")) is None
