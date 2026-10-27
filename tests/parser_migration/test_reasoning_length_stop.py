# SPDX-License-Identifier: Apache-2.0
"""A length stop before the reasoning end marker (#5295), in all six modes.

The unfinished output stays in reasoning and content stays empty: no notice
and no reasoning tail is copied into content, buffered or streaming. Chat
reports ``finish_reason="length"``, Responses ``status="incomplete"`` and
Messages ``stop_reason="max_tokens"``. The reasoning block is either opened by
the output or, with no opener, by the rendered prompt; the request-local owner
resolves the latter from the prompt's initial channel.
"""

import pytest

from rapid_mlx.prompt_parse_context import PromptParseContext
from tests.parser_migration.harness import route

TRACE = "Step one: 17 * 20 = 340. Step two: 17 * 3 ="
# Rendered Qwen thinking prompt: the assistant turn opens ``<think>``.
THINKING_PROMPT = PromptParseContext("<think>\n", None, "reasoning", True, "chatml")
# Rendered thinking-off prompt: the empty block is already closed.
CLOSED_PROMPT = PromptParseContext(
    "<think>\n\n</think>\n\n", None, "content", True, "chatml"
)

SURFACES = ["chat", "responses", "messages"]
SPLITS = [1, 7, 1000]


@pytest.fixture(autouse=True)
def _retired_notice_knobs_on(monkeypatch):
    # The former opt-out knobs must not bring a notice back.
    monkeypatch.setenv("RAPID_MLX_REASONING_RESCUE", "on")
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")


def _run(raw, surface, stream, split, tools_mode, context):
    return route(
        "qwen3",
        "hermes",
        raw,
        surface,
        stream,
        chunks=[raw[i : i + split] for i in range(0, len(raw), split)],
        finish="length",
        tools_mode=tools_mode,
        prompt_context=context,
        capture=True,
    )


def _terminal(result, surface, stream):
    """Return the transport's structured stop signal."""
    packets = result["packets"]
    if surface == "chat":
        reasons = [c.get("finish_reason") for p in packets for c in p["choices"]]
        return [r for r in reasons if r][-1]
    if surface == "responses":
        if not stream:
            return packets[0]["status"]
        terminal = [p for p in packets if p.get("type", "").startswith("response.")]
        return [
            p["response"]["status"]
            for p in terminal
            if p["type"] in {"response.completed", "response.incomplete"}
        ][-1]
    if not stream:
        return packets[0]["stop_reason"]
    return [
        p["delta"]["stop_reason"] for p in packets if p.get("type") == "message_delta"
    ][-1]


EXPECTED_STOP = {"chat": "length", "responses": "incomplete", "messages": "max_tokens"}


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("tools_mode", ["absent", "enabled"])
@pytest.mark.parametrize(
    "raw,context",
    [
        ("<think>" + TRACE, None),
        ("<think>" + TRACE, THINKING_PROMPT),
        (TRACE, THINKING_PROMPT),
    ],
    ids=["output-opened", "output-opened-prompt", "prompt-opened"],
)
def test_unfinished_reasoning_stays_reasoning(
    raw, context, tools_mode, surface, stream, split
):
    result = _run(raw, surface, stream, split, tools_mode, context)

    assert result["content"] == ""
    assert result["content_deltas"] == 0
    assert result["reasoning"] == TRACE
    assert result["calls"] == []
    assert _terminal(result, surface, stream) == EXPECTED_STOP[surface]


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("tools_mode", ["absent", "enabled"])
def test_answer_after_closed_prompt_block_stays_content(
    tools_mode, surface, stream, split
):
    """Control: a truncated answer after a closed block is content."""
    result = _run(TRACE, surface, stream, split, tools_mode, CLOSED_PROMPT)

    assert result["content"] == TRACE
    assert result["reasoning"] == ""
    assert _terminal(result, surface, stream) == EXPECTED_STOP[surface]


def test_unknown_prompt_state_keeps_existing_tool_free_policy():
    """Without rendered prompt state a tool-free request keeps its parser.

    Selection needs the engine's current-turn initial channel; a request
    flag alone is not evidence of the rendered prompt state.
    """
    from rapid_mlx.config import reset_config
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.parsing.prompt_owner import PromptOutputOwner

    cfg = reset_config()
    cfg.reasoning_parser_name = "qwen3"
    try:
        request = {"messages": [{"role": "user", "content": "hi"}]}
        owner = PromptOutputOwner.for_request(cfg, request, thinking=True)
        assert owner.reasoning_only
        assert owner.tools is None
        output = GenerationOutput(text=TRACE, new_text=TRACE)
        assert owner.observe(output) is None
        assert not owner.active
        owner = PromptOutputOwner.for_request(cfg, request, thinking=False)
        assert not owner.reasoning_only
    finally:
        reset_config()
