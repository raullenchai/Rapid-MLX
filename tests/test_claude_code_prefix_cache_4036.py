# SPDX-License-Identifier: Apache-2.0
"""Issue #4036: Claude Code's per-turn token counter must not break the prefix.

Claude Code >= 2.1.287 appends ``{"role": "system", "content":
"<total_tokens>N tokens left</total_tokens>"}`` to ``messages`` on every
request. Hoisted into the leading system block, its per-request value changed
the front of the prompt each turn, so every tool-loop turn missed the hybrid
prefix cache (~17.6k tokens re-prefilled, ~25 s TTFT on qwen3.6-35b).

The Anthropic adapter now drops a system message whose whole text is that
counter. Everything else must convert exactly as on origin/main: the golden
file was captured from origin/main 9221ff773.
"""

from __future__ import annotations

import json
import pathlib

import jinja2
import jinja2.sandbox
import pytest

from rapid_mlx.api import anthropic_adapter
from rapid_mlx.api.anthropic_adapter import (
    _is_token_budget_counter,
    anthropic_to_openai,
)
from rapid_mlx.api.anthropic_models import AnthropicRequest
from rapid_mlx.api.utils import extract_multimodal_content
from rapid_mlx.config import get_config
from rapid_mlx.utils.chat_template import apply_chat_template

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CAPTURED = json.loads((FIXTURES / "claude_code_2_1_287_turns.json").read_text())
GOLDEN = json.loads((FIXTURES / "anthropic_system_merge_golden_main.json").read_text())


@pytest.fixture(params=[False, True], ids=["hoist-default", "relocate-flag"])
def relocate(request, monkeypatch):
    monkeypatch.setattr(get_config(), "relocate_mid_conversation_system", request.param)
    return request.param


class _Qwen35Template:
    """The checked-in Qwen3.5 chat template, rendered the way transformers does."""

    def __init__(self):
        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
        env.globals["raise_exception"] = raise_exception
        self.chat_template = (FIXTURES / "qwen35_chat_template.jinja").read_text()
        self._template = env.from_string(self.chat_template)

    def apply_chat_template(self, messages, **kwargs):
        kwargs.setdefault("add_generation_prompt", True)
        return self._template.render(messages=messages, **kwargs)


def _prompt(request: dict, *, add_generation_prompt: bool) -> str:
    """Render a /v1/messages body to the prompt string the engine prefills."""
    openai_request = anthropic_to_openai(AnthropicRequest(**request))
    messages, _, _ = extract_multimodal_content(
        openai_request.messages, preserve_native_format=True
    )
    tools = [t.model_dump(exclude_none=True) for t in openai_request.tools or []]
    return apply_chat_template(
        _Qwen35Template(),
        messages,
        tools=tools,
        enable_thinking=False,
        model_name="qwen3.6-35b-4bit",
        add_generation_prompt=add_generation_prompt,
    )


def _next_turn(previous: dict, counter: str, *, keep_old_counter: bool) -> dict:
    """Synthesise Claude Code's following request: one more Read round."""
    history = list(previous["messages"])
    if not keep_old_counter:
        history = history[:-1]
    n = len(history)
    history += [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"toolu_{n:08x}",
                    "name": "Read",
                    "input": {"file_path": "/work/fixture/wordutil/__init__.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"toolu_{n:08x}",
                    "content": "1\tfrom .count import count_vowels\n",
                }
            ],
        },
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": counter,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]
    return {**previous, "messages": history}


class TestCapturedTurnsSharePrefix:
    def test_fixture_is_the_counter_shape(self):
        turn1, turn2 = CAPTURED["turns"]
        assert turn1["messages"][-1]["role"] == "system"
        assert "total_tokens" not in json.dumps(turn1)
        assert turn2["messages"][-1]["content"][0]["text"] == (
            "<total_tokens>14982239 tokens left</total_tokens>"
        )
        assert turn2["system"] == turn1["system"]
        assert turn2["tools"] == turn1["tools"]

    def test_turn_one_prompt_is_a_prefix_of_turn_two(self, relocate):
        turn1, turn2 = CAPTURED["turns"]
        first = _prompt(turn1, add_generation_prompt=False)
        second = _prompt(turn2, add_generation_prompt=True)
        assert second.startswith(first)
        assert "total_tokens" not in second

    def test_red_without_the_drop(self, relocate, monkeypatch):
        """The prefix assertion above can fail: origin/main's hoist breaks it."""
        monkeypatch.setattr(
            anthropic_adapter,
            "_is_token_budget_counter",
            lambda messages, index: False,
        )
        turn1, turn2 = CAPTURED["turns"]
        first = _prompt(turn1, add_generation_prompt=False)
        second = _prompt(turn2, add_generation_prompt=True)
        assert not second.startswith(first)
        assert "<total_tokens>14982239 tokens left</total_tokens>" in second

    @pytest.mark.parametrize("keep_old_counter", [False, True])
    def test_later_turns_keep_extending_the_prefix(self, relocate, keep_old_counter):
        turn2 = CAPTURED["turns"][1]
        turn3 = _next_turn(
            turn2,
            "<total_tokens>14966103 tokens left</total_tokens>",
            keep_old_counter=keep_old_counter,
        )
        turn4 = _next_turn(
            turn3,
            "<total_tokens>14950871 tokens left</total_tokens>",
            keep_old_counter=keep_old_counter,
        )
        for earlier, later in ((turn2, turn3), (turn3, turn4)):
            assert _prompt(later, add_generation_prompt=True).startswith(
                _prompt(earlier, add_generation_prompt=False)
            )


@pytest.mark.parametrize(
    "case",
    GOLDEN["cases"],
    ids=[f"{c['name']}-relocate={c['relocate']}" for c in GOLDEN["cases"]],
)
def test_requests_without_the_counter_convert_as_on_main(case, monkeypatch):
    monkeypatch.setattr(
        get_config(), "relocate_mid_conversation_system", case["relocate"]
    )
    converted = anthropic_to_openai(AnthropicRequest(**case["request"])).messages
    assert [m.model_dump(exclude_none=True) for m in converted] == case["expected"]


def _counter_at(messages: list[dict], index: int) -> bool:
    request = AnthropicRequest(model="m", max_tokens=8, messages=messages)
    return _is_token_budget_counter(request.messages, index)


COUNTER = "<total_tokens>14982239 tokens left</total_tokens>"
USER = {"role": "user", "content": "go"}
ASSISTANT = {"role": "assistant", "content": "ok"}


def _block(text: str, **extra) -> dict:
    return {"role": "system", "content": [{"type": "text", "text": text, **extra}]}


@pytest.mark.parametrize(
    "text",
    [
        COUNTER,
        "  <total_tokens> 1,234,567 tokens left </total_tokens>\n",
        "<total_tokens>1 token left</total_tokens>",
    ],
)
def test_counter_shapes_are_recognised(text):
    assert _counter_at([USER, {"role": "system", "content": text}], 1)
    assert _counter_at([USER, _block(text, cache_control={"type": "ephemeral"})], 1)


def test_counter_kept_in_history_before_the_assistant_reply_is_recognised():
    """Later turns keep the old counter where it was: before that turn's reply."""
    messages = [USER, _block(COUNTER), ASSISTANT, USER, _block(COUNTER)]
    assert _counter_at(messages, 1)
    assert _counter_at(messages, 4)


@pytest.mark.parametrize(
    "messages, index",
    [
        # Not Claude Code's position: a user turn follows.
        ([USER, _block(COUNTER), USER], 1),
        ([USER, {"role": "system", "content": COUNTER}, USER], 1),
        # Not the counter's shape.
        ([{"role": "user", "content": COUNTER}], 0),
        ([USER, _block("<total_tokens>see runbook</total_tokens>")], 1),
        ([USER, _block(COUNTER + " Wrap up now.")], 1),
        ([USER, _block("<total_tokens>tokens left</total_tokens>")], 1),
        (
            [
                USER,
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": COUNTER},
                        {"type": "text", "text": "Wrap up now."},
                    ],
                },
            ],
            1,
        ),
    ],
)
def test_anything_else_is_not_the_counter(messages, index):
    assert not _counter_at(messages, index)
