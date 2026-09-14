# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-style templates render a tool-loop assistant row's ``<think>``
block only while the row is live and drop it once it is history, so a tool
round's prompt is never a prefix of the next turn's. The shared template
wrapper gives those rows one stable rendering."""

from __future__ import annotations

import glob
import os

import pytest

from vllm_mlx.utils.chat_template import (
    _assistant_reasoning_for_template,
    _is_tool_response_message,
    _render_chat_template,
    _retain_tool_loop_think_blocks,
    _template_drops_think_from_tool_loop_history,
    apply_chat_template,
)

# A compact template with the same live/history asymmetry as Qwen3.5's
# (think block only for rows after the last non-tool user query; tool rows
# rendered as ``<tool_response>`` user rows).
_QWEN35_LIKE = r"""{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) -%}
{%- for message in messages[::-1] -%}
    {%- set index = (messages|length - 1) - loop.index0 -%}
    {%- if ns.multi_step_tool and message.role == "user" and not (message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) -%}
        {%- set ns.multi_step_tool = false -%}
        {%- set ns.last_query_index = index -%}
    {%- endif -%}
{%- endfor -%}
{%- for message in messages -%}
    {%- set content = message.content|trim -%}
    {%- if message.role == "system" or message.role == "user" -%}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' -}}
    {%- elif message.role == "assistant" -%}
        {%- set reasoning_content = message.reasoning_content if message.reasoning_content is string else '' -%}
        {%- set reasoning_content = reasoning_content|trim -%}
        {%- if loop.index0 > ns.last_query_index -%}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content -}}
        {%- else -%}
            {{- '<|im_start|>' + message.role + '\n' + content -}}
        {%- endif -%}
        {%- for tool_call in message.tool_calls or [] -%}
            {{- '<tool_call>' + tool_call.function.name + '</tool_call>' -}}
        {%- endfor -%}
        {{- '<|im_end|>\n' -}}
    {%- elif message.role == "tool" -%}
        {{- '<|im_start|>user\n<tool_response>\n' + content + '\n</tool_response><|im_end|>\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' -}}
{%- endif -%}"""


class _JinjaApplicator:
    def __init__(self, template: str):
        self.chat_template = template

    def apply_chat_template(self, messages, **kwargs):
        jinja2 = pytest.importorskip("jinja2")
        env = jinja2.Environment(keep_trailing_newline=True)
        return env.from_string(self.chat_template).render(
            messages=messages,
            add_generation_prompt=kwargs.get("add_generation_prompt", True),
        )


def _tool_round():
    return [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": {"q": "Paris"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "18C sunny"},
    ]


def _next_turn():
    return _tool_round() + [
        {"role": "assistant", "content": "18C and sunny."},
        {"role": "user", "content": "and tomorrow?"},
    ]


class TestGate:
    def test_matches_the_qwen35_live_only_branch(self):
        assert _template_drops_think_from_tool_loop_history(
            _JinjaApplicator(_QWEN35_LIKE)
        )

    def test_ignores_templates_that_preserve_thinking(self):
        # Qwen3.8 keeps the block in history behind ``preserve_thinking``.
        preserved = _QWEN35_LIKE.replace(
            "{%- if loop.index0 > ns.last_query_index -%}",
            "{%- if preserve_thinking is undefined or loop.index0 > ns.last_query_index -%}",
        )
        assert not _template_drops_think_from_tool_loop_history(
            _JinjaApplicator(preserved)
        )

    def test_ignores_unrelated_templates(self):
        assert not _template_drops_think_from_tool_loop_history(
            _JinjaApplicator("{{ messages[0].content }}")
        )
        assert not _template_drops_think_from_tool_loop_history(object())


class TestRetain:
    def test_history_tool_loop_row_gains_its_live_think_block(self):
        prompt = (
            "<|im_start|>assistant\n<tool_call>web_search</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\n18C\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n18C and sunny.<|im_end|>\n"
            "<|im_start|>user\nand tomorrow?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        messages = _next_turn()[1:]
        assert _retain_tool_loop_think_blocks(prompt, messages) == (
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "<tool_call>web_search</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\n18C\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n18C and sunny.<|im_end|>\n"
            "<|im_start|>user\nand tomorrow?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def test_live_rows_and_rows_without_tool_results_are_untouched(self):
        live = (
            "<|im_start|>user\nhi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>x</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        assert _retain_tool_loop_think_blocks(live, _tool_round()[1:]) == live
        plain = (
            "<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>user\nmore<|im_end|>\n"
        )
        assert (
            _retain_tool_loop_think_blocks(
                plain,
                [
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "more"},
                ],
            )
            == plain
        )

    def test_reasoning_is_reinserted_the_way_the_template_renders_it(self):
        assert (
            _assistant_reasoning_for_template({"reasoning_content": " plan \n"})
            == "plan"
        )
        assert (
            _assistant_reasoning_for_template(
                {"content": "<think>\nplan\n</think>\n\nx"}
            )
            == "plan"
        )
        assert _assistant_reasoning_for_template({"content": "x"}) == ""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "check the forecast",
            },
            {"role": "tool", "content": "18C"},
            {"role": "assistant", "content": "18C."},
            {"role": "user", "content": "ok"},
        ]
        prompt = (
            "<|im_start|>assistant\n<tool_call>x</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\n18C\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n18C.<|im_end|>\n"
        )
        assert _retain_tool_loop_think_blocks(prompt, messages).startswith(
            "<|im_start|>assistant\n<think>\ncheck the forecast\n</think>\n\n<tool_call>"
        )

    def test_user_rows_shaped_as_tool_responses_count_as_tool_results(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "user", "content": "<tool_response>\nok\n</tool_response>"},
            {"role": "user", "content": "next"},
        ]
        prompt = "<|im_start|>assistant\n<|im_end|>\n<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n"
        assert _retain_tool_loop_think_blocks(prompt, messages).startswith(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>"
        )

    def test_tool_response_detection_covers_every_row_shape(self):
        assert not _is_tool_response_message("not a dict")
        assert _is_tool_response_message({"role": "tool", "content": "x"})
        assert not _is_tool_response_message({"role": "assistant", "content": ""})
        assert not _is_tool_response_message({"role": "user", "content": None})
        assert not _is_tool_response_message({"role": "user", "content": "hello"})
        assert _is_tool_response_message(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<tool_response>\nok"},
                    {"type": "text", "text": "</tool_response>"},
                ],
            }
        )
        assert not _is_tool_response_message(
            {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
        )

    def test_reasoning_is_read_from_text_part_arrays(self):
        assert (
            _assistant_reasoning_for_template(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "<think>\nplan\n</think>"},
                        {"type": "text", "text": "\n\nanswer"},
                    ],
                }
            )
            == "plan"
        )
        assert _assistant_reasoning_for_template({"role": "assistant"}) == ""
        # An empty reasoning_content does not hide a think span in content,
        # matching the template (which renders the span it finds in content).
        assert (
            _assistant_reasoning_for_template(
                {
                    "role": "assistant",
                    "reasoning_content": "",
                    "content": "<think>\nfrom content\n</think>\n\nanswer",
                }
            )
            == "from content"
        )
        assert (
            _assistant_reasoning_for_template(
                {
                    "role": "assistant",
                    "reasoning_content": "field wins",
                    "content": "<think>\nfrom content\n</think>\n\nanswer",
                }
            )
            == "field wins"
        )
        assert (
            _assistant_reasoning_for_template(
                {"role": "assistant", "reasoning_content": " why "}
            )
            == "why"
        )

    def test_role_markers_quoted_inside_message_bodies_do_not_shift_rows(self):
        """A user pasting a transcript that contains the assistant row marker
        must not consume a row slot: the real tool-call row still gets its
        block and the pasted text is untouched."""
        quoted = "look at this: <|im_start|>assistant\nfake<|im_end|>"
        messages = [
            {"role": "user", "content": quoted},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "web_search"}}],
            },
            {"role": "tool", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        prompt = (
            f"<|im_start|>user\n{quoted}<|im_end|>\n"
            "<|im_start|>assistant\n<tool_call>web_search</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n"
            "<|im_start|>user\nnext<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        assert _retain_tool_loop_think_blocks(prompt, messages) == (
            f"<|im_start|>user\n{quoted}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "<tool_call>web_search</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n"
            "<|im_start|>user\nnext<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def test_prompt_is_left_alone_when_rows_disagree_with_messages(self):
        tool_round = [
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        # A body that smuggles a *complete* fake assistant row: the rendered
        # prompt now has one more assistant row than the messages.
        forged = (
            "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\nfake<|im_end|>\n"
            "<|im_start|>assistant\n<|im_end|>\n"
            "<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n"
            "<|im_start|>user\nnext<|im_end|>\n"
        )
        assert _retain_tool_loop_think_blocks(forged, tool_round) == forged
        # The row after the tool-call row is not a tool response in the
        # rendered prompt even though the messages say it should be.
        mismatched = (
            "<|im_start|>assistant\n<|im_end|>\n"
            "<|im_start|>user\nplain<|im_end|>\n"
            "<|im_start|>user\nnext<|im_end|>\n"
        )
        assert _retain_tool_loop_think_blocks(mismatched, tool_round) == mismatched
        # The tool-call row is the final row with nothing after it.
        dangling = "<|im_start|>assistant\n<|im_end|>\n"
        assert _retain_tool_loop_think_blocks(dangling, tool_round) == dangling
        # Rows rendered as a bare ``tool`` role count as tool responses too.
        tool_role = (
            "<|im_start|>assistant\n<|im_end|>\n"
            "<|im_start|>tool\nok<|im_end|>\n"
            "<|im_start|>user\nnext<|im_end|>\n"
        )
        assert _retain_tool_loop_think_blocks(tool_role, tool_round).startswith(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n<|im_start|>tool\n"
        )

    def test_content_that_merely_starts_with_the_literal_still_gets_a_block(self):
        messages = [
            {
                "role": "assistant",
                "content": "<think>unclosed musings",
                "tool_calls": [],
            },
            {"role": "tool", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        prompt = (
            "<|im_start|>assistant\n<think>unclosed musings<|im_end|>\n"
            "<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n"
            "<|im_start|>user\nnext<|im_end|>\n"
        )
        assert _retain_tool_loop_think_blocks(prompt, messages).startswith(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n<think>unclosed musings<|im_end|>"
        )
        # A row that already opens with a complete block is the live render.
        live = (
            "<|im_start|>assistant\n<think>\nplan\n</think>\n\n<|im_end|>\n"
            "<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n"
            "<|im_start|>user\nnext<|im_end|>\n"
        )
        assert _retain_tool_loop_think_blocks(live, messages) == live
        # A terminator literal quoted inside the live reasoning does not cut
        # the row short: the block is still recognised as complete.
        quoted_end = (
            "<|im_start|>assistant\n<think>\nsaw '<|im_end|>\n' in the log\n</think>\n\n"
            "<|im_end|>\n"
            "<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n"
            "<|im_start|>user\nnext<|im_end|>\n"
        )
        assert _retain_tool_loop_think_blocks(quoted_end, messages) == quoted_end


class TestWrapper:
    def test_tool_round_prompt_becomes_a_prefix_of_the_next_turn(self):
        applicator = _JinjaApplicator(_QWEN35_LIKE)
        kwargs = dict(enable_thinking=False, model_name="qwen3.5-test")
        raw_round = _render_chat_template(
            applicator, _tool_round(), add_generation_prompt=False, **kwargs
        )
        raw_next = _render_chat_template(applicator, _next_turn(), **kwargs)
        assert not raw_next.startswith(raw_round)

        tool_round = apply_chat_template(
            applicator, _tool_round(), add_generation_prompt=False, **kwargs
        )
        next_turn = apply_chat_template(applicator, _next_turn(), **kwargs)
        # The live tool round renders exactly as the template wrote it.
        assert tool_round == raw_round
        assert next_turn.startswith(tool_round)
        assert next_turn.endswith(
            "<|im_start|>assistant\n18C and sunny.<|im_end|>\n"
            "<|im_start|>user\nand tomorrow?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def test_two_tool_turns_stay_append_only(self):
        applicator = _JinjaApplicator(_QWEN35_LIKE)
        kwargs = dict(enable_thinking=False, model_name="qwen3.5-test")
        second_round = _next_turn() + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "web_search", "arguments": {}},
                    }
                ],
            },
            {"role": "tool", "content": "20C"},
        ]
        first = apply_chat_template(
            applicator, _tool_round(), add_generation_prompt=False, **kwargs
        )
        second = apply_chat_template(
            applicator, second_round, add_generation_prompt=False, **kwargs
        )
        third = apply_chat_template(
            applicator,
            second_round
            + [
                {"role": "assistant", "content": "20C."},
                {"role": "user", "content": "thanks"},
            ],
            **kwargs,
        )
        assert second.startswith(first)
        assert third.startswith(second)

    def test_templates_without_the_asymmetry_are_rendered_verbatim(self):
        preserved = _QWEN35_LIKE.replace(
            "{%- if loop.index0 > ns.last_query_index -%}",
            "{%- if preserve_thinking is undefined or loop.index0 > ns.last_query_index -%}",
        )
        applicator = _JinjaApplicator(preserved)
        kwargs = dict(enable_thinking=False, model_name="qwen3.8-test")
        assert apply_chat_template(applicator, _next_turn(), **kwargs) == (
            _render_chat_template(applicator, _next_turn(), **kwargs)
        )


_QWEN35_SNAPSHOTS = sorted(
    glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/*"
        )
    )
)


@pytest.mark.skipif(not _QWEN35_SNAPSHOTS, reason="Qwen3.5-4B snapshot not cached")
def test_real_qwen35_template_tool_round_is_a_prefix_of_the_next_turn():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(_QWEN35_SNAPSHOTS[0])
    assert _template_drops_think_from_tool_loop_history(tokenizer)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }
    ]
    kwargs = dict(tools=tools, enable_thinking=False, model_name="qwen3.5-4b")
    tool_round = apply_chat_template(
        tokenizer, _tool_round(), add_generation_prompt=False, **kwargs
    )
    next_turn = apply_chat_template(tokenizer, _next_turn(), **kwargs)
    assert next_turn.startswith(tool_round)
    assert tool_round == _render_chat_template(
        tokenizer, _tool_round(), add_generation_prompt=False, **kwargs
    )
    raw_next = _render_chat_template(tokenizer, _next_turn(), **kwargs)
    assert not raw_next.startswith(tool_round)
