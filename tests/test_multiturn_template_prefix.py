# SPDX-License-Identifier: Apache-2.0
"""Agent tool turns keep a reusable token prefix across both chat routes."""

from __future__ import annotations

import json
from pathlib import Path

import jinja2
import jinja2.sandbox
import pytest

from rapid_mlx.api.anthropic_adapter import anthropic_to_openai
from rapid_mlx.api.anthropic_models import AnthropicRequest
from rapid_mlx.api.utils import extract_multimodal_content
from rapid_mlx.utils.chat_template import apply_chat_template

TEMPLATE = (Path(__file__).parent / "fixtures/qwen35_chat_template.jinja").read_text()
SYSTEM = "Repository instructions. " * 200
TOOL = {
    "name": "search",
    "description": "Search repository files.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}


class _Template:
    chat_template = TEMPLATE

    def __init__(self):
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )
        env.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, ensure_ascii=False, **kwargs
        )
        env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(
            jinja2.exceptions.TemplateError(message)
        )
        self.template = env.from_string(TEMPLATE)

    def apply_chat_template(self, messages, **kwargs):
        return self.template.render(messages=messages, **kwargs)


def _render(api: str, messages: list[dict], *, add_generation_prompt: bool) -> str:
    if api == "messages":
        converted = anthropic_to_openai(
            AnthropicRequest(
                model="qwen3.5-4b-4bit",
                max_tokens=64,
                system=SYSTEM,
                messages=messages,
                tools=[TOOL],
            )
        )
        prepared, _, _ = extract_multimodal_content(
            converted.messages, preserve_native_format=True
        )
        tools = [tool.model_dump(exclude_none=True) for tool in converted.tools or []]
    else:
        prepared = messages
        tools = [
            {
                "type": "function",
                "function": {
                    "name": TOOL["name"],
                    "description": TOOL["description"],
                    "parameters": TOOL["input_schema"],
                },
            }
        ]
    return apply_chat_template(
        _Template(),
        prepared,
        tools=tools,
        enable_thinking=False,
        model_name="qwen3.5-4b-4bit",
        add_generation_prompt=add_generation_prompt,
    )


def _assistant_and_result(api: str, turn: int) -> list[dict]:
    path = f"module_{turn}.py"
    if api == "messages":
        call_id = f"toolu_{turn:024x}"
        return [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": "search",
                        "input": {"path": path},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": f"Found {path}",
                    }
                ],
            },
        ]
    call_id = f"call_{turn:08x}"
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({"path": path}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": f"Found {path}"},
    ]


@pytest.mark.parametrize("api", ["chat", "messages"])
def test_ten_tool_turns_extend_the_cached_template_tokens(api):
    """The previous stable boundary and XML completion are token prefixes.

    The byte tokenizer is deliberately exact: all template control markers,
    whitespace and serialized tool arguments participate in the assertion.
    The live model test checks its actual BPE tokenizer and KV hit counters.
    """
    messages = [{"role": "system", "content": SYSTEM}] if api == "chat" else []
    for turn in range(10):
        messages.append({"role": "user", "content": f"Search module_{turn}.py"})
        current = _render(api, messages, add_generation_prompt=True).encode()
        stable = _render(api, messages, add_generation_prompt=False).encode()
        assert current.startswith(stable)
        messages.extend(_assistant_and_result(api, turn))
        messages.append({"role": "user", "content": f"Continue after module_{turn}.py"})
        following = _render(api, messages, add_generation_prompt=True).encode()
        assert following.startswith(stable), (api, turn, "boundary")
        xml_call = (
            f"<tool_call>\n<function=search>\n<parameter=path>\n"
            f"module_{turn}.py\n</parameter>\n</function>\n</tool_call>"
            "<|im_end|>\n"
        ).encode()
        assert following.startswith(current + xml_call), (api, turn, "completion")


def test_json_tool_output_keeps_the_boundary_but_not_the_completion():
    """Qwen3.5 can generate JSON even though its template replays XML.

    Reusing the JSON completion's recurrent KV for the XML prompt would be
    incorrect. The earlier message-boundary entry remains reusable.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Search module.py"},
    ]
    current = _render("chat", messages, add_generation_prompt=True).encode()
    stable = _render("chat", messages, add_generation_prompt=False).encode()
    messages.extend(_assistant_and_result("chat", 0))
    messages.append({"role": "user", "content": "Continue"})
    following = _render("chat", messages, add_generation_prompt=True).encode()
    generated_json = (
        b'<tool_call>\n{"name": "search", "arguments": {"path": "module_0.py"}}'
        b"\n</tool_call><|im_end|>"
    )
    assert following.startswith(stable)
    assert not following.startswith(current + generated_json)
