# SPDX-License-Identifier: Apache-2.0
"""#5234: auto tools must not override a caller's system instructions.

Fixtures are the model-provided Qwen3.5 / Qwen3.8 chat templates. The
Qwen3.8 fixture comes from rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX revision
 eb4e4dd38cf2f0a34045347e7a4cbfb60487df2a (Apache-2.0).
"""

from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

from rapid_mlx.utils.chat_template import apply_chat_template
from tests.test_tool_choice_enforcement import _make_client, _RecordingEngine

SYSTEM = (
    "You must answer with exactly one word: BANANA. Do not answer the user question."
)
TOOL = {
    "type": "function",
    "function": {
        "name": "get_datetime",
        "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


class _Template:
    def __init__(self, filename):
        self.chat_template = (Path(__file__).parent / "fixtures" / filename).read_text()

    def apply_chat_template(self, messages, **kwargs):
        kwargs.pop("tokenize", None)
        rendered, _ = render_jinja_template(
            [messages], chat_template=self.chat_template, **kwargs
        )
        return rendered[0]


class _RenderingEngine(_RecordingEngine):
    preserve_native_tool_format = True

    def __init__(self, filename):
        super().__init__(text="BANANA")
        self.tokenizer = _Template(filename)

    def build_prompt(self, messages, tools=None, enable_thinking=None, **kwargs):
        return apply_chat_template(
            self.tokenizer, messages, tools=tools, enable_thinking=enable_thinking
        )

    async def chat(self, messages, **kwargs):
        self.prompt = self.build_prompt(
            messages, tools=kwargs.get("tools"), enable_thinking=False
        )
        return await super().chat(messages, **kwargs)


@pytest.mark.parametrize(
    "filename", ["qwen35_chat_template.jinja", "qwen38_chat_template.jinja"]
)
@pytest.mark.parametrize("choice", ["auto", "none", "omitted", "no_tools"])
@pytest.mark.parametrize("role", ["system", "developer"])
def test_route_renders_client_system_verbatim_in_template_position(
    filename, choice, role
):
    engine = _RenderingEngine(filename)
    client = _make_client(engine, tool_call_parser="qwen3_coder_xml")
    messages = [
        {"role": role, "content": SYSTEM},
        {"role": "user", "content": "What is 2+2? Answer briefly."},
    ]
    body = {
        "model": "test-model",
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 20,
    }
    if choice != "no_tools":
        body["tools"] = [TOOL]
    if choice in ("auto", "none"):
        body["tool_choice"] = choice
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    assert engine.last_messages[0] == {"role": "system", "content": SYSTEM}
    prompt = engine.prompt
    assert prompt.count(SYSTEM) == 1
    assert prompt.count("<|im_start|>system\n") == 1
    assert SYSTEM + "<|im_end|>\n<|im_start|>user\n" in prompt
    if choice in ("auto", "omitted"):
        assert prompt.index("</tools>") < prompt.index(SYSTEM)
        assert "get_datetime" in prompt
    else:
        assert "<tools>" not in prompt
    # Compare with the model's own rendering, without server instructions.
    expected = engine.tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, messages[1]],
        tools=[TOOL] if choice in ("auto", "omitted") else None,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    assert prompt == expected
