# SPDX-License-Identifier: Apache-2.0
"""Thinking-off compatibility for the byte-pinned Flint chat template."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rapid_mlx.api.models import ChatCompletionRequest
from rapid_mlx.service.helpers import _resolve_enable_thinking
from rapid_mlx.utils.chat_template import apply_chat_template

TEMPLATE = (Path(__file__).parent / "fixtures/flint_chat_template.jinja").read_text()
MESSAGES = [{"role": "user", "content": "What is 2 + 2?"}]


@pytest.fixture
def tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        chat_template=TEMPLATE,
    )


@pytest.mark.parametrize("thinking", [False, True, None])
@pytest.mark.parametrize("generation", [False, True])
@pytest.mark.parametrize(
    "model", ["flint-40b-a2.5b-4bit", "RecursiveSI/flint-int4", "local-copy"]
)
def test_flint_thinking_prefix(tokenizer, thinking, generation, model):
    original = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=generation
    )
    rendered = apply_chat_template(
        tokenizer,
        MESSAGES,
        enable_thinking=thinking,
        add_generation_prompt=generation,
        model_name=model,
    )
    assert rendered == original
    assert tokenizer.chat_template == TEMPLATE


@pytest.mark.parametrize(
    "no_thinking,requested", [(False, False), (True, True), (True, None)]
)
def test_request_and_server_switches_reach_flint(
    tokenizer, monkeypatch, no_thinking, requested
):
    monkeypatch.setattr(
        "rapid_mlx.service.helpers.get_config",
        lambda: SimpleNamespace(no_thinking=no_thinking),
    )
    request = ChatCompletionRequest(
        messages=MESSAGES,
        chat_template_kwargs={"enable_thinking": requested}
        if requested is not None
        else None,
    )
    thinking = _resolve_enable_thinking(request)
    assert thinking is False
    assert apply_chat_template(
        tokenizer,
        MESSAGES,
        enable_thinking=thinking,
        chat_template_kwargs=request.chat_template_kwargs,
    ).endswith("<|assistant|>")


def test_flint_tool_history_and_request_isolation(tokenizer):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = MESSAGES + [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Check first.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "4", "tool_call_id": "call_1"},
    ]
    baseline = apply_chat_template(
        tokenizer, messages, tools=tools, enable_thinking=True
    )
    for thinking in [False, True, False, None]:
        rendered = apply_chat_template(
            tokenizer, messages, tools=tools, enable_thinking=thinking
        )
        assert rendered == baseline
        assert "<think>Check first.</think>" in rendered
        assert "<|observation|><tool_response>4</tool_response>" in rendered
    assert tokenizer.chat_template == TEMPLATE


@pytest.mark.parametrize(
    "template",
    [
        "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}",
        TEMPLATE + "\n{# custom template #}",
    ],
)
def test_unrelated_and_custom_templates_unchanged(tokenizer, template):
    tokenizer.chat_template = template
    expected = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True
    )
    assert apply_chat_template(tokenizer, MESSAGES, enable_thinking=False) == expected


def test_explicit_template_override_preserved(tokenizer):
    custom = "{{ messages[0].content }}<|assistant|>"
    assert (
        apply_chat_template(
            tokenizer,
            MESSAGES,
            enable_thinking=False,
            chat_template_kwargs={"chat_template": custom},
        )
        == "What is 2 + 2?<|assistant|>"
    )


def test_missing_template_and_non_generation_suffix_unchanged():
    stub = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "fallback")
    assert apply_chat_template(stub, MESSAGES, enable_thinking=False) == "fallback"
    stub.chat_template = TEMPLATE
    assert apply_chat_template(stub, MESSAGES, enable_thinking=False) == "fallback"


def test_client_tokenize_override_cannot_bypass_thinking_off(tokenizer):
    # The shared renderer always returns text; encoding happens downstream.
    rendered = apply_chat_template(
        tokenizer,
        MESSAGES,
        enable_thinking=False,
        chat_template_kwargs={"tokenize": True},
    )
    assert isinstance(rendered, str)
    assert rendered.endswith("<|assistant|>")


def test_null_template_override_uses_configured_flint_template(tokenizer):
    rendered = apply_chat_template(
        tokenizer,
        MESSAGES,
        enable_thinking=False,
        chat_template_kwargs={"chat_template": None},
    )
    assert rendered.endswith("<|assistant|>")
