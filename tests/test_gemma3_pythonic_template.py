# SPDX-License-Identifier: Apache-2.0
"""Gemma 3 profile and tool-template contracts without model weights."""

import json

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from rapid_mlx.model_aliases import list_profiles, resolve_profile
from rapid_mlx.tool_parsers import PythonicToolParser
from rapid_mlx.utils.chat_template import apply_chat_template
from rapid_mlx.utils.chat_template_registry import (
    bundled_chat_template,
    resolve_profile_chat_template,
)

# Measured on real weights: 12B follows the bracketed call-list format for
# single, parallel and streamed calls; 1B and 4B ignore it (fenced or
# positional calls), so they do not advertise tool support.
TOOL_CAPABLE = {"gemma3-12b-4bit", "gemma3-27b-4bit", "gemma3-27b-qat-4bit"}
NO_TOOLS = {"gemma3-1b-4bit", "gemma3-1b-qat-4bit", "gemma3-4b-qat-4bit"}


def test_every_gemma3_alias_is_classified():
    assert {a for a in list_profiles() if a.startswith("gemma3-")} == (
        TOOL_CAPABLE | NO_TOOLS
    )


@pytest.mark.parametrize("alias", sorted(TOOL_CAPABLE))
def test_alias_and_exact_repository(alias):
    profile = resolve_profile(alias)
    assert profile.tool_call_parser == "pythonic"
    assert profile.chat_template_id == "gemma3_pythonic"
    assert resolve_profile(profile.hf_path) == profile
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")),
        bos_token="<bos>",
    )
    assert resolve_profile_chat_template(tokenizer, alias)
    assert tokenizer.chat_template == bundled_chat_template("gemma3_pythonic")


@pytest.mark.parametrize("alias", sorted(NO_TOOLS))
def test_small_gemma3_aliases_do_not_advertise_tools(alias):
    profile = resolve_profile(alias)
    assert profile.tool_call_parser is None
    assert profile.chat_template_id is None


@pytest.fixture
def tokenizer():
    result = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")),
        bos_token="<bos>",
    )
    resolve_profile_chat_template(result, "gemma3-12b-4bit")
    return result


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


def test_tools_prompt(tokenizer):
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Weather?"},
        ],
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert text.startswith("<bos><start_of_turn>user\nBe helpful.\n\n")
    assert "python list of the function calls" in text
    assert "DO NOT USE MARKDOWN SYNTAX" in text
    assert '"name": "weather"' in text
    assert text.endswith("Weather?<end_of_turn>\n<start_of_turn>model\n")


@pytest.mark.parametrize("content", [None, "", "omitted"])
def test_tool_history(tokenizer, content):
    args = {
        "city": "Paris",
        "options": {"metric": True, "default": None},
        "days": [1, 2],
    }
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": json.dumps(args)},
            }
        ],
    }
    if content != "omitted":
        assistant["content"] = content
    messages = [
        {"role": "user", "content": "Weather?"},
        assistant,
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        {"role": "assistant", "content": "It is sunny."},
        {"role": "user", "content": "And tomorrow?"},
    ]
    text = apply_chat_template(
        tokenizer, messages, tools=TOOLS, add_generation_prompt=True
    )
    call_text = text.split("<start_of_turn>model\n", 1)[1].split("<end_of_turn>", 1)[0]
    parsed = PythonicToolParser().extract_tool_calls(call_text)
    assert parsed.tools_called
    assert json.loads(parsed.tool_calls[0]["arguments"]) == args
    assert (
        "<start_of_turn>user\n<tool_response>\nSunny</tool_response><end_of_turn>"
        in text
    )
    assert text.endswith("<start_of_turn>model\n")


def test_without_tools_and_multimodal(tokenizer):
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": [{"type": "text", "text": "Context"}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "Describe"}],
            },
        ],
        tokenize=False,
    )
    assert (
        text
        == "<bos><start_of_turn>user\nContext\n\n<start_of_image>Describe<end_of_turn>\n"
    )
    assert "Tools (functions)" not in text


def test_invalid_roles(tokenizer):
    with pytest.raises(Exception, match="Conversation roles must alternate"):
        tokenizer.apply_chat_template(
            [{"role": "assistant", "content": "Hello"}], tokenize=False
        )


@pytest.mark.parametrize("tool_calls", [[], None])
def test_empty_tool_calls_are_regular_assistant_turns(tokenizer, tool_calls):
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "tool_calls": tool_calls},
            {"role": "user", "content": "Continue"},
        ],
        tokenize=False,
    )
    assert "<start_of_turn>model\nHi<end_of_turn>" in text
    assert "[]" not in text
