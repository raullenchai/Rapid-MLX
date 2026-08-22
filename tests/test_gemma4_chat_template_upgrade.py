# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for stale converted Gemma 4 chat templates."""

from __future__ import annotations

from hashlib import sha256

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from vllm_mlx.utils.chat_template import apply_chat_template
from vllm_mlx.utils.gemma4_chat_template import (
    _KNOWN_STALE_TEMPLATE_VARIANTS,
    _canonical_template,
    upgrade_stale_gemma4_chat_template,
)

_STALE_COMPACT = """
{%- macro format_argument(argument, escape_keys=True) -%}{%- endmacro -%}
{%- set ns = namespace(prev_message_type=None) -%}
<|tool_call>call:
"""

_STALE_FULL = (
    _STALE_COMPACT + "\n{%- if not enable_thinking | default(false) -%}{%- endif -%}\n"
)


def _tokenizer(template: str) -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = template
    return tokenizer


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "limit": {"type": ["integer", "null"]},
                    },
                    "required": ["command"],
                },
            },
        }
    ]


def _allow_fixture_template(monkeypatch, template: str, variant: str) -> None:
    monkeypatch.setitem(
        _KNOWN_STALE_TEMPLATE_VARIANTS,
        sha256(template.encode("utf-8")).hexdigest(),
        variant,
    )


def test_compact_and_full_stale_templates_select_matching_canonical_variant(
    monkeypatch,
) -> None:
    _allow_fixture_template(monkeypatch, _STALE_COMPACT, "compact")
    _allow_fixture_template(monkeypatch, _STALE_FULL, "full")
    compact = _tokenizer(_STALE_COMPACT)
    full = _tokenizer(_STALE_FULL)
    metadata_only = _tokenizer(_STALE_FULL)
    metadata_only.init_kwargs["name_or_path"] = "local/gemma-4-e2b"

    assert upgrade_stale_gemma4_chat_template(compact, "gemma-4-e2b-4bit")
    assert upgrade_stale_gemma4_chat_template(full, "gemma-4-26b-4bit")
    assert upgrade_stale_gemma4_chat_template(metadata_only)
    assert compact.chat_template == _canonical_template("compact")
    assert full.chat_template == _canonical_template("full")
    assert metadata_only.chat_template == _canonical_template("compact")


def test_current_canonical_and_unknown_custom_templates_are_preserved() -> None:
    canonical = _tokenizer(_canonical_template("compact"))
    custom = _tokenizer(_STALE_COMPACT + "\ncustom {{ messages }}")
    canonical_before = canonical.chat_template
    custom_before = custom.chat_template

    assert not upgrade_stale_gemma4_chat_template(canonical, "gemma-4-e2b")
    assert not upgrade_stale_gemma4_chat_template(custom, "qwen-custom")
    assert canonical.chat_template == canonical_before
    assert custom.chat_template == custom_before


def test_official_template_renders_null_and_normalized_openai_arguments(
    monkeypatch,
) -> None:
    pytest.importorskip("jinja2")
    _allow_fixture_template(monkeypatch, _STALE_FULL, "full")
    tokenizer = _tokenizer(_STALE_FULL)
    messages = [
        {"role": "user", "content": "List /tmp"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"ls /tmp","limit":null}',
                    },
                }
            ],
        },
    ]

    rendered = apply_chat_template(
        tokenizer,
        messages,
        tools=_tools(),
        enable_thinking=False,
        model_name="gemma-4-26b-4bit",
        add_generation_prompt=False,
    )

    assert 'command:<|"|>ls /tmp<|"|>' in rendered
    assert "limit:null" in rendered
    assert 'type:[<|"|>INTEGER<|"|>,<|"|>NULL<|"|>]' in rendered
    assert "limit:None" not in rendered
    assert '{{"command"' not in rendered


def test_official_template_restores_thinking_continuation_after_tool_result(
    monkeypatch,
) -> None:
    pytest.importorskip("jinja2")
    _allow_fixture_template(monkeypatch, _STALE_FULL, "full")
    tokenizer = _tokenizer(_STALE_FULL)
    messages = [
        {"role": "user", "content": "List /tmp"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "ls /tmp"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "alpha.txt"},
    ]

    rendered = apply_chat_template(
        tokenizer,
        messages,
        tools=_tools(),
        enable_thinking=True,
        model_name="gemma-4-26b-4bit",
    )

    assert rendered.endswith(
        '<|tool_response>response:bash{value:<|"|>alpha.txt<|"|>}'
        "<tool_response|><|channel>thought\n"
    )
