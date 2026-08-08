# SPDX-License-Identifier: Apache-2.0
"""GH-973 regression tests — assistant tool_call.arguments dict invariant.

Every mainstream HF chat template renders prior assistant tool_calls by
iterating ``tool_call.arguments|items`` (Qwen3 / Hermes / Llama3 / GLM4 /
Nemotron / minimax). The OpenAI wire contract encodes
``tool_calls[i].function.arguments`` as a JSON *string*, so any client
that replays an assistant tool_call in the history verbatim (notably
``pydantic_ai``'s structured-output retry path — GH-973) crashes the
Jinja render with:

    TypeError: Can only get item pairs from a mapping.

The fix normalises assistant ``tool_call.function.arguments`` to a dict
at the shared ``apply_chat_template`` boundary, which is the single choke
point every model / route / engine funnels through. This test module
pins:

1. The helper's behaviour (dict / str-json / str-non-json / already-dict
   / mixed).
2. Cross-parser end-to-end coverage — hermes / qwen3 / qwen3-coder /
   glm4.7 / harmony / minimax / deepseek-v3. Each parser gets a
   template shape that would have crashed pre-fix (``|items``,
   ``|tojson``, dict-attribute access) and we assert:
     * no exception is raised
     * the rendered prompt contains the expected argument values
     * the dict-form input yields the same output as the JSON-string
       input (backward-compat)

The parser suite uses *minimal fake tokenizers* that carry the shape of
each real Jinja template — we do NOT depend on the HF hub or on any
mlx-lm model weights, so this suite runs in unit CI.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest

from vllm_mlx.utils.chat_template import (
    _normalize_assistant_tool_call_arguments,
    apply_chat_template,
)

# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


class TestNormalizeAssistantToolCallArguments:
    def test_json_string_dict_becomes_dict(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris", "unit": "C"}',
                        },
                    }
                ],
            }
        ]
        out = _normalize_assistant_tool_call_arguments(messages)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Paris",
            "unit": "C",
        }
        # Original untouched — the API surface keeps the wire shape.
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
            '{"city": "Paris", "unit": "C"}'
        )

    def test_already_dict_is_identity(self):
        original_args = {"city": "Tokyo"}
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": original_args}}
                ],
            }
        ]
        out = _normalize_assistant_tool_call_arguments(messages)
        # Short-circuit: nothing to normalise, so the list is returned
        # by reference (idempotent — layering on top of upstream
        # normalisers costs nothing).
        assert out is messages
        assert out[0]["tool_calls"][0]["function"]["arguments"] is original_args

    def test_json_string_non_dict_is_wrapped(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "echo", "arguments": '["a", "b"]'}}
                ],
            }
        ]
        out = _normalize_assistant_tool_call_arguments(messages)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {"value": ["a", "b"]}

    def test_malformed_json_is_wrapped(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "bad", "arguments": "{not valid json"}}
                ],
            }
        ]
        out = _normalize_assistant_tool_call_arguments(messages)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {
            "value": "{not valid json"
        }

    def test_non_assistant_messages_untouched(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "x", "content": "result"},
            {"role": "system", "content": "you are a helper"},
        ]
        out = _normalize_assistant_tool_call_arguments(messages)
        assert out is messages  # no-op short-circuit

    def test_assistant_without_tool_calls_untouched(self):
        messages = [{"role": "assistant", "content": "hello"}]
        out = _normalize_assistant_tool_call_arguments(messages)
        assert out is messages

    def test_multi_tool_call_batch(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "a", "arguments": '{"x": 1}'}},
                    {"function": {"name": "b", "arguments": {"y": 2}}},
                    {"function": {"name": "c", "arguments": '{"z": 3}'}},
                ],
            }
        ]
        out = _normalize_assistant_tool_call_arguments(messages)
        args = [tc["function"]["arguments"] for tc in out[0]["tool_calls"]]
        assert args == [{"x": 1}, {"y": 2}, {"z": 3}]

    def test_partial_mutation_preserves_untouched_messages(self):
        # Only the last message needs mutation. The middle assistant
        # message is dict-form and MUST pass through by reference.
        user_msg = {"role": "user", "content": "q"}
        dict_form_msg = {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "x", "arguments": {"a": 1}}}],
        }
        tool_msg = {"role": "tool", "content": "r"}
        str_form_msg = {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "y", "arguments": '{"b": 2}'}}],
        }
        messages = [user_msg, dict_form_msg, tool_msg, str_form_msg]
        out = _normalize_assistant_tool_call_arguments(messages)
        assert out[0] is user_msg
        assert out[1] is dict_form_msg
        assert out[2] is tool_msg
        # Touched message got copied so the caller's message list is
        # left with the OpenAI-wire string form.
        assert out[3] is not str_form_msg
        assert out[3]["tool_calls"][0]["function"]["arguments"] == {"b": 2}
        assert str_form_msg["tool_calls"][0]["function"]["arguments"] == '{"b": 2}'

    def test_empty_and_degenerate_shapes(self):
        # Empty list
        assert _normalize_assistant_tool_call_arguments([]) == []
        # Non-list
        assert _normalize_assistant_tool_call_arguments("not a list") == "not a list"  # type: ignore[arg-type]
        # Non-dict message
        assert _normalize_assistant_tool_call_arguments([42]) == [42]  # type: ignore[list-item]
        # Assistant with tool_calls not a list
        m = [{"role": "assistant", "tool_calls": "oops"}]
        assert _normalize_assistant_tool_call_arguments(m) is m
        # Assistant with tool_call whose function is not a dict
        m2 = [{"role": "assistant", "tool_calls": [{"function": "oops"}]}]
        assert _normalize_assistant_tool_call_arguments(m2) is m2

    def test_absent_arguments_key_is_not_invented(self):
        """Missing ``arguments`` key MUST NOT be materialised as
        ``{"value": None}`` — that would silently invent a payload the
        client never sent (codex r1 NIT on PR #981).
        """
        # Missing on nested function
        m1 = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "x"}}],
            }
        ]
        assert _normalize_assistant_tool_call_arguments(m1) is m1
        # Missing on top-level (no ``function`` envelope)
        m2 = [{"role": "assistant", "tool_calls": [{"name": "x"}]}]
        assert _normalize_assistant_tool_call_arguments(m2) is m2
        # Present but None on nested — treat as present (get_arguments
        # explicitly wrapped) — this IS the SDK-bug case the wrapper
        # exists for; verify we wrap None as {"value": None} only when
        # the key is actually present.
        m3 = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "x", "arguments": None}}],
            }
        ]
        out3 = _normalize_assistant_tool_call_arguments(m3)
        assert out3[0]["tool_calls"][0]["function"]["arguments"] == {"value": None}

    def test_top_level_arguments_string_becomes_dict(self):
        """Codex r1 BLOCKING: some templates access ``tool_call.arguments``
        directly (no ``tool_call.function`` unwrap). The nested-only
        normaliser leaked the JSON-string form to those templates.
        """
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    }
                ],
            }
        ]
        out = _normalize_assistant_tool_call_arguments(messages)
        assert out[0]["tool_calls"][0]["arguments"] == {"city": "Paris"}
        # Original untouched.
        assert messages[0]["tool_calls"][0]["arguments"] == '{"city": "Paris"}'

    def test_mixed_shape_normalizes_both_nested_and_top_level(self):
        """Codex r3 BLOCKING on PR #981 — a mixed-shape replay that
        carries BOTH ``function.arguments`` and top-level ``arguments``
        (some SDKs mirror the field for template compatibility) must
        normalise both, otherwise a template that iterates
        ``tc.arguments|items`` still crashes even when
        ``tc.function.arguments`` was normalised.
        """
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {"name": "x", "arguments": '{"a": 1}'},
                        "arguments": '{"a": 1}',
                    }
                ],
            }
        ]
        out = _normalize_assistant_tool_call_arguments(messages)
        # Both shapes end up dict-form; templates that read either
        # location render without crashing.
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
        assert out[0]["tool_calls"][0]["arguments"] == {"a": 1}

    def test_top_level_arguments_malformed_wrapped(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"name": "x", "arguments": "not json"}],
            }
        ]
        out = _normalize_assistant_tool_call_arguments(messages)
        assert out[0]["tool_calls"][0]["arguments"] == {"value": "not json"}


# ---------------------------------------------------------------------------
# Cross-parser end-to-end coverage — apply_chat_template with fake
# tokenizers that mimic each real family's Jinja arguments-access shape.
# ---------------------------------------------------------------------------


def _fake_tokenizer_with_template(chat_template_str: str) -> MagicMock:
    """Return a tokenizer stub that renders ``chat_template_str`` via a real
    Jinja environment.

    The stub exposes the same surface the sanitiser wants (all-special-tokens
    lists), so the sanitisation layer and the tool-call normalisation layer
    both run end-to-end.
    """
    # The minimal PR-validation environment intentionally omits optional
    # template dependencies. Helper-level normalisation tests above remain
    # runnable there; only real rendering cases skip when Jinja is unavailable.
    jinja2 = pytest.importorskip("jinja2")

    env = jinja2.Environment(  # noqa: S701 - test scaffolding
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(
        jinja2.TemplateError(message)
    )
    template = env.from_string(chat_template_str)

    tok = MagicMock()
    tok.all_special_tokens = []
    tok.additional_special_tokens = []
    tok.special_tokens_map = {}
    # Prevent recursive processor-unwrap in _collect_role_markers.
    del tok.tokenizer

    def _apply(messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    tok.apply_chat_template.side_effect = _apply
    return tok


# --- The template shapes (one per parser family). ------------------------
#
# These are minimal — only the bits that touch ``tool_call.arguments``.
# Each shape is representative of the family's real chat_template.jinja
# on HuggingFace (Qwen3 / Hermes / Llama3 / GLM4.7 / harmony / minimax /
# DeepSeek-V3 as of 2026-Q2). The goal is not to fully render the
# family's prompt — just to trigger the ``|items`` / ``|tojson`` /
# dict-access path that GH-973's traceback shows.

# Qwen3 / Nemotron shape — ``arguments|items`` (the actual crashing form).
QWEN3_LIKE_TEMPLATE = """\
{% for m in messages -%}
{% if m.role == 'assistant' and m.tool_calls -%}
{%- for tc in m.tool_calls -%}
<tool_call>
name={{ tc.function.name }}
{% for k, v in tc.function.arguments|items -%}
{{ k }}={{ v }}
{% endfor -%}
</tool_call>
{% endfor -%}
{% else -%}
{{ m.role }}: {{ m.content }}
{% endif -%}
{% endfor -%}
"""

# Hermes shape — dict-attribute access via `|tojson` (used by NousResearch
# Hermes 2/3 tool-use templates).
HERMES_LIKE_TEMPLATE = """\
{% for m in messages -%}
{% if m.role == 'assistant' and m.tool_calls -%}
{% for tc in m.tool_calls -%}
<tool_call>{"name": "{{ tc.function.name }}", "arguments": {{ tc.function.arguments | tojson }}}</tool_call>
{% endfor -%}
{% else -%}
{{ m.role }}: {{ m.content }}
{% endif -%}
{% endfor -%}
"""

# GLM 4.7 shape — same ``|items`` pattern (GLM-4.6 / 4.7 templates).
GLM47_LIKE_TEMPLATE = QWEN3_LIKE_TEMPLATE

# Harmony (gpt-oss) shape — dict access + ``|tojson``.
HARMONY_LIKE_TEMPLATE = """\
{% for m in messages -%}
{% if m.role == 'assistant' and m.tool_calls -%}
{% for tc in m.tool_calls -%}
<|start|>assistant to=functions.{{ tc.function.name }}<|channel|>commentary<|message|>{{ tc.function.arguments | tojson }}<|end|>
{% endfor -%}
{% else -%}
{{ m.role }}: {{ m.content }}
{% endif -%}
{% endfor -%}
"""

# MiniMax shape — mixed access (dict-key + iteration).
MINIMAX_LIKE_TEMPLATE = """\
{% for m in messages -%}
{% if m.role == 'assistant' and m.tool_calls -%}
{% for tc in m.tool_calls -%}
[function_call]{{ tc.function.name }}
{% for k, v in tc.function.arguments|items -%}
{{ k }} = {{ v }}
{% endfor -%}
[/function_call]
{% endfor -%}
{% else -%}
{{ m.role }}: {{ m.content }}
{% endif -%}
{% endfor -%}
"""

# DeepSeek V3 shape — dict-attribute access with fenced JSON body.
DEEPSEEK_V3_LIKE_TEMPLATE = """\
{% for m in messages -%}
{% if m.role == 'assistant' and m.tool_calls -%}
{% for tc in m.tool_calls -%}
```json
{"name": "{{ tc.function.name }}", "arguments": {{ tc.function.arguments | tojson }}}
```
{% endfor -%}
{% else -%}
{{ m.role }}: {{ m.content }}
{% endif -%}
{% endfor -%}
"""

# Qwen3-coder XML shape — attribute access + string-safe cast (matches the
# in-tree Nemotron template's `<parameter=NAME>VALUE</parameter>` output,
# which is the closest thing to the qwen3-coder-xml family in the repo).
QWEN3_CODER_XML_LIKE_TEMPLATE = """\
{% for m in messages -%}
{% if m.role == 'assistant' and m.tool_calls -%}
{% for tc in m.tool_calls -%}
<function={{ tc.function.name }}>
{% for k, v in tc.function.arguments|items -%}
<parameter={{ k }}>{{ v | string }}</parameter>
{% endfor -%}
</function>
{% endfor -%}
{% else -%}
{{ m.role }}: {{ m.content }}
{% endif -%}
{% endfor -%}
"""

# Top-level (flat) shape — some templates access ``tc.arguments`` directly
# without an ``if tool_call.function is defined`` unwrap. Codex r1
# BLOCKING on PR #981: the nested-only fix left this shape crashing.
FLAT_TOP_LEVEL_TEMPLATE = """\
{% for m in messages -%}
{% if m.role == 'assistant' and m.tool_calls -%}
{% for tc in m.tool_calls -%}
<tool_call name={{ tc.name }}>
{% for k, v in tc.arguments|items -%}
{{ k }}={{ v }}
{% endfor -%}
</tool_call>
{% endfor -%}
{% else -%}
{{ m.role }}: {{ m.content }}
{% endif -%}
{% endfor -%}
"""

# Table of (parser_family_id, template_str, expected_substrings_dict_form).
# Each parser family gets exercised for both the dict-form input (baseline)
# and the JSON-string input (the GH-973 crash form). Post-fix, both
# inputs render the same output.
PARSER_TEMPLATE_MATRIX = [
    ("hermes", HERMES_LIKE_TEMPLATE, ['"city":', '"Paris"']),
    ("qwen3_xml", QWEN3_LIKE_TEMPLATE, ["city=Paris", "unit=C"]),
    ("qwen3_coder_xml", QWEN3_CODER_XML_LIKE_TEMPLATE, ["<parameter=city>Paris"]),
    ("glm4_7", GLM47_LIKE_TEMPLATE, ["city=Paris", "unit=C"]),
    ("harmony", HARMONY_LIKE_TEMPLATE, ['"city":', '"Paris"']),
    ("minimax", MINIMAX_LIKE_TEMPLATE, ["city = Paris", "unit = C"]),
    ("deepseek_v3", DEEPSEEK_V3_LIKE_TEMPLATE, ['"city":', '"Paris"']),
]


def _messages_with_assistant_tool_call(arguments):
    return [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": arguments},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "sunny, 22C in Paris",
        },
        {"role": "user", "content": "and tomorrow?"},
    ]


@pytest.mark.parametrize(
    ("parser_id", "template_str", "expected_substrings"),
    PARSER_TEMPLATE_MATRIX,
    ids=[row[0] for row in PARSER_TEMPLATE_MATRIX],
)
def test_cross_parser_json_string_arguments_renders_without_crash(
    parser_id, template_str, expected_substrings
):
    """The GH-973 crash form: ``arguments`` = OpenAI-wire JSON string.

    Every mainstream template family iterates or dict-accesses
    ``tool_call.function.arguments``, so any of these would raise a
    ``TypeError`` on the JSON-string form pre-fix. Post-fix, the shared
    ``apply_chat_template`` normalises to dict before the Jinja render.
    """
    tokenizer = _fake_tokenizer_with_template(template_str)
    messages = _messages_with_assistant_tool_call('{"city": "Paris", "unit": "C"}')

    prompt = apply_chat_template(tokenizer, messages)

    for expected in expected_substrings:
        assert expected in prompt, (
            f"[{parser_id}] expected substring {expected!r} not found in "
            f"prompt:\n{prompt}"
        )


@pytest.mark.parametrize(
    ("parser_id", "template_str", "expected_substrings"),
    PARSER_TEMPLATE_MATRIX,
    ids=[row[0] for row in PARSER_TEMPLATE_MATRIX],
)
def test_cross_parser_dict_arguments_backward_compat(
    parser_id, template_str, expected_substrings
):
    """Backward-compat: dict-form input renders exactly like it did pre-fix.

    Callers that already pass dict-form (the ``routes/chat.py`` non-MLLM
    path, the ``engine/batched.py`` upstream normaliser) MUST NOT
    regress. The normaliser short-circuits when nothing needs mutation
    (returns the caller's list by reference), so this is also the
    perf-critical common case.
    """
    tokenizer = _fake_tokenizer_with_template(template_str)
    messages = _messages_with_assistant_tool_call({"city": "Paris", "unit": "C"})

    prompt = apply_chat_template(tokenizer, messages)

    for expected in expected_substrings:
        assert expected in prompt, (
            f"[{parser_id}] expected substring {expected!r} not found in "
            f"prompt:\n{prompt}"
        )


@pytest.mark.parametrize(
    ("parser_id", "template_str", "expected_substrings"),
    PARSER_TEMPLATE_MATRIX,
    ids=[row[0] for row in PARSER_TEMPLATE_MATRIX],
)
def test_cross_parser_dict_and_string_render_identically(
    parser_id, template_str, expected_substrings
):
    """The GH-973 invariant: str-form and dict-form MUST render to the
    same prompt post-normalisation. This is the property that makes
    ``pydantic_ai``'s structured-output retry path work — the retry
    replay uses the JSON-string wire shape, the initial call uses dict.
    """
    tokenizer_a = _fake_tokenizer_with_template(template_str)
    tokenizer_b = _fake_tokenizer_with_template(template_str)

    dict_form = apply_chat_template(
        tokenizer_a,
        _messages_with_assistant_tool_call({"city": "Paris", "unit": "C"}),
    )
    string_form = apply_chat_template(
        tokenizer_b,
        _messages_with_assistant_tool_call('{"city": "Paris", "unit": "C"}'),
    )
    assert dict_form == string_form, (
        f"[{parser_id}] dict-form and str-form renders diverged:\n"
        f"dict-form:\n{dict_form}\n"
        f"str-form:\n{string_form}"
    )


def _messages_with_flat_top_level_tool_call(arguments):
    """Legacy / flat wire shape: ``tool_call`` carries ``name`` + ``arguments``
    directly, no ``function`` envelope. Some templates access it via
    ``tc.arguments`` without an unwrap step; this is the codex r1
    BLOCKING case on PR #981.
    """
    return [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "name": "get_weather",
                    "arguments": arguments,
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "sunny, 22C in Paris",
        },
        {"role": "user", "content": "and tomorrow?"},
    ]


def test_flat_top_level_template_json_string_arguments_renders():
    """Codex r1 BLOCKING on PR #981 — templates that iterate
    ``tc.arguments|items`` on the FLAT wire shape (no ``function``
    unwrap) MUST also be normalised at the boundary.
    """
    tokenizer = _fake_tokenizer_with_template(FLAT_TOP_LEVEL_TEMPLATE)
    messages = _messages_with_flat_top_level_tool_call('{"city": "Paris", "unit": "C"}')
    prompt = apply_chat_template(tokenizer, messages)
    assert "city=Paris" in prompt
    assert "unit=C" in prompt


def test_flat_top_level_template_dict_arguments_identical():
    """The dict-form input and JSON-string input render identically on
    the flat wire shape too."""
    tokenizer_a = _fake_tokenizer_with_template(FLAT_TOP_LEVEL_TEMPLATE)
    tokenizer_b = _fake_tokenizer_with_template(FLAT_TOP_LEVEL_TEMPLATE)
    dict_form = apply_chat_template(
        tokenizer_a,
        _messages_with_flat_top_level_tool_call({"city": "Paris", "unit": "C"}),
    )
    string_form = apply_chat_template(
        tokenizer_b,
        _messages_with_flat_top_level_tool_call('{"city": "Paris", "unit": "C"}'),
    )
    assert dict_form == string_form


def test_pydantic_ai_style_retry_message_list_does_not_crash():
    """End-to-end shape of the GH-973 repro: replayed assistant tool_call
    whose ``arguments`` is a JSON string (the pydantic_ai retry path
    against Qwen3.5 / 3.6). Pre-fix this raised ``TypeError`` mid-render
    and the route returned 500. Post-fix the render succeeds and the
    prompt contains the tool call's parsed arguments.
    """
    tokenizer = _fake_tokenizer_with_template(QWEN3_LIKE_TEMPLATE)
    messages = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "Extract: 'Alice is 30 years old'"},
        {
            # pydantic_ai's retry replay — arguments as JSON string per
            # the OpenAI wire contract.
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_final_result_1",
                    "type": "function",
                    "function": {
                        "name": "final_result",
                        "arguments": '{"name": "Alice", "age": 30}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_final_result_1",
            "content": "validation_error: age must be int",
        },
        {"role": "user", "content": "Please retry."},
    ]
    prompt = apply_chat_template(tokenizer, messages)
    assert "name=Alice" in prompt
    assert "age=30" in prompt


DEEPSEEK_R1_LIKE_TEMPLATE = r"""
{%- for message in messages %}
  {%- if message.role == 'assistant' and message.tool_calls is defined %}
    {%- for tool in message.tool_calls %}
      {{ tool.function.name + '\n```json\n' + tool.function.arguments + '\n```' }}
    {%- endfor %}
  {%- elif message.content is defined %}{{ message.content }}
  {%- endif %}
{%- endfor %}
"""


def test_deepseek_r1_string_concat_template_replays_tool_result():
    """#1676: DeepSeek-R1's official template requires JSON strings.

    The shared normaliser first converts OpenAI-wire strings to mappings for
    Qwen/Hermes-style templates.  The exact string-concatenation TypeError must
    trigger the compatibility retry rather than escape as an HTTP 500 on the
    tool-result turn.
    """
    tokenizer = _fake_tokenizer_with_template(DEEPSEEK_R1_LIKE_TEMPLATE)
    messages = _messages_with_assistant_tool_call('{"city": "Paris"}')
    original = copy.deepcopy(messages)

    prompt = apply_chat_template(tokenizer, messages)

    assert 'get_weather\n```json\n{"city":"Paris"}\n```' in prompt
    assert messages == original


def test_deepseek_r1_retry_accepts_internal_dict_arguments():
    tokenizer = _fake_tokenizer_with_template(DEEPSEEK_R1_LIKE_TEMPLATE)
    prompt = apply_chat_template(
        tokenizer, _messages_with_assistant_tool_call({"city": "東京"})
    )
    assert '{"city":"東京"}' in prompt


def test_deepseek_serialized_retry_survives_unsupported_kwarg():
    """The serialized candidate must survive later kwarg compatibility retries."""
    tokenizer = _fake_tokenizer_with_template(DEEPSEEK_R1_LIKE_TEMPLATE)
    render = tokenizer.apply_chat_template.side_effect
    calls = 0

    def concat_then_reject_kwarg(messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1 and "enable_thinking" in kwargs:
            raise TypeError(
                "apply_chat_template() got an unexpected keyword argument "
                "'enable_thinking'"
            )
        return render(messages, **kwargs)

    tokenizer.apply_chat_template.side_effect = concat_then_reject_kwarg
    prompt = apply_chat_template(
        tokenizer, _messages_with_assistant_tool_call('{"city":"Paris"}')
    )
    assert 'get_weather\n```json\n{"city":"Paris"}\n```' in prompt


ALTERNATING_ONLY_TEMPLATE = r"""
{%- set loop_messages = messages[1:] if messages and messages[0].role == 'system' else messages -%}
{%- for message in loop_messages %}
  {%- if (message.role == 'user') != (loop.index0 % 2 == 0) -%}
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
  {%- endif -%}
  {{ message.role + ': ' + message.get('content', '') + '\n' }}
{%- endfor %}
"""


def test_gemma_alternating_template_replays_openai_tool_history():
    tokenizer = _fake_tokenizer_with_template(ALTERNATING_ONLY_TEMPLATE)
    prompt = apply_chat_template(
        tokenizer,
        _messages_with_assistant_tool_call('{"city":"Paris"}'),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    assert 'Tool call get_weather: {"city":"Paris"}' in prompt
    assert "Tool result call_1: sunny, 22C in Paris" in prompt
    assert "and tomorrow?" in prompt
    assert "tool:" not in prompt


def test_alternating_fallback_after_unsupported_template_kwarg():
    """A keyword retry must not bypass the strict-role compatibility path."""
    tokenizer = _fake_tokenizer_with_template(ALTERNATING_ONLY_TEMPLATE)
    render = tokenizer.apply_chat_template.side_effect

    def reject_enable_thinking(messages, **kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError(
                "apply_chat_template() got an unexpected keyword argument "
                "'enable_thinking'"
            )
        return render(messages, **kwargs)

    tokenizer.apply_chat_template.side_effect = reject_enable_thinking
    prompt = apply_chat_template(
        tokenizer,
        _messages_with_assistant_tool_call('{"city":"Paris"}'),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    assert 'Tool call get_weather: {"city":"Paris"}' in prompt
    assert "Tool result call_1: sunny, 22C in Paris" in prompt


def test_unrelated_template_error_is_not_hidden():
    tokenizer = _fake_tokenizer_with_template(
        "{{ raise_exception('different template failure') }}"
    )
    with pytest.raises(Exception, match="different template failure"):
        apply_chat_template(tokenizer, _messages_with_assistant_tool_call("{}"))
