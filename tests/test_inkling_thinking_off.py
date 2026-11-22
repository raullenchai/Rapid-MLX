# SPDX-License-Identifier: Apache-2.0
"""Inkling's native effort switch honors a resolved thinking opt-out (#5916)."""

import jinja2
import pytest

from rapid_mlx.utils.chat_template import (
    apply_chat_template,
    detect_native_reasoning_effort_levels,
)

# Trimmed from Inkling 2 Mini: effort macro and message/generation shape.
INKLING_TEMPLATE = """
{%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2,
                      "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99} -%}
{%- macro emit_thinking_effort() -%}
    {%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}
    {%- if eff is string -%}
        {%- set key = eff | trim -%}
        {%- if key not in effort_map -%}
            {{- raise_exception("Unknown reasoning_effort: " ~ eff) -}}
        {%- endif -%}
        {%- set num = effort_map[key] -%}
    {%- else -%}
        {%- set num = eff | float -%}
    {%- endif -%}
    {%- if num < 0.0 or num > 0.99 -%}
        {{- raise_exception("reasoning_effort must be in [0.0, 0.99]") -}}
    {%- endif -%}
    {{- "<|message_system|><|content_text|>Thinking effort level: " -}}
    {%- if num == 0.0 -%}0{%- else -%}{{ num }}{%- endif -%}
    {{- "<|end_message|>" -}}
{%- endmacro -%}
{%- set state = namespace(effort_emitted=false) -%}
{%- for message in messages -%}
    {%- if not state.effort_emitted and message.role != "system" -%}
        {{- emit_thinking_effort() -}}
        {%- set state.effort_emitted = true -%}
    {%- endif -%}
    {{- message.role ~ "<|content_text|>" ~ message.content ~ "<|end_message|>" -}}
{%- endfor -%}
{%- if not state.effort_emitted -%}{{- emit_thinking_effort() -}}{%- endif -%}
{%- if add_generation_prompt -%}<|message_model|>{%- endif -%}
"""
LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class RenderingTokenizer:
    def __init__(self, template=INKLING_TEMPLATE):
        self.chat_template = template
        self.kwargs = {}

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        env = jinja2.Environment()

        def raise_exception(message):
            raise ValueError(message)

        env.globals["raise_exception"] = raise_exception
        return env.from_string(self.chat_template).render(messages=messages, **kwargs)


@pytest.mark.parametrize(
    "enabled,effort,expected",
    [
        (False, None, "0"),
        (None, None, "0.9"),
        (True, None, "0.9"),
        (False, "high", "0.9"),
    ],
)
def test_render_thinking_effort(enabled, effort, expected):
    tok = RenderingTokenizer()
    rendered = apply_chat_template(
        tok,
        [{"role": "user", "content": "Hi"}],
        enable_thinking=enabled,
        chat_template_kwargs={} if effort is None else {"reasoning_effort": effort},
    )
    assert f"Thinking effort level: {expected}<|end_message|>" in rendered


def test_detect_inkling_levels():
    assert detect_native_reasoning_effort_levels(INKLING_TEMPLATE) == LEVELS


def test_macro_dictionary_rebound_before_call():
    template = (
        "{% set levels = {'none': 0, 'high': 0.9} %}"
        "{% macro validate() %}"
        "{% if reasoning_effort not in levels %}"
        "{{ raise_exception('bad reasoning_effort') }}{% endif %}"
        "{{ levels[reasoning_effort] }}{% endmacro %}"
        "{% set levels = {'low': 0.2, 'xhigh': 0.99} %}"
        "{{ validate() }}"
    )
    tok = RenderingTokenizer(template)
    assert tok.apply_chat_template([], reasoning_effort="low") == "0.2"
    with pytest.raises(ValueError, match="bad reasoning_effort"):
        tok.apply_chat_template([], reasoning_effort="none")
    assert detect_native_reasoning_effort_levels(template) == ("low", "xhigh")


@pytest.mark.parametrize(
    "template",
    [
        "{{ reasoning_effort | default('medium') }}",
        INKLING_TEMPLATE.replace('"none": 0.0, ', ""),
        INKLING_TEMPLATE + "{{ enable_thinking }}",
    ],
)
def test_other_templates_keep_their_default(template):
    tok = RenderingTokenizer(template)
    apply_chat_template(tok, [{"role": "user", "content": "Hi"}], enable_thinking=False)
    assert "reasoning_effort" not in tok.kwargs


def test_gpt_oss_low_path():
    tok = RenderingTokenizer(
        "<|start|><|channel|><|message|>Reasoning: "
        "{{ reasoning_effort | default('medium') }}"
    )
    rendered = apply_chat_template(
        tok, [{"role": "user", "content": "Hi"}], enable_thinking=False
    )
    assert rendered.endswith("Reasoning: low")


@pytest.mark.parametrize(
    "template",
    [
        # A local value is not the context's effort.
        INKLING_TEMPLATE.replace("eff | trim", "'high'"),
        # A map must have string keys and actually reject invalid efforts.
        INKLING_TEMPLATE.replace('"none": 0.0', "0: 0.0"),
        INKLING_TEMPLATE.replace(
            '{{- raise_exception("Unknown reasoning_effort: " ~ eff) -}}', "ignored"
        ),
        # Deferred validation in an uncalled macro proves nothing.
        INKLING_TEMPLATE[: INKLING_TEMPLATE.index("{%- set state")],
        # Conditional rebinding invalidates the dictionary / effort alias.
        INKLING_TEMPLATE.replace(
            "{%- if key not in effort_map -%}",
            "{%- if tools %}{% set effort_map = other %}{% endif %}"
            "{%- if key not in effort_map -%}",
        ),
        INKLING_TEMPLATE.replace(
            "{%- if key not in effort_map -%}",
            "{%- if tools %}{% set key = 'high' %}{% endif %}"
            "{%- if key not in effort_map -%}",
        ),
        # Loop targets shadow outer names; do not infer validation inside loops.
        "{% set effort_map = {'none': 0, 'high': 0.9} %}"
        "{% for reasoning_effort in messages %}"
        "{% if reasoning_effort not in effort_map %}"
        "{{ raise_exception('bad') }}{% endif %}{% endfor %}",
    ],
)
def test_dictionary_detection_fails_closed(template):
    assert detect_native_reasoning_effort_levels(template) is None


@pytest.mark.parametrize("effort", ["unknown", 1.0])
def test_explicit_invalid_effort_still_raises(effort):
    tok = RenderingTokenizer()
    with pytest.raises(Exception, match="reasoning_effort"):
        apply_chat_template(
            tok,
            [{"role": "user", "content": "Hi"}],
            enable_thinking=False,
            chat_template_kwargs={"reasoning_effort": effort},
        )


def test_top_level_dictionary_validation():
    template = (
        "{% set levels = {'none': 0, 'high': 0.9} %}"
        "{% if reasoning_effort not in levels %}"
        "{{ raise_exception('bad') }}{% endif %}"
        "{{ levels[reasoning_effort] }}"
    )
    assert detect_native_reasoning_effort_levels(template) == ("none", "high")
    tok = RenderingTokenizer(template)
    assert (
        apply_chat_template(
            tok, [{"role": "user", "content": "Hi"}], enable_thinking=False
        )
        == "0"
    )
