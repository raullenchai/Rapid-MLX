# SPDX-License-Identifier: Apache-2.0
"""Template-native thinking switch for templates without ``enable_thinking`` (#3045).

Cohere's North Mini Code template ignores ``enable_thinking``; its switch is
a boolean ``reasoning`` (default on, also derived from
``reasoning_effort == "none"``) and the generation prompt seeds an empty
thinking block when it is false. Desktop's default (thinking off) and
``rapid-mlx chat`` without ``--think`` resolve ``enable_thinking=False``,
which was silently inert for this family. ``apply_chat_template`` now maps
that off flag onto the template's own switch, found by reading the Jinja AST
for the variables the template consults from its context and branches on as
booleans.
"""

from __future__ import annotations

import jinja2
import pytest

from vllm_mlx.utils.chat_template import (
    _template_context_facts_for_source,
    apply_chat_template,
    template_thinking_switch,
)

# The verbatim control clauses of mlx-community/North-Mini-Code-1.0-4bit's
# chat_template.jinja (the ``reasoning`` derivation and the generation
# prompt), with a minimal message loop in between.
NORTH_CLAUSE = (
    "{%- set reasoning = reasoning if reasoning is not undefined else "
    '(false if reasoning_effort is defined and reasoning_effort | lower == "none" '
    "else true) %}"
    "{%- for message in messages %}{{ message.role }}: {{ message.content }}\n"
    "{%- endfor %}"
    "{%- if add_generation_prompt -%}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    "{% if reasoning %}<|START_THINKING|>{% else %}"
    "<|START_THINKING|><|END_THINKING|>{% endif %}{%- endif %}"
)
THINKING_PREFIX = "<|CHATBOT_TOKEN|><|START_THINKING|>"
NO_THINKING_PREFIX = "<|CHATBOT_TOKEN|><|START_THINKING|><|END_THINKING|>"

ENABLE_THINKING_TEMPLATE = (
    "{%- for m in messages %}{{ m.content }}{%- endfor %}"
    "{%- if enable_thinking is defined and not enable_thinking %}<think></think>"
    "{%- endif %}"
)
HARMONY_LIKE_TEMPLATE = (
    "{%- set effort = reasoning_effort | default('medium') %}"
    "Reasoning: {{ effort }}{%- for m in messages %}{{ m.content }}{%- endfor %}"
)


class _RenderingTokenizer:
    """Applicator that renders its template with jinja2 like transformers does."""

    def __init__(self, template):
        self.chat_template = template
        self.received_kwargs: dict | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.received_kwargs = dict(kwargs)
        kwargs.pop("tokenize", None)
        env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
        return env.from_string(self.chat_template).render(messages=messages, **kwargs)

    def encode(self, _text):
        return [1, 2]


MESSAGES = [{"role": "user", "content": "hi"}]


class TestContextReads:
    """Which names a template reads from its render context (Jinja scoping)."""

    @pytest.mark.parametrize(
        "template",
        [
            # attribute access on a message is not a context read
            "{%- for m in messages %}{% if m.reasoning %}{{ m.reasoning }}{% endif %}"
            "{%- endfor %}",
            # a loop variable named ``reasoning`` shadows the context in its body
            "{%- for reasoning in messages %}{{ reasoning }}{%- endfor %}",
            # a macro argument named ``reasoning`` shadows the context in its body
            "{%- macro show(reasoning) %}{{ reasoning }}{%- endmacro %}{{ show(1) }}",
            # a call-block argument likewise
            "{%- macro m(caller) %}{{ caller(1) }}{% endmacro %}"
            "{%- call(reasoning) m() %}{{ reasoning }}{% endcall %}",
            # assigned before being read: the template's own local, not a knob
            "{%- set reasoning = true %}{% if reasoning %}x{% endif %}",
            # a block assignment likewise
            "{%- set reasoning %}yes{% endset %}{% if reasoning %}x{% endif %}",
            # bound in every branch before the read
            "{%- if x %}{% set reasoning = 1 %}{% else %}{% set reasoning = 0 %}"
            "{%- endif %}{% if reasoning %}x{% endif %}",
            # imported under that name
            "{%- import 'x.jinja' as reasoning %}{{ reasoning.thing }}",
            # a ``with`` binding, read only inside the block
            "{%- with reasoning = 1 %}{{ reasoning }}{% endwith %}",
        ],
    )
    def test_bound_or_attribute_uses_are_not_reads(self, template):
        assert "reasoning" not in _template_context_facts_for_source(template)[0]

    @pytest.mark.parametrize(
        "template",
        [
            # North's own line: the read happens before the binding
            "{%- set reasoning = reasoning if reasoning is not undefined else true %}",
            # read after a loop that only shadowed it inside the body
            "{%- for reasoning in messages %}{{ reasoning }}{%- endfor %}{{ reasoning }}",
            # read after a ``with`` block that only bound it inside
            "{%- with reasoning = 1 %}{{ reasoning }}{% endwith %}{{ reasoning }}",
            # bound in only one branch: the later read may still be the context
            "{%- if x %}{% set reasoning = 1 %}{% endif %}{{ reasoning }}",
            # read inside a macro body that does not bind it
            "{%- macro show() %}{{ reasoning }}{%- endmacro %}{{ show() }}",
            # the loop's own filter clause
            "{%- for m in messages if reasoning %}{{ m }}{%- endfor %}",
            # a ``{% set %}`` whose value reads it inside a nested expression
            "{%- set flags = namespace(on=reasoning) %}{{ flags.on }}",
            # a macro default value is evaluated in the outer scope
            "{%- macro show(on=reasoning) %}{{ on }}{%- endmacro %}{{ show() }}",
            # a ``with`` value is evaluated in the outer scope
            "{%- with reasoning = reasoning %}{{ reasoning }}{% endwith %}",
        ],
    )
    def test_context_reads(self, template):
        assert "reasoning" in _template_context_facts_for_source(template)[0]

    def test_unparseable_source_reads_nothing(self):
        assert _template_context_facts_for_source("{% if reasoning %}") == (
            frozenset(),
            frozenset(),
        )


class TestTemplateThinkingSwitch:
    def test_north_reads_and_branches_on_reasoning(self):
        assert template_thinking_switch(NORTH_CLAUSE) == "reasoning"

    def test_enable_thinking_template_keeps_its_own_switch(self):
        assert template_thinking_switch(ENABLE_THINKING_TEMPLATE) is None

    def test_reads_of_both_prefer_enable_thinking(self):
        assert template_thinking_switch(NORTH_CLAUSE + ENABLE_THINKING_TEMPLATE) is None

    def test_harmony_style_effort_only_template_has_no_switch(self):
        assert template_thinking_switch(HARMONY_LIKE_TEMPLATE) is None

    @pytest.mark.parametrize(
        "template",
        [
            # a bare truthiness test
            "{% if reasoning %}<think>{% endif %}",
            # its negation
            "{% if not reasoning %}<think></think>{% endif %}",
            # a conditional expression
            "{{ '<think>' if reasoning else '<think></think>' }}",
            # an elif branch
            "{% if x %}a{% elif reasoning %}b{% endif %}",
            # read as data somewhere else too, but branched on as a boolean
            "{{ reasoning }}{% if reasoning %}x{% endif %}",
            # North's shape: a definedness-guarded self-rebinding keeps the
            # context value, so the later branch is on the knob
            "{%- set reasoning = reasoning if reasoning is not undefined else true %}"
            "{% if reasoning %}x{% endif %}",
            "{%- set reasoning = reasoning if reasoning is defined else true %}"
            "{% if reasoning %}x{% endif %}",
            "{%- set reasoning = true if reasoning is undefined else reasoning %}"
            "{% if reasoning %}x{% endif %}",
            "{%- set reasoning = reasoning if reasoning is not none else true %}"
            "{% if reasoning %}x{% endif %}",
            "{%- set reasoning = true if reasoning is not defined else reasoning %}"
            "{% if reasoning %}x{% endif %}",
            # the ``default`` filter idiom likewise (a defined False survives)
            "{%- set reasoning = reasoning | default(true) %}{% if reasoning %}x{% endif %}",
            "{%- set reasoning = reasoning | default(true, false) %}"
            "{% if reasoning %}x{% endif %}",
            "{%- set reasoning = reasoning | default(true, boolean=false) %}"
            "{% if reasoning %}x{% endif %}",
        ],
    )
    def test_a_boolean_branch_on_a_context_read_is_a_switch(self, template):
        assert template_thinking_switch(template) == "reasoning"

    @pytest.mark.parametrize(
        "template",
        [
            # rendered as data, never branched on
            "{{ reasoning }}",
            # only a definedness check
            "{% if reasoning is defined %}x{% endif %}",
            # compared against a value: not an on/off switch
            "{% if reasoning == 'high' %}x{% endif %}",
            # combined with something else (not a plain boolean test)
            "{% if reasoning and tools %}x{% endif %}",
            # branched on, but as an attribute
            "{% if reasoning.enabled %}x{% endif %}",
            # branched on the template's own local, not the context
            "{%- set reasoning = true %}{% if reasoning %}x{% endif %}",
            # read as data, then shadowed by a local before the branch
            "{{ reasoning }}{% set reasoning = true %}{% if reasoning %}x{% endif %}",
            # a rebinding that does not preserve the value is a fresh local
            "{%- set reasoning = reasoning if tools else true %}{% if reasoning %}x{% endif %}",
            "{%- set reasoning = reasoning | lower %}{% if reasoning %}x{% endif %}",
            # the arm taken for a defined value does not carry the name
            "{%- set reasoning = reasoning if reasoning is undefined else true %}"
            "{% if reasoning %}x{% endif %}",
            "{%- set reasoning = true if reasoning is defined else reasoning %}"
            "{% if reasoning %}x{% endif %}",
            "{%- set reasoning = reasoning if reasoning is none else true %}"
            "{% if reasoning %}x{% endif %}",
            # ``default`` with the boolean flag replaces a defined False
            "{%- set reasoning = reasoning | default(true, true) %}"
            "{% if reasoning %}x{% endif %}",
            "{%- set reasoning = reasoning | default(true, boolean=true) %}"
            "{% if reasoning %}x{% endif %}",
            "{%- set reasoning = reasoning | default(true, boolean=flag) %}"
            "{% if reasoning %}x{% endif %}",
            # a preserving rebinding of a name already shadowed stays shadowed
            "{%- for reasoning in messages %}{% set reasoning = reasoning | default(1) %}"
            "{% if reasoning %}x{% endif %}{%- endfor %}",
            # the branch is inside a macro that binds the name
            "{%- macro show(reasoning) %}{% if reasoning %}x{% endif %}{%- endmacro %}"
            "{{ show(reasoning) }}",
            # branched on a loop variable of that name
            "{%- for reasoning in messages %}{% if reasoning %}x{% endif %}{%- endfor %}",
        ],
    )
    def test_data_definedness_or_local_uses_are_not_a_switch(self, template):
        assert template_thinking_switch(template) is None

    def test_unparseable_and_missing_templates(self):
        assert template_thinking_switch("{% if reasoning %}") is None
        assert template_thinking_switch(None) is None
        assert template_thinking_switch({"default": NORTH_CLAUSE}) == "reasoning"
        assert (
            template_thinking_switch(
                {"tool_use": NORTH_CLAUSE, "default": "x"}, tools=[{}]
            )
            == "reasoning"
        )

    @pytest.mark.parametrize("order", ["switch-first", "enable-thinking-first"])
    def test_enable_thinking_in_any_selected_source_wins(self, monkeypatch, order):
        """``_chat_template_strings`` selects one source today; should it ever
        return several, a read of ``enable_thinking`` anywhere still wins."""
        from vllm_mlx.utils import chat_template as module

        sources = [NORTH_CLAUSE, ENABLE_THINKING_TEMPLATE]
        if order == "enable-thinking-first":
            sources.reverse()
        monkeypatch.setattr(
            module, "_chat_template_strings", lambda template, *, tools: sources
        )
        assert template_thinking_switch({"default": "x"}) is None

    def test_without_jinja2_no_switch_is_reported(self, monkeypatch):
        from vllm_mlx.utils import chat_template as module

        module._template_context_facts_for_source.cache_clear()
        monkeypatch.setattr(module, "_jinja_nodes", lambda: (None, None))
        try:
            assert template_thinking_switch(NORTH_CLAUSE) is None
        finally:
            module._template_context_facts_for_source.cache_clear()


class TestApplyChatTemplateOnNorth:
    def test_thinking_off_seeds_an_empty_thinking_block(self):
        tok = _RenderingTokenizer(NORTH_CLAUSE)
        prompt = apply_chat_template(
            tok, MESSAGES, enable_thinking=False, model_name="north-mini-code-4bit"
        )
        assert tok.received_kwargs["reasoning"] is False
        assert prompt.endswith(NO_THINKING_PREFIX)

    def test_thinking_on_keeps_the_template_default(self):
        tok = _RenderingTokenizer(NORTH_CLAUSE)
        prompt = apply_chat_template(
            tok, MESSAGES, enable_thinking=True, model_name="north-mini-code-4bit"
        )
        assert "reasoning" not in tok.received_kwargs
        assert prompt.endswith(THINKING_PREFIX)

    def test_unset_thinking_defaults_on_for_this_family(self):
        tok = _RenderingTokenizer(NORTH_CLAUSE)
        prompt = apply_chat_template(tok, MESSAGES, model_name="north-mini-code-4bit")
        assert tok.received_kwargs["enable_thinking"] is True
        assert "reasoning" not in tok.received_kwargs
        assert prompt.endswith(THINKING_PREFIX)

    def test_desktop_shape_client_kwarg_off_is_honoured(self):
        """Desktop sends ``chat_template_kwargs.enable_thinking=false``; the
        route resolves it to ``enable_thinking=False`` before rendering."""
        tok = _RenderingTokenizer(NORTH_CLAUSE)
        prompt = apply_chat_template(
            tok,
            MESSAGES,
            enable_thinking=False,
            chat_template_kwargs={"enable_thinking": False},
            model_name="north-mini-code-4bit",
        )
        assert prompt.endswith(NO_THINKING_PREFIX)

    def test_explicit_client_reasoning_wins(self):
        tok = _RenderingTokenizer(NORTH_CLAUSE)
        prompt = apply_chat_template(
            tok,
            MESSAGES,
            enable_thinking=False,
            chat_template_kwargs={"reasoning": True},
            model_name="north-mini-code-4bit",
        )
        assert tok.received_kwargs["reasoning"] is True
        assert prompt.endswith(THINKING_PREFIX)

    def test_explicit_client_reasoning_effort_wins(self):
        tok = _RenderingTokenizer(NORTH_CLAUSE)
        prompt = apply_chat_template(
            tok,
            MESSAGES,
            enable_thinking=False,
            chat_template_kwargs={"reasoning_effort": "high"},
            model_name="north-mini-code-4bit",
        )
        assert "reasoning" not in tok.received_kwargs
        assert prompt.endswith(THINKING_PREFIX)
        tok = _RenderingTokenizer(NORTH_CLAUSE)
        prompt = apply_chat_template(
            tok,
            MESSAGES,
            enable_thinking=True,
            chat_template_kwargs={"reasoning_effort": "none"},
        )
        assert prompt.endswith(NO_THINKING_PREFIX)


class TestOtherTemplatesUnaffected:
    def test_enable_thinking_template_gets_no_reasoning_kwarg(self):
        tok = _RenderingTokenizer(ENABLE_THINKING_TEMPLATE)
        prompt = apply_chat_template(tok, MESSAGES, enable_thinking=False)
        assert "reasoning" not in tok.received_kwargs
        assert prompt.endswith("<think></think>")

    def test_harmony_style_template_keeps_the_low_effort_path(self):
        tok = _RenderingTokenizer(HARMONY_LIKE_TEMPLATE)
        apply_chat_template(
            tok, MESSAGES, enable_thinking=False, model_name="gpt-oss-20b"
        )
        assert "reasoning" not in tok.received_kwargs
        assert tok.received_kwargs.get("reasoning_effort") == "low"
