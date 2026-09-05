# SPDX-License-Identifier: Apache-2.0
"""#3043 — graded ``reasoning_effort`` maps onto a template's native levels.

Qwen3.8's chat template validates ``reasoning_effort`` against
``('xhigh', 'medium', 'low')`` and defaults to ``xhigh``. Before #3043 the
OpenAI ``reasoning_effort`` knob never reached that variable: every graded
value became a ``reasoning_max_tokens`` cap, so ``reasoning_effort="low"``
rendered the template's *xhigh* instruction ("think carefully through the
task…") and was then force-closed at 512 thinking tokens — the prompt and
the budget contradicted each other — while ``xhigh`` itself was a 400.

This file pins the post-fix contract:

  1. Detection is data-driven from the template source: only a template
     that *validates* ``reasoning_effort`` against a literal set publishes
     a vocabulary. Harmony (interpolation only) and North Mini Code
     (``== "none"`` sentinel) do not, and keep the token-cap fallback.
  2. Mapping ranks both sides on the OpenAI ladder, uses a native value
     verbatim when the template accepts it, and rounds ties *up*.
  3. ``maybe_apply_reasoning_effort`` writes the native level into
     ``chat_template_kwargs`` and layers no cap; an explicit client
     ``chat_template_kwargs.reasoning_effort`` wins; templates without a
     vocabulary (Qwen3.5 / 3.6, Gemma 4, …) and callers that pass no
     template keep the pre-#3043 cap tiers byte-for-byte.
  4. The mapped value is always one the real Jinja template accepts
     (rendered through jinja2 with the checkpoint's own validation clause).
  5. ``xhigh`` is accepted on both OpenAI surfaces and sits in the cap
     table for templates that need the fallback.
  6. End to end through ``/v1/chat/completions`` and ``/v1/responses`` with
     an engine whose tokenizer serves the Qwen3.8 template: ``engine.chat``
     receives the native level and *no* ``reasoning_max_tokens`` cap (so the
     request never grows a budget logits processor and stays MTP-eligible),
     while the same request against an on/off-only template keeps the cap.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vllm_mlx.api.models import (
    _VALID_REASONING_EFFORTS,
    OPENAI_REASONING_EFFORT_TO_MAX_TOKENS,
    ChatCompletionRequest,
)
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.service.helpers import (
    maybe_apply_reasoning_effort,
    served_chat_template,
)
from vllm_mlx.utils.chat_template import (
    REASONING_EFFORT_LADDER,
    detect_native_reasoning_effort_levels,
    map_reasoning_effort_to_native,
)

# Verbatim validation + instruction clause from
# rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX ``chat_template.jinja`` (2026-09-04).
# Kept as a fixture so the contract is pinned against the real shape, not a
# hand-simplified one; the trailing user/assistant turn is the minimal
# scaffolding needed to render it standalone.
QWEN38_TEMPLATE = """\
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
{%- if reasoning_instructions %}
    {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
    {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- endfor %}
{{- '<|im_start|>assistant\\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\\n\\n</think>\\n\\n' }}
{%- else %}
    {{- '<think>\\n' }}
{%- endif %}
"""

# Hy3 shape: validates against a list literal that includes a non-ladder
# name (``no_think``).
HY3_TEMPLATE_CLAUSE = (
    "{%- if not reasoning_effort is defined or reasoning_effort not in "
    "['high', 'low', 'no_think'] %}{%- set reasoning_effort = 'no_think' %}{%- endif %}"
)

# GPT-OSS / Harmony shape: interpolates the value, never validates it.
HARMONY_TEMPLATE_CLAUSE = (
    '{%- if reasoning_effort is not defined %}{%- set reasoning_effort = "medium" %}'
    '{%- endif %}{{- "Reasoning: " + reasoning_effort + "\\n\\n" }}'
)

# North Mini Code shape: a single-sentinel comparison, no vocabulary.
NORTH_TEMPLATE_CLAUSE = (
    "{%- set reasoning = reasoning if reasoning is not undefined else "
    '(false if reasoning_effort is defined and reasoning_effort | lower == "none" '
    "else true) -%}"
)

# Qwen3.5 / 3.6 / Gemma 4 shape: only an on/off switch.
ENABLE_THINKING_ONLY_TEMPLATE = (
    "{%- if enable_thinking is defined and not enable_thinking %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
)


def _request(**overrides):
    base = dict(
        reasoning_effort=None,
        chat_template_kwargs=None,
        enable_thinking=None,
        reasoning_max_tokens=None,
        tools=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# (1) Detection is data-driven from the template source
# ---------------------------------------------------------------------------


class TestDetectNativeLevels:
    def test_qwen38_publishes_its_vocabulary_in_template_order(self):
        assert detect_native_reasoning_effort_levels(QWEN38_TEMPLATE) == (
            "xhigh",
            "medium",
            "low",
        )

    def test_hy3_list_literal_is_detected(self):
        assert detect_native_reasoning_effort_levels(HY3_TEMPLATE_CLAUSE) == (
            "high",
            "low",
            "no_think",
        )

    def test_harmony_interpolation_only_publishes_nothing(self):
        """Harmony renders whatever it is given; it never validates, so it
        does not declare a vocabulary and keeps the cap fallback."""
        assert detect_native_reasoning_effort_levels(HARMONY_TEMPLATE_CLAUSE) is None

    def test_north_mini_code_sentinel_compare_publishes_nothing(self):
        assert detect_native_reasoning_effort_levels(NORTH_TEMPLATE_CLAUSE) is None

    def test_enable_thinking_only_template_publishes_nothing(self):
        assert (
            detect_native_reasoning_effort_levels(ENABLE_THINKING_ONLY_TEMPLATE) is None
        )

    @pytest.mark.parametrize("template", [None, "", 42])
    def test_absent_or_foreign_template_publishes_nothing(self, template):
        assert detect_native_reasoning_effort_levels(template) is None

    def test_dict_template_prefers_tool_use_variant_when_tools_given(self):
        template = {
            "default": ENABLE_THINKING_ONLY_TEMPLATE,
            "tool_use": QWEN38_TEMPLATE,
        }
        assert detect_native_reasoning_effort_levels(template) is None
        assert detect_native_reasoning_effort_levels(
            template, tools=[{"type": "function"}]
        ) == ("xhigh", "medium", "low")

    def test_default_filter_between_name_and_membership_is_tolerated(self):
        clause = (
            "{% if reasoning_effort|default('low') not in ['low', 'high'] %}"
            "{{ raise_exception('x') }}{% endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("low", "high")

    def test_duplicate_literals_collapse(self):
        clause = (
            "{% if reasoning_effort not in ('low', 'low', 'high') %}"
            "{{ raise_exception('x') }}{% endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("low", "high")


class TestDetectionRequiresAValidationBlock:
    """Codex r1/r2 BLOCKING: a bare membership test is not a declaration. A
    template that merely *branches* on a subset must not have that subset
    mistaken for its accepted vocabulary (``medium`` would be upgraded and
    lose its cap). The proof is made on the Jinja AST: ``<var> not in
    (...)`` where ``<var>`` derives from ``reasoning_effort``, with an
    unconditional ``raise_exception`` or ``set <var>`` at the top level of
    the block body."""

    def test_positive_subset_branch_is_not_a_vocabulary(self):
        clause = "{%- if reasoning_effort in ('high', 'xhigh') %}deep{%- endif %}"
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_negative_branch_without_rejection_is_not_a_vocabulary(self):
        clause = (
            "{%- if reasoning_effort not in ('high', 'xhigh') %}shallow{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_rejection_block_declares(self):
        clause = (
            "{%- if reasoning_effort not in ('a', 'b') %}"
            "{{- raise_exception('bad') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("a", "b")

    def test_defaulting_block_declares(self):
        clause = (
            "{%- if not reasoning_effort is defined or reasoning_effort not in ['a', 'b'] %}"
            "{%- set reasoning_effort = 'a' %}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("a", "b")

    def test_setting_an_unrelated_variable_does_not_declare(self):
        clause = (
            "{%- if reasoning_effort not in ['a', 'b'] %}"
            "{%- set other = 'a' %}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_rejection_in_a_later_block_does_not_leak_backwards(self):
        clause = (
            "{%- if reasoning_effort not in ['a'] %}x{%- endif %}"
            "{%- if y %}{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_elif_terminates_the_block_body(self):
        clause = (
            "{%- if reasoning_effort not in ['a'] %}x"
            "{%- elif z %}{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_whitespace_control_and_derived_variable(self):
        clause = (
            "{%- set resolved_reasoning_effort = reasoning_effort|default('b') -%}"
            "{%- if resolved_reasoning_effort not in ('a', 'b') -%}"
            "{{- raise_exception('bad') -}}{%- endif -%}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("a", "b")

    def test_rejection_nested_under_an_inner_condition_is_not_unconditional(self):
        """Codex r2: a ``raise_exception`` that only fires under a further
        ``if`` inside the block may never fire, so the set is not proven."""
        clause = (
            "{%- if reasoning_effort not in ['a'] %}"
            "{%- if strict %}{{ raise_exception('z') }}{%- endif %}"
            "{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_unrelated_variable_with_a_similar_name_does_not_count(self):
        """Codex r2: ``my_reasoning_effort`` is not derived from the request
        knob, so validating it says nothing about ``reasoning_effort``."""
        clause = (
            "{%- set my_reasoning_effort = 'a' %}"
            "{%- if my_reasoning_effort not in ['a', 'b'] %}"
            "{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_variable_derived_through_a_chain_counts(self):
        clause = (
            "{%- set r1 = reasoning_effort %}{%- set r2 = r1 | lower %}"
            "{%- if r2 not in ['a'] %}{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("a",)

    def test_negated_positive_membership_is_left_alone(self):
        clause = (
            "{%- if not (reasoning_effort in ['a', 'b']) %}"
            "{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_non_literal_set_is_not_a_vocabulary(self):
        clause = (
            "{%- if reasoning_effort not in allowed %}"
            "{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_validation_nested_under_enable_thinking_counts(self):
        """Qwen3.8 validates only while thinking is on; the vocabulary still
        applies whenever the level matters."""
        clause = (
            "{%- if enable_thinking is undefined or enable_thinking is true %}"
            "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
            "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"
            "{{- raise_exception('bad') }}{%- endif %}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == (
            "xhigh",
            "medium",
            "low",
        )

    def test_unparseable_template_publishes_nothing(self):
        clause = "{% if reasoning_effort not in ['a'] %}{{ raise_exception('x') }}"
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_transformers_generation_tag_is_parsed(self):
        """HF templates may wrap assistant turns in ``{% generation %}``
        (transformers' assistant-mask extension); the detector must not
        choke on it."""
        clause = (
            "{%- for m in messages %}{% generation %}{{ m.content }}{% endgeneration %}"
            "{%- endfor %}{%- if reasoning_effort not in ('a', 'b') %}"
            "{{ raise_exception('x') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("a", "b")

    def test_conditional_derivation_does_not_count(self):
        """Codex r3: an assignment inside a sibling ``if`` may not have run."""
        clause = (
            "{%- if x %}{%- set r = reasoning_effort %}{%- endif %}"
            "{%- if r not in ['a'] %}{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_later_derivation_does_not_count(self):
        clause = (
            "{%- if r not in ['a'] %}{{ raise_exception('z') }}{%- endif %}"
            "{%- set r = reasoning_effort %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_derivation_inside_a_loop_does_not_leak(self):
        clause = (
            "{%- for m in messages %}{%- set r = reasoning_effort %}{%- endfor %}"
            "{%- if r not in ['a'] %}{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_derivation_in_an_enclosing_block_counts(self):
        clause = (
            "{%- if enable_thinking is undefined or enable_thinking is true %}"
            "{%- set r = reasoning_effort %}"
            "{%- if r not in ['a'] %}{{ raise_exception('z') }}{%- endif %}"
            "{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("a",)

    @pytest.mark.parametrize(
        "clause",
        [
            "{%- if legacy_mode %}fixed-xhigh{%- else %}"
            "{%- if reasoning_effort not in ['low', 'high'] %}"
            "{{ raise_exception('bad') }}{%- endif %}{%- endif %}",
            "{%- if legacy_mode %}fixed-xhigh"
            "{%- elif reasoning_effort not in ['low', 'high'] %}"
            "{{ raise_exception('bad') }}{%- endif %}",
            "{%- if enable_thinking %}{%- if tools %}"
            "{%- if reasoning_effort not in ['low', 'high'] %}"
            "{{ raise_exception('bad') }}{%- endif %}{%- endif %}{%- endif %}",
            "{%- if enable_thinking %}"
            "{%- if reasoning_effort not in ['low', 'high'] %}"
            "{{ raise_exception('bad') }}{%- endif %}{%- endif %}",
            "{%- if enable_thinking is true %}"
            "{%- if reasoning_effort not in ['low', 'high'] %}"
            "{{ raise_exception('bad') }}{%- endif %}{%- endif %}",
        ],
    )
    def test_path_conditional_validation_does_not_publish_globally(self, clause):
        """A vocabulary is not global when an unrelated runtime branch can
        skip its validation while reasoning remains enabled."""
        assert detect_native_reasoning_effort_levels(clause) is None

    @pytest.mark.parametrize(
        "binding",
        [
            "{%- macro raise_exception(message) %}ok{%- endmacro %}",
            "{%- set raise_exception = harmless %}",
            "{%- import 'helpers.jinja' as raise_exception %}",
            "{%- from 'helpers.jinja' import harmless as raise_exception %}",
        ],
    )
    def test_template_local_raise_exception_binding_fails_closed(self, binding):
        clause = (
            binding + "{%- if reasoning_effort not in ['low', 'high'] %}"
            "{{ raise_exception('bad') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_and_conjunct_does_not_guarantee_rejection(self):
        """Codex r4: with ``strict and …`` the branch is skipped when
        ``strict`` is false, so the membership test proves nothing."""
        clause = (
            "{%- if strict and reasoning_effort not in ('a', 'b') %}"
            "{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_overwritten_derivation_is_forgotten(self):
        """Codex r4: ``set r = reasoning_effort`` then ``set r = 'c'``."""
        clause = (
            "{%- set r = reasoning_effort %}{%- set r = 'c' %}"
            "{%- if r not in ['a'] %}{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_conditionally_overwritten_derivation_is_forgotten(self):
        """A Jinja ``if`` body leaks its assignments, so the overwrite may
        have happened."""
        clause = (
            "{%- set r = reasoning_effort %}"
            "{%- if x %}{%- set r = 'c' %}{%- endif %}"
            "{%- if r not in ['a'] %}{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_block_assignment_overwrite_is_forgotten(self):
        clause = (
            "{%- set r = reasoning_effort %}{%- set r %}c{%- endset %}"
            "{%- if r not in ['a'] %}{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_loop_variable_shadowing_is_forgotten(self):
        clause = (
            "{%- for reasoning_effort in efforts %}{%- endfor %}"
            "{%- if reasoning_effort not in ['a'] %}{{ raise_exception('z') }}"
            "{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_macro_body_is_not_searched(self):
        """Codex r4: a macro that is never invoked proves nothing."""
        clause = (
            "{%- macro check() %}{%- if reasoning_effort not in ['a'] %}"
            "{{ raise_exception('z') }}{%- endif %}{%- endmacro %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_call_block_body_is_not_searched(self):
        clause = (
            "{%- macro wrap() %}{{ caller() }}{%- endmacro %}"
            "{%- call wrap() %}{%- if reasoning_effort not in ['a'] %}"
            "{{ raise_exception('z') }}{%- endif %}{%- endcall %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_loop_body_is_not_searched(self):
        """A loop may run zero times, so a validation inside it may never
        execute."""
        clause = (
            "{%- for m in messages %}{%- if reasoning_effort not in ['a'] %}"
            "{{ raise_exception('z') }}{%- endif %}{%- endfor %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_unrelated_disjunct_disqualifies_the_test(self):
        """Codex r5: ``mode not in ['x'] or …`` enters the block for a valid
        effort whenever ``mode`` is off, so the set proves nothing."""
        clause = (
            "{%- if mode not in ['x'] or reasoning_effort not in ['a', 'b'] %}"
            "{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None
        clause = (
            "{%- if debug or reasoning_effort not in ['a', 'b'] %}"
            "{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    @pytest.mark.parametrize(
        "guard",
        [
            "not reasoning_effort is defined",
            "reasoning_effort is undefined",
            "reasoning_effort is none",
            "not reasoning_effort",
        ],
    )
    def test_definedness_guard_disjunct_is_allowed(self, guard):
        clause = (
            "{%- if " + guard + " or reasoning_effort not in ['a', 'b'] %}"
            "{%- set reasoning_effort = 'a' %}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("a", "b")

    def test_definedness_guard_on_another_variable_disqualifies(self):
        clause = (
            "{%- if not mode is defined or reasoning_effort not in ['a', 'b'] %}"
            "{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_elif_after_an_effort_dependent_test_is_path_constrained(self):
        """Codex r5: reached only when ``reasoning_effort != 'x'``, so the
        accepted set is really ``{'x'} ∪ [...]``."""
        clause = (
            "{%- if reasoning_effort == 'x' %}ok"
            "{%- elif reasoning_effort not in ['a'] %}{{ raise_exception('z') }}"
            "{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_nested_validation_under_an_effort_dependent_branch_is_skipped(self):
        clause = (
            "{%- if reasoning_effort == 'x' %}ok{%- else %}"
            "{%- if reasoning_effort not in ['a'] %}{{ raise_exception('z') }}"
            "{%- endif %}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_assignment_inside_a_skipped_branch_still_forgets(self):
        clause = (
            "{%- set r = reasoning_effort %}"
            "{%- if reasoning_effort == 'x' %}{%- set r = 'c' %}{%- endif %}"
            "{%- if r not in ['a'] %}{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    @pytest.mark.parametrize(
        "expr",
        [
            "reasoning_effort == 'high'",
            "'a' if reasoning_effort in ['x'] else 'b'",
            "reasoning_effort | replace('x', 'a')",
            "reasoning_effort | upper",
            "reasoning_effort ~ ''",
            "[reasoning_effort]",
        ],
    )
    def test_domain_changing_derivation_does_not_count(self, expr):
        """Codex r5: mentioning ``reasoning_effort`` is not deriving from it."""
        clause = (
            "{%- set r = " + expr + " %}"
            "{%- if r not in ['a'] %}{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_a_bare_effort_test_makes_the_branch_path_constrained(self):
        clause = (
            "{%- if reasoning_effort %}"
            "{%- if reasoning_effort not in ['a'] %}{{ raise_exception('z') }}"
            "{%- endif %}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    @pytest.mark.parametrize(
        "clause",
        [
            # the membership's left side is not a value-preserving name
            "{%- if (reasoning_effort | upper) not in ['A'] %}"
            "{{ raise_exception('z') }}{%- endif %}",
            "{%- if 'x' not in ['a'] %}{{ raise_exception('z') }}{%- endif %}",
            # the set mixes in a non-string / non-literal item
            "{%- if reasoning_effort not in ['a', 1] %}"
            "{{ raise_exception('z') }}{%- endif %}",
            "{%- if reasoning_effort not in ['a', other] %}"
            "{{ raise_exception('z') }}{%- endif %}",
        ],
    )
    def test_ill_formed_membership_tests_publish_nothing(self, clause):
        assert detect_native_reasoning_effort_levels(clause) is None

    @pytest.mark.parametrize(
        "rebind",
        [
            # tuple unpacking on the path
            "{%- set r, q = 'c', 'd' %}",
            # tuple unpacking inside a skipped (effort-dependent) branch
            "{%- if reasoning_effort == 'x' %}{%- set r, q = 'c', 'd' %}{%- endif %}",
            # names bound by import / from-import (str and alias-tuple targets)
            "{%- import 'x.jinja' as r %}",
            "{%- from 'x.jinja' import r %}",
            "{%- from 'x.jinja' import q as r %}",
            # a tuple loop target
            "{%- for r, q in items %}{%- endfor %}",
        ],
    )
    def test_every_rebinding_shape_forgets_the_derived_name(self, rebind):
        clause = (
            "{%- set r = reasoning_effort %}" + rebind + "{%- if r not in ['a'] %}"
            "{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_without_jinja2_detection_publishes_nothing(self, monkeypatch):
        from vllm_mlx.utils import chat_template as module

        monkeypatch.setattr(module, "_jinja_nodes", lambda: (None, None))
        module._template_parser.cache_clear()
        module._native_reasoning_effort_levels_for_source.cache_clear()
        try:
            assert detect_native_reasoning_effort_levels(QWEN38_TEMPLATE) is None
        finally:
            module._template_parser.cache_clear()
            module._native_reasoning_effort_levels_for_source.cache_clear()

    @pytest.mark.parametrize(
        "expr",
        [
            "reasoning_effort",
            "reasoning_effort | default('a')",
            "reasoning_effort | trim | lower",
            "reasoning_effort | string",
        ],
    )
    def test_value_preserving_derivation_counts(self, expr):
        clause = (
            "{%- set r = " + expr + " %}"
            "{%- if r not in ['a'] %}{{ raise_exception('z') }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("a",)

    def test_conditional_raise_expression_does_not_count(self):
        """Codex r3: ``{{ raise_exception(...) if strict else '' }}`` may
        never raise."""
        clause = (
            "{%- if reasoning_effort not in ['a'] %}"
            "{{ raise_exception('bad') if strict else '' }}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_default_outside_the_vocabulary_does_not_count(self):
        """Codex r3: ``set reasoning_effort = 'unsupported'`` proves nothing
        about which values the template accepts."""
        clause = (
            "{%- if reasoning_effort not in ['a', 'b'] %}"
            "{%- set reasoning_effort = 'unsupported' %}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_non_literal_default_does_not_count(self):
        clause = (
            "{%- if reasoning_effort not in ['a', 'b'] %}"
            "{%- set reasoning_effort = fallback %}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_validation_in_an_elif_test_counts(self):
        clause = (
            "{%- if enable_thinking is false %}off"
            "{%- elif reasoning_effort not in ['a', 'b'] %}{{ raise_exception('z') }}"
            "{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("a", "b")

    def test_validation_in_enabled_else_branch_counts(self):
        clause = (
            "{%- if enable_thinking is false %}off{%- else %}"
            "{%- if reasoning_effort not in ['a', 'b'] %}"
            "{{ raise_exception('z') }}{%- endif %}{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) == ("a", "b")

    def test_bare_falsey_enable_thinking_branch_does_not_prove_elif_reachable(self):
        clause = (
            "{%- if not enable_thinking %}off"
            "{%- elif reasoning_effort not in ['a', 'b'] %}{{ raise_exception('z') }}"
            "{%- endif %}"
        )
        assert detect_native_reasoning_effort_levels(clause) is None

    def test_first_validation_block_wins_over_a_later_branch(self):
        clause = QWEN38_TEMPLATE + "{%- if reasoning_effort in ('zzz',) %}q{%- endif %}"
        assert detect_native_reasoning_effort_levels(clause) == (
            "xhigh",
            "medium",
            "low",
        )


# ---------------------------------------------------------------------------
# (2) Mapping ranks both sides on the OpenAI ladder
# ---------------------------------------------------------------------------


class TestMapToNative:
    QWEN38 = ("xhigh", "medium", "low")

    @pytest.mark.parametrize(
        ("effort", "expected"),
        [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "xhigh"),
            ("xhigh", "xhigh"),
        ],
    )
    def test_qwen38_mapping(self, effort, expected):
        assert map_reasoning_effort_to_native(effort, self.QWEN38) == expected

    def test_high_rounds_up_to_the_template_ceiling(self):
        """``high`` is equidistant from ``medium`` and ``xhigh`` on the
        ladder; a client asking for the top of OpenAI's range wants the
        template's ceiling, so ties round up."""
        assert map_reasoning_effort_to_native("high", ("low", "medium", "xhigh")) == (
            "xhigh"
        )

    @pytest.mark.parametrize(
        ("effort", "expected"),
        [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),
        ],
    )
    def test_harmony_shaped_vocabulary_maps_identity_and_clamps(self, effort, expected):
        assert (
            map_reasoning_effort_to_native(effort, ("low", "medium", "high"))
            == expected
        )

    def test_hy3_ignores_non_ladder_names_but_still_maps(self):
        levels = ("high", "low", "no_think")
        assert map_reasoning_effort_to_native("low", levels) == "low"
        assert map_reasoning_effort_to_native("medium", levels) == "high"
        assert map_reasoning_effort_to_native("xhigh", levels) == "high"

    def test_unknown_effort_name_is_unmapped(self):
        assert map_reasoning_effort_to_native("banana", self.QWEN38) is None

    def test_vocabulary_without_ladder_names_is_unmapped(self):
        assert map_reasoning_effort_to_native("high", ("no_think", "think")) is None

    def test_none_is_never_a_native_level(self):
        """``none`` is the on/off dimension (``enable_thinking=False``),
        never a template level — even if a template happened to list it."""
        assert "none" not in REASONING_EFFORT_LADDER
        assert map_reasoning_effort_to_native("none", self.QWEN38) is None


# ---------------------------------------------------------------------------
# (3) Helper contract: native level in, no cap on top
# ---------------------------------------------------------------------------


class TestHelperWithNativeTemplate:
    @pytest.mark.parametrize(
        ("effort", "native"),
        [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "xhigh"),
            ("xhigh", "xhigh"),
        ],
    )
    def test_graded_writes_native_level_and_no_cap(self, effort, native):
        req = _request(reasoning_effort=effort)
        assert maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE) is True
        assert req.chat_template_kwargs == {"reasoning_effort": native}
        assert req.reasoning_max_tokens is None
        assert req.enable_thinking is None

    def test_none_still_disables_thinking_on_native_template(self):
        req = _request(reasoning_effort="none")
        assert maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}
        assert req.reasoning_max_tokens is None

    def test_explicit_client_template_kwarg_wins_and_no_cap_is_layered(self):
        req = _request(
            reasoning_effort="high",
            chat_template_kwargs={"reasoning_effort": "low"},
        )
        assert maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE) is False
        assert req.chat_template_kwargs == {"reasoning_effort": "low"}
        assert req.reasoning_max_tokens is None

    def test_merge_preserves_forward_compat_keys(self):
        req = _request(
            reasoning_effort="low",
            chat_template_kwargs={"future_key": "x", "enable_thinking": True},
        )
        assert maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE) is True
        assert req.chat_template_kwargs == {
            "future_key": "x",
            "enable_thinking": True,
            "reasoning_effort": "low",
        }

    def test_explicit_cap_is_orthogonal_to_the_native_level(self):
        """A client that sets both ``reasoning_effort`` and its own
        ``reasoning_max_tokens`` drives two dimensions: the prompt level is
        still translated and the explicit cap is left exactly as sent."""
        req = _request(reasoning_effort="high", reasoning_max_tokens=1000)
        assert maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE) is True
        assert req.chat_template_kwargs == {"reasoning_effort": "xhigh"}
        assert req.reasoning_max_tokens == 1000

    def test_idempotent_second_call(self):
        req = _request(reasoning_effort="high")
        assert maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE) is True
        assert maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE) is False
        assert req.chat_template_kwargs == {"reasoning_effort": "xhigh"}
        assert req.reasoning_max_tokens is None

    def test_tools_variant_of_dict_template_is_consulted(self):
        template = {
            "default": ENABLE_THINKING_ONLY_TEMPLATE,
            "tool_use": QWEN38_TEMPLATE,
        }
        req = _request(reasoning_effort="low", tools=[{"type": "function"}])
        assert maybe_apply_reasoning_effort(req, chat_template=template) is True
        assert req.chat_template_kwargs == {"reasoning_effort": "low"}
        assert req.reasoning_max_tokens is None


class TestHelperFallbackUnchanged:
    """Templates without a vocabulary, and callers that pass no template,
    keep the pre-#3043 ``reasoning_max_tokens`` tiers byte-for-byte."""

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh"])
    def test_enable_thinking_only_template_keeps_cap_tiers(self, effort):
        req = _request(reasoning_effort=effort)
        assert (
            maybe_apply_reasoning_effort(
                req, chat_template=ENABLE_THINKING_ONLY_TEMPLATE
            )
            is True
        )
        assert req.chat_template_kwargs is None
        assert req.reasoning_max_tokens == OPENAI_REASONING_EFFORT_TO_MAX_TOKENS[effort]

    @pytest.mark.parametrize("template", [None, "", HARMONY_TEMPLATE_CLAUSE])
    def test_no_template_or_harmony_keeps_cap(self, template):
        req = _request(reasoning_effort="high")
        assert maybe_apply_reasoning_effort(req, chat_template=template) is True
        assert req.chat_template_kwargs is None
        assert req.reasoning_max_tokens == 8192

    def test_legacy_positional_call_signature_still_works(self):
        req = _request(reasoning_effort="medium")
        assert maybe_apply_reasoning_effort(req) is True
        assert req.reasoning_max_tokens == 2048

    def test_none_on_cap_template_disables_thinking(self):
        req = _request(reasoning_effort="none")
        assert (
            maybe_apply_reasoning_effort(
                req, chat_template=ENABLE_THINKING_ONLY_TEMPLATE
            )
            is True
        )
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_explicit_cap_wins_on_cap_template(self):
        req = _request(reasoning_effort="high", reasoning_max_tokens=64)
        assert (
            maybe_apply_reasoning_effort(
                req, chat_template=ENABLE_THINKING_ONLY_TEMPLATE
            )
            is False
        )
        assert req.reasoning_max_tokens == 64


class TestServedChatTemplateAccessor:
    def test_reads_tokenizer_chat_template(self):
        engine = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=QWEN38_TEMPLATE)
        )
        assert served_chat_template(engine) == QWEN38_TEMPLATE

    def test_reads_the_processor_template_used_by_a_multimodal_engine(self):
        processor_template = HY3_TEMPLATE_CLAUSE
        engine = SimpleNamespace(
            _is_mllm=True,
            _processor=SimpleNamespace(
                chat_template=processor_template,
                apply_chat_template=lambda *args, **kwargs: None,
            ),
            tokenizer=SimpleNamespace(chat_template=QWEN38_TEMPLATE),
        )
        assert served_chat_template(engine) == processor_template

    @pytest.mark.parametrize(
        "processor",
        [
            SimpleNamespace(chat_template=HY3_TEMPLATE_CLAUSE),
            SimpleNamespace(apply_chat_template=lambda *args, **kwargs: None),
            SimpleNamespace(
                chat_template=None,
                apply_chat_template=lambda *args, **kwargs: None,
            ),
        ],
    )
    def test_multimodal_processor_without_a_usable_template_falls_back(self, processor):
        engine = SimpleNamespace(
            _is_mllm=True,
            _processor=processor,
            tokenizer=SimpleNamespace(chat_template=QWEN38_TEMPLATE),
        )
        assert served_chat_template(engine) == QWEN38_TEMPLATE

    @pytest.mark.parametrize(
        "engine",
        [
            SimpleNamespace(),
            SimpleNamespace(tokenizer=None),
            SimpleNamespace(tokenizer=SimpleNamespace()),
            None,
        ],
    )
    def test_missing_tokenizer_or_template_is_none(self, engine):
        assert served_chat_template(engine) is None


# ---------------------------------------------------------------------------
# (4) The mapped value is always one the real template accepts
# ---------------------------------------------------------------------------


def _render_qwen38(**template_kwargs) -> str:
    """Render the verbatim Qwen3.8 clause through jinja2 the way
    ``transformers`` does (``raise_exception`` bound as a global)."""
    jinja2 = pytest.importorskip("jinja2")

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(QWEN38_TEMPLATE).render(
        messages=[{"role": "user", "content": "hi"}], **template_kwargs
    )


class TestRealTemplateRender:
    def test_unmapped_openai_value_is_rejected_by_the_template(self):
        """The pre-#3043 failure shape: forwarding ``high`` verbatim raises
        inside the template (this is what surfaced as a 400 Chat template
        error). The mapping exists precisely so this never happens."""
        jinja2 = pytest.importorskip("jinja2")
        with pytest.raises(
            jinja2.exceptions.TemplateError, match="Unexpected reasoning effort high"
        ):
            _render_qwen38(reasoning_effort="high")

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh"])
    def test_every_mapped_value_renders(self, effort):
        req = _request(reasoning_effort=effort)
        maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE)
        rendered = _render_qwen38(**req.chat_template_kwargs)
        assert rendered.endswith("<|im_start|>assistant\n<think>\n")

    def test_low_renders_the_brief_instruction(self):
        req = _request(reasoning_effort="low")
        maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE)
        rendered = _render_qwen38(**req.chat_template_kwargs)
        assert "Reasoning effort is set to low. Keep your thinking brief" in rendered
        assert "xhigh" not in rendered

    def test_high_renders_the_xhigh_instruction(self):
        req = _request(reasoning_effort="high")
        maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE)
        rendered = _render_qwen38(**req.chat_template_kwargs)
        assert "Reasoning effort is set to xhigh. Please think carefully" in rendered

    def test_medium_renders_no_instruction_line(self):
        req = _request(reasoning_effort="medium")
        maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE)
        rendered = _render_qwen38(**req.chat_template_kwargs)
        assert "Reasoning effort is set to" not in rendered
        assert rendered.endswith("<|im_start|>assistant\n<think>\n")

    def test_none_renders_the_empty_think_block(self):
        req = _request(reasoning_effort="none")
        maybe_apply_reasoning_effort(req, chat_template=QWEN38_TEMPLATE)
        rendered = _render_qwen38(**req.chat_template_kwargs)
        assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert "Reasoning effort is set to" not in rendered


# ---------------------------------------------------------------------------
# (5) ``xhigh`` on both OpenAI surfaces + cap table
# ---------------------------------------------------------------------------


class TestXhighSchema:
    def test_xhigh_is_in_the_closed_set(self):
        assert "xhigh" in _VALID_REASONING_EFFORTS

    def test_xhigh_cap_tier_matches_anthropic(self):
        from vllm_mlx.api.anthropic_models import (
            ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS,
        )

        assert OPENAI_REASONING_EFFORT_TO_MAX_TOKENS["xhigh"] == 24000
        assert (
            OPENAI_REASONING_EFFORT_TO_MAX_TOKENS["xhigh"]
            == ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS["xhigh"]
        )

    def test_cap_tiers_stay_monotonic_with_xhigh(self):
        caps = [
            OPENAI_REASONING_EFFORT_TO_MAX_TOKENS[e]
            for e in ("minimal", "low", "medium", "high", "xhigh")
        ]
        assert caps == sorted(caps) and len(set(caps)) == len(caps)

    def test_chat_request_accepts_xhigh(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="xhigh",
        )
        assert req.reasoning_effort == "xhigh"

    def test_responses_request_accepts_xhigh_top_level_and_nested(self):
        top = ResponsesRequest(model="m", input="hi", reasoning_effort="xhigh")
        assert top.reasoning_effort == "xhigh"
        nested = ResponsesRequest(model="m", input="hi", reasoning={"effort": "xhigh"})
        assert nested.reasoning == {"effort": "xhigh"}

    def test_garbage_still_400s_at_schema(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                reasoning_effort="xxhigh",
            )


# ---------------------------------------------------------------------------
# (6) Route level: the engine sees the native level, never a cap
# ---------------------------------------------------------------------------


class _RouteEngine:
    """Thinking-model-shaped mock whose tokenizer serves ``chat_template``
    and which records the kwargs the route forwards to ``engine.chat``."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    supports_tool_calls = True

    def __init__(self, chat_template):
        self.tokenizer = (
            SimpleNamespace(chat_template=chat_template)
            if chat_template is not None
            else None
        )
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages=None, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
        )

    async def stream_chat(self, messages=None, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        for i, piece in enumerate(("o", "k")):
            yield GenerationOutput(
                text="ok"[: i + 1],
                new_text=piece,
                prompt_tokens=4 if i == 0 else 0,
                completion_tokens=i + 1,
                finished=i == 1,
                finish_reason="stop" if i == 1 else None,
            )

    @property
    def kwargs(self) -> dict:
        assert self.chat_calls, "engine.chat was not called"
        return self.chat_calls[0]["kwargs"]

    @property
    def stream_kwargs(self) -> dict:
        assert self.stream_calls, "engine.stream_chat was not called"
        return self.stream_calls[0]["kwargs"]


@pytest.fixture
def _rate_limiter_state():
    from vllm_mlx.middleware.auth import rate_limiter

    saved_enabled = rate_limiter.enabled
    saved_rpm = rate_limiter.requests_per_minute
    saved_requests = dict(rate_limiter._requests)
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()
    yield rate_limiter
    rate_limiter.enabled = saved_enabled
    rate_limiter.requests_per_minute = saved_rpm
    rate_limiter._requests.clear()
    rate_limiter._requests.update(saved_requests)


@pytest.fixture
def _chat_cap_probe():
    """Record the ``reasoning_max_tokens`` the chat route hands its
    generation-time budget builder. ``None`` means no cap was translated, so
    no ``ReasoningBudgetLogitsProcessor`` can ever be built for the request
    (the builder is the only place one is created on this surface)."""
    seen: list = []

    def _probe(engine, request, cfg, messages, resolved_thinking, **_kw):
        seen.append(getattr(request, "reasoning_max_tokens", None))
        return None

    with patch(
        "vllm_mlx.routes.chat._build_reasoning_budget_processor", side_effect=_probe
    ):
        yield seen


@pytest.fixture
def _responses_cap_probe():
    """Record the ``reasoning_max_tokens`` the responses route hands its
    post-hoc finalizer (the generic cap is enforced post-hoc on this surface)."""
    import vllm_mlx.routes.responses as responses_mod

    seen: list = []
    original = responses_mod._finalize_content_and_reasoning

    def _probe(*args, **kwargs):
        seen.append(kwargs.get("reasoning_max_tokens"))
        return original(*args, **kwargs)

    with patch(
        "vllm_mlx.routes.responses._finalize_content_and_reasoning",
        side_effect=_probe,
    ):
        yield seen


def _client(engine: _RouteEngine, *, surface: str) -> TestClient:
    if surface == "chat":
        from vllm_mlx.routes.chat import router
    else:
        from vllm_mlx.routes.responses import router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    return TestClient(app)


def _chat_body(**extra) -> dict:
    body = {
        "model": "test-model",
        "max_tokens": 80,
        "messages": [{"role": "user", "content": "In 8 words, what is rapid-mlx?"}],
    }
    body.update(extra)
    return body


def _responses_body(**extra) -> dict:
    body = {
        "model": "test-model",
        "max_output_tokens": 80,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "In 8 words, what is rapid-mlx?"}
                ],
            }
        ],
    }
    body.update(extra)
    return body


class TestChatRouteNativeLevel:
    @pytest.mark.parametrize(
        ("effort", "native"),
        [("low", "low"), ("medium", "medium"), ("high", "xhigh"), ("xhigh", "xhigh")],
    )
    def test_graded_reaches_engine_as_template_level_without_cap(
        self, _rate_limiter_state, _chat_cap_probe, effort, native
    ):
        engine = _RouteEngine(QWEN38_TEMPLATE)
        resp = _client(engine, surface="chat").post(
            "/v1/chat/completions", json=_chat_body(reasoning_effort=effort)
        )
        assert resp.status_code == 200, resp.text
        assert engine.kwargs.get("chat_template_kwargs") == {"reasoning_effort": native}
        # No cap was translated, so no budget processor can be built: the
        # request stays on the MTP-eligible path (#3044 is about the cap path).
        assert _chat_cap_probe == [None]
        assert "reasoning_budget_logits_processor" not in engine.kwargs
        # ``reasoning_effort`` is an explicit reasoning signal: the casual-chat
        # auto-disable must step aside so the native level actually applies.
        assert engine.kwargs.get("enable_thinking") is not False

    def test_client_template_kwarg_wins_over_reasoning_effort(
        self, _rate_limiter_state, _chat_cap_probe
    ):
        engine = _RouteEngine(QWEN38_TEMPLATE)
        resp = _client(engine, surface="chat").post(
            "/v1/chat/completions",
            json=_chat_body(
                reasoning_effort="high",
                chat_template_kwargs={"reasoning_effort": "low"},
            ),
        )
        assert resp.status_code == 200, resp.text
        assert engine.kwargs.get("chat_template_kwargs") == {"reasoning_effort": "low"}
        assert _chat_cap_probe == [None]

    def test_none_still_switches_thinking_off(
        self, _rate_limiter_state, _chat_cap_probe
    ):
        engine = _RouteEngine(QWEN38_TEMPLATE)
        resp = _client(engine, surface="chat").post(
            "/v1/chat/completions", json=_chat_body(reasoning_effort="none")
        )
        assert resp.status_code == 200, resp.text
        ctk = engine.kwargs.get("chat_template_kwargs") or {}
        assert ctk.get("enable_thinking") is False
        assert "reasoning_effort" not in ctk
        assert _chat_cap_probe == [None]

    def test_on_off_only_template_keeps_the_cap(
        self, _rate_limiter_state, _chat_cap_probe
    ):
        engine = _RouteEngine(ENABLE_THINKING_ONLY_TEMPLATE)
        resp = _client(engine, surface="chat").post(
            "/v1/chat/completions", json=_chat_body(reasoning_effort="high")
        )
        assert resp.status_code == 200, resp.text
        assert "reasoning_effort" not in (
            engine.kwargs.get("chat_template_kwargs") or {}
        )
        assert _chat_cap_probe == [8192]

    def test_engine_without_tokenizer_keeps_the_cap(
        self, _rate_limiter_state, _chat_cap_probe
    ):
        engine = _RouteEngine(None)
        resp = _client(engine, surface="chat").post(
            "/v1/chat/completions", json=_chat_body(reasoning_effort="low")
        )
        assert resp.status_code == 200, resp.text
        assert _chat_cap_probe == [512]

    def test_unknown_effort_is_a_400_not_a_template_error(self, _rate_limiter_state):
        engine = _RouteEngine(QWEN38_TEMPLATE)
        resp = _client(engine, surface="chat").post(
            "/v1/chat/completions", json=_chat_body(reasoning_effort="ultra")
        )
        assert resp.status_code == 400, resp.text
        assert not engine.chat_calls


class TestResponsesRouteNativeLevel:
    @pytest.mark.parametrize(
        "body",
        [
            {"reasoning": {"effort": "high"}},
            {"reasoning_effort": "high"},
        ],
        ids=["nested-reasoning.effort", "top-level-reasoning_effort"],
    )
    def test_high_reaches_engine_as_xhigh_without_cap(
        self, _rate_limiter_state, _responses_cap_probe, body
    ):
        engine = _RouteEngine(QWEN38_TEMPLATE)
        resp = _client(engine, surface="responses").post(
            "/v1/responses", json=_responses_body(**body)
        )
        assert resp.status_code == 200, resp.text
        assert engine.kwargs.get("chat_template_kwargs") == {
            "reasoning_effort": "xhigh"
        }
        assert _responses_cap_probe == [None]
        assert "reasoning_budget_logits_processor" not in engine.kwargs
        assert engine.kwargs.get("enable_thinking") is not False

    def test_xhigh_is_accepted_on_the_nested_field(self, _rate_limiter_state):
        engine = _RouteEngine(QWEN38_TEMPLATE)
        resp = _client(engine, surface="responses").post(
            "/v1/responses", json=_responses_body(reasoning={"effort": "xhigh"})
        )
        assert resp.status_code == 200, resp.text
        assert engine.kwargs.get("chat_template_kwargs") == {
            "reasoning_effort": "xhigh"
        }

    def test_client_template_kwarg_wins(
        self, _rate_limiter_state, _responses_cap_probe
    ):
        engine = _RouteEngine(QWEN38_TEMPLATE)
        resp = _client(engine, surface="responses").post(
            "/v1/responses",
            json=_responses_body(
                reasoning={"effort": "high"},
                chat_template_kwargs={"reasoning_effort": "medium"},
            ),
        )
        assert resp.status_code == 200, resp.text
        assert engine.kwargs.get("chat_template_kwargs") == {
            "reasoning_effort": "medium"
        }
        assert _responses_cap_probe == [None]

    def test_client_template_kwargs_reach_the_engine_at_all(self, _rate_limiter_state):
        """Regression guard for the /v1/responses passthrough itself: before
        #3043 the route resolved ``enable_thinking`` from the client's
        ``chat_template_kwargs`` but never forwarded the dict to the engine,
        so any other template variable silently vanished on this surface."""
        engine = _RouteEngine(ENABLE_THINKING_ONLY_TEMPLATE)
        resp = _client(engine, surface="responses").post(
            "/v1/responses",
            json=_responses_body(chat_template_kwargs={"custom_flag": True}),
        )
        assert resp.status_code == 200, resp.text
        assert engine.kwargs.get("chat_template_kwargs", {}).get("custom_flag") is True

    def test_on_off_only_template_keeps_the_cap(
        self, _rate_limiter_state, _responses_cap_probe
    ):
        engine = _RouteEngine(ENABLE_THINKING_ONLY_TEMPLATE)
        resp = _client(engine, surface="responses").post(
            "/v1/responses", json=_responses_body(reasoning={"effort": "medium"})
        )
        assert resp.status_code == 200, resp.text
        assert "reasoning_effort" not in (
            engine.kwargs.get("chat_template_kwargs") or {}
        )
        assert _responses_cap_probe == [2048]


class TestStreamingRoutesNativeLevel:
    """Codex r1 NIT: the streaming builders on both surfaces are separate
    code paths (``/v1/responses`` had its own ``chat_kwargs`` block that
    dropped ``chat_template_kwargs``), so exercise them through SSE."""

    def test_chat_stream_forwards_native_level(
        self, _rate_limiter_state, _chat_cap_probe
    ):
        engine = _RouteEngine(QWEN38_TEMPLATE)
        with _client(engine, surface="chat").stream(
            "POST",
            "/v1/chat/completions",
            json=_chat_body(reasoning_effort="high", stream=True),
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert "data: [DONE]" in body
        assert engine.stream_kwargs.get("chat_template_kwargs") == {
            "reasoning_effort": "xhigh"
        }
        assert "reasoning_budget_logits_processor" not in engine.stream_kwargs
        assert _chat_cap_probe == [None]

    def test_responses_stream_forwards_native_level(self, _rate_limiter_state):
        engine = _RouteEngine(QWEN38_TEMPLATE)
        with _client(engine, surface="responses").stream(
            "POST",
            "/v1/responses",
            json=_responses_body(reasoning={"effort": "low"}, stream=True),
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert "response.completed" in body
        assert engine.stream_kwargs.get("chat_template_kwargs") == {
            "reasoning_effort": "low"
        }
        assert "reasoning_budget_logits_processor" not in engine.stream_kwargs

    def test_responses_stream_forwards_client_template_kwargs(
        self, _rate_limiter_state
    ):
        engine = _RouteEngine(ENABLE_THINKING_ONLY_TEMPLATE)
        with _client(engine, surface="responses").stream(
            "POST",
            "/v1/responses",
            json=_responses_body(
                chat_template_kwargs={"custom_flag": True}, stream=True
            ),
        ) as resp:
            assert resp.status_code == 200
            "".join(resp.iter_text())
        assert (
            engine.stream_kwargs.get("chat_template_kwargs", {}).get("custom_flag")
            is True
        )
