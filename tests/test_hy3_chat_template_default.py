# SPDX-License-Identifier: Apache-2.0
"""
Regression test for the Hy3 ``reasoning_effort`` default override.

The Tencent Hunyuan 3 (Hy3) chat_template.jinja defaults ``reasoning_effort``
to ``no_think`` which empirically returns "France" instead of "Paris" for
factual-recall questions (upstream PR #1211 comment 4927711484, observed
2026-07-09 during the vendor drop's boot-time spike on
``mlx-community/Hy3-preview-4bit``). Rapid-mlx overrides this default to
``low`` at the ``apply_chat_template`` boundary so out-of-the-box requests
produce correct answers without the client having to learn the template
kwarg.

This test uses a mock template applicator (rather than a live tokenizer)
so it stays hermetic — no HY3 weights required.
"""

from __future__ import annotations

import pytest

from vllm_mlx.utils.chat_template import _looks_like_hy3, apply_chat_template


@pytest.mark.parametrize(
    "name,expected",
    [
        ("hy3-preview-4bit", True),
        ("mlx-community/Hy3-preview-4bit", True),
        ("Hy3-Preview-4bit", True),  # case-insensitive
        ("Hunyuan-3-Preview", True),  # dash separator
        ("hunyuan3", True),
        ("hy-v3-experimental", True),
        # Negatives — must NOT match.
        ("qwen3.5-4b-4bit", False),
        ("qwen3.6-35b-a3b-mtp-4bit", False),
        ("mlx-community/Qwen3.6-27B-4bit", False),
        ("gemma4-27b-8bit", False),
        ("", False),
        # Codex round-3 NIT (PR #1070 finding #4): unanchored substring
        # matches leaked family detection into unrelated names. Lock the
        # boundary behaviour so an incidental ``hunyuanx3`` /
        # ``mymodelhy3embedded`` substring doesn't get Hy3-only kwarg
        # injection.
        ("not-hunyuanx3-test", False),
        ("mymodelhy3embedded", False),
        ("mlx-community/hunyuan5-preview", False),
        # codex R13 BLOCKING: ``hy3`` as a PARENT / org / namespace segment must
        # NOT match — only the FINAL path segment (repo/alias name) keys the
        # family. Mirrors the model_auto_config.py R11 fix.
        ("hy3/qwen-model", False),
        ("some/hy3/nested-qwen", False),
        # ``org/hy3`` (a repo literally named hy3) DOES match — hy3 is the
        # basename there.
        ("org/hy3", True),
    ],
)
def test_looks_like_hy3_predicate(name: str, expected: bool) -> None:
    assert _looks_like_hy3(name) is expected


class _CapturingTokenizer:
    """Mock template applicator that captures the kwargs it was called with.

    Mirrors the ``PreTrainedTokenizerBase.apply_chat_template`` shape well
    enough for our purposes — has an ``apply_chat_template`` method that
    returns a stub prompt and records the kwargs so the test can assert
    on the presence / value of ``reasoning_effort``.
    """

    def __init__(self):
        self.captured_kwargs: dict = {}

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.captured_kwargs = kwargs
        return "<stub prompt>"


def test_hy3_default_reasoning_effort_low_is_injected():
    """The load-bearing fix. When ``model_name`` looks like Hy3 and the
    caller does NOT pass ``enable_thinking=False``, the default template
    kwarg ``reasoning_effort='low'`` MUST be injected."""
    tok = _CapturingTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "Capital of France?"}],
        model_name="mlx-community/Hy3-preview-4bit",
    )
    assert tok.captured_kwargs.get("reasoning_effort") == "low"


def test_hy3_default_not_injected_when_enable_thinking_false():
    """Client explicitly disabled thinking — the ``no_think`` default is
    what they want. The parser MUST respect that intent and NOT override
    ``reasoning_effort``."""
    tok = _CapturingTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "Capital of France?"}],
        enable_thinking=False,
        model_name="mlx-community/Hy3-preview-4bit",
    )
    assert "reasoning_effort" not in tok.captured_kwargs


def test_hy3_reasoning_effort_template_still_honors_enable_thinking_false():
    """Hy3 also uses a ``reasoning_effort`` template knob, but
    ``enable_thinking=False`` must leave its native ``no_think`` default
    untouched instead of borrowing GPT-OSS's ``low`` workaround."""

    class Hy3LikeTokenizer(_CapturingTokenizer):
        chat_template = (
            "{% if reasoning_effort is not defined %}"
            "{% set reasoning_effort = 'no_think' %}{% endif %}"
            "Reasoning: {{ reasoning_effort }}"
        )

    tok = Hy3LikeTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=False,
        model_name="mlx-community/Hy3-preview-4bit",
    )
    assert tok.captured_kwargs.get("enable_thinking") is False
    assert "reasoning_effort" not in tok.captured_kwargs


def test_gpt_oss_template_gets_low_effort_when_thinking_disabled():
    """GPT-OSS/Harmony templates ignore ``enable_thinking`` but honor
    ``reasoning_effort``. A route-level thinking-off decision should
    render as the closest native setting: ``Reasoning: low``."""

    class GptOssLikeTokenizer(_CapturingTokenizer):
        chat_template = (
            "{% if reasoning_effort is not defined %}"
            "{% set reasoning_effort = 'medium' %}{% endif %}"
            "Reasoning: {{ reasoning_effort }}"
        )

    tok = GptOssLikeTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=False,
        model_name="66ton99/gpt-oss-120b",
    )
    assert tok.captured_kwargs.get("enable_thinking") is False
    assert tok.captured_kwargs.get("reasoning_effort") == "low"


def test_gpt_oss_harmony_template_alias_gets_low_effort_when_thinking_disabled():
    """Served aliases should still get the GPT-OSS/Harmony low-effort
    mapping when the template itself exposes the Harmony shape."""

    class HarmonyAliasTokenizer(_CapturingTokenizer):
        chat_template = (
            "<|start|>{{ role }}<|channel|>{{ channel }}<|message|>"
            "{% if reasoning_effort is not defined %}"
            "{% set reasoning_effort = 'medium' %}{% endif %}"
            "Reasoning: {{ reasoning_effort }}"
        )

    tok = HarmonyAliasTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=False,
        model_name="prod-alias",
    )
    assert tok.captured_kwargs.get("enable_thinking") is False
    assert tok.captured_kwargs.get("reasoning_effort") == "low"


def test_gpt_oss_dict_chat_template_gets_low_effort_when_thinking_disabled():
    """HF tokenizers may expose ``chat_template`` as a named-template dict."""

    class DictTemplateTokenizer(_CapturingTokenizer):
        chat_template = {
            "default": (
                "{% if reasoning_effort is not defined %}"
                "{% set reasoning_effort = 'medium' %}{% endif %}"
                "Reasoning: {{ reasoning_effort }}"
            )
        }

    tok = DictTemplateTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=False,
        model_name="66ton99/gpt-oss-120b",
    )
    assert tok.captured_kwargs.get("enable_thinking") is False
    assert tok.captured_kwargs.get("reasoning_effort") == "low"


def test_gpt_oss_low_effort_survives_enable_thinking_retry():
    """Some template applicators reject unknown ``enable_thinking`` kwargs.
    The first retry drops only that kwarg and must keep
    ``reasoning_effort='low'``."""

    class EnableThinkingRejectingGptOssTokenizer:
        chat_template = (
            "{% if reasoning_effort is not defined %}"
            "{% set reasoning_effort = 'medium' %}{% endif %}"
            "Reasoning: {{ reasoning_effort }}"
        )

        def __init__(self):
            self.calls: list[dict] = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(dict(kwargs))
            if "enable_thinking" in kwargs:
                raise TypeError(
                    "apply_chat_template() got unexpected keyword "
                    "argument 'enable_thinking'"
                )
            return "<gpt-oss prompt>"

    tok = EnableThinkingRejectingGptOssTokenizer()
    result = apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=False,
        model_name="66ton99/gpt-oss-120b",
    )
    assert result == "<gpt-oss prompt>"
    assert len(tok.calls) == 2
    assert tok.calls[0].get("reasoning_effort") == "low"
    assert tok.calls[0].get("enable_thinking") is False
    assert tok.calls[1].get("reasoning_effort") == "low"
    assert "enable_thinking" not in tok.calls[1]


def test_non_hy3_model_never_sees_reasoning_effort_kwarg():
    """Every other family's chat template rejects unknown kwargs with
    ``TypeError``. The Hy3-only injection MUST NOT fire for other models
    or the fallback retry chain would kick in unnecessarily."""
    tok = _CapturingTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "Hello"}],
        model_name="qwen3.5-4b-4bit",
    )
    assert "reasoning_effort" not in tok.captured_kwargs


def test_hy3_parent_segment_model_does_not_get_effort_injected():
    """codex R13 BLOCKING: a non-Hy3 model living under an org / parent
    directory named ``hy3`` (``hy3/qwen-model``) must NOT get
    ``reasoning_effort='low'`` injected — the family root must be the repo
    BASENAME, not a parent path segment. Otherwise a Qwen (or any) model served
    from such a path would silently receive the Hy3-only kwarg and hit the
    TypeError fallback chain (or, worse, mis-condition a template that happens
    to accept it)."""
    tok = _CapturingTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "Capital of France?"}],
        model_name="hy3/qwen-model",
    )
    assert "reasoning_effort" not in tok.captured_kwargs


def test_hy3_default_survives_enable_thinking_true():
    """Explicit ``enable_thinking=True`` is orthogonal — thinking is on
    AND the default effort should be ``low`` (not ``no_think``)."""
    tok = _CapturingTokenizer()
    apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        enable_thinking=True,
        model_name="hy3-preview-4bit",
    )
    assert tok.captured_kwargs.get("reasoning_effort") == "low"
    assert tok.captured_kwargs.get("enable_thinking") is True


def test_hy3_default_dropped_only_after_second_type_error():
    """Two-stage retry (codex round-1 NIT fix). The first retry drops
    ``enable_thinking`` and KEEPS ``reasoning_effort`` — a Hy3
    checkpoint that supports the effort override but rejects
    ``enable_thinking`` MUST still see the ``low`` value on retry.
    Only the SECOND TypeError drops ``reasoning_effort``."""

    class EnableThinkingFlakyTokenizer:
        """Rejects ``enable_thinking`` only; accepts ``reasoning_effort``."""

        def __init__(self):
            self.calls: list[dict] = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(dict(kwargs))
            if "enable_thinking" in kwargs:
                raise TypeError(
                    "apply_chat_template() got unexpected keyword "
                    "argument 'enable_thinking'"
                )
            return "<enable_thinking-dropped ok>"

    tok = EnableThinkingFlakyTokenizer()
    result = apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        model_name="hy3-preview-4bit",
    )
    # First call had both; retry dropped enable_thinking but preserved
    # reasoning_effort=low. Codex round-3 NIT (PR #1070 finding #5):
    # assert the exact expected value (``True`` — the default when the
    # caller doesn't pass ``enable_thinking`` and the model_name isn't
    # a coder alias) rather than the looser ``is not None`` which
    # would silently accept an invalid value.
    assert len(tok.calls) == 2
    assert tok.calls[0].get("enable_thinking") is True
    assert tok.calls[0].get("reasoning_effort") == "low"
    assert "enable_thinking" not in tok.calls[1]
    assert tok.calls[1].get("reasoning_effort") == "low"
    assert result == "<enable_thinking-dropped ok>"


def test_hy3_default_dropped_when_reasoning_effort_alone_is_rejected():
    """Realistic degradation path — checkpoint accepts ``enable_thinking``
    but not ``reasoning_effort``. First call fails because of
    ``reasoning_effort``; the two-stage retry drops it and Step 2's
    tools-restore path (which re-adds ``enable_thinking``) succeeds."""

    class ReasoningEffortFlakyTokenizer:
        """Rejects ``reasoning_effort`` only; accepts ``enable_thinking``."""

        def __init__(self):
            self.calls: list[dict] = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(dict(kwargs))
            if "reasoning_effort" in kwargs:
                raise TypeError(
                    "apply_chat_template() got unexpected keyword "
                    "argument 'reasoning_effort'"
                )
            return "<reasoning_effort-dropped ok>"

    tok = ReasoningEffortFlakyTokenizer()
    result = apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        model_name="hy3-preview-4bit",
    )
    # First call had reasoning_effort=low and enable_thinking; failed.
    # Retry-1 dropped enable_thinking but kept reasoning_effort; failed.
    # Second-TypeError handler dropped reasoning_effort; Step 2
    # tools-restore re-added enable_thinking; the final call succeeded.
    assert tok.calls[0].get("reasoning_effort") == "low"
    assert "reasoning_effort" not in tok.calls[-1]
    assert result == "<reasoning_effort-dropped ok>"


def test_hy3_reasoning_effort_survives_tools_fallback():
    """codex R8 BLOCKING: when the template rejects ``tools`` (NOT
    ``reasoning_effort``), the two-stage retry must NOT drop
    ``reasoning_effort`` before the prompt-injection tools fallback — otherwise
    a Hy3 request on a tools-rejecting template silently loses the load-bearing
    ``reasoning_effort='low'`` override. The tools fallback must carry it."""

    class ToolsRejectingTokenizer:
        """Rejects ``enable_thinking`` and ``tools``; accepts
        ``reasoning_effort``. The first call fails on enable_thinking, retry-1
        fails on tools, and the prompt-injection fallback (no tools kwarg)
        must succeed WITH reasoning_effort still present."""

        def __init__(self):
            self.calls: list[dict] = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(dict(kwargs))
            if "enable_thinking" in kwargs:
                raise TypeError(
                    "apply_chat_template() got unexpected keyword "
                    "argument 'enable_thinking'"
                )
            if "tools" in kwargs:
                raise TypeError(
                    "apply_chat_template() got unexpected keyword argument 'tools'"
                )
            return "<tools-injected ok>"

    tok = ToolsRejectingTokenizer()
    result = apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        model_name="hy3-preview-4bit",
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    assert result == "<tools-injected ok>"
    # The FINAL (prompt-injection) call MUST still carry reasoning_effort=low —
    # the second TypeError was about tools, not reasoning_effort.
    assert tok.calls[-1].get("reasoning_effort") == "low"
    assert "tools" not in tok.calls[-1]


def test_reasoning_effort_not_dropped_on_unrelated_error_mentioning_it():
    """codex R9 NIT: the drop decision matches Python's ACTUAL unexpected-kwarg
    error text, not a loose substring. A ``tools`` failure whose message merely
    MENTIONS ``reasoning_effort`` in passing must NOT drop the override — only
    ``unexpected keyword argument 'reasoning_effort'`` triggers the drop."""

    class MisleadingErrorTokenizer:
        """Rejects enable_thinking, then tools with a message that incidentally
        mentions reasoning_effort (but is NOT an unexpected-kwarg error for
        it)."""

        def __init__(self):
            self.calls: list[dict] = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(dict(kwargs))
            if "enable_thinking" in kwargs:
                raise TypeError(
                    "apply_chat_template() got unexpected keyword "
                    "argument 'enable_thinking'"
                )
            if "tools" in kwargs:
                # Message mentions reasoning_effort but the CULPRIT is tools.
                raise TypeError(
                    "template does not support tools when reasoning_effort is set"
                )
            return "<injected ok>"

    tok = MisleadingErrorTokenizer()
    result = apply_chat_template(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        model_name="hy3-preview-4bit",
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    assert result == "<injected ok>"
    # reasoning_effort must SURVIVE — the error text was not the exact
    # unexpected-kwarg shape for reasoning_effort.
    assert tok.calls[-1].get("reasoning_effort") == "low"
