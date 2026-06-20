# SPDX-License-Identifier: Apache-2.0
"""
H-16 (Pavel r3): parse-time validation of ``tool_choice`` on
``/v1/chat/completions``.

Before the fix, ``tool_choice={"foo":"bar"}`` (no ``type`` field) and
``tool_choice={"type":"banana"}`` HTTP 200'd as a free-form chat
completion — the typed ``str | dict`` union accepted the dict arm and
the chat-route ``type=='function'`` guard (``vllm_mlx/routes/chat.py``
L756) didn't match, so the request silently degraded with no tool
forcing. PR #766 (M-03) closed the symmetric gap on ``/v1/messages``;
this file pins the same contract for the OpenAI surface.

The validator lives on ``ChatCompletionRequest`` (api/models.py), so
the assertions here use the model directly (no HTTP fixture needed) —
the global validation handler maps Pydantic ``ValidationError`` →
400 ``invalid_request_error`` (middleware/exception_handlers.py).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vllm_mlx.api.models import ChatCompletionRequest


def _base_request(**overrides):
    """Minimum-valid OpenAI body; callers layer ``tool_choice`` on top."""
    body = {
        "model": "qwen3-0.6b-8bit",
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Negative path — unknown / malformed tool_choice
# ---------------------------------------------------------------------------


def test_no_type_field_object_rejected():
    """The H-16 repro: ``tool_choice={"foo":"bar"}`` (no ``type``
    field) must 400. Previously HTTP 200'd as a free-form chat
    completion because the typed dict arm swallowed the shape and
    the chat-route ``type=='function'`` guard didn't fire."""
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(**_base_request(tool_choice={"foo": "bar"}))
    msg = str(exc_info.value)
    assert "tool_choice" in msg
    assert "type" in msg


def test_unknown_object_type_rejected():
    """``{"type":"banana"}`` — an unknown ``type`` value must 400 at
    parse, not silently degrade."""
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(**_base_request(tool_choice={"type": "banana"}))
    msg = str(exc_info.value)
    assert "tool_choice" in msg
    assert "banana" in msg


@pytest.mark.parametrize(
    "bad_type",
    [
        "tool",  # Anthropic's word, NOT OpenAI's
        "any",  # Anthropic's word for "required"
        "FUNCTION",  # case-sensitive per spec
        "",  # empty string
        " function",  # leading whitespace
        "Function",
        # The legacy bare-string ``"function"`` literal is allowed
        # as a string-form tool_choice (route honors it at L1220),
        # but NOT as an object's ``type``. Object form must be
        # the canonical ``{"type":"function",...}``; a typo'd
        # ``{"type":"Function"}`` is a separate hazard.
    ],
)
def test_other_unknown_object_types_rejected(bad_type):
    """Strict-equality on ``type`` — catches cross-API confusions
    (Anthropic-shape values landing on the OpenAI surface) and
    typos / case variants that the silent-degrade path used to
    swallow."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**_base_request(tool_choice={"type": bad_type}))


@pytest.mark.parametrize(
    "bad_string",
    [
        "banana",
        "any",  # Anthropic's word
        "tool",  # Anthropic's word
        "AUTO",  # case-sensitive per spec
        " required",  # leading whitespace
        "Auto",
        "",
    ],
)
def test_unknown_strings_rejected(bad_string):
    """The string form is closed-set: only ``none`` / ``auto`` /
    ``required``. Reject anything else so a client typo or
    cross-API confusion 400s instead of HTTP-200'ing as silent
    free-form generation."""
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(**_base_request(tool_choice=bad_string))
    msg = str(exc_info.value)
    assert "tool_choice" in msg


@pytest.mark.parametrize(
    "bad_value",
    [
        42,
        3.14,
        [1, 2, 3],
        ["function"],
        # ``True`` would coerce-or-skip the union arms; flat reject.
        True,
        False,
    ],
)
def test_non_string_non_object_rejected(bad_value):
    """``tool_choice`` is either a string or an object — anything
    else (numbers, lists, booleans) is malformed. Before the fix,
    the pydantic union-attempt error leaked both arm names
    (``tool_choice.str: ...; tool_choice.dict[any,any]: ...``);
    after the fix, a single ``tool_choice`` ValueError points
    at the field with a clean message."""
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(**_base_request(tool_choice=bad_value))
    assert "tool_choice" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Positive path — every spec-legal shape must still pass parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("legal_string", ["none", "auto", "required"])
def test_legal_strings_accepted(legal_string):
    """The full OpenAI string-form set is accepted. ``"required"``
    additionally requires a non-empty ``tools`` array (F-034) —
    layer one on so the ``_validate_tool_choice_against_tools``
    after-validator doesn't fire."""
    req = ChatCompletionRequest(
        **_base_request(
            tool_choice=legal_string,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "ping", "parameters": {"type": "object"}},
                }
            ],
        )
    )
    assert req.tool_choice == legal_string


def test_legacy_function_literal_accepted():
    """Pre-2024 OpenAI SDKs sent ``tool_choice="function"`` (bare
    string literal) to mean "force any function call" before the
    dict form was added. The chat-route ``_forced_tool_choice``
    gate (routes/chat.py L1212, codex r9 NIT #1 on #551) still
    honors it — so the schema must let it pass to the route. Pin
    by test_diffusion_engine.py::
    test_engine_opts_out_blocks_legacy_function_literal_tool_choice
    which would regress if we tightened to the modern triplet."""
    req = ChatCompletionRequest(
        **_base_request(
            tool_choice="function",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "ping", "parameters": {"type": "object"}},
                }
            ],
        )
    )
    assert req.tool_choice == "function"


def test_legal_object_form_accepted():
    """``{"type":"function","function":{"name":"X"}}`` — the
    canonical OpenAI named-tool form."""
    tc = {"type": "function", "function": {"name": "ping"}}
    req = ChatCompletionRequest(
        **_base_request(
            tool_choice=tc,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "ping", "parameters": {"type": "object"}},
                }
            ],
        )
    )
    assert req.tool_choice == tc


def test_legal_object_form_with_extra_keys_accepted():
    """Extra keys on the object form are tolerated — OpenAI's
    contract is "extra keys ignored" (forward-compat). Mirror the
    same wording M-03's validator uses on the Anthropic surface."""
    tc = {
        "type": "function",
        "function": {"name": "ping"},
        "extra": "preserved",
    }
    req = ChatCompletionRequest(
        **_base_request(
            tool_choice=tc,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "ping", "parameters": {"type": "object"}},
                }
            ],
        )
    )
    assert req.tool_choice == tc


def test_omitted_tool_choice_accepted():
    """Absent ``tool_choice`` (the most common case) — server picks
    the default policy."""
    req = ChatCompletionRequest(**_base_request())
    assert req.tool_choice is None


def test_explicit_null_tool_choice_accepted():
    """``tool_choice=null`` (JSON null) — same semantics as
    omission. Mirrors M-03's ``test_valid_tool_choice_shapes_accepted``
    null entry."""
    req = ChatCompletionRequest(**_base_request(tool_choice=None))
    assert req.tool_choice is None


# ---------------------------------------------------------------------------
# Boundary: the route-level guard at chat.py L756 still owns
# ``type=='function'`` without ``function.name`` (more informative
# message including the F-145 case-insensitive name hint). The
# schema validator must NOT pre-empt that path.
# ---------------------------------------------------------------------------


def test_function_type_without_name_passes_schema():
    """``{"type":"function"}`` with no ``function`` field — schema
    accepts (the route 400s with a more informative
    ``tool_choice with type='function' requires function.name``
    message). Mirrors the M-03 deferral discipline: don't duplicate
    a downstream check that already produces a better message."""
    tc = {"type": "function"}
    req = ChatCompletionRequest(**_base_request(tool_choice=tc))
    assert req.tool_choice == tc


def test_function_type_with_empty_name_passes_schema():
    """Same as above — empty ``function.name`` is the route's job.
    Schema only catches the shapes the route silently accepts."""
    tc = {"type": "function", "function": {"name": ""}}
    req = ChatCompletionRequest(**_base_request(tool_choice=tc))
    assert req.tool_choice == tc


# ---------------------------------------------------------------------------
# Cross-cutting: validation must NOT mutate the field
# ---------------------------------------------------------------------------


def test_validator_does_not_mutate_tool_choice():
    """The validator is a gate, not a normalizer. The chat route
    reads ``request.tool_choice`` as the raw value — silently
    rewriting it here would mask cross-layer contract drift.
    Mirrors M-03's ``test_validator_does_not_mutate_tool_choice``."""
    original = {
        "type": "function",
        "function": {"name": "ping"},
        "extra": "preserved",
    }
    req = ChatCompletionRequest(
        **_base_request(
            tool_choice=original,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "ping", "parameters": {"type": "object"}},
                }
            ],
        )
    )
    assert req.tool_choice == original
    assert req.tool_choice is not original  # Pydantic copies on construction
