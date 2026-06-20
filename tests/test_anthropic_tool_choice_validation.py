# SPDX-License-Identifier: Apache-2.0
"""
M-03 (#742 follow-up): parse-time validation of ``tool_choice.type`` on
``/v1/messages``.

Before the fix, ``tool_choice={"type":"banana"}`` HTTP 200'd with plain
text — the adapter's ``_convert_tool_choice`` fell through to
``return "auto"`` and the request silently degraded to a no-forcing
generation. This file pins the Anthropic-side schema contract:

* Unknown ``type`` → 400 at schema parse, before the adapter runs.
* ``type`` values from the public spec (``auto`` / ``any`` / ``tool``
  / ``none``) → accepted.
* ``type=="tool"`` without a non-empty ``name`` → 400.

The validator lives on ``AnthropicRequest`` (api/anthropic_models.py),
so the assertions here use the model directly (no HTTP fixture
needed); the route already maps Pydantic ``ValidationError`` → 400
(routes/anthropic.py L132-L135), so a model-level 422 IS the wire 400.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vllm_mlx.api.anthropic_models import AnthropicRequest


def _base_request(**overrides):
    """Minimum-valid Anthropic body; callers layer ``tool_choice`` on top."""
    body = {
        "model": "qwen3.5-4b-4bit",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Negative path — unknown / malformed tool_choice
# ---------------------------------------------------------------------------


def test_unknown_tool_choice_type_rejected():
    """The M-03 repro: ``tool_choice={"type":"banana"}`` must 400."""
    with pytest.raises(ValidationError) as exc_info:
        AnthropicRequest(**_base_request(tool_choice={"type": "banana"}))
    msg = str(exc_info.value)
    assert "tool_choice.type" in msg
    assert "banana" in msg


@pytest.mark.parametrize(
    "bad_type",
    [
        "function",  # OpenAI's word, NOT Anthropic's
        "required",  # OpenAI's word for "any"
        "TOOL",  # case-sensitive per spec
        "",  # empty string
        " auto",  # leading whitespace
    ],
)
def test_other_unknown_tool_choice_types_rejected(bad_type):
    """Verify the validator is strict-equality on the spec set, not
    a substring / lowercase match. Catches the class of typos /
    cross-API confusions (OpenAI-shape values landing on the
    Anthropic surface) the M-03 gap silently absorbed."""
    with pytest.raises(ValidationError):
        AnthropicRequest(**_base_request(tool_choice={"type": bad_type}))


def test_non_dict_tool_choice_rejected():
    """``tool_choice`` MUST be an object per the Anthropic spec.
    A string (OpenAI shape) should 400 with a clear shape error
    rather than a downstream AttributeError when the adapter calls
    ``.get("type")``."""
    with pytest.raises(ValidationError) as exc_info:
        AnthropicRequest(**_base_request(tool_choice="auto"))
    assert "tool_choice" in str(exc_info.value)


def test_tool_type_without_name_rejected():
    """``{"type":"tool"}`` with no ``name`` is malformed per the
    Anthropic spec. The Anthropic SDK enforces this client-side,
    but a raw HTTP client could omit ``name`` and the adapter would
    build ``function.name=""`` — let the route 400 here so the
    error message points at the actual missing field."""
    with pytest.raises(ValidationError) as exc_info:
        AnthropicRequest(**_base_request(tool_choice={"type": "tool"}))
    msg = str(exc_info.value)
    assert "name" in msg


def test_tool_type_with_empty_name_rejected():
    """Whitespace-only ``name`` is the same hazard as missing ``name``."""
    with pytest.raises(ValidationError):
        AnthropicRequest(**_base_request(tool_choice={"type": "tool", "name": "   "}))


def test_tool_type_with_non_string_name_rejected():
    """A typo'd ``name: 42`` would otherwise coerce-or-skip silently."""
    with pytest.raises(ValidationError):
        AnthropicRequest(**_base_request(tool_choice={"type": "tool", "name": 42}))


# ---------------------------------------------------------------------------
# Positive path — every spec-legal shape must still pass parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "auto"},
        {"type": "any"},
        {"type": "none"},
        {"type": "tool", "name": "search"},
        # Spec-legal: extra keys are tolerated (Anthropic SDK adds e.g.
        # ``disable_parallel_tool_use``); the validator must not be
        # strict-keys.
        {"type": "auto", "disable_parallel_tool_use": True},
        {"type": "tool", "name": "search", "disable_parallel_tool_use": True},
        # Empty dict — adapter defaults to "auto" (back-compat with
        # TestConvertToolChoice.test_missing_type_defaults_to_auto).
        {},
        # Omitted entirely — null path.
        None,
    ],
)
def test_valid_tool_choice_shapes_accepted(tool_choice):
    req = AnthropicRequest(**_base_request(tool_choice=tool_choice))
    assert req.tool_choice == tool_choice


# ---------------------------------------------------------------------------
# Cross-cutting: validation must NOT mutate the field
# ---------------------------------------------------------------------------


def test_validator_does_not_mutate_tool_choice():
    """The validator is a gate, not a normalizer. The adapter and the
    downstream chat route both read ``request.tool_choice`` as the
    raw dict — silently rewriting it here would mask cross-layer
    contract drift (and a future ``output_config`` may carry a
    similarly-shaped struct we don't want to fold)."""
    original = {"type": "tool", "name": "search", "extra": "preserved"}
    req = AnthropicRequest(**_base_request(tool_choice=original))
    assert req.tool_choice == original
    assert req.tool_choice is not original  # Pydantic copies on construction
