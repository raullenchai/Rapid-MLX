# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for /v1/messages thinking-control parity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rapid_mlx.api.anthropic_adapter import anthropic_to_openai
from rapid_mlx.api.anthropic_models import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicToolDef,
)
from rapid_mlx.routes.anthropic import _apply_anthropic_thinking_defaults
from rapid_mlx.service.helpers import _extract_thinking_from_request


def _request(**kwargs) -> AnthropicRequest:
    payload = {
        "model": "default",
        "messages": [AnthropicMessage(role="user", content="Say hi")],
        "max_tokens": 800,
    }
    payload.update(kwargs)
    return AnthropicRequest(**payload)


def test_omitted_thinking_uses_casual_chat_default_off(monkeypatch) -> None:
    monkeypatch.setattr(
        "rapid_mlx.service.helpers.get_config",
        lambda: SimpleNamespace(reasoning_parser_name="qwen3", no_thinking=False),
    )
    converted = anthropic_to_openai(_request())
    assert _extract_thinking_from_request(converted) is None

    _apply_anthropic_thinking_defaults(converted)

    assert _extract_thinking_from_request(converted) is False


def test_tools_use_shared_default_off(monkeypatch) -> None:
    # Isolate the tools policy: this must keep passing even if the independent
    # casual-chat default is disabled or removed.
    monkeypatch.setattr(
        "rapid_mlx.routes.anthropic.maybe_auto_disable_thinking_for_casual_chat",
        lambda request: False,
    )
    converted = anthropic_to_openai(
        _request(
            tools=[
                AnthropicToolDef(
                    name="lookup",
                    description="Look something up",
                    input_schema={"type": "object"},
                )
            ]
        )
    )

    _apply_anthropic_thinking_defaults(converted)

    assert _extract_thinking_from_request(converted) is False


@pytest.mark.parametrize("explicit", [True, False])
def test_tools_preserve_explicit_extension_preference(explicit: bool) -> None:
    converted = anthropic_to_openai(
        _request(
            enable_thinking=explicit,
            tools=[
                AnthropicToolDef(
                    name="lookup",
                    description="Look something up",
                    input_schema={"type": "object"},
                )
            ],
        )
    )

    _apply_anthropic_thinking_defaults(converted)

    assert _extract_thinking_from_request(converted) is explicit


def test_native_enabled_budget_preserves_reasoning_intent() -> None:
    converted = anthropic_to_openai(
        _request(thinking={"type": "enabled", "budget_tokens": 16})
    )

    _apply_anthropic_thinking_defaults(converted)

    assert _extract_thinking_from_request(converted) is True
    assert converted.reasoning_max_tokens == 16


def test_legacy_budget_only_shape_preserves_reasoning_intent() -> None:
    converted = anthropic_to_openai(_request(thinking={"budget_tokens": 16}))

    _apply_anthropic_thinking_defaults(converted)

    assert _extract_thinking_from_request(converted) is True
    assert converted.reasoning_max_tokens == 16


def test_native_adaptive_thinking_is_accepted_and_enabled() -> None:
    converted = anthropic_to_openai(_request(thinking={"type": "adaptive"}))

    _apply_anthropic_thinking_defaults(converted)

    assert _extract_thinking_from_request(converted) is True


def test_output_effort_wins_over_legacy_disabled_thinking() -> None:
    converted = anthropic_to_openai(
        _request(
            output_config={"effort": "low"},
            thinking={"type": "disabled"},
        )
    )

    _apply_anthropic_thinking_defaults(converted)

    assert _extract_thinking_from_request(converted) is True
    assert converted.reasoning_max_tokens == 512


@pytest.mark.parametrize("thinking_type", ["disabled", "adaptive"])
def test_non_fixed_thinking_modes_ignore_legacy_budget(thinking_type) -> None:
    converted = anthropic_to_openai(
        _request(thinking={"type": thinking_type, "budget_tokens": 16})
    )

    assert converted.reasoning_max_tokens is None
