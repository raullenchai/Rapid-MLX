# SPDX-License-Identifier: Apache-2.0
"""Tests for system-prompt canonicalization."""

from vllm_mlx.api.prompt_canonicalize import canonicalize_system_prompt


def test_canonicalize_system_prompt_strips_anthropic_billing_header_line():
    text = (
        "You are a coding assistant.\n"
        "x-anthropic-billing-header: account=abc; cch=rotating-hash\n"
        "Follow the repository instructions."
    )

    expected = "\n".join(
        ["You are a coding assistant.", "Follow the repository instructions."]
    )
    assert canonicalize_system_prompt(text) == expected


def test_canonicalize_system_prompt_is_idempotent_for_clean_input():
    text = "You are a coding assistant.\nCurrent time: 2026-05-12T01:00:00Z"

    assert canonicalize_system_prompt(text) == text


def test_canonicalize_system_prompt_does_not_strip_user_visible_timestamp():
    text = "Current time: 2026-05-12T01:00:00Z\nUse it when answering."

    assert canonicalize_system_prompt(text) == text


def test_canonicalize_system_prompt_handles_empty_and_none():
    assert canonicalize_system_prompt("") == ""
    assert canonicalize_system_prompt(None) is None
