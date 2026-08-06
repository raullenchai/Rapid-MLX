# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the final sanitize_output catch-all.

This sanitizer is the LAST defense against markup leakage in the
non-streaming response path. Any token-level markup that bypasses
the streaming filter (e.g. Gemma 4's `<|tool_call>call:name{...}<tool_call|>`
which is emitted as a small handful of tokens at the end of generation)
must be stripped here, otherwise it leaks into `message.content` even
when `tool_calls` is correctly populated.

Fix history:
- 2026-04-07: regex stripped the bare `<|tool_call>` and `<tool_call|>`
  tokens but left the inner `call:name{...}` body in place. Also, the
  non-streaming chat path never called sanitize_output at all.
"""

import pytest

from vllm_mlx.api.utils import sanitize_output

# ---------------------------------------------------------------------------
# The exact regression cases from the bug report
# ---------------------------------------------------------------------------


def test_strips_full_gemma4_tool_call_bare_args():
    """Original leak: numeric args left the entire markup in content."""
    out = sanitize_output("<|tool_call>call:add{a:3,b:4}<tool_call|>")
    assert out is None  # nothing but markup → collapse to None


def test_strips_full_gemma4_tool_call_quoted_args():
    out = sanitize_output(
        '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    )
    assert out is None


def test_strips_gemma4_markup_inside_content():
    """When markup is sandwiched between real content, leave the content."""
    out = sanitize_output("Result: <|tool_call>call:add{a:3,b:4}<tool_call|> done")
    assert out is not None
    assert "<|tool_call>" not in out
    assert "<tool_call|>" not in out
    assert "call:add" not in out  # body must also be stripped
    assert "Result:" in out
    assert "done" in out


def test_strips_multiple_gemma4_tool_calls():
    out = sanitize_output(
        "First: <|tool_call>call:a{x:1}<tool_call|>"
        " then: <|tool_call>call:b{y:2}<tool_call|>"
    )
    assert out is not None
    assert "call:a" not in out
    assert "call:b" not in out
    assert "First:" in out
    assert "then:" in out


# ---------------------------------------------------------------------------
# Existing token strippers — must still work after the regex changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_clean",
    [
        # symmetric special tokens
        ("Hello<|im_end|>", "Hello"),
        # text-format tool call
        ("[Calling tool: foo]", None),
        ('[Calling tool="foo"]', None),
        # NOTE: ``("answer</tool_call>", "answer")`` used to live here.
        # The content channel no longer strips the ``</tool_call>``
        # closer — see ``test_tool_call_closer_content_integrity.py`` for
        # why (it was deleting the token out of files coding agents had
        # been asked to write). The reasoning channel still strips it and
        # is covered by ``test_reasoning_channel_still_strips_closer``
        # below.
    ],
)
def test_legacy_strippers_still_work(raw, expected_clean):
    out = sanitize_output(raw)
    assert out == expected_clean


# ---------------------------------------------------------------------------
# Pass-through cases — sanitizer must NOT touch normal text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "Hello world",
        "The answer is 42.",
        "Multi\nline\noutput.",
        "",  # empty
        "Code: a = b + c  # comment",
    ],
)
def test_pass_through_normal_text(raw):
    assert sanitize_output(raw) == raw


def test_none_input():
    assert sanitize_output(None) is None


# ---------------------------------------------------------------------------
# Empty-after-strip collapses to None (so callers see message.content=None)
# ---------------------------------------------------------------------------


def test_only_markup_collapses_to_none():
    """If all that's left after stripping is whitespace, return None."""
    cases = [
        "<|tool_call>call:a{x:1}<tool_call|>",
        "<|im_end|>",
        "[Calling tool: foo]",
    ]
    for c in cases:
        assert sanitize_output(c) is None, f"failed: {c!r}"


# ---------------------------------------------------------------------------
# Channel split for the ``</tool_call>`` closer
# ---------------------------------------------------------------------------


def test_content_channel_keeps_closer():
    """Content may legitimately contain the literal closer.

    Full rationale and the end-to-end Claude Code / Codex reproduction
    live in ``test_tool_call_closer_content_integrity.py``.
    """
    assert sanitize_output("answer</tool_call>") == "answer</tool_call>"


def test_reasoning_channel_still_strips_closer():
    """A bare closer inside a reasoning trace is parser residue."""
    from vllm_mlx.api.utils import sanitize_reasoning_content

    assert sanitize_reasoning_content("answer</tool_call>") == "answer"
