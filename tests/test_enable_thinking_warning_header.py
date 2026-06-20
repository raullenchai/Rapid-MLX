# SPDX-License-Identifier: Apache-2.0
"""Regression tests for L-05 — ``X-RapidMLX-Warning`` header surfaces
silent ``enable_thinking`` drops on non-Qwen reasoning parsers.

Pre-fix: ``chat_template_kwargs={"enable_thinking": false}`` was honored
by the ``qwen3`` parser (its chat template skipped the ``<think>`` pre-
injection and the parser's Case-4 fallback only fired under ``True``).
All other registered parsers either:
  * accepted the flag for signature parity but ``del enable_thinking``
    immediately (``gemma4``, ``gpt_oss``, ``harmony``, ``minimax``,
    ``glm4``), or
  * only consulted ``enable_thinking=True`` and ignored ``False``
    entirely (``deepseek_r1``, ``vibethinker``, ``think_parser``).

A client setting ``enable_thinking=False`` on a phi-4-mini-reasoning
deployment (deepseek_r1 parser) got reasoning_content back anyway with
no signal that their hint was unhonored.

Post-fix: ``enable_thinking_warning_header(request, parser_name)`` builds
``{"X-RapidMLX-Warning": "enable_thinking ignored for parser=<name>"}``
when the client EXPLICITLY set ``chat_template_kwargs.enable_thinking``
AND the parser is not in ``_THINKING_FLAG_HONORING_PARSERS``. The chat
route propagates this dict into ``Response(headers=...)`` (non-stream)
and ``StreamingResponse(headers=...)`` (stream).

These tests pin the pure helper. End-to-end response wiring is exercised
by the existing route tests + this file's helper-level coverage of the
matrix (parsers × request shapes).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_mlx.service.helpers import (
    _THINKING_FLAG_HONORING_PARSERS,
    enable_thinking_warning_header,
)


# ──────────────────────────────────────────────────────────────────────
# Source of truth: only ``qwen3`` honors ``enable_thinking``
# ──────────────────────────────────────────────────────────────────────


def test_honoring_parsers_set_is_just_qwen3() -> None:
    """The whitelist is intentionally narrow — every other registered
    parser either ``del enable_thinking`` for signature parity or only
    consults ``True`` for Case-4 routing. If a future parser truly
    honors ``False`` as a strict switch, it MUST be added here AND its
    chat template MUST skip the ``<think>`` pre-injection."""
    assert _THINKING_FLAG_HONORING_PARSERS == frozenset({"qwen3"})


# ──────────────────────────────────────────────────────────────────────
# Headline fires: ctk.enable_thinking + non-Qwen parser
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "parser",
    [
        "deepseek_r1",
        "vibethinker",
        "glm4",
        "gemma4",
        "gpt_oss",
        "harmony",
        "minimax",
    ],
)
def test_warning_fires_for_non_qwen_parser_with_explicit_false(parser: str) -> None:
    """Headline L-05: client sent ``chat_template_kwargs.enable_thinking=False``
    on a non-Qwen parser → ``X-RapidMLX-Warning`` carries the parser
    name so the client can decide what to do (downgrade reasoning_content
    handling, retry against a different deployment, etc.)."""
    request = SimpleNamespace(
        chat_template_kwargs={"enable_thinking": False}, enable_thinking=None
    )
    headers = enable_thinking_warning_header(request, parser)
    assert headers == {
        "X-RapidMLX-Warning": f"enable_thinking ignored for parser={parser}"
    }


def test_warning_also_fires_when_explicit_true_on_non_qwen() -> None:
    """The warning is about "your hint was dropped" regardless of which
    bool the hint was. A client explicitly opting INTO thinking on a
    parser that doesn't gate on the flag still gets the same drop —
    and still wants the signal."""
    request = SimpleNamespace(
        chat_template_kwargs={"enable_thinking": True}, enable_thinking=None
    )
    headers = enable_thinking_warning_header(request, "deepseek_r1")
    assert (
        headers.get("X-RapidMLX-Warning")
        == "enable_thinking ignored for parser=deepseek_r1"
    )


# ──────────────────────────────────────────────────────────────────────
# Silent paths: qwen3 (honors), no ctk (no client opinion), no parser
# ──────────────────────────────────────────────────────────────────────


def test_no_warning_for_qwen3_parser() -> None:
    """qwen3 actually honors the flag — no warning."""
    request = SimpleNamespace(
        chat_template_kwargs={"enable_thinking": False}, enable_thinking=None
    )
    assert enable_thinking_warning_header(request, "qwen3") == {}


def test_no_warning_when_chat_template_kwargs_absent() -> None:
    """L-05 fires only when the client EXPLICITLY set the OpenAI-ext
    ``chat_template_kwargs.enable_thinking`` key. A request without
    any ctk shouldn't get the noise — there's no ignored hint to
    warn about."""
    request = SimpleNamespace(chat_template_kwargs=None, enable_thinking=None)
    assert enable_thinking_warning_header(request, "deepseek_r1") == {}


def test_no_warning_when_ctk_present_but_no_enable_thinking_key() -> None:
    """``chat_template_kwargs={"some_other_kwarg": "x"}`` — no
    enable_thinking on this request, so nothing was dropped silently."""
    request = SimpleNamespace(
        chat_template_kwargs={"some_other_kwarg": "x"}, enable_thinking=None
    )
    assert enable_thinking_warning_header(request, "deepseek_r1") == {}


def test_no_warning_for_top_level_enable_thinking_only() -> None:
    """Top-level ``request.enable_thinking`` is the rapid-mlx-extension
    field. Surface area for L-05 is the OpenAI-spec ``ctk`` channel —
    clients using the top-level field are already mlx-aware and the
    matrix is documented in the parser-support table. Keeping the
    warning narrow avoids header noise on every rapid-mlx-native
    request."""
    request = SimpleNamespace(chat_template_kwargs=None, enable_thinking=False)
    assert enable_thinking_warning_header(request, "deepseek_r1") == {}


def test_no_warning_when_parser_name_is_none() -> None:
    """No parser configured → no parser to ignore the hint. Silent."""
    request = SimpleNamespace(
        chat_template_kwargs={"enable_thinking": False}, enable_thinking=None
    )
    assert enable_thinking_warning_header(request, None) == {}


# ──────────────────────────────────────────────────────────────────────
# Header shape contract — spec-aligned, single key, ASCII-safe
# ──────────────────────────────────────────────────────────────────────


def test_header_name_is_x_rapidmlx_warning() -> None:
    """The header MUST be exactly ``X-RapidMLX-Warning`` — the L-05 spec
    pins this name. Renaming would break any client that sniffs it."""
    request = SimpleNamespace(
        chat_template_kwargs={"enable_thinking": False}, enable_thinking=None
    )
    headers = enable_thinking_warning_header(request, "deepseek_r1")
    assert list(headers.keys()) == ["X-RapidMLX-Warning"]


def test_header_value_is_ascii_safe_and_carries_parser_name() -> None:
    """The header value carries the parser name so the client can route
    differently (e.g. retry against a deployment running ``qwen3``).
    Must be plain ASCII so it survives every HTTP/1.1 hop."""
    request = SimpleNamespace(
        chat_template_kwargs={"enable_thinking": False}, enable_thinking=None
    )
    value = enable_thinking_warning_header(request, "vibethinker")[
        "X-RapidMLX-Warning"
    ]
    assert value == "enable_thinking ignored for parser=vibethinker"
    # ASCII-only — no smart quotes, no unicode dashes that could
    # confuse an HTTP/1.1 hop or a header-sniffing client.
    assert value.encode("ascii")  # raises if it contains non-ASCII bytes
