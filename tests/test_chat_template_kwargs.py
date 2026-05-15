# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #387: chat_template_kwargs.enable_thinking passthrough.

Before this fix, ``ChatCompletionRequest`` did not declare a
``chat_template_kwargs`` field at all, so Pydantic silently dropped the
key on parse and the chat template ran with thinking enabled by default.
External user @smallhadroncollider hit this on v0.6.49 with qwen3.6-27b-8bit
(2996 reasoning tokens for a one-line joke). See issue #387.

The contract under test is the precedence in ``_resolve_enable_thinking``:
    1. server ``--no-thinking`` (cfg.no_thinking) → False
    2. ``request.chat_template_kwargs["enable_thinking"]`` (OpenAI ext spec)
    3. ``request.enable_thinking`` (top-level field, our extension)
    4. None (template default)

These tests guard the helper, the request-model field, and the call sites
in routes/chat.py + routes/anthropic.py + speculative/dflash/server.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.service.helpers import (
    _extract_thinking_from_request,
    _resolve_enable_thinking,
)


def _fake_cfg(no_thinking: bool = False) -> SimpleNamespace:
    """Minimal stand-in for the ServerConfig singleton.

    The helper only reads ``cfg.no_thinking`` — keeping this a SimpleNamespace
    avoids dragging the full singleton + cascade machinery into a unit test.
    """
    return SimpleNamespace(no_thinking=no_thinking)


# ── Pydantic model: chat_template_kwargs is now declared ────────────────


class TestRequestModel:
    def test_chat_template_kwargs_field_accepted(self):
        """Before #387 fix: this dict was silently dropped on parse."""
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": False},
        )
        assert r.chat_template_kwargs == {"enable_thinking": False}

    def test_chat_template_kwargs_default_is_none(self):
        r = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        assert r.chat_template_kwargs is None

    def test_chat_template_kwargs_arbitrary_keys_preserved(self):
        """Unknown keys are accepted — caller may inspect them later."""
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": True, "future_key": "x"},
        )
        assert r.chat_template_kwargs == {"enable_thinking": True, "future_key": "x"}


# ── _resolve_enable_thinking precedence ─────────────────────────────────


class TestResolveEnableThinking:
    def test_chat_template_kwargs_false_overrides_default(self):
        """The actual #387 bug: this used to be silently dropped."""
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": False},
        )
        with patch("vllm_mlx.service.helpers.get_config", return_value=_fake_cfg()):
            assert _resolve_enable_thinking(r) is False

    def test_chat_template_kwargs_true_propagates(self):
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        with patch("vllm_mlx.service.helpers.get_config", return_value=_fake_cfg()):
            assert _resolve_enable_thinking(r) is True

    def test_top_level_enable_thinking_used_when_no_ctk(self):
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            enable_thinking=True,
        )
        with patch("vllm_mlx.service.helpers.get_config", return_value=_fake_cfg()):
            assert _resolve_enable_thinking(r) is True

    def test_chat_template_kwargs_wins_over_top_level(self):
        """Both set: nested key wins (matches OpenAI spec semantics)."""
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": False},
            enable_thinking=True,
        )
        with patch("vllm_mlx.service.helpers.get_config", return_value=_fake_cfg()):
            assert _resolve_enable_thinking(r) is False

    def test_server_no_thinking_overrides_everything(self):
        """Operator-level --no-thinking is the hard kill switch."""
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": True},
            enable_thinking=True,
        )
        with patch(
            "vllm_mlx.service.helpers.get_config",
            return_value=_fake_cfg(no_thinking=True),
        ):
            assert _resolve_enable_thinking(r) is False

    def test_unset_returns_none(self):
        """None lets the chat template apply its own default."""
        r = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        with patch("vllm_mlx.service.helpers.get_config", return_value=_fake_cfg()):
            assert _resolve_enable_thinking(r) is None

    def test_string_form_false_tolerated(self):
        """Some HTTP clients stringify booleans — accept "false" / "true"."""
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": "false"},
        )
        with patch("vllm_mlx.service.helpers.get_config", return_value=_fake_cfg()):
            assert _resolve_enable_thinking(r) is False

    def test_string_form_true_tolerated(self):
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": "TRUE"},
        )
        with patch("vllm_mlx.service.helpers.get_config", return_value=_fake_cfg()):
            assert _resolve_enable_thinking(r) is True

    def test_garbage_value_in_ctk_falls_through(self):
        """Non-bool, non-string-bool values should not poison the precedence."""
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": 42},
            enable_thinking=True,
        )
        with patch("vllm_mlx.service.helpers.get_config", return_value=_fake_cfg()):
            # ctk value rejected, fall through to top-level
            assert _resolve_enable_thinking(r) is True

    def test_chat_template_kwargs_without_enable_thinking_key_falls_through(self):
        """A ctk dict that doesn't carry the key is not opinionated about thinking."""
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"some_other_key": "x"},
            enable_thinking=False,
        )
        with patch("vllm_mlx.service.helpers.get_config", return_value=_fake_cfg()):
            assert _resolve_enable_thinking(r) is False


# ── _extract_thinking_from_request: shared with dflash route ────────────


class TestExtractThinkingFromRequest:
    """The dflash route shares this sub-helper directly. The OpenAI/anthropic
    helper layers ``cfg.no_thinking`` on top; dflash layers its own closure
    ``no_thinking`` arg on top. The string-bool tolerance lives here once.
    """

    def test_ctk_false(self):
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": False},
        )
        assert _extract_thinking_from_request(r) is False

    def test_ctk_wins_over_top_level(self):
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": False},
            enable_thinking=True,
        )
        assert _extract_thinking_from_request(r) is False

    def test_top_level_used_when_no_ctk(self):
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            enable_thinking=True,
        )
        assert _extract_thinking_from_request(r) is True

    def test_neither_set_returns_none(self):
        r = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        assert _extract_thinking_from_request(r) is None

    def test_string_bool_tolerance_lives_here(self):
        """If we ever extend the tolerance (e.g. "1"/"0") this is the one
        function to update — both routes pick it up automatically."""
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": "false"},
        )
        assert _extract_thinking_from_request(r) is False

    def test_does_not_consult_global_config(self):
        """The dflash route relies on this — must not touch get_config().

        Even with cfg.no_thinking=True patched, the request-only helper must
        return what's in the request (the cfg consult happens in the OpenAI/
        anthropic wrapper, not here).
        """
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        with patch(
            "vllm_mlx.service.helpers.get_config",
            return_value=_fake_cfg(no_thinking=True),
        ):
            assert _extract_thinking_from_request(r) is True


# ── dflash inline precedence (closure no_thinking + extractor) ──────────


class TestDflashPrecedence:
    """Mirror of the dflash route's branch in
    ``vllm_mlx/speculative/dflash/server.py`` — kept in sync via the
    shared ``_extract_thinking_from_request`` helper.
    """

    @staticmethod
    def _resolve_dflash(no_thinking: bool, request) -> bool | None:
        """Replicates the dflash route's enable_thinking resolution."""
        if no_thinking:
            return False
        return _extract_thinking_from_request(request)

    def test_closure_no_thinking_overrides_ctk_true(self):
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        assert self._resolve_dflash(no_thinking=True, request=r) is False

    def test_closure_no_thinking_false_lets_ctk_through(self):
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": False},
        )
        assert self._resolve_dflash(no_thinking=False, request=r) is False

    def test_closure_off_top_level_true_returns_true(self):
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            enable_thinking=True,
        )
        assert self._resolve_dflash(no_thinking=False, request=r) is True

    def test_dflash_does_not_consult_cfg_no_thinking(self):
        """Regression for codex round 1 finding #2: dflash must NOT inherit
        the OpenAI route's cfg.no_thinking semantics. Even if cfg says off,
        dflash with closure no_thinking=False must respect ctk=True.
        """
        r = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        with patch(
            "vllm_mlx.service.helpers.get_config",
            return_value=_fake_cfg(no_thinking=True),
        ):
            assert self._resolve_dflash(no_thinking=False, request=r) is True
