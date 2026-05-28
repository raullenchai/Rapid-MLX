# SPDX-License-Identifier: Apache-2.0
"""Tests for `_is_model_loaded` and `ensure_model_loaded` in service.helpers.

These helpers unify the previously inconsistent "is the request model name
something we can serve?" check across /v1/chat/completions, /v1/completions,
and /v1/messages routes.

Future on-demand auto-loading (see follow-up PR) will hook into
`ensure_model_loaded`; today it strictly raises 404 for unloaded models
(and 400 for empty model strings, preserving the existing OpenAI parity).
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from vllm_mlx.config import get_config
from vllm_mlx.runtime.model_registry import ModelRegistry
from vllm_mlx.service.helpers import _is_model_loaded, ensure_model_loaded


@pytest.fixture
def reset_config():
    """Snapshot mutated config fields and restore after the test."""
    cfg = get_config()
    fields = ("model_name", "model_alias", "model_path", "model_registry")
    snap = {f: getattr(cfg, f) for f in fields}
    yield cfg
    for f, v in snap.items():
        setattr(cfg, f, v)


def _make_registry(*names: str) -> MagicMock:
    """Spec'd registry mock — catches interface drift if ModelRegistry changes."""
    reg = MagicMock(spec=ModelRegistry)
    reg.__contains__ = lambda self, x: x in names
    reg.list_model_names.return_value = list(names)
    return reg


# ---------------------------------------------------------------------------
# _is_model_loaded — single-model mode
# ---------------------------------------------------------------------------


class TestIsModelLoadedSingleMode:
    def test_matches_model_name(self, reset_config):
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        assert _is_model_loaded("qwen3.5-4b") is True

    def test_matches_model_alias(self, reset_config):
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = "qwen"
        reset_config.model_path = None
        reset_config.model_registry = None
        assert _is_model_loaded("qwen") is True

    def test_matches_model_path(self, reset_config):
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = "mlx-community/Qwen2.5-4B-Instruct-4bit"
        reset_config.model_registry = None
        assert _is_model_loaded("mlx-community/Qwen2.5-4B-Instruct-4bit") is True

    def test_rejects_unknown_model(self, reset_config):
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        assert _is_model_loaded("kimi-48b") is False

    def test_default_accepted_in_single_mode(self, reset_config):
        """P2-1 fix: 'default' is loaded in BOTH single-model and registry mode.

        Pre-fix, `_validate_model_name` accepted 'default' only when a registry
        was configured — single-model servers 404'd on `model: "default"` even
        though the request unambiguously targets the one model that's loaded.
        """
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        assert _is_model_loaded("default") is True

    def test_default_rejected_when_no_model_configured(self, reset_config):
        """Helper contract: True iff `model_name` maps to a served model.

        Without this gate, `_is_model_loaded("default")` returns True even
        on an unconfigured server, contradicting the contract. The route-
        level short-circuit covers downstream behavior, but tightening the
        primitive here keeps the helper honest on its own.
        """
        reset_config.model_name = None
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        assert _is_model_loaded("default") is False

    def test_returns_true_for_none(self, reset_config):
        """``None`` = "no specific model in the request" — the default serves it."""
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        assert _is_model_loaded(None) is True

    def test_returns_false_when_no_model_configured(self, reset_config):
        reset_config.model_name = None
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        assert _is_model_loaded("anything") is False


# ---------------------------------------------------------------------------
# _is_model_loaded — registry (multi-model) mode
# ---------------------------------------------------------------------------


class TestIsModelLoadedRegistryMode:
    def test_matches_registry_entry(self, reset_config):
        reset_config.model_registry = _make_registry("qwen3.5-4b", "phi4-14b")
        reset_config.model_name = "qwen3.5-4b"
        assert _is_model_loaded("phi4-14b") is True

    def test_rejects_non_registry_entry(self, reset_config):
        reset_config.model_registry = _make_registry("qwen3.5-4b", "phi4-14b")
        reset_config.model_name = "qwen3.5-4b"
        assert _is_model_loaded("kimi-48b") is False

    def test_default_accepted_in_registry_mode(self, reset_config):
        reset_config.model_registry = _make_registry("qwen3.5-4b")
        reset_config.model_name = "qwen3.5-4b"
        assert _is_model_loaded("default") is True


# ---------------------------------------------------------------------------
# ensure_model_loaded — strict-404 (+ 400-on-empty) semantics
# ---------------------------------------------------------------------------


class TestEnsureModelLoaded:
    @pytest.mark.asyncio
    async def test_noop_when_model_is_loaded(self, reset_config):
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        # Should not raise.
        await ensure_model_loaded("qwen3.5-4b")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", [None, "default"])
    async def test_noop_for_none_or_default(self, reset_config, name):
        """Both ``None`` (no model specified) and ``"default"`` should be
        treated as the loaded model — parametrized so a regression on one
        case points at the right line."""
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        await ensure_model_loaded(name)

    @pytest.mark.asyncio
    async def test_empty_string_raises_400(self, reset_config):
        """Empty ``model`` field is a client bug — match upstream OpenAI 400.

        Pre-fix, the empty string silently fell through to the default model,
        masking a typo'd env var or unset client config.
        """
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        with pytest.raises(HTTPException) as exc:
            await ensure_model_loaded("")
        assert exc.value.status_code == 400
        assert "empty" in exc.value.detail.lower()

    @pytest.mark.asyncio
    async def test_raises_404_when_model_not_loaded(self, reset_config):
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        with pytest.raises(HTTPException) as exc:
            await ensure_model_loaded("kimi-48b")
        assert exc.value.status_code == 404
        assert "kimi-48b" in exc.value.detail

    @pytest.mark.asyncio
    async def test_404_detail_lists_available_single_mode(self, reset_config):
        """Locks the 404 contract: detail must include the `Available:` hint.

        Mirrors `_validate_model_name`'s message shape so the two helpers are
        interchangeable on the strict-404 path — and so #319's swap logic can
        replace this `raise` without changing what clients see on a miss.
        """
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        with pytest.raises(HTTPException) as exc:
            await ensure_model_loaded("kimi-48b")
        assert exc.value.status_code == 404
        assert "Available:" in exc.value.detail
        assert "qwen3.5-4b" in exc.value.detail

    @pytest.mark.asyncio
    async def test_404_detail_lists_available_registry_mode(self, reset_config):
        reset_config.model_registry = _make_registry("qwen3.5-4b", "phi4-14b")
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = None
        with pytest.raises(HTTPException) as exc:
            await ensure_model_loaded("kimi-48b")
        assert exc.value.status_code == 404
        assert "Available:" in exc.value.detail
        assert "phi4-14b" in exc.value.detail
        assert "qwen3.5-4b" in exc.value.detail

    @pytest.mark.asyncio
    async def test_noop_when_server_unconfigured(self, reset_config):
        """Theoretical unconfigured-server case — preserves parity with
        `_validate_model_name`, which silently returns when neither
        `model_name` nor `model_registry` is set."""
        reset_config.model_name = None
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        # Should not raise even for a clearly-unknown name.
        await ensure_model_loaded("anything")
