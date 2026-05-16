# SPDX-License-Identifier: Apache-2.0
"""Tests for `_is_model_loaded` and `ensure_model_loaded` in service.helpers.

These helpers unify the previously inconsistent "is the request model name
something we can serve?" check across /v1/chat/completions, /v1/completions,
and /v1/messages routes.

Future on-demand auto-loading (see follow-up PR) will hook into
`ensure_model_loaded`; today it strictly raises 404 for unloaded models.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from vllm_mlx.config import get_config
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
    reg = MagicMock()
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
# ensure_model_loaded — strict-404 semantics (PR #1 baseline)
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
    async def test_noop_for_empty_or_default(self, reset_config):
        reset_config.model_name = "qwen3.5-4b"
        reset_config.model_alias = None
        reset_config.model_path = None
        reset_config.model_registry = None
        await ensure_model_loaded(None)
        await ensure_model_loaded("")
        await ensure_model_loaded("default")

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
