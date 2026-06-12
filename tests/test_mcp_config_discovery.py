# SPDX-License-Identifier: Apache-2.0
"""Tests for MCP config discovery + back-compat after the vllm-mlx → rapid-mlx
rename.

The rename moved the config search dir and env var to ``rapid-mlx`` /
``RAPID_MLX_MCP_CONFIG`` while keeping the pre-rename ``vllm-mlx`` /
``VLLM_MLX_MCP_CONFIG`` locations as fallbacks so existing users' configs keep
working. These tests pin that contract: new location wins, old location still
resolves, and resolution is existence-aware (a stale new path must not shadow a
working legacy one).
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.mcp import config as mcp_config


@pytest.fixture(autouse=True)
def _clear_mcp_env(monkeypatch):
    """Both MCP env vars start unset so the host environment can't leak in."""
    monkeypatch.delenv("RAPID_MLX_MCP_CONFIG", raising=False)
    monkeypatch.delenv("VLLM_MLX_MCP_CONFIG", raising=False)


def _write_cfg(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": {}}))
    return path


# --- Search-path discovery (HOME-rooted) ----------------------------------


def test_rapid_mlx_config_dir_is_discovered(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = _write_cfg(tmp_path / ".config" / "rapid-mlx" / "mcp.json")
    assert mcp_config._find_config_file() == cfg


def test_legacy_vllm_mlx_config_dir_still_discovered(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = _write_cfg(tmp_path / ".config" / "vllm-mlx" / "mcp.json")
    assert mcp_config._find_config_file() == cfg


def test_rapid_mlx_dir_preferred_over_legacy(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    new = _write_cfg(tmp_path / ".config" / "rapid-mlx" / "mcp.json")
    _write_cfg(tmp_path / ".config" / "vllm-mlx" / "mcp.json")
    assert mcp_config._find_config_file() == new


# --- Env-var discovery -----------------------------------------------------


def test_rapid_mlx_env_var_honored(monkeypatch, tmp_path):
    cfg = _write_cfg(tmp_path / "explicit.json")
    monkeypatch.setenv("RAPID_MLX_MCP_CONFIG", str(cfg))
    assert mcp_config._find_config_file() == cfg


def test_legacy_env_var_still_honored(monkeypatch, tmp_path):
    cfg = _write_cfg(tmp_path / "explicit.json")
    monkeypatch.setenv("VLLM_MLX_MCP_CONFIG", str(cfg))
    assert mcp_config._find_config_file() == cfg


def test_new_env_var_wins_when_both_point_to_real_files(monkeypatch, tmp_path):
    new = _write_cfg(tmp_path / "new.json")
    old = _write_cfg(tmp_path / "old.json")
    monkeypatch.setenv("RAPID_MLX_MCP_CONFIG", str(new))
    monkeypatch.setenv("VLLM_MLX_MCP_CONFIG", str(old))
    assert mcp_config._find_config_file() == new


def test_stale_new_env_var_falls_through_to_working_legacy(monkeypatch, tmp_path):
    """A new env var pointing at a missing file must not shadow a legacy var
    that points at a real one — the loader's lookup is existence-aware."""
    old = _write_cfg(tmp_path / "old.json")
    monkeypatch.setenv("RAPID_MLX_MCP_CONFIG", str(tmp_path / "does-not-exist.json"))
    monkeypatch.setenv("VLLM_MLX_MCP_CONFIG", str(old))
    assert mcp_config._find_config_file() == old


def test_directory_env_var_falls_through_to_working_legacy(monkeypatch, tmp_path):
    """An env var pointing at a *directory* (e.g. the config dir itself) must
    not be treated as a valid config file — it would raise IsADirectoryError
    in read_text(). It should fall through to the legacy file."""
    a_dir = tmp_path / "rapid-mlx"
    a_dir.mkdir()
    old = _write_cfg(tmp_path / "old.json")
    monkeypatch.setenv("RAPID_MLX_MCP_CONFIG", str(a_dir))
    monkeypatch.setenv("VLLM_MLX_MCP_CONFIG", str(old))
    assert mcp_config._find_config_file() == old


def test_no_config_anywhere_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert mcp_config._find_config_file() is None
