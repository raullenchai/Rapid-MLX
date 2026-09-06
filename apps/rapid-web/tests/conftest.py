# SPDX-License-Identifier: Apache-2.0
"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_home(tmp_path, monkeypatch):
    """Point ``Path.home()`` at a throwaway directory for every test.

    ``WebConfig`` builds a :class:`ConnectorStore` by default, and that store
    resolves ``~/.config/rapid-mlx/mcp.json`` — a real file other tools on
    this Mac read. Without this, a test asserting on connector state would
    depend on whatever the developer happens to have configured, and a test
    exercising the write routes would EDIT it.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
