# SPDX-License-Identifier: Apache-2.0
"""Tests for env-only authentication of local HTTP clients."""

from unittest.mock import Mock, patch

from vllm_mlx.agents.testing import AgentTestRunner
from vllm_mlx.http_auth import rapid_mlx_auth_headers


def test_auth_headers_are_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    assert rapid_mlx_auth_headers() == {}


def test_auth_headers_use_bearer_without_exposing_key_in_argv(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_API_KEY", "local-secret")
    assert rapid_mlx_auth_headers() == {
        "Authorization": "Bearer local-secret",
    }


def test_harness_model_discovery_uses_same_bearer(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_API_KEY", "harness-secret")
    response = Mock()
    response.json.return_value = {"data": [{"id": "secured-model"}]}

    with patch("httpx.get", return_value=response) as get:
        runner = AgentTestRunner(Mock(), base_url="http://127.0.0.1:8000/v1")

    assert runner.model_id == "secured-model"
    get.assert_called_once_with(
        "http://127.0.0.1:8000/v1/models",
        headers={"Authorization": "Bearer harness-secret"},
        timeout=5,
    )
