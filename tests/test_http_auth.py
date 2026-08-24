# SPDX-License-Identifier: Apache-2.0
"""Tests for env-only authentication of local HTTP clients."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

from vllm_mlx.agents.testing import AgentTestRunner
from vllm_mlx.http_auth import rapid_mlx_auth_headers

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def test_langchain_integration_propagates_bearer_to_discovery_and_client():
    """Keep both LangChain HTTP paths on the shared env-only auth contract."""
    source = (
        REPO_ROOT / "vllm_mlx" / "_integration_tests" / "test_langchain.py"
    ).read_text()

    assert "headers=auth_headers" in source
    assert 'os.environ.get("RAPID_MLX_API_KEY") or "not-needed"' in source


def test_hermes_cli_receives_process_scoped_auth_without_persisting_it():
    """Keep the external Hermes CLI on the env-only auth contract."""
    source = (
        REPO_ROOT / "vllm_mlx" / "_integration_tests" / "test_hermes.py"
    ).read_text()

    assert 'env["OPENAI_API_KEY"] = api_key' in source
    assert 'env["CUSTOM_API_KEY"] = api_key' in source
    assert "env=_hermes_subprocess_env()" in source
    config_block = source.split("def ensure_hermes_config", 1)[1].split(
        "def _hermes_subprocess_env", 1
    )[0]
    assert "api_key" not in config_block
    assert 'os.environ.get("HERMES_HOME")' in config_block
    assert 'or os.path.expanduser("~/.hermes")' in config_block


def test_hermes_harness_writes_config_to_overridden_home(monkeypatch, tmp_path):
    """The release gate fixture must never fall back to operator state."""
    source = REPO_ROOT / "vllm_mlx" / "_integration_tests" / "test_hermes.py"
    spec = importlib.util.spec_from_file_location("hermes_home_contract", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    isolated_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(isolated_home))
    monkeypatch.setattr(module, "_detect_context_window", lambda: 65_536)

    module.ensure_hermes_config()

    config = (isolated_home / "config.yaml").read_text()
    assert f'base_url: "{module.BASE_URL}"' in config
    assert "context_length: 65536" in config
