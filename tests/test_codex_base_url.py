# SPDX-License-Identifier: Apache-2.0
"""Codex server-root input must reach the same API in setup and tests."""

from argparse import Namespace

import pytest
import tomllib

from rapid_mlx.agents import get_profile
from rapid_mlx.agents.adapter import setup_agent_config
from rapid_mlx.agents.testing import AgentTestRunner


@pytest.mark.parametrize(
    "supplied,expected",
    [
        ("http://127.0.0.1:8899", "http://127.0.0.1:8899/v1"),
        ("http://localhost:8899/", "http://localhost:8899/v1"),
        ("https://inference.example", "https://inference.example/v1"),
        ("http://[::1]:8899/", "http://[::1]:8899/v1"),
        ("http://localhost:8899/v1", "http://localhost:8899/v1"),
        ("http://localhost:8899/v1/", "http://localhost:8899/v1/"),
        ("https://inference.example/local/v1", "https://inference.example/local/v1"),
        ("https://inference.example/api", "https://inference.example/api"),
    ],
)
def test_setup_and_runner_agree_on_api_base(tmp_path, monkeypatch, supplied, expected):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    profile = get_profile("codex")
    setup_agent_config(profile, supplied, "local-qwen")
    config = tomllib.loads((tmp_path / "config.toml").read_text())
    assert config["model_providers"]["rapid-mlx"]["base_url"] == expected
    runner = AgentTestRunner(profile, base_url=supplied, model_id="local-qwen")
    assert runner.base_url == expected
    assert profile.normalize_base_url(expected) == expected


def test_runner_discovers_model_at_normalized_api(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return type(
            "Response", (), {"json": lambda _: {"data": [{"id": "local-qwen"}]}}
        )()

    monkeypatch.setattr("rapid_mlx.agents.testing.httpx.get", get)
    runner = AgentTestRunner(get_profile("codex"), "http://127.0.0.1:8899")
    assert runner.model_id == "local-qwen"
    assert calls == ["http://127.0.0.1:8899/v1/models"]


def test_cli_normalizes_before_discovery_verification_and_write(tmp_path, monkeypatch):
    from rapid_mlx.cli import agents_command

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    calls = []

    def discover(url):
        calls.append(("discover", url))
        return "local-qwen", 32768

    def verify(url, model, **kwargs):
        calls.append(("verify", url))
        assert model == "local-qwen"
        return model

    monkeypatch.setattr("rapid_mlx.agents.adapter._detect_running_model", discover)
    monkeypatch.setattr("rapid_mlx.agents.setup.verify_server", verify)
    agents_command(
        Namespace(
            agent_name="codex",
            base_url="http://127.0.0.1:8899/",
            model=None,
            setup=True,
            test=False,
            agent_version=None,
            dry_run=False,
            no_check=False,
            yes=True,
        )
    )
    assert calls == [
        ("discover", "http://127.0.0.1:8899/v1"),
        ("verify", "http://127.0.0.1:8899/v1"),
    ]
    config = tomllib.loads((tmp_path / "config.toml").read_text())
    assert (
        config["model_providers"]["rapid-mlx"]["base_url"] == "http://127.0.0.1:8899/v1"
    )


@pytest.mark.parametrize("name", ["claude-code", "hermes", "qwen-code"])
def test_other_agent_url_contracts_unchanged(name):
    assert (
        get_profile(name).normalize_base_url("http://localhost:8899/")
        == "http://localhost:8899/"
    )
