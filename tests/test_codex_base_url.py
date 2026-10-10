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


@pytest.mark.parametrize(
    "suffix", ["?token=secret", "/?token=secret", "#fragment", "?", "#", "/?#"]
)
def test_root_query_or_fragment_rejected_before_setup_or_discovery(
    tmp_path, monkeypatch, suffix
):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    profile = get_profile("codex")
    supplied = "http://localhost:8899" + suffix
    with pytest.raises(ValueError, match="must not contain a query or fragment"):
        setup_agent_config(profile, supplied, "local-qwen")
    with pytest.raises(ValueError, match="must not contain a query or fragment"):
        AgentTestRunner(profile, supplied)
    assert not (tmp_path / "config.toml").exists()


def test_cli_rejects_root_query_without_echoing_secret(tmp_path, monkeypatch, capsys):
    from rapid_mlx.cli import agents_command

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        agents_command(
            Namespace(
                agent_name="codex",
                base_url="http://localhost:8899?token=secret",
                setup=True,
                test=False,
            )
        )
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "must not contain a query or fragment" in output
    assert "secret" not in output
    assert not (tmp_path / "config.toml").exists()


@pytest.mark.parametrize(
    "supplied",
    [
        "http://user:credential-canary-4436＠localhost:8899",
        "http://user:credential-canary-4436@localhost：8899/v1",
        "http://user:credential-canary-4436@[::1",
    ],
)
def test_malformed_url_rejected_without_echoing_secret(
    tmp_path, monkeypatch, capsys, supplied
):
    import traceback

    from rapid_mlx.cli import agents_command

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    profile = get_profile("codex")
    for operation in (
        lambda: setup_agent_config(profile, supplied, "local-qwen"),
        lambda: AgentTestRunner(profile, supplied),
    ):
        with pytest.raises(ValueError) as exc:
            operation()
        assert "credential-canary-4436" not in "".join(
            traceback.format_exception(exc.value)
        )
        assert "valid HTTP(S)" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        agents_command(
            Namespace(agent_name="codex", base_url=supplied, setup=True, test=False)
        )
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "credential-canary-4436" not in output
    assert "valid HTTP(S)" in output
    assert not (tmp_path / "config.toml").exists()


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
