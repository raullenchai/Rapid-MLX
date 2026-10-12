"""Current Goose and Crush configurations for a local OpenAI server."""

import json
from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.agents import get_profile
from rapid_mlx.agents.adapter import setup_agent_config
from rapid_mlx.agents.testing import AgentTestRunner, TestResult, TestStatus


@pytest.mark.parametrize("keyed", [False, True])
def test_goose_provider_setup(tmp_path, monkeypatch, keyed):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    if keyed:
        monkeypatch.setenv("RAPID_MLX_API_KEY", "local-secret")
    else:
        monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    profile = get_profile("goose")
    assert profile is not None
    summary = setup_agent_config(
        profile, "http://127.0.0.1:18740/v1", "qwen3.5-4b-4bit", context_length=32768
    )
    path = tmp_path / ".config/goose/custom_providers/custom_rapid_mlx.json"
    assert str(path) in summary
    data = json.loads(path.read_text())
    assert data["base_url"] == "http://127.0.0.1:18740/v1/chat/completions"
    assert data["models"] == [{"name": "qwen3.5-4b-4bit", "context_limit": 32768}]
    assert data["requires_auth"] is keyed
    assert data["api_key_env"] == ("RAPID_MLX_API_KEY" if keyed else "")
    assert "local-secret" not in path.read_text()


def test_goose_xdg_config_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    profile = get_profile("goose")
    setup_agent_config(profile, "http://127.0.0.1:18740/v1", "qwen3.5-4b-4bit")
    assert (tmp_path / "xdg/goose/custom_providers/custom_rapid_mlx.json").exists()


def test_goose_path_root_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("GOOSE_PATH_ROOT", str(tmp_path / "goose-root"))
    setup_agent_config(
        get_profile("goose"), "http://127.0.0.1:18740/v1", "qwen3.5-4b-4bit"
    )
    assert (
        tmp_path / "goose-root/config/custom_providers/custom_rapid_mlx.json"
    ).exists()


@pytest.mark.parametrize("keyed", [False, True])
def test_crush_provider_setup(tmp_path, monkeypatch, keyed):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CRUSH_GLOBAL_CONFIG", str(tmp_path / "crush-config"))
    if keyed:
        monkeypatch.setenv("RAPID_MLX_API_KEY", "local-secret")
    else:
        monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    profile = get_profile("crush")
    assert profile is not None
    setup_agent_config(
        profile, "http://127.0.0.1:18741/v1", "qwen3.5-4b-4bit", context_length=32768
    )
    path = tmp_path / "crush-config/crush.json"
    data = json.loads(path.read_text())
    provider = data["providers"]["rapid-mlx"]
    assert provider["type"] == "openai-compat"
    assert provider["base_url"] == "http://127.0.0.1:18741/v1"
    assert provider["api_key"] == ("$RAPID_MLX_API_KEY" if keyed else "not-needed")
    assert provider["models"][0]["context_window"] == 32768
    assert data["models"]["large"] == {
        "provider": "rapid-mlx",
        "model": "qwen3.5-4b-4bit",
    }
    assert data["options"]["request_timeout"] == 300
    assert "local-secret" not in path.read_text()


def test_crush_xdg_config_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("CRUSH_GLOBAL_CONFIG", raising=False)
    setup_agent_config(
        get_profile("crush"), "http://127.0.0.1:18740/v1", "qwen3.5-4b-4bit"
    )
    assert (tmp_path / "xdg/crush/crush.json").exists()


def test_agents_setup_keeps_public_alias(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.fetch_context_window", lambda *_: 32768
    )
    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.setup_agent_config",
        lambda _profile, _url, model_id, **_kwargs: (
            seen.append(model_id) or "Would write config (nothing written)"
        ),
    )
    args = SimpleNamespace(
        agent_name="aider",
        base_url="http://127.0.0.1:18740/v1",
        model="mlx-community/Qwen3.5-4B-MLX-4bit",
        _original_alias="qwen3.5-4b-4bit",
        test=False,
        setup=True,
        agent_version=None,
        dry_run=True,
        no_check=True,
    )
    cli.agents_command(args)
    assert seen == ["qwen3.5-4b-4bit"]


def test_agents_test_keeps_public_alias(monkeypatch):
    seen = []

    class FakeRunner:
        def __init__(self, _profile, *, model_id, **_kwargs):
            seen.append(model_id)

        def _server_available(self):
            return True

        def run(self):
            return SimpleNamespace(print_summary=lambda: True)

    monkeypatch.setattr("rapid_mlx.agents.testing.AgentTestRunner", FakeRunner)
    args = SimpleNamespace(
        agent_name="aider",
        base_url="http://127.0.0.1:18740/v1",
        model="mlx-community/Qwen3.5-4B-MLX-4bit",
        _original_alias="qwen3.5-4b-4bit",
        test=True,
        setup=False,
        agent_version=None,
    )
    with pytest.raises(SystemExit) as result:
        cli.agents_command(args)
    assert result.value.code == 0
    assert seen == ["qwen3.5-4b-4bit"]


@pytest.mark.parametrize("agent", ["goose", "crush"])
def test_agent_test_uses_isolated_config_and_data(agent, monkeypatch, tmp_path):
    from rapid_mlx.agents import testing

    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "operator-xdg"))
    monkeypatch.setenv("GOOSE_PATH_ROOT", str(tmp_path / "operator-goose"))
    monkeypatch.setenv(
        "GOOSE_ADDITIONAL_CONFIG_FILES", str(tmp_path / "operator-config.yaml")
    )
    monkeypatch.setenv("CRUSH_GLOBAL_DATA", str(tmp_path / "operator-crush"))
    seen_setup = []
    seen_child = []
    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.fetch_context_window", lambda *_: 32768
    )
    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.setup_agent_config",
        lambda *_args, **_kwargs: seen_setup.append(dict(testing.os.environ)),
    )
    for name in (
        "plain_chat",
        "single_tool_call",
        "tool_choice",
        "multi_turn_tool",
        "no_tool_leak",
        "no_tool_needed",
        "streaming_tool_call",
        "many_tools",
        "streaming_basic",
        "stress_no_leak",
    ):
        monkeypatch.setattr(
            testing,
            f"_test_{name}",
            lambda *_args, _name=name: TestResult(_name, TestStatus.PASS),
        )

    def fake_e2e(*_args, env_overrides, **_kwargs):
        seen_child.append(env_overrides)
        return TestResult("e2e", TestStatus.PASS)

    for name in ("chat", "file_read", "terminal"):
        monkeypatch.setattr(testing, f"_test_e2e_{name}", fake_e2e)
    monkeypatch.setattr(AgentTestRunner, "_server_available", lambda _self: True)
    monkeypatch.setattr(AgentTestRunner, "_agent_binary_available", lambda _self: True)
    report = AgentTestRunner(
        get_profile(agent), "http://127.0.0.1:18740/v1", "qwen3.5-4b-4bit"
    ).run()
    assert report.results
    assert seen_setup and seen_child
    assert seen_setup[0]["HOME"] != str(tmp_path / "operator")
    if agent == "goose":
        assert seen_setup[0]["GOOSE_PATH_ROOT"] == ""
        assert seen_setup[0]["XDG_CONFIG_HOME"].startswith(seen_setup[0]["HOME"])
        assert seen_child[0]["GOOSE_ADDITIONAL_CONFIG_FILES"] == ""
    else:
        assert seen_child[0]["CRUSH_GLOBAL_DATA"].startswith(seen_child[0]["HOME"])
