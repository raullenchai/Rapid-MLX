# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the opencode profile's headless claim (#4042).

The profile used to claim opencode was "interactive-only" and set
``query_cmd: null`` — but ``opencode run '<prompt>'`` is a headless one-shot
mode, verified against opencode 1.18.34 (T1–T6 + follow-up turn pass). The
claim cost a skipped e2e gate for no reason.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rapid_mlx.agents import get_profile, load_profiles
from rapid_mlx.agents.adapter import get_setup_instructions, setup_agent_config
from rapid_mlx.agents.opencode_version import installed_version
from rapid_mlx.agents.testing import (
    E2E_FIRST_LINE_TOKEN,
    AgentTestRunner,
    TestStatus,
    _agent_query,
    _test_e2e_file_read,
)


def setup_function():
    load_profiles()


def test_opencode_has_a_headless_query_cmd():
    profile = get_profile("opencode")
    assert profile is not None
    assert profile.testing.binary == "opencode"
    assert profile.testing.query_cmd == "opencode run '{query}'"


def test_opencode_known_issues_do_not_claim_interactive_only():
    profile = get_profile("opencode")
    assert profile is not None
    for issue in profile.known_issues:
        assert "interactive-only" not in issue, issue


def test_opencode_config_supports_v1_and_v2_custom_providers():
    profile = get_profile("opencode")
    assert profile is not None
    config = json.loads(
        profile.render_config(
            "http://127.0.0.1:18610/v1",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
            context_length=32768,
        )
    )
    model = "mlx-community/Qwen3.5-4B-MLX-4bit"
    legacy = config["provider"]["rapid-mlx"]
    native = config["providers"]["rapid-mlx"]

    assert legacy["npm"] == "@ai-sdk/openai-compatible"
    assert legacy["options"]["baseURL"] == "http://127.0.0.1:18610/v1"
    assert model in legacy["models"]
    assert native["package"] == "@opencode/ai/providers/openai-compatible"
    assert native["settings"] == legacy["options"]
    assert native["models"][model]["limit"] == {"context": 32768, "output": 8192}
    assert native["models"][model]["capabilities"]["tools"] is True
    assert config["model"] == f"rapid-mlx/{model}"
    old = json.loads(
        profile.render_config(
            "http://127.0.0.1:18610/v1", model, "1.2.27", context_length=32768
        )
    )
    assert "providers" not in old
    assert old["provider"]["rapid-mlx"] == legacy


def test_opencode_setup_replaces_own_v2_config_when_using_v1(tmp_path, monkeypatch):
    profile = get_profile("opencode")
    assert profile is not None
    monkeypatch.setenv("HOME", str(tmp_path))
    with patch(
        "rapid_mlx.agents.opencode_version.installed_version", return_value="2.0.26"
    ):
        setup_agent_config(profile, model_id="local-model")
    path = tmp_path / ".config/opencode/opencode.json"
    config = json.loads(path.read_text())
    assert "providers" in config
    config["user_setting"] = "preserved"
    path.write_text(json.dumps(config))
    with patch(
        "rapid_mlx.agents.opencode_version.installed_version", return_value="1.2.27"
    ):
        setup_agent_config(profile, model_id="local-model")
    config = json.loads(path.read_text())
    assert "providers" not in config
    assert config["provider"]["rapid-mlx"]["options"]["apiKey"] == "not-needed"
    assert config["user_setting"] == "preserved"


def test_opencode_upgrade_keeps_legacy_models_and_options(tmp_path, monkeypatch):
    profile = get_profile("opencode")
    assert profile is not None
    monkeypatch.setenv("HOME", str(tmp_path))
    setup_agent_config(profile, model_id="old-model", agent_version="1.18.35")
    path = tmp_path / ".config/opencode/opencode.json"
    old = json.loads(path.read_text())
    old_provider = old["provider"]["rapid-mlx"]
    old_provider["options"]["timeout"] = 600000
    old_provider["models"]["extra-model"] = {
        "id": "upstream-extra-model",
        "tool_call": False,
        "limit": {"context": 16384, "output": 4096},
    }
    path.write_text(json.dumps(old))

    setup_agent_config(profile, model_id="new-model", agent_version="2.0.26")
    native = json.loads(path.read_text())["providers"]["rapid-mlx"]
    assert native["settings"]["timeout"] == 600000
    assert native["models"]["extra-model"] == {
        "modelID": "upstream-extra-model",
        "capabilities": {"tools": False},
        "limit": {"context": 16384, "output": 4096},
    }
    assert "new-model" in native["models"]


def test_opencode_downgrade_reports_other_native_providers(tmp_path, monkeypatch):
    profile = get_profile("opencode")
    assert profile is not None
    monkeypatch.setenv("HOME", str(tmp_path))
    setup_agent_config(profile, model_id="local-model", agent_version="2.0.26")
    path = tmp_path / ".config/opencode/opencode.json"
    config = json.loads(path.read_text())
    config["providers"]["other"] = {"models": {"other-model": {}}}
    original = json.dumps(config)
    path.write_text(original)
    message = setup_agent_config(
        profile, model_id="local-model", agent_version="1.2.27"
    )
    assert "OpenCode 1.x cannot read" in message
    assert path.read_text() == original


def test_opencode_instructions_detect_old_major():
    profile = get_profile("opencode")
    assert profile is not None
    with patch(
        "rapid_mlx.agents.opencode_version.installed_version", return_value="1.2.27"
    ):
        guide = get_setup_instructions(profile, model_id="local-model")
    assert '"provider"' in guide
    assert '"providers"' not in guide


def test_opencode_version_detection_parses_major_from_start():
    with (
        patch(
            "rapid_mlx.agents.opencode_version.shutil.which",
            return_value="/bin/opencode",
        ) as which,
        patch("rapid_mlx.agents.opencode_version.subprocess.run") as run,
    ):
        run.return_value = SimpleNamespace(stdout="1.2.27\n")
        assert installed_version() == "1.2.27"
        run.return_value = SimpleNamespace(stdout="opencode v2.0.26\n")
        assert installed_version() == "2.0.26"
        run.return_value = SimpleNamespace(stdout="unknown")
        assert installed_version() is None
        run.side_effect = FileNotFoundError("opencode")
        assert installed_version() is None
        which.return_value = None
        assert installed_version() is None


def test_opencode_test_runner_uses_private_server_for_v2():
    profile = get_profile("opencode")
    assert profile is not None
    with patch("rapid_mlx.agents.testing.installed_version", return_value="2.0.26"):
        runner = AgentTestRunner(profile, model_id="local-model")
        assert runner._opencode_query_cmd("opencode run '{query}'") == (
            "opencode run --standalone '{query}'"
        )
    runner = AgentTestRunner(profile, model_id="local-model", agent_version="1.2.27")
    assert runner._opencode_query_cmd("opencode run '{query}'") == (
        "opencode run '{query}'"
    )


def test_opencode_query_pwd_matches_throwaway_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("PWD", "/some/other/repo")
    with (
        patch("rapid_mlx.agents.testing.shutil.which", return_value="/bin/opencode"),
        patch("rapid_mlx.agents.testing.subprocess.run") as run,
    ):
        run.return_value = SimpleNamespace(stdout="ok", stderr="", returncode=0)
        assert _agent_query(
            "opencode", "opencode run '{query}'", "hello", cwd=str(tmp_path)
        ) == ("ok", None)
    assert run.call_args.kwargs["cwd"] == str(tmp_path)
    assert run.call_args.kwargs["env"]["PWD"] == str(tmp_path)


def test_opencode_file_read_names_the_disposable_file():
    def answer(binary, query_cmd, query, timeout, cwd, env_overrides):
        path = Path(cwd, "pyproject.toml")
        assert binary == "opencode"
        assert str(path) in query
        assert path.read_text().startswith(E2E_FIRST_LINE_TOKEN)
        return E2E_FIRST_LINE_TOKEN, None

    with patch("rapid_mlx.agents.testing._agent_query", side_effect=answer):
        result = _test_e2e_file_read("opencode", "opencode run '{query}'", 120)
    assert result.status == TestStatus.PASS
