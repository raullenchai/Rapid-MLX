"""Keep the shipped OpenHands profile aligned with its 1.x CLI contract."""

import subprocess

from rapid_mlx.agents import PROFILES_DIR, _load_profile_from_yaml
from rapid_mlx.agents.adapter import get_setup_instructions, setup_agent_config
from rapid_mlx.agents.testing import _agent_query


def test_openhands_setup_and_test_use_current_cli():
    profile = _load_profile_from_yaml(PROFILES_DIR / "openhands.yaml")
    assert profile.render_config("http://localhost:18750/v1", "qwen3.5-4b-4bit") == {
        "LLM_BASE_URL": "http://localhost:18750/v1",
        "LLM_API_KEY": "not-needed",
        "LLM_MODEL": "openai/qwen3.5-4b-4bit",
    }
    testing = profile.get_testing_for_version(None)
    assert testing.binary == "openhands"
    assert "--override-with-envs" in testing.query_cmd
    assert "--headless" in testing.query_cmd
    assert profile.needs_function_calling
    setup = setup_agent_config(
        profile, "http://localhost:18750/v1", "qwen3.5-4b-4bit", dry_run=True
    )
    assert "openhands --override-with-envs" in setup
    assert "openhands --override-with-envs" in get_setup_instructions(
        profile, "http://localhost:18750/v1", "qwen3.5-4b-4bit"
    )


def test_openhands_e2e_paths_stay_in_throwaway_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    work = tmp_path / "work"
    monkeypatch.setenv("OPENHANDS_PERSISTENCE_DIR", "/real/settings")
    monkeypatch.setenv("OPENHANDS_CONVERSATIONS_DIR", "/real/conversations")
    monkeypatch.setenv("OPENHANDS_WORK_DIR", "/real/work")
    monkeypatch.setattr(
        "rapid_mlx.agents.testing.shutil.which", lambda _: "/bin/openhands"
    )
    observed = {}

    def fake_run(_cmd, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(_cmd, 0, stdout="done", stderr="")

    monkeypatch.setattr("rapid_mlx.agents.testing.subprocess.run", fake_run)
    _agent_query(
        "openhands",
        "openhands --headless -t '{query}'",
        "hello",
        cwd=str(work),
        env_overrides={"HOME": str(home)},
    )

    env = observed["env"]
    assert env["OPENHANDS_PERSISTENCE_DIR"] == str(home / ".openhands")
    assert env["OPENHANDS_CONVERSATIONS_DIR"] == str(home / ".openhands/conversations")
    assert env["OPENHANDS_WORK_DIR"] == str(work)
