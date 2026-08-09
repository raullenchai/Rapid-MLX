"""Regression coverage for the Claude Code agent discovery surface (#1531)."""

from vllm_mlx.agents import get_profile, list_profiles, load_profiles
from vllm_mlx.agents.adapter import setup_agent_config


def setup_function():
    # Keep these tests independent of registry state left by profile-loader tests.
    load_profiles()


def test_claude_code_is_listed_once_and_claude_is_an_alias():
    profile = get_profile("claude-code")

    assert profile is not None
    assert get_profile("claude") is profile
    assert [p.name for p in list_profiles()].count("claude-code") == 1


def test_claude_code_setup_uses_bare_anthropic_base_url():
    profile = get_profile("claude-code")
    assert profile is not None

    summary = setup_agent_config(
        profile,
        base_url="http://localhost:8000/v1",
        model_id="qwen3.5-9b-4bit",
    )

    assert "export ANTHROPIC_BASE_URL=http://localhost:8000" in summary
    assert "export ANTHROPIC_API_KEY=not-needed" in summary
    assert "export ANTHROPIC_MODEL=qwen3.5-9b-4bit" in summary
    assert "ANTHROPIC_BASE_URL=http://localhost:8000/v1" not in summary


def test_claude_code_profile_has_runnable_test_command():
    profile = get_profile("claude-code")
    assert profile is not None
    assert profile.testing.binary == "claude"
    assert profile.testing.query_cmd == "claude -p '{query}'"
