# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the pi coding-agent profile.

The config contract pinned here is the one verified against rapid-mlx 0.15.4 /
qwen3.6-35b-4bit in the 2026-10-02 harness-lab run: pi 1.0.0 passed T1–T6 plus
the follow-up turn with exactly this ``~/.pi/agent/models.json`` shape and was
the fastest harness in the run.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from rapid_mlx.agents import get_profile, list_profiles, load_profiles
from rapid_mlx.agents.adapter import setup_agent_config


def setup_function():
    # Keep these tests independent of registry state left by profile-loader tests.
    load_profiles()


def test_pi_is_listed_once_and_resolvable():
    profile = get_profile("pi")
    assert profile is not None
    assert profile.name == "pi"
    assert [p.name for p in list_profiles()].count("pi") == 1


def test_pi_profile_matches_the_verified_models_json_contract():
    profile = get_profile("pi")
    assert profile is not None

    rendered = json.loads(
        profile.render_config(
            "http://127.0.0.1:8153/v1",
            "qwen3.6-35b-4bit",
            context_length=262144,
        )
    )
    provider = rendered["providers"]["rapid-mlx"]
    assert provider["baseUrl"] == "http://127.0.0.1:8153/v1"
    # pi speaks the OpenAI chat-completions wire; the key is a loopback
    # sentinel — the server never validates it.
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "not-needed"
    (model,) = provider["models"]
    assert model == {
        "id": "qwen3.6-35b-4bit",
        "name": "qwen3.6-35b-4bit (Rapid-MLX)",
        "contextWindow": 262144,
        "maxTokens": 8192,
        "input": ["text"],
        "reasoning": False,
    }


def test_pi_setup_writes_verified_models_json_into_the_agents_home(
    tmp_path, monkeypatch
):
    """`agents pi --setup` writes <home>/models.json, honouring PI_CODING_AGENT_DIR.

    The redirected-home contract is what keeps the release gate (and any
    scripted setup) away from the operator's real ~/.pi/agent/models.json.
    """
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))

    profile = get_profile("pi")
    assert profile is not None
    summary = setup_agent_config(
        profile,
        base_url="http://127.0.0.1:8153/v1",
        model_id="qwen3.6-35b-4bit",
        context_length=262144,
    )

    assert "Cannot" not in summary
    written = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))
    provider = written["providers"]["rapid-mlx"]
    assert provider["baseUrl"] == "http://127.0.0.1:8153/v1"
    assert provider["models"][0]["id"] == "qwen3.6-35b-4bit"
    assert provider["models"][0]["contextWindow"] == 262144


def test_pi_setup_merge_preserves_existing_providers(tmp_path, monkeypatch):
    """Only providers.rapid-mlx is (re)written; the user's own providers stay."""
    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "glm-spark": {
                        "baseUrl": "http://elsewhere:8888/v1",
                        "api": "openai-completions",
                        "apiKey": "sk-noauth",
                        "models": [{"id": "GLM-5.3", "contextWindow": 200000}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))

    profile = get_profile("pi")
    assert profile is not None
    setup_agent_config(
        profile,
        base_url="http://127.0.0.1:8153/v1",
        model_id="qwen3.5-9b-4bit",
        context_length=65536,
    )

    written = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))
    assert written["providers"]["glm-spark"]["baseUrl"] == "http://elsewhere:8888/v1"
    assert written["providers"]["rapid-mlx"]["models"][0]["id"] == "qwen3.5-9b-4bit"


def test_pi_profile_has_runnable_headless_query():
    profile = get_profile("pi")
    assert profile is not None
    assert profile.testing.binary == "pi"
    assert profile.testing.query_cmd == "pi --print --no-session '{query}'"
    assert (
        profile.testing.install_cmd == "npm install -g @earendil-works/pi-coding-agent"
    )


def test_pi_e2e_receives_relocated_agent_dir_and_home(tmp_path):
    """The e2e gate must drive pi inside the throwaway home, never the real one.

    The runner redirects HOME universally and layers the profile's own
    relocation variable on top; for pi that variable is PI_CODING_AGENT_DIR.
    """
    isolated_home = tmp_path / "isolated-pi-home"
    isolated_home.mkdir()
    temporary_home = SimpleNamespace(name=str(isolated_home), cleanup=lambda: None)

    with (
        patch(
            "rapid_mlx.agents.testing.tempfile.TemporaryDirectory",
            return_value=temporary_home,
        ),
        patch(
            "rapid_mlx.agents.testing.AgentTestRunner._server_available",
            return_value=True,
        ),
        patch(
            "rapid_mlx.agents.testing.AgentTestRunner._agent_binary_available",
            return_value=True,
        ),
        patch("rapid_mlx.agents.testing._test_plain_chat"),
        patch("rapid_mlx.agents.testing._test_single_tool_call"),
        patch("rapid_mlx.agents.testing._test_tool_choice"),
        patch("rapid_mlx.agents.testing._test_multi_turn_tool"),
        patch("rapid_mlx.agents.testing._test_no_tool_leak"),
        patch("rapid_mlx.agents.testing._test_no_tool_needed"),
        patch("rapid_mlx.agents.testing._test_streaming_tool_call"),
        patch("rapid_mlx.agents.testing._test_streaming_basic"),
        patch("rapid_mlx.agents.testing._test_stress_no_leak"),
        patch("rapid_mlx.agents.testing._test_e2e_chat") as e2e_chat,
        patch("rapid_mlx.agents.testing._test_e2e_file_read"),
        patch("rapid_mlx.agents.testing._test_e2e_terminal"),
    ):
        from rapid_mlx.agents.testing import AgentTestRunner, TestResult, TestStatus

        e2e_chat.return_value = TestResult("e2e_chat", TestStatus.PASS)
        AgentTestRunner(
            get_profile("pi"),
            base_url="http://localhost:8000/v1",
            model_id="qwen3.6-35b-4bit",
        ).run()

    env = e2e_chat.call_args.kwargs["env_overrides"]
    assert env["HOME"] == str(isolated_home)
    assert env["PI_CODING_AGENT_DIR"] == str(isolated_home)
