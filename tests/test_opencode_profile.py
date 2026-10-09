# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the opencode profile's headless claim (#4042).

The profile used to claim opencode was "interactive-only" and set
``query_cmd: null`` — but ``opencode run '<prompt>'`` is a headless one-shot
mode, verified against opencode 1.18.34 (T1–T6 + follow-up turn pass). The
claim cost a skipped e2e gate for no reason.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from rapid_mlx.agents import get_profile, load_profiles
from rapid_mlx.agents.testing import AgentTestRunner


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


def test_opencode_test_runner_uses_private_server_for_v2():
    profile = get_profile("opencode")
    assert profile is not None
    with patch("rapid_mlx.agents.testing.subprocess.run") as run:
        run.return_value = SimpleNamespace(stdout="opencode v2.0.26\n")
        runner = AgentTestRunner(profile, model_id="local-model")
        assert runner._opencode_query_cmd("opencode run '{query}'") == (
            "opencode run --standalone '{query}'"
        )
        run.return_value = SimpleNamespace(stdout="1.18.35\n")
        assert runner._opencode_query_cmd("opencode run '{query}'") == (
            "opencode run '{query}'"
        )
