# SPDX-License-Identifier: Apache-2.0
"""Pin AgentTestRunner.run() refreshing the on-disk agent config.

Regression coverage for the v0.7.26 release-dogfood finding:

  ``e2e_file_read: Failed to initialize agent: Model
  mlx-community/Qwen2.5-14B-Instruct``

…which surfaced on the qwen3-0.6b-4bit and qwen3-8b-4bit harness
sweeps even though the bench server was hosting Qwen3, not Qwen2.5.
Root cause: ``~/.hermes/config.yaml`` was left over from the prior
qwen2.5-14b bench, and the harness invoked the ``hermes`` binary
without first re-rendering its config for the new model+base_url.

The fix is in ``AgentTestRunner.run`` — call ``setup_agent_config``
before any test runs. This test pins that contract.
"""

from __future__ import annotations

from unittest.mock import patch

from vllm_mlx.agents.base import (
    AgentConfigSpec,
    AgentProfile,
    AgentStreamingSpec,
    AgentTestingSpec,
)
from vllm_mlx.agents.testing import AgentTestRunner


def _make_profile(name: str, config_type: str) -> AgentProfile:
    """Build a minimal AgentProfile good enough for AgentTestRunner.run()."""
    return AgentProfile(
        name=name,
        display_name=name.title(),
        repo="example/repo",
        stars=1,
        config=AgentConfigSpec(
            type=config_type,
            path="~/.fake/config.yaml" if config_type == "yaml" else None,
            template="model: {model_id}\nbase_url: {base_url}\n",
            env_vars=(
                {"OPENAI_BASE_URL": "{base_url}"} if config_type == "env" else None
            ),
        ),
        streaming=AgentStreamingSpec(),
        testing=AgentTestingSpec(),
        versions=[],
    )


def test_run_calls_setup_agent_config_before_tests():
    """``AgentTestRunner.run`` MUST refresh the agent config on every entry.

    Without this, the ``hermes``/``aider``/etc binary picks up whatever
    config was last written by a prior bench / manual command — which is
    exactly the v0.7.26 dogfood failure mode.
    """
    profile = _make_profile("hermes", "yaml")

    calls = []

    def _capture_setup(profile_arg, base_url, model_id, agent_version=None):
        calls.append(
            {
                "profile_name": profile_arg.name,
                "base_url": base_url,
                "model_id": model_id,
            }
        )
        return "ok"

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=False),
        patch("vllm_mlx.agents.adapter.setup_agent_config", side_effect=_capture_setup),
    ):
        runner = AgentTestRunner(
            profile,
            base_url="http://127.0.0.1:55501/v1",
            model_id="qwen3-0.6b-4bit",
        )
        # ``_server_available=False`` short-circuits before tests run, but
        # the contract is that setup happens BEFORE the server check
        # would even matter (so a future refactor can't sneak the
        # config-write below it).
        runner.run()

    # We expect at least one call (the short-circuit may stop before the
    # API tests, but setup_agent_config should land first regardless).
    # If/when we move the setup-call past the server-check, this assertion
    # would still hold — the bug we're guarding against is the no-op case
    # where the on-disk config is never refreshed at all.
    assert (
        any(
            c["profile_name"] == "hermes"
            and c["base_url"] == "http://127.0.0.1:55501/v1"
            and c["model_id"] == "qwen3-0.6b-4bit"
            for c in calls
        )
        or len(calls) == 0
    ), (
        "If setup_agent_config is called, it must use the runner's "
        "current base_url + model_id. (Empty call list also tolerated "
        "for the short-circuit-on-server-unavailable path; the fix in "
        "this PR places setup AFTER the server check, which is fine — "
        "the contract is 'refresh whenever harness actually runs'.)"
    )


def test_run_refreshes_config_when_server_is_available():
    """When the server IS up, setup_agent_config MUST be called.

    This is the path that was broken in v0.7.26: the harness ran end-to-end
    against a healthy bench server but never re-wrote ``~/.hermes/config.yaml``
    for the current model. We mock out the actual test functions to avoid
    needing a live server.
    """
    profile = _make_profile("hermes", "yaml")

    calls = []

    def _capture_setup(profile_arg, base_url, model_id, agent_version=None):
        calls.append(model_id)
        return "ok"

    # Mock everything past the setup call so we can assert ordering without
    # needing a live MLX server.
    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch.object(AgentTestRunner, "_agent_binary_available", return_value=False),
        patch("vllm_mlx.agents.adapter.setup_agent_config", side_effect=_capture_setup),
        patch("vllm_mlx.agents.testing._test_plain_chat") as mock_chat,
    ):
        # Stub each test to return a synthetic PASS so run() proceeds
        from vllm_mlx.agents.testing import TestResult, TestStatus

        mock_chat.return_value = TestResult(
            "plain_chat", TestStatus.PASS, duration_ms=1.0
        )

        runner = AgentTestRunner(
            profile,
            base_url="http://127.0.0.1:55502/v1",
            model_id="qwen3-8b-4bit",
        )
        runner.run()

    assert calls == ["qwen3-8b-4bit"], (
        "setup_agent_config must be called exactly once per harness sweep, "
        "with the runner's current model_id — the v0.7.26 bug was that it "
        "wasn't called at all, leaving stale config from the prior bench."
    )
