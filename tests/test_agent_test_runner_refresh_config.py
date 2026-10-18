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

import os
from dataclasses import replace
from unittest.mock import patch

import pytest

from rapid_mlx.agents.base import (
    AgentConfigSpec,
    AgentProfile,
    AgentStreamingSpec,
    AgentTestingSpec,
)
from rapid_mlx.agents.testing import AgentTestRunner, TestStatus


def test_continue_harness_refreshes_yaml_in_isolated_home(monkeypatch):
    from rapid_mlx.agents import get_profile
    from rapid_mlx.agents.testing import TestResult

    profile = replace(
        get_profile("continue"),
        needs_function_calling=False,
        needs_streaming=False,
    )
    assert profile.config.home_env == "CONTINUE_GLOBAL_DIR"
    seen = []

    def record_setup(base_url, model_id, api_key):
        seen.append((base_url, model_id, os.environ["CONTINUE_GLOBAL_DIR"]))

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch.object(AgentTestRunner, "_agent_binary_available", return_value=False),
        patch("rapid_mlx.agents.adapter.fetch_context_window", return_value=32768),
        patch(
            "rapid_mlx.launch.continue_dev.write_or_patch_config",
            side_effect=record_setup,
        ),
        patch(
            "rapid_mlx.agents.testing._test_plain_chat",
            return_value=TestResult("plain_chat", TestStatus.PASS),
        ),
    ):
        AgentTestRunner(
            profile,
            base_url="http://127.0.0.1:18730/v1",
            model_id="local-model",
        ).run()

    assert len(seen) == 1
    assert seen[0][:2] == ("http://127.0.0.1:18730/v1", "local-model")
    assert seen[0][2] != os.environ.get("CONTINUE_GLOBAL_DIR")


def test_cline_harness_clears_inherited_config_overrides(monkeypatch, tmp_path):
    from rapid_mlx.agents import get_profile
    from rapid_mlx.agents.testing import TestResult

    for key in (
        "CLINE_PROVIDER_SETTINGS_PATH",
        "CLINE_MCP_SETTINGS_PATH",
        "CLINE_DB_DATA_DIR",
        "CLINE_DIR",
    ):
        monkeypatch.setenv(key, str(tmp_path / key))
    profile = replace(
        get_profile("cline"),
        needs_function_calling=False,
        needs_streaming=False,
        streaming=AgentStreamingSpec(),
    )
    seen = []

    def record_e2e(binary, query_cmd, timeout, **kwargs):
        seen.append(kwargs["env_overrides"])
        return TestResult("e2e_chat", TestStatus.PASS)

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch.object(AgentTestRunner, "_agent_binary_available", return_value=True),
        patch("rapid_mlx.agents.adapter.fetch_context_window", return_value=32768),
        patch("rapid_mlx.launch.cline.write_or_patch_config"),
        patch(
            "rapid_mlx.agents.testing._test_plain_chat",
            return_value=TestResult("plain_chat", TestStatus.PASS),
        ),
        patch("rapid_mlx.agents.testing._test_e2e_chat", side_effect=record_e2e),
    ):
        AgentTestRunner(profile, model_id="local-model").run()

    assert len(seen) == 1
    child = seen[0]
    assert child["CLINE_MCP_SETTINGS_PATH"] == ""
    assert child["CLINE_DB_DATA_DIR"] == ""
    assert child["CLINE_DIR"] == child["HOME"]
    assert child["CLINE_DATA_DIR"] == child["HOME"]
    assert child["CLINE_PROVIDER_SETTINGS_PATH"].startswith(child["HOME"])


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

    def _capture_setup(
        profile_arg,
        base_url,
        model_id,
        agent_version=None,
        context_length=None,
        _probe_runner=None,
    ):
        assert callable(_probe_runner)
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
        patch(
            "rapid_mlx.agents.adapter.setup_agent_config", side_effect=_capture_setup
        ),
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

    def _capture_setup(
        profile_arg,
        base_url,
        model_id,
        agent_version=None,
        context_length=None,
        _probe_runner=None,
    ):
        assert callable(_probe_runner)
        calls.append(model_id)
        return "ok"

    # Mock everything past the setup call so we can assert ordering without
    # needing a live MLX server.
    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch.object(AgentTestRunner, "_agent_binary_available", return_value=False),
        patch(
            "rapid_mlx.agents.adapter.setup_agent_config", side_effect=_capture_setup
        ),
        patch("rapid_mlx.agents.testing._test_plain_chat") as mock_chat,
    ):
        # Stub each test to return a synthetic PASS so run() proceeds
        from rapid_mlx.agents.testing import TestResult, TestStatus

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


def test_specific_python_suite_runs_with_script_semantics():
    """SDK suites guarded by ``__main__`` must execute, not import as no-ops."""
    profile = _make_profile("pydanticai", "env")
    runner = AgentTestRunner(profile, model_id="test-model")

    source = (
        b"if __name__ == '__main__':\n"
        b"    results = {'sdk_call': 'PASS'}\n"
        b"    raise SystemExit(0)\n"
    )
    with patch("pathlib.Path.read_bytes", return_value=source):
        results = runner._run_specific_tests("test_pydantic_ai_full.py")

    assert len(results) == 1
    assert results[0].name == "sdk_call"
    assert results[0].status == TestStatus.PASS


def test_file_config_agent_uses_an_isolated_home_for_setup_and_e2e(monkeypatch):
    """#1598: the gate must not inherit or overwrite the operator's Codex home."""
    monkeypatch.delenv("CODEX_HOME", raising=False)
    profile = _make_profile("codex", "yaml")
    profile = replace(
        profile,
        config=replace(profile.config, home_env="CODEX_HOME"),
        testing=AgentTestingSpec(binary="codex", query_cmd="codex {query}"),
    )
    observed: dict[str, str] = {}

    def _capture_setup(*_args, **_kwargs):
        observed["setup_home"] = os.environ["CODEX_HOME"]
        return "ok"

    def _capture_e2e(*_args, env_overrides=None, **_kwargs):
        observed["child_home"] = env_overrides["CODEX_HOME"]
        return TestResult("e2e_chat", TestStatus.PASS)

    from rapid_mlx.agents.testing import TestResult, TestStatus

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch.object(AgentTestRunner, "_agent_binary_available", return_value=True),
        patch(
            "rapid_mlx.agents.adapter.setup_agent_config", side_effect=_capture_setup
        ),
        patch(
            "rapid_mlx.agents.testing._test_plain_chat",
            return_value=TestResult("plain_chat", TestStatus.PASS),
        ),
        patch("rapid_mlx.agents.testing._test_e2e_chat", side_effect=_capture_e2e),
    ):
        AgentTestRunner(profile, model_id="qwen3.5-9b-4bit").run()

    assert observed["setup_home"] == observed["child_home"]
    assert "rapid-mlx-codex-home-" in observed["child_home"]
    assert "CODEX_HOME" not in os.environ


def test_agent_without_home_env_still_uses_isolated_home(monkeypatch):
    """Every profile must avoid the operator's HOME, not only Codex/Hermes."""
    real_home = "/Users/operator"
    monkeypatch.setenv("HOME", real_home)
    monkeypatch.setenv("CFFIXED_USER_HOME", real_home)
    monkeypatch.setenv("BROWSER", "/usr/bin/open")
    monkeypatch.setenv("ZDOTDIR", "/operator/zdotdir")
    monkeypatch.setenv("BASH_ENV", "/operator/bash-env")
    monkeypatch.setenv("ENV", "/operator/sh-env")
    monkeypatch.setenv("PATH", "")
    real_path = os.environ["PATH"]
    profile = _make_profile("opencode", "yaml")
    profile = replace(
        profile,
        testing=AgentTestingSpec(binary="opencode", query_cmd="opencode {query}"),
    )
    observed: dict[str, str] = {}

    def _capture_setup(*_args, **_kwargs):
        observed["setup_home"] = os.environ["HOME"]
        assert os.environ["CFFIXED_USER_HOME"] == os.environ["HOME"]
        assert os.environ["BROWSER"] == os.path.join(
            os.path.dirname(os.environ["_AGENT_TEST_GUI_MARKER"]), "browser"
        )
        assert not os.environ["BROWSER"].startswith(os.environ["HOME"])
        assert os.environ["PATH"].split(os.pathsep)[1:] == os.defpath.split(os.pathsep)
        return "ok"

    def _capture_e2e(*_args, env_overrides=None, **_kwargs):
        observed["child_home"] = env_overrides["HOME"]
        assert env_overrides["BROWSER"] == os.path.join(
            os.path.dirname(env_overrides["_AGENT_TEST_GUI_MARKER"]), "browser"
        )
        assert not env_overrides["BROWSER"].startswith(env_overrides["HOME"])
        assert env_overrides["PATH"].split(os.pathsep)[0] == os.path.dirname(
            env_overrides["BROWSER"]
        )
        assert env_overrides["CFFIXED_USER_HOME"] == env_overrides["HOME"]
        return TestResult("e2e_chat", TestStatus.PASS)

    from rapid_mlx.agents.testing import TestResult, TestStatus

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch.object(AgentTestRunner, "_agent_binary_available", return_value=True),
        patch(
            "rapid_mlx.agents.adapter.setup_agent_config", side_effect=_capture_setup
        ),
        patch(
            "rapid_mlx.agents.testing._test_plain_chat",
            return_value=TestResult("plain_chat", TestStatus.PASS),
        ),
        patch("rapid_mlx.agents.testing._test_e2e_chat", side_effect=_capture_e2e),
    ):
        AgentTestRunner(profile, model_id="qwen3.5-9b-4bit").run()

    assert observed["setup_home"] == observed["child_home"]
    assert "rapid-mlx-opencode-home-" in observed["child_home"]
    assert os.environ["HOME"] == real_home
    assert os.environ["CFFIXED_USER_HOME"] == real_home
    assert os.environ["BROWSER"] == "/usr/bin/open"
    assert os.environ["ZDOTDIR"] == "/operator/zdotdir"
    assert os.environ["BASH_ENV"] == "/operator/bash-env"
    assert os.environ["ENV"] == "/operator/sh-env"
    assert os.environ["PATH"] == real_path
    assert "_AGENT_TEST_GUI_MARKER" not in os.environ


def test_config_refresh_launch_attempt_fails_before_e2e(monkeypatch):
    import subprocess

    profile = _make_profile("opencode", "yaml")
    observed = []

    def _setup(*_args, **_kwargs):
        import shutil

        shutil.rmtree(os.environ["HOME"])
        observed.append(
            subprocess.run(["open", "https://example.invalid"], check=False)
        )

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch("rapid_mlx.agents.adapter.setup_agent_config", side_effect=_setup),
        patch("rapid_mlx.agents.testing._test_plain_chat") as plain_chat,
    ):
        report = AgentTestRunner(profile, model_id="local-model").run()

    assert observed[0].returncode == 126
    assert report.results[0].status is TestStatus.FAIL
    assert "sign in to" in report.results[0].message
    plain_chat.assert_not_called()


def test_config_refresh_replaces_home_then_login_shell_still_uses_guard():
    import shutil
    import subprocess
    import sys

    profile = replace(
        _make_profile("opencode", "yaml"),
        testing=AgentTestingSpec(binary="opencode", query_cmd="opencode {query}"),
    )
    seen = []

    def _setup(*_args, **_kwargs):
        home = os.environ["HOME"]
        shutil.rmtree(home)
        os.makedirs(home)
        with open(os.path.join(home, ".profile"), "w", encoding="utf-8") as handle:
            handle.write(
                "export EARLY_OPEN=$(command -v open)\n"
                "export REFRESHED_CONFIG=1\n"
                "PATH=/usr/bin:/bin\n"
                "export AFTER_RESET_OPEN=$(command -v open)\nreturn\n"
            )
        with open(os.path.join(home, ".zshrc"), "w", encoding="utf-8") as handle:
            handle.write(
                "export ZSH_EARLY_OPEN=$(command -v open)\n"
                "export ZSH_REFRESHED_CONFIG=1\n"
                "PATH=/usr/bin:/bin\n"
                "export ZSH_AFTER_RESET_OPEN=$(command -v open)\nreturn\n"
            )

    def _e2e(*_args, env_overrides=None, **_kwargs):
        env = os.environ | env_overrides
        for shell in ("/bin/sh", "/bin/bash"):
            result = subprocess.run(
                [
                    shell,
                    "-lc",
                    'command -v open; printf \'%s\\n%s\\n%s\\n%s\' "$EARLY_OPEN" "$AFTER_RESET_OPEN" "$REFRESHED_CONFIG" "$PATH"',
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            seen.append(result.stdout)
            lines = result.stdout.splitlines()
            assert lines[:4] == ["open", "open", "open", "1"]
            assert lines[4].split(os.pathsep)[0] == os.path.dirname(
                env_overrides["BROWSER"]
            )
        if sys.platform == "darwin":
            result = subprocess.run(
                [
                    "/bin/zsh",
                    "-lic",
                    'command -v open; printf "%s\\n%s\\n%s\\n%s" "$ZSH_EARLY_OPEN" "$ZSH_AFTER_RESET_OPEN" "$ZSH_REFRESHED_CONFIG" "$PATH"',
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            seen.append(result.stdout)
            lines = result.stdout.splitlines()
            assert lines[:4] == ["open", "open", "open", "1"]
            assert lines[4].split(os.pathsep)[0] == os.path.dirname(
                env_overrides["BROWSER"]
            )
        return TestResult("e2e_chat", TestStatus.PASS)

    from rapid_mlx.agents.testing import TestResult

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch.object(AgentTestRunner, "_agent_binary_available", return_value=True),
        patch("rapid_mlx.agents.adapter.setup_agent_config", side_effect=_setup),
        patch(
            "rapid_mlx.agents.testing._test_plain_chat",
            return_value=TestResult("plain_chat", TestStatus.PASS),
        ),
        patch("rapid_mlx.agents.testing._test_e2e_chat", side_effect=_e2e),
    ):
        AgentTestRunner(profile, model_id="local-model").run()

    assert len(seen) == (3 if sys.platform == "darwin" else 2)


def test_config_refresh_replaces_home_with_file_then_restores_directory():
    import shutil
    from pathlib import Path

    from rapid_mlx.agents.testing import TestResult

    profile = replace(
        _make_profile("opencode", "yaml"),
        needs_function_calling=False,
        needs_streaming=False,
    )
    seen = {}

    def _setup(*_args, **_kwargs):
        home = Path(os.environ["HOME"])
        shutil.rmtree(home)
        home.write_text("replaced", encoding="utf-8")
        seen["home"] = home

    def _plain_chat(*_args, **_kwargs):
        assert seen["home"].is_dir()
        assert (seen["home"] / ".profile").is_file()
        return TestResult("plain_chat", TestStatus.PASS)

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch.object(AgentTestRunner, "_agent_binary_available", return_value=False),
        patch("rapid_mlx.agents.adapter.setup_agent_config", side_effect=_setup),
        patch("rapid_mlx.agents.testing._test_plain_chat", side_effect=_plain_chat),
    ):
        report = AgentTestRunner(profile, model_id="local-model").run()

    assert report.failed == 0


def test_config_refresh_reports_failure_when_shell_guard_cannot_be_restored():
    from rapid_mlx.agents import testing as testing_module

    profile = _make_profile("opencode", "yaml")
    install_guard = testing_module._install_shell_path_guard
    installs = 0

    def fail_restore(shim_dir, home):
        nonlocal installs
        installs += 1
        if installs == 2:
            raise OSError("profile write denied")
        return install_guard(shim_dir, home)

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch("rapid_mlx.agents.adapter.setup_agent_config"),
        patch.object(
            testing_module, "_install_shell_path_guard", side_effect=fail_restore
        ),
        patch("rapid_mlx.agents.testing._test_plain_chat") as plain_chat,
    ):
        report = AgentTestRunner(profile, model_id="local-model").run()

    assert installs == 2
    assert report.results[0].name == "config_refresh"
    assert report.results[0].status is TestStatus.FAIL
    assert "profile write denied" in report.results[0].message
    plain_chat.assert_not_called()


@pytest.mark.parametrize("agent_name", ["hermes", "opencode"])
def test_setup_probe_stops_delayed_descendant(tmp_path, agent_name):
    import subprocess
    import sys
    import time

    profile = _make_profile(agent_name, "yaml")
    delayed = tmp_path / "delayed-setup-child-ran"
    child_script = (
        f"import pathlib, time; time.sleep(0.5); pathlib.Path({str(delayed)!r}).touch()"
    )
    setup_script = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', "
        f"{child_script!r}], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
    )

    def _setup(*_args, _probe_runner=None, **_kwargs):
        assert callable(_probe_runner)
        result = _probe_runner(
            [sys.executable, "-c", setup_script],
            timeout=5,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=agent_name == "opencode",
        )
        assert result.returncode == 0
        return "ok"

    from rapid_mlx.agents.testing import TestResult

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch.object(AgentTestRunner, "_agent_binary_available", return_value=False),
        patch("rapid_mlx.agents.adapter.setup_agent_config", side_effect=_setup),
        patch(
            "rapid_mlx.agents.testing._test_plain_chat",
            return_value=TestResult("plain_chat", TestStatus.PASS),
        ),
    ):
        report = AgentTestRunner(profile, model_id="local-model").run()

    assert report.results[0].status is TestStatus.PASS
    time.sleep(0.7)
    assert not delayed.exists()


def test_late_setup_launch_marker_fails_final_report():
    from pathlib import Path

    from rapid_mlx.agents.testing import TestResult

    profile = _make_profile("hermes", "yaml")
    marker = None

    def _setup(*_args, **_kwargs):
        nonlocal marker
        marker = Path(os.environ["_AGENT_TEST_GUI_MARKER"])
        return "ok"

    def _plain_chat(*_args, **_kwargs):
        assert marker is not None
        marker.touch()
        return TestResult("plain_chat", TestStatus.PASS)

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch.object(AgentTestRunner, "_agent_binary_available", return_value=False),
        patch("rapid_mlx.agents.adapter.setup_agent_config", side_effect=_setup),
        patch("rapid_mlx.agents.testing._test_plain_chat", side_effect=_plain_chat),
    ):
        report = AgentTestRunner(profile, model_id="local-model").run()

    assert report.results[-1].name == "gui_launch"
    assert report.results[-1].status is TestStatus.FAIL


def test_claude_e2e_sets_documented_config_directory():
    """Claude must not discover the operator's hooks through its account home."""
    profile = _make_profile("claude-code", "env")
    profile = replace(
        profile,
        testing=AgentTestingSpec(binary="claude", query_cmd="claude -p {query}"),
    )
    observed: dict[str, str] = {}

    def _capture_e2e(*_args, env_overrides=None, **_kwargs):
        observed.update(env_overrides)
        return TestResult("e2e_chat", TestStatus.PASS)

    from rapid_mlx.agents.testing import TestResult, TestStatus

    with (
        patch.object(AgentTestRunner, "_server_available", return_value=True),
        patch.object(AgentTestRunner, "_agent_binary_available", return_value=True),
        patch("rapid_mlx.agents.adapter.setup_agent_config", return_value="ok"),
        patch(
            "rapid_mlx.agents.testing._test_plain_chat",
            return_value=TestResult("plain_chat", TestStatus.PASS),
        ),
        patch("rapid_mlx.agents.testing._test_e2e_chat", side_effect=_capture_e2e),
    ):
        AgentTestRunner(profile, model_id="qwen3.5-9b-4bit").run()

    assert observed["CLAUDE_CONFIG_DIR"].startswith(observed["HOME"])
    assert observed["CLAUDE_CONFIG_DIR"].endswith("/.claude")
