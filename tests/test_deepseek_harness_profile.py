from __future__ import annotations

import json
import stat
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from vllm_mlx.agents import get_profile, load_profiles
from vllm_mlx.agents.setup import apply_setup_plan, build_setup_plan
from vllm_mlx.agents.testing import (
    AgentTestRunner,
    TestResult,
    TestStatus,
    _agent_query,
)


def test_dsh_alias_resolves_first_class_profile():
    load_profiles()
    profile = get_profile("dsh")
    assert profile is not None
    assert profile.name == "deepseek-harness"
    assert profile.config.home_env == "DSH_HOME"
    assert profile.testing.binary == "dsh"
    assert profile.testing.query_cmd == "dsh --profile headless '{query}'"


def test_hermes_binary_uses_path_instead_of_one_obsolete_install_layout():
    load_profiles()
    profile = get_profile("hermes")
    assert profile is not None
    assert profile.testing.binary == "hermes"


def test_aider_profile_runs_the_real_cli():
    load_profiles()
    profile = get_profile("aider")
    assert profile is not None
    assert profile.testing.binary == "aider"
    assert profile.testing.query_cmd is not None


def test_qwen_code_profile_matches_current_provider_schema():
    load_profiles()
    profile = get_profile("qwen-code")
    assert profile is not None
    rendered = json.loads(
        profile.render_config(
            "http://127.0.0.1:8153/v1",
            "qwen3.5-9b-4bit",
            context_length=262144,
        )
    )
    model = rendered["modelProviders"]["openai"][0]
    assert model["baseUrl"] == "http://127.0.0.1:8153/v1"
    assert model["generationConfig"]["contextWindowSize"] == 262144
    assert rendered["security"]["auth"]["selectedType"] == "openai"
    assert profile.testing.install_cmd == "npm install -g @qwen-code/qwen-code@latest"


def test_kilo_profile_matches_current_cli_path_and_schema():
    load_profiles()
    profile = get_profile("kilo-code")
    assert profile is not None
    assert profile.config.path == "~/.config/kilo/kilo.json"
    rendered = json.loads(
        profile.render_config(
            "http://127.0.0.1:8153/v1",
            "qwen3.5-9b-4bit",
            context_length=262144,
        )
    )
    assert rendered["model"] == "rapid-mlx/qwen3.5-9b-4bit"
    model = rendered["provider"]["rapid-mlx"]["models"]["qwen3.5-9b-4bit"]
    assert model["limit"] == {"context": 262144, "output": 8192}
    assert profile.testing.install_cmd == "npm install -g @kilocode/cli"
    assert "--dir '{cwd}'" in profile.testing.query_cmd
    assert "--pure" in profile.testing.query_cmd


def test_dsh_setup_plan_is_side_effect_free_and_uses_real_capacity(
    tmp_path, monkeypatch
):
    dsh_home = tmp_path / "dsh"
    settings = dsh_home / "settings.yaml"
    dsh_home.mkdir()
    settings.write_text(
        "ui-theme:\n  theme: dark\nllm-pi-ai:\n  providers:\n    existing:\n      baseURL: https://example.test/v1\n"
    )
    original = settings.read_text()
    monkeypatch.setenv("DSH_HOME", str(dsh_home))

    plan = build_setup_plan(
        "deepseek-harness",
        "http://127.0.0.1:8152/v1",
        "qwen3.5-9b-4bit",
        context_length=262144,
    )

    assert settings.read_text() == original
    assert plan.after["ui-theme"] == {"theme": "dark"}
    assert "existing" in plan.after["llm-pi-ai"]["providers"]
    rapid = plan.after["llm-pi-ai"]["providers"]["rapid-mlx"]
    assert rapid["baseURL"] == "http://127.0.0.1:8152/v1"
    assert rapid["apiKeyEnv"] == "RAPID_MLX_API_KEY"
    assert rapid["models"][0]["contextWindow"] == 262144
    assert rapid["models"][0]["reasoningEfforts"]["off"] == "none"
    assert plan.after["agent-default-model"] == {
        "provider": "rapid-mlx",
        "model": "qwen3.5-9b-4bit",
    }
    assert plan.credentials_after == {"RAPID_MLX_API_KEY": "not-needed"}


def test_dsh_setup_apply_is_atomic_and_backed_up(tmp_path, monkeypatch):
    dsh_home = tmp_path / "dsh"
    settings = dsh_home / "settings.yaml"
    dsh_home.mkdir()
    settings.write_text("ui-theme:\n  theme: dark\n")
    monkeypatch.setenv("DSH_HOME", str(dsh_home))

    plan = build_setup_plan("dsh", "http://127.0.0.1:8152/v1", "qwen3.5-9b-4bit", 65536)
    apply_setup_plan(plan)

    written = yaml.safe_load(settings.read_text())
    assert written["ui-theme"] == {"theme": "dark"}
    assert written["agent-default-model"]["provider"] == "rapid-mlx"
    assert (
        written["llm-pi-ai"]["providers"]["rapid-mlx"]["models"][0]["contextWindow"]
        == 65536
    )
    assert len(list(dsh_home.glob("settings.yaml.bak.*"))) == 1
    credentials = dsh_home / ".credentials.yaml"
    assert yaml.safe_load(credentials.read_text()) == {
        "RAPID_MLX_API_KEY": "not-needed"
    }
    assert stat.S_IMODE(credentials.stat().st_mode) == 0o600


def test_dsh_setup_repairs_missing_credentials_when_settings_match(
    tmp_path, monkeypatch
):
    """A settings-only prior setup must not make credential repair a no-op."""
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    initial = build_setup_plan(
        "deepseek-harness",
        "http://127.0.0.1:8000/v1",
        "test-model",
        context_length=65536,
    )
    initial.path.write_text(
        yaml.safe_dump(initial.after, sort_keys=False), encoding="utf-8"
    )

    repair = build_setup_plan(
        "deepseek-harness",
        "http://127.0.0.1:8000/v1",
        "test-model",
        context_length=65536,
    )
    assert repair.before == repair.after
    assert repair.changed
    apply_setup_plan(repair)
    assert yaml.safe_load((tmp_path / ".credentials.yaml").read_text()) == {
        "RAPID_MLX_API_KEY": "not-needed"
    }


def test_dsh_setup_preserves_and_redacts_existing_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    secret = "real-local-auth-secret"
    credentials = tmp_path / ".credentials.yaml"
    credentials.write_text(
        yaml.safe_dump(
            {"RAPID_MLX_API_KEY": secret, "REMOTE_PROVIDER_KEY": "other-secret"}
        ),
        encoding="utf-8",
    )

    plan = build_setup_plan("dsh", "http://127.0.0.1:8000/v1", "test-model", 65536)
    assert plan.credentials_after == {
        "RAPID_MLX_API_KEY": secret,
        "REMOTE_PROVIDER_KEY": "other-secret",
    }
    preview = plan.diff()
    assert secret not in preview
    assert "other-secret" not in preview
    assert "REMOTE_PROVIDER_KEY" not in preview
    apply_setup_plan(plan)
    assert yaml.safe_load(credentials.read_text())["RAPID_MLX_API_KEY"] == secret


def test_dsh_setup_refuses_concurrent_change(tmp_path, monkeypatch):
    dsh_home = tmp_path / "dsh"
    dsh_home.mkdir()
    settings = dsh_home / "settings.yaml"
    settings.write_text("ui-theme:\n  theme: dark\n")
    monkeypatch.setenv("DSH_HOME", str(dsh_home))
    plan = build_setup_plan("dsh", "http://127.0.0.1:8152/v1", "model")
    settings.write_text("ui-theme:\n  theme: light\n")

    import pytest

    with pytest.raises(RuntimeError, match="changed after preview"):
        apply_setup_plan(plan)
    assert yaml.safe_load(settings.read_text()) == {"ui-theme": {"theme": "light"}}


def test_dsh_test_runner_supplies_only_dummy_loopback_credential(monkeypatch):
    load_profiles()
    profile = get_profile("deepseek-harness")
    assert profile is not None
    observed = {}

    def capture(*_args, env_overrides=None, **_kwargs):
        observed.update(env_overrides)
        return TestResult("e2e_chat", TestStatus.PASS)

    monkeypatch.setattr(AgentTestRunner, "_server_available", lambda _self: True)
    monkeypatch.setattr(AgentTestRunner, "_agent_binary_available", lambda _self: True)
    monkeypatch.setattr(
        "vllm_mlx.agents.adapter.setup_agent_config", lambda *_args, **_kwargs: "ok"
    )
    monkeypatch.setattr(
        "vllm_mlx.agents.testing._test_plain_chat",
        lambda *_args, **_kwargs: TestResult("plain_chat", TestStatus.PASS),
    )
    monkeypatch.setattr("vllm_mlx.agents.testing._test_e2e_chat", capture)
    monkeypatch.setattr("vllm_mlx.agents.testing._test_e2e_file_read", capture)
    monkeypatch.setattr("vllm_mlx.agents.testing._test_e2e_terminal", capture)
    monkeypatch.setattr(
        "vllm_mlx.agents.testing._test_single_tool_call",
        lambda *_args, **_kwargs: TestResult("tool", TestStatus.PASS),
    )
    monkeypatch.setattr(
        "vllm_mlx.agents.testing._test_tool_choice",
        lambda *_args, **_kwargs: TestResult("tool", TestStatus.PASS),
    )
    monkeypatch.setattr(
        "vllm_mlx.agents.testing._test_multi_turn_tool",
        lambda *_args, **_kwargs: TestResult("tool", TestStatus.PASS),
    )
    monkeypatch.setattr(
        "vllm_mlx.agents.testing._test_no_tool_leak",
        lambda *_args, **_kwargs: TestResult("tool", TestStatus.PASS),
    )
    monkeypatch.setattr(
        "vllm_mlx.agents.testing._test_no_tool_needed",
        lambda *_args, **_kwargs: TestResult("tool", TestStatus.PASS),
    )
    monkeypatch.setattr(
        "vllm_mlx.agents.testing._test_streaming_tool_call",
        lambda *_args, **_kwargs: TestResult("tool", TestStatus.PASS),
    )
    monkeypatch.setattr(
        "vllm_mlx.agents.testing._test_many_tools",
        lambda *_args, **_kwargs: TestResult("tool", TestStatus.PASS),
    )
    monkeypatch.setattr(
        "vllm_mlx.agents.testing._test_streaming_basic",
        lambda *_args, **_kwargs: TestResult("stream", TestStatus.PASS),
    )
    monkeypatch.setattr(
        "vllm_mlx.agents.testing._test_stress_no_leak",
        lambda *_args, **_kwargs: TestResult("stress", TestStatus.PASS),
    )

    AgentTestRunner(profile, model_id="qwen3.5-9b-4bit").run()
    assert observed["RAPID_MLX_API_KEY"] == "not-needed"
    assert (
        observed["DSH_HOME"].startswith("/var/") or "rapid-mlx-" in observed["DSH_HOME"]
    )


def test_dsh_reports_old_node_before_opaque_plugin_boot_failure():
    with (
        patch(
            "vllm_mlx.agents.testing.shutil.which",
            side_effect=["/fake/dsh", "/fake/node"],
        ),
        patch(
            "vllm_mlx.agents.testing.subprocess.run",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr=""),
        ) as run,
    ):
        output, error = _agent_query("dsh", "dsh {query}", "hello")

    assert output is None
    assert "Node 22.15+" in error
    assert run.call_count == 1
