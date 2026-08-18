from __future__ import annotations

import contextlib
import json
import stat
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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


# --------------------------------------------------------------------------- #
# reasoningEfforts must describe the served model, not every model.
# --------------------------------------------------------------------------- #
#
# DSH renders a reasoning-effort control from this block. Advertising the
# off/low/medium/high ladder unconditionally gave users a selector that did
# nothing on a model with no reasoning parser. pi-ai accepts
# ``reasoningEfforts: false`` for exactly that case.


def _models_payload(entries: list[dict]) -> bytes:
    return json.dumps({"object": "list", "data": entries}).encode()


@contextlib.contextmanager
def _serving(entries: list[dict]):
    """Run a throwaway /v1/models server and yield its base_url.

    A real socket rather than a mocked urlopen: the thing under test is
    "what do we conclude from what the server actually sent", and a mock
    would let a wrong field name pass.
    """
    payload = _models_payload(entries)

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler API
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):  # keep pytest output clean
            return

    srv = HTTPServer(("127.0.0.1", 0), _H)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}/v1"
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _dsh_model_entry(plan) -> dict:
    return plan.after["llm-pi-ai"]["providers"]["rapid-mlx"]["models"][0]


def _dsh_provider(plan) -> dict:
    return plan.after["llm-pi-ai"]["providers"]["rapid-mlx"]


def test_reasoning_support_is_three_state(tmp_path, monkeypatch):
    """True / False / None must be distinguishable at the source."""
    from vllm_mlx.agents.adapter import fetch_reasoning_support

    with _serving([{"id": "m", "reasoning_parser": "qwen3"}]) as url:
        assert fetch_reasoning_support(url, "m") is True
    # Explicit null — the model has no reasoning parser.
    with _serving([{"id": "m", "reasoning_parser": None}]) as url:
        assert fetch_reasoning_support(url, "m") is False
    # Key absent — a rapid-mlx too old to report it. NOT the same as False.
    with _serving([{"id": "m"}]) as url:
        assert fetch_reasoning_support(url, "m") is None
    # Multi-model serve with no exact match must not describe another model.
    with _serving(
        [{"id": "a", "reasoning_parser": None}, {"id": "b", "reasoning_parser": None}]
    ) as url:
        assert fetch_reasoning_support(url, "missing") is None
    # Unreachable server.
    assert fetch_reasoning_support("http://127.0.0.1:9/v1", "m") is None


def test_dsh_setup_declines_reasoning_for_a_model_without_a_parser(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    plan = build_setup_plan(
        "dsh",
        "http://127.0.0.1:8152/v1",
        "no-think-1b",
        65536,
        supports_reasoning=False,
    )
    assert _dsh_model_entry(plan)["reasoningEfforts"] is False
    assert _dsh_provider(plan)["compat"]["supportsReasoningEffort"] is False


def test_dsh_setup_keeps_the_ladder_for_a_reasoning_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    plan = build_setup_plan(
        "dsh",
        "http://127.0.0.1:8152/v1",
        "qwen3.6-35b-8bit",
        65536,
        supports_reasoning=True,
    )
    assert _dsh_model_entry(plan)["reasoningEfforts"]["off"] == "none"
    assert _dsh_provider(plan)["compat"]["supportsReasoningEffort"] is True


def test_dsh_setup_does_not_downgrade_on_unknown(tmp_path, monkeypatch):
    """``None`` means we could not find out — never delete a real control on a guess."""
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    for unknown in (None,):
        plan = build_setup_plan(
            "dsh", "http://127.0.0.1:8152/v1", "m", 65536, supports_reasoning=unknown
        )
        assert _dsh_model_entry(plan)["reasoningEfforts"]["high"] == "high"
        assert _dsh_provider(plan)["compat"]["supportsReasoningEffort"] is True
    # Default argument must behave the same as an explicit None.
    plan = build_setup_plan("dsh", "http://127.0.0.1:8152/v1", "m", 65536)
    assert _dsh_model_entry(plan)["reasoningEfforts"]["high"] == "high"


def test_dsh_declined_reasoning_survives_yaml_round_trip(tmp_path, monkeypatch):
    """``false`` must reach the file as a YAML boolean, not the string 'False'.

    pi-ai validates ``reasoningEfforts`` as ``false | {level: wire}``; a
    quoted string would fail its schema and DSH would refuse the provider.
    """
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    plan = build_setup_plan(
        "dsh",
        "http://127.0.0.1:8000/v1",
        "no-think-1b",
        65536,
        supports_reasoning=False,
    )
    apply_setup_plan(plan)
    written = yaml.safe_load((tmp_path / "settings.yaml").read_text())
    entry = written["llm-pi-ai"]["providers"]["rapid-mlx"]["models"][0]
    assert entry["reasoningEfforts"] is False
    raw = (tmp_path / "settings.yaml").read_text()
    assert "reasoningEfforts: false" in raw, raw
