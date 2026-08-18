from __future__ import annotations

import json

import pytest

from vllm_mlx.agents.setup import (
    apply_setup_plan,
    build_setup_plan,
    verify_server,
)
from vllm_mlx.launch import claude_code, continue_dev


@pytest.fixture
def setup_paths(tmp_path, monkeypatch):
    claude_path = tmp_path / "claude" / "settings.json"
    continue_path = tmp_path / "continue" / "config.json"
    monkeypatch.setattr(claude_code, "current_config_path", lambda: claude_path)
    monkeypatch.setattr(continue_dev, "current_config_path", lambda: continue_path)
    return claude_path, continue_path


def test_claude_plan_is_side_effect_free_and_uses_bare_base(setup_paths):
    claude_path, _ = setup_paths
    claude_path.parent.mkdir(parents=True)
    claude_path.write_text('{"permissions":{"allow":["Read"]}}')

    plan = build_setup_plan(
        "claude-code", "http://localhost:8000/v1", "qwen3.6-35b-4bit"
    )

    assert json.loads(claude_path.read_text()) == {"permissions": {"allow": ["Read"]}}
    assert plan.after["permissions"] == {"allow": ["Read"]}
    assert plan.after["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:8000"
    assert "ANTHROPIC_API_KEY" in plan.diff()


def test_continue_apply_preserves_models_and_creates_backup(setup_paths):
    _, continue_path = setup_paths
    continue_path.parent.mkdir(parents=True)
    continue_path.write_text(
        json.dumps({"models": [{"title": "Existing", "provider": "ollama"}]})
    )
    plan = build_setup_plan("continue", "http://localhost:8000", "qwen3.5-9b-4bit")

    apply_setup_plan(plan)

    data = json.loads(continue_path.read_text())
    assert any(model["title"] == "Existing" for model in data["models"])
    rapid = next(model for model in data["models"] if model["title"] == "rapid-mlx")
    assert rapid["apiBase"] == "http://localhost:8000/v1"
    assert rapid["model"] == "qwen3.5-9b-4bit"
    assert len(list(continue_path.parent.glob("config.json.bak.*"))) == 1


def test_apply_refuses_file_changed_after_preview(setup_paths):
    claude_path, _ = setup_paths
    claude_path.parent.mkdir(parents=True)
    claude_path.write_text("{}")
    plan = build_setup_plan("claude-code", "http://localhost:8000", "model")
    claude_path.write_text('{"new":"concurrent edit"}')

    with pytest.raises(RuntimeError, match="changed after preview"):
        apply_setup_plan(plan)
    assert json.loads(claude_path.read_text()) == {"new": "concurrent edit"}


def test_verify_server_checks_health_and_models(monkeypatch):
    class Response:
        status = 200

        def __init__(self, body=b""):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.body

    urls: list[str] = []

    def fake_open(url, timeout):
        urls.append(url)
        if url.endswith("/health"):
            return Response()
        return Response(json.dumps({"data": [{"id": "served-model"}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    assert verify_server("http://localhost:8000/v1", "default") == "served-model"
    assert urls == [
        "http://localhost:8000/health",
        "http://localhost:8000/v1/models",
    ]


def test_dsh_plan_and_profile_template_agree_on_the_provider_contract(monkeypatch):
    """The two DSH provider definitions must not drift apart.

    ``agents dsh --setup`` builds the provider block in ``agents/setup.py``,
    while ``agents dsh --test`` renders the template in
    ``profiles/deepseek-harness.yaml``. They are deliberately separate — only
    the plan adapts ``reasoningEfforts`` to the served model — but every other
    key is the same contract, and nothing but this test notices when an edit
    lands in one and not the other.
    """
    import yaml

    from vllm_mlx.agents import get_profile
    from vllm_mlx.agents.setup import build_setup_plan

    base_url = "http://localhost:8000/v1"
    model = "qwen3.6-35b-4bit"
    context = 131072

    monkeypatch.setattr(
        "vllm_mlx.agents.setup._dsh_settings_path",
        lambda: __import__("pathlib").Path("/nonexistent/settings.yaml"),
    )
    plan = build_setup_plan("dsh", base_url, model, context_length=context)
    planned = plan.after["llm-pi-ai"]["providers"]["rapid-mlx"]

    profile = get_profile("deepseek-harness")
    rendered = yaml.safe_load(
        profile.render_config(base_url, model, context_length=context)
    )
    templated = rendered["llm-pi-ai"]["providers"]["rapid-mlx"]

    for key in (
        "displayName",
        "apiKeyEnv",
        "api",
        "baseURL",
        "defaultContextWindow",
        "defaultMaxTokens",
    ):
        assert planned[key] == templated[key], f"DSH provider key drifted: {key}"

    assert plan.after["agent-default-model"] == rendered["agent-default-model"]

    planned_model = planned["models"][0]
    templated_model = templated["models"][0]
    for key in ("id", "name", "contextWindow", "maxTokens"):
        assert planned_model[key] == templated_model[key], (
            f"DSH model key drifted: {key}"
        )


def test_cli_parser_exposes_setup_safety_flags(monkeypatch):
    import vllm_mlx.cli as cli

    captured = {}
    monkeypatch.setattr(cli, "agents_command", lambda args: captured.update(vars(args)))
    monkeypatch.setattr(
        "sys.argv", ["rapid-mlx", "agents", "continue", "--setup", "--dry-run"]
    )
    cli.main()
    assert captured["agent_name"] == "continue"
    assert captured["setup"] is True
    assert captured["dry_run"] is True
    assert captured["yes"] is False


def test_cli_reports_saved_config_when_connection_check_fails(
    setup_paths, monkeypatch, capsys
):
    import vllm_mlx.cli as cli

    _, continue_path = setup_paths
    monkeypatch.setattr(
        "sys.argv",
        [
            "rapid-mlx",
            "agents",
            "continue",
            "--setup",
            "--yes",
            "--model",
            "qwen3.5-4b-4bit",
        ],
    )
    monkeypatch.setattr(
        "vllm_mlx.agents.setup.verify_server",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("connection refused")),
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    assert continue_path.exists()
    output = capsys.readouterr().out
    assert "Configuration was saved, but the connection check failed" in output
    assert "Setup incomplete" not in output
