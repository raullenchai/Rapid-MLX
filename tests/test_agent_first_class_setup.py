from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import Request

import pytest

from rapid_mlx.agents.setup import (
    apply_setup_plan,
    build_setup_plan,
    verify_server,
)
from rapid_mlx.launch import claude_code, continue_dev


@pytest.fixture
def setup_paths(tmp_path, monkeypatch):
    claude_path = tmp_path / "claude" / "settings.json"
    continue_path = tmp_path / "continue" / "config.yaml"
    monkeypatch.setattr(claude_code, "current_config_path", lambda: claude_path)
    monkeypatch.setattr(continue_dev, "current_config_path", lambda: continue_path)
    return claude_path, continue_path


def test_claude_plan_is_side_effect_free_and_uses_bare_base(setup_paths):
    claude_path, _ = setup_paths
    claude_path.parent.mkdir(parents=True)
    claude_path.write_text('{"permissions":{"allow":["Read"]}}')

    plan = build_setup_plan(
        "claude-code",
        "http://localhost:8000/v1",
        "qwen3.6-35b-4bit",
        context_length=131072,
    )

    assert json.loads(claude_path.read_text()) == {"permissions": {"allow": ["Read"]}}
    assert plan.after["permissions"] == {"allow": ["Read"]}
    assert plan.after["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:8000"
    assert plan.after["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
    assert "ANTHROPIC_API_KEY" in plan.diff()


def test_claude_plan_preserves_context_override_when_server_has_no_limit(setup_paths):
    claude_path, _ = setup_paths
    claude_path.parent.mkdir(parents=True)
    claude_path.write_text('{"env":{"CLAUDE_CODE_MAX_CONTEXT_TOKENS":"65536"}}')

    plan = build_setup_plan("claude-code", "http://localhost:8000/v1", "local-model")

    assert plan.after["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"


def test_claude_preview_hides_unrelated_environment_credentials(setup_paths):
    claude_path, _ = setup_paths
    claude_path.parent.mkdir(parents=True)
    credentials = {
        "OPENAI_API_KEY": "private-openai-value",
        "AWS_SECRET_ACCESS_KEY": "private-aws-value",
        "CUSTOM_CREDENTIAL": "private-custom-value",
    }
    claude_path.write_text(json.dumps({"env": credentials}))

    plan = build_setup_plan("claude-code", "http://localhost:8000/v1", "local-model")

    preview = plan.diff()
    for name, value in credentials.items():
        assert value not in preview
        assert plan.after["env"][name] == value
    assert "http://localhost:8000" in preview


def test_claude_cli_setup_fetches_live_context_for_local_model(
    setup_paths, monkeypatch, capsys
):
    from rapid_mlx import cli

    monkeypatch.setattr(
        "rapid_mlx.agents.adapter._detect_running_model",
        lambda _url: ("local-model", 131072),
    )

    cli.agents_command(
        SimpleNamespace(
            agent_name="claude-code",
            base_url="http://localhost:8000/v1",
            test=False,
            setup=True,
            model=None,
            agent_version=None,
            dry_run=True,
            yes=False,
            no_check=True,
        )
    )

    assert '"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "131072"' in capsys.readouterr().out


def test_continue_apply_preserves_models_and_creates_backup(setup_paths):
    import yaml

    _, continue_path = setup_paths
    continue_path.parent.mkdir(parents=True)
    continue_path.write_text(
        "name: Mine\nversion: 1.0.0\nschema: v1\n"
        "models:\n- name: Existing\n  provider: ollama\n  model: llama3\n"
    )
    plan = build_setup_plan("continue", "http://localhost:8000", "qwen3.5-9b-4bit")

    apply_setup_plan(plan)

    data = yaml.safe_load(continue_path.read_text())
    assert any(model["name"] == "Existing" for model in data["models"])
    rapid = next(model for model in data["models"] if model["name"] == "rapid-mlx")
    assert rapid["apiBase"] == "http://localhost:8000/v1"
    assert rapid["model"] == "qwen3.5-9b-4bit"
    assert len(list(continue_path.parent.glob("config.yaml.bak.*"))) == 1
    assert continue_path.stat().st_mode & 0o777 == 0o600


def test_continue_plan_migrates_legacy_json_and_leaves_it_untouched(setup_paths):
    import yaml

    _, continue_path = setup_paths
    legacy = continue_path.with_name("config.json")
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "models": [
                    {"title": "Existing", "provider": "anthropic", "apiKey": "real-key"}
                ],
                "systemMessage": "be brief",
            }
        )
    )
    legacy_bytes = legacy.read_bytes()

    plan = build_setup_plan("continue", "http://localhost:8000", "qwen3.5-9b-4bit")
    preview = plan.diff()
    assert "real-key" not in preview
    assert "config.json is left unchanged" in preview
    assert plan.path == continue_path and plan.migrated_from == legacy

    apply_setup_plan(plan)

    data = yaml.safe_load(continue_path.read_text())
    assert [m["name"] for m in data["models"]] == ["rapid-mlx", "Existing"]
    assert data["models"][1]["apiKey"] == "real-key"
    assert data["rules"] == ["be brief"]
    assert legacy.read_bytes() == legacy_bytes
    # Re-running is a no-op plan.
    assert not build_setup_plan(
        "continue", "http://localhost:8000", "qwen3.5-9b-4bit"
    ).changed


def test_continue_apply_refuses_legacy_json_changed_after_preview(setup_paths):
    _, continue_path = setup_paths
    legacy = continue_path.with_name("config.json")
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"models": [{"title": "A", "provider": "ollama"}]}')
    plan = build_setup_plan("continue", "http://localhost:8000", "model")
    legacy.write_text('{"models": [{"title": "B", "provider": "ollama"}]}')

    with pytest.raises(RuntimeError, match="changed after preview"):
        apply_setup_plan(plan)
    assert not continue_path.exists()


def test_continue_apply_treats_unreadable_legacy_json_as_changed(setup_paths):
    _, continue_path = setup_paths
    legacy = continue_path.with_name("config.json")
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"models": [{"title": "A", "provider": "ollama"}]}')
    plan = build_setup_plan("continue", "http://localhost:8000", "model")
    legacy.write_text('{"models": [')

    with pytest.raises(RuntimeError, match="changed after preview"):
        apply_setup_plan(plan)
    assert not continue_path.exists()


def test_continue_plan_reports_invalid_yaml_as_value_error(setup_paths):
    _, continue_path = setup_paths
    continue_path.parent.mkdir(parents=True)
    continue_path.write_text("models: [unclosed\n")

    with pytest.raises(ValueError, match="not valid YAML"):
        build_setup_plan("continue", "http://localhost:8000", "model")


def test_qwen_code_apply_preserves_providers_and_creates_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = tmp_path / ".qwen" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "modelProviders": {
                    "openai": [
                        {"id": "existing", "name": "Existing provider"},
                        {
                            "id": "local-model",
                            "name": "Custom name",
                            "custom": True,
                        },
                    ]
                },
                "custom": {"preserved": True},
            }
        )
    )

    plan = build_setup_plan(
        "qwen-code",
        "http://localhost:8000/v1",
        "local-model",
        context_length=131072,
    )

    assert json.loads(settings_path.read_text())["custom"] == {"preserved": True}
    assert plan.after["custom"] == {"preserved": True}
    providers = plan.after["modelProviders"]["openai"]
    assert [provider["id"] for provider in providers] == ["existing", "local-model"]
    rapid = providers[1]
    assert rapid["custom"] is True
    assert rapid["name"] == "local-model (Rapid-MLX)"
    assert rapid["baseUrl"] == "http://localhost:8000/v1"
    assert rapid["generationConfig"]["contextWindowSize"] == 131072

    apply_setup_plan(plan)

    written = json.loads(settings_path.read_text())
    assert written == plan.after
    assert written["custom"] == {"preserved": True}
    assert len(list(settings_path.parent.glob("settings.json.bak.*"))) == 1


@pytest.mark.parametrize(
    "content", [b"{not json", b"[1, 2]", b"\xff\xfe{}", b"[" * 100_000]
)
def test_qwen_code_plan_refuses_an_unmergeable_settings_file(
    tmp_path, monkeypatch, content
):
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = tmp_path / ".qwen" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_bytes(content)

    with (
        patch("rapid_mlx.agents.setup.track_agent_configure_failed") as failed,
        pytest.raises(ValueError),
    ):
        build_setup_plan("qwen-code", "http://localhost:8000/v1", "local-model")

    failed.assert_called_once_with("config_invalid", "qwen-code")
    assert settings_path.read_bytes() == content
    assert not list(settings_path.parent.glob("settings.json.bak.*"))


def test_qwen_code_plan_reports_an_unreadable_settings_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # A directory where the file belongs exists but cannot be read as text.
    (tmp_path / ".qwen" / "settings.json").mkdir(parents=True)

    with (
        patch("rapid_mlx.agents.setup.track_agent_configure_failed") as failed,
        pytest.raises(OSError),
    ):
        build_setup_plan("qwen-code", "http://localhost:8000/v1", "local-model")

    failed.assert_called_once_with("other", "qwen-code")


@pytest.mark.parametrize(
    "config",
    [
        {"type": "env", "path": None, "template": None},
        {"template": '{"model": {"name": "{model_id}"}}'},
        {"template": '{"modelProviders": {"openai": [{"name": "no id"}]}}'},
        {"template": "[]"},
        {"template": {"modelProviders": {"openai": [{"id": "{model_id}"}]}}},
        {"path": ["~/.qwen/settings.json"]},
    ],
)
def test_qwen_code_plan_refuses_a_shadowing_profile_it_cannot_merge(
    tmp_path, monkeypatch, config
):
    from dataclasses import replace

    from rapid_mlx.agents import get_profile

    monkeypatch.setenv("HOME", str(tmp_path))
    shipped = get_profile("qwen-code")
    shadow = replace(shipped, config=replace(shipped.config, **config))
    monkeypatch.setattr("rapid_mlx.agents.setup._qwen_code_profile", lambda: shadow)

    with pytest.raises(ValueError, match="installed qwen-code profile"):
        build_setup_plan("qwen-code", "http://localhost:8000/v1", "local-model")

    assert not (tmp_path / ".qwen").exists()


def test_qwen_code_plan_leaves_entries_it_cannot_key_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = tmp_path / ".qwen" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    unkeyed = ["text", 7, None, ["nested"], {"name": "no id"}, {"id": ["odd"]}]
    settings_path.write_text(json.dumps({"modelProviders": {"openai": unkeyed}}))

    plan = build_setup_plan("qwen-code", "http://localhost:8000/v1", "local-model")
    apply_setup_plan(plan)

    written = json.loads(settings_path.read_text())["modelProviders"]["openai"]
    assert written[:-1] == unkeyed
    assert written[-1]["id"] == "local-model"


def test_qwen_code_plan_selects_the_config_for_the_agent_version(tmp_path, monkeypatch):
    from dataclasses import replace

    from rapid_mlx.agents import get_profile
    from rapid_mlx.agents.base import AgentVersionSpec

    monkeypatch.setenv("HOME", str(tmp_path))
    shipped = get_profile("qwen-code")
    versioned = replace(
        shipped,
        versions=[
            AgentVersionSpec(
                version_range=">=9.0",
                config=replace(shipped.config, path="~/.qwen-next/settings.json"),
            )
        ],
    )
    monkeypatch.setattr("rapid_mlx.agents.setup._qwen_code_profile", lambda: versioned)

    def plan_path(agent_version):
        return build_setup_plan(
            "qwen-code",
            "http://localhost:8000/v1",
            "local-model",
            agent_version=agent_version,
        ).path

    assert plan_path(None) == (tmp_path / ".qwen" / "settings.json").resolve()
    assert plan_path("1.0.0") == (tmp_path / ".qwen" / "settings.json").resolve()
    assert plan_path("9.1.0") == (tmp_path / ".qwen-next" / "settings.json").resolve()


def test_qwen_code_plan_refuses_an_unresolvable_settings_path(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("HOME", str(tmp_path))

    def symlink_loop(self, strict=False):
        raise RuntimeError(f"Symlink loop from {str(self)!r}")

    # Raised by Path.resolve() on a symlink loop before Python 3.13.
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "resolve", symlink_loop)
        with pytest.raises(ValueError, match="cannot resolve"):
            build_setup_plan("qwen-code", "http://localhost:8000/v1", "local-model")

    assert not (tmp_path / ".qwen").exists()


@pytest.mark.parametrize("content", [b"{not json", b"\xff\xfe{}"])
def test_apply_refuses_file_that_stopped_parsing_after_preview(
    tmp_path, monkeypatch, content
):
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = tmp_path / ".qwen" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{}")
    plan = build_setup_plan("qwen-code", "http://localhost:8000/v1", "local-model")
    settings_path.write_bytes(content)

    with (
        patch("rapid_mlx.agents.setup.track_agent_configure_failed") as failed,
        pytest.raises(RuntimeError, match="changed after preview"),
    ):
        apply_setup_plan(plan)

    failed.assert_called_once_with("config_changed", "qwen-code")
    assert settings_path.read_bytes() == content
    assert not list(settings_path.parent.glob("settings.json.bak.*"))


@pytest.mark.parametrize(
    "existing",
    [
        None,
        {"custom": {"preserved": True}},
        {"modelProviders": "legacy", "custom": {"preserved": True}},
        {"modelProviders": {"openai": {"id": "legacy"}, "anthropic": [{"id": "a"}]}},
    ],
)
def test_qwen_code_plan_writes_its_entry_when_no_provider_list_exists(
    tmp_path, monkeypatch, existing
):
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = tmp_path / ".qwen" / "settings.json"
    if existing is not None:
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps(existing))

    plan = build_setup_plan("qwen-code", "http://localhost:8000/v1", "local-model")
    apply_setup_plan(plan)

    written = json.loads(settings_path.read_text())
    assert [entry["id"] for entry in written["modelProviders"]["openai"]] == [
        "local-model"
    ]
    assert written["model"] == {"name": "local-model"}
    if existing and "custom" in existing:
        assert written["custom"] == {"preserved": True}
    if existing and isinstance(existing.get("modelProviders"), dict):
        assert written["modelProviders"]["anthropic"] == [{"id": "a"}]


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
        actual_url = url.full_url if isinstance(url, Request) else url
        urls.append(actual_url)
        if actual_url.endswith("/health"):
            return Response()
        return Response(json.dumps({"data": [{"id": "served-model"}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    assert (
        verify_server("http://localhost:8000/v1", "default", agent="deepseek-harness")
        == "served-model"
    )
    assert urls == [
        "http://localhost:8000/health",
        "http://localhost:8000/v1/models",
    ]


def test_keyed_model_probe_uses_exported_server_key(monkeypatch):
    from rapid_mlx.agents.adapter import _detect_running_model

    key = "test-agent-key"
    monkeypatch.setenv("RAPID_MLX_API_KEY", key)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                return
            if (
                self.path == "/v1/models"
                and self.headers.get("Authorization") == f"Bearer {key}"
            ):
                body = json.dumps(
                    {"data": [{"id": "local-model", "context_window": 131072}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(401)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        assert verify_server(base_url, "local-model", agent="codex") == "local-model"
        assert _detect_running_model(base_url) == ("local-model", 131072)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "requested,expected",
    [("local-alias", "local-alias"), ("default", "organization/full-model")],
)
def test_verify_server_reports_the_requested_advertised_alias(
    monkeypatch, requested, expected
):
    from io import BytesIO

    class Response(BytesIO):
        status = 200

    def open_url(request, **kwargs):
        if isinstance(request, str):
            return Response(b'{"ready": true}')
        return Response(
            b'{"data": [{"id": "organization/full-model"}, {"id": "local-alias"}, {"id": "default"}]}'
        )

    monkeypatch.setattr("rapid_mlx.agents.setup.urllib.request.urlopen", open_url)
    assert (
        verify_server("http://localhost:8000/v1", requested, agent="codex") == expected
    )


def test_verify_server_reports_malformed_url_as_setup_failure():
    with pytest.raises(RuntimeError, match="server is not ready"):
        verify_server("localhost:8000/v1", "default", agent="codex")


def test_claude_plan_uses_keyed_server_credential(setup_paths, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_API_KEY", "test-agent-key")
    plan = build_setup_plan("claude-code", "http://localhost:8000/v1", "local-model")
    assert plan.after["env"]["ANTHROPIC_API_KEY"] == "test-agent-key"
    assert "test-agent-key" not in plan.diff()
    assert "values hidden" in plan.diff()


def test_dsh_plan_and_profile_template_agree_on_the_provider_contract(monkeypatch):
    """The two DSH provider definitions must not drift apart.

    ``agents dsh --setup`` builds the provider block in ``agents/setup.py``
    (as a Cordis patch-layer list for dsh >= 0.2), while ``agents dsh --test``
    renders the template in ``profiles/deepseek-harness.yaml``. They are
    deliberately separate — only the plan adapts ``reasoningEfforts`` to the
    served model — but every other key is the same contract, and nothing but
    this test notices when an edit lands in one and not the other.
    """
    import yaml

    from rapid_mlx.agents import get_profile
    from rapid_mlx.agents.setup import build_setup_plan

    base_url = "http://localhost:8000/v1"
    model = "qwen3.6-35b-4bit"
    context = 131072

    monkeypatch.setattr(
        "rapid_mlx.agents.setup._dsh_patch_path",
        lambda: __import__("pathlib").Path("/nonexistent/cordis.patch.yml"),
    )
    plan = build_setup_plan("dsh", base_url, model, context_length=context)
    planned = {layer["id"]: layer["config"] for layer in plan.after}["llm-pi-ai"][
        "providers"
    ]["rapid-mlx"]

    profile = get_profile("deepseek-harness")
    rendered = yaml.safe_load(
        profile.render_config(base_url, model, context_length=context)
    )
    assert isinstance(rendered, list), "dsh patch template must be a top-level list"
    templated = {layer["id"]: layer["config"] for layer in rendered}["llm-pi-ai"][
        "providers"
    ]["rapid-mlx"]

    for key in (
        "displayName",
        "apiKeyEnv",
        "api",
        "baseURL",
        "defaultContextWindow",
        "defaultMaxTokens",
    ):
        assert planned[key] == templated[key], f"DSH provider key drifted: {key}"

    assert {layer["id"]: layer["config"] for layer in plan.after}[
        "agent-default-model"
    ] == {layer["id"]: layer["config"] for layer in rendered}["agent-default-model"]

    planned_model = planned["models"][0]
    templated_model = templated["models"][0]
    for key in ("id", "name", "contextWindow", "maxTokens"):
        assert planned_model[key] == templated_model[key], (
            f"DSH model key drifted: {key}"
        )

    planned_model = planned["models"][0]
    templated_model = templated["models"][0]
    for key in ("id", "name", "contextWindow", "maxTokens"):
        assert planned_model[key] == templated_model[key], (
            f"DSH model key drifted: {key}"
        )


@pytest.fixture
def builtin_profiles(tmp_path, monkeypatch):
    """Reload the agent registry from the repo's built-in profiles only.

    ``load_profiles`` also overlays ``~/.rapid-mlx/agents``; a machine
    with user profiles there would change the counts the footer test
    pins. Point HOME at an empty tmp dir for the reload, then restore
    the real registry afterwards.
    """
    from rapid_mlx import agents as agents_registry

    monkeypatch.setenv("HOME", str(tmp_path))
    agents_registry.load_profiles()
    yield agents_registry
    monkeypatch.undo()
    agents_registry.load_profiles()


def test_agents_continue_dev_resolves_to_the_continue_profile(builtin_profiles):
    """``agents continue-dev`` is the launch registry's slug for the same
    product; it must resolve to the exact profile ``agents continue``
    uses (#2082)."""
    from rapid_mlx.agents import get_profile

    canonical = get_profile("continue")
    aliased = get_profile("continue-dev")
    assert canonical is not None
    assert aliased is canonical


def test_framework_kind_comes_from_profile_metadata(builtin_profiles):
    """Exactly the three framework profiles declare ``kind: framework``;
    every other profile defaults to ``agent`` (#2082)."""
    from rapid_mlx.agents import list_profiles

    profiles = list_profiles()
    frameworks = {p.name for p in profiles if p.kind == "framework"}
    assert frameworks == {"langchain", "pydanticai", "smolagents"}
    assert all(p.kind in {"agent", "framework"} for p in profiles)


def test_agents_footer_counts_agents_and_frameworks_separately(
    builtin_profiles, monkeypatch, capsys
):
    """The ``rapid-mlx agents`` footer must not count frameworks as
    agents: 14 rows are 11 agents + 3 frameworks (#2082)."""
    import rapid_mlx.cli as cli

    monkeypatch.setattr("sys.argv", ["rapid-mlx", "agents"])
    cli.main()
    out = capsys.readouterr().out
    assert "11 agents + 3 frameworks supported" in out
    assert "14 agents supported" not in out
    assert "GitHub" in out
    assert "tools" in out
    assert "FC = function calling" in out


def test_cli_parser_exposes_setup_safety_flags(monkeypatch):
    import rapid_mlx.cli as cli

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
    import rapid_mlx.cli as cli

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
        "rapid_mlx.agents.setup.verify_server",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("connection refused")
        ),
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    assert continue_path.exists()
    output = capsys.readouterr().out
    assert "Configuration was saved, but the connection check failed" in output
    assert "Setup incomplete" not in output


def test_cli_reports_a_failed_config_write_without_a_traceback(
    setup_paths, monkeypatch, capsys
):
    import rapid_mlx.cli as cli

    _, continue_path = setup_paths
    monkeypatch.setattr(
        "sys.argv",
        [
            "rapid-mlx",
            "agents",
            "continue",
            "--setup",
            "--yes",
            "--no-check",
            "--model",
            "qwen3.5-4b-4bit",
        ],
    )
    monkeypatch.setattr(
        "rapid_mlx.agents.setup.apply_setup_plan",
        lambda _plan: (_ for _ in ()).throw(PermissionError("read-only directory")),
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    assert not continue_path.exists()
    assert "setup failed: read-only directory" in capsys.readouterr().out


def test_user_continue_dev_overlay_wins_over_the_builtin_alias(tmp_path, monkeypatch):
    """The ``continue-dev`` -> ``continue`` alias must be a FALLBACK only: a
    user who installs their own ``~/.rapid-mlx/agents/continue-dev.yaml``
    gets that profile, not the aliased built-in (#2082 codex review)."""
    from rapid_mlx import agents as agents_registry
    from rapid_mlx.agents import get_profile

    user_dir = tmp_path / ".rapid-mlx" / "agents"
    user_dir.mkdir(parents=True)
    (user_dir / "continue-dev.yaml").write_text(
        "name: continue-dev\ndisplay_name: My Custom Continue\nconfig:\n  type: env\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    agents_registry.load_profiles()
    try:
        profile = get_profile("continue-dev")
        assert profile is not None
        assert profile.display_name == "My Custom Continue", (
            "user overlay must beat the built-in continue-dev alias"
        )
        # The alias still works when no overlay exists for the other slug.
        assert get_profile("continue") is not None
    finally:
        monkeypatch.undo()
        agents_registry.load_profiles()
