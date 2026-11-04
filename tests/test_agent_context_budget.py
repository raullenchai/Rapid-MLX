"""Agent writers share served and offline memory-fit context budgets."""

import json
from types import SimpleNamespace

import pytest

from rapid_mlx.agents import adapter, context_budget, get_profile


@pytest.fixture(autouse=True)
def cpu_memory_probe(monkeypatch):
    monkeypatch.setattr(
        "rapid_mlx.byom.preflight.mlx_recommended_working_set_bytes", lambda: None
    )


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "_fetch_models",
        lambda *_a, **_k: [
            {"id": "local-model", "context_window": 262144, "max_model_len": 60000}
        ],
    )


@pytest.mark.parametrize(
    "agent, expected",
    [
        ("claude-code", 60000),
        ("codex", 60000),
        ("opencode", 54000),
        ("pi", 54000),
        ("dsh", 54000),
        ("qwen-code", 54000),
        ("generic", 54000),
    ],
)
def test_client_policy(served, agent, expected):
    assert (
        context_budget.agent_context_budget(
            agent, "http://localhost:8000/v1", "local-model"
        )
        == expected
    )


def test_hermes_supported_range(served):
    with pytest.raises(ValueError, match="64,000"):
        context_budget.agent_context_budget(
            "hermes", "http://localhost/v1", "local-model"
        )
    assert context_budget.context_budget_for_ceiling("hermes", 262144) == 131072
    assert context_budget.context_budget_for_ceiling("hermes", 98304) == 98304


@pytest.mark.parametrize(
    "agent, expected",
    [("opencode", 54000), ("codex", 60000), ("pi", 54000), ("qwen-code", 54000)],
)
def test_profile_and_setup_instructions_use_memory_window(served, agent, expected):
    profile = get_profile(agent)
    rendered = profile.render_config(
        "http://localhost/v1", "local-model", "1.2.0" if agent == "opencode" else None
    )
    assert str(expected) in rendered
    assert "262144" not in rendered
    assert str(expected) in adapter.get_setup_instructions(
        profile,
        "http://localhost/v1",
        "local-model",
        "1.2.0" if agent == "opencode" else None,
    )


def test_small_opencode_output_is_inside_window():
    profile = get_profile("opencode")
    rendered = json.loads(
        profile.render_config(
            "http://localhost/v1", "local-model", "1.2.0", context_length=100
        )
    )
    limit = rendered["provider"]["rapid-mlx"]["models"]["local-model"]["limit"]
    assert limit == {"context": 100, "output": 25}


def test_large_opencode_output_keeps_historical_reservation():
    profile = get_profile("opencode")
    rendered = json.loads(
        profile.render_config(
            "http://localhost/v1", "local-model", "1.2.0", context_length=200_000
        )
    )
    limit = rendered["provider"]["rapid-mlx"]["models"]["local-model"]["limit"]
    assert limit == {"context": 200_000, "output": 8192}


def test_model_selection_and_validity(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "_fetch_models",
        lambda *_a, **_k: [
            {"id": "a", "context_window": 100, "max_model_len": 200},
            {"id": "b", "context_window": 200, "max_model_len": 150},
        ],
    )
    assert context_budget.served_context_ceiling("http://localhost", "a") == 100
    assert (
        context_budget.served_context_ceiling("http://localhost", "b", api_key="secret")
        == 150
    )
    assert context_budget.served_context_ceiling("http://localhost", "missing") is None
    monkeypatch.setattr(
        adapter, "_fetch_models", lambda *_a, **_k: [{"context_window": 200}]
    )
    assert context_budget.served_context_ceiling("http://localhost", "alias") == 200
    assert (
        context_budget.served_context_ceiling(
            "http://localhost", "alias", require_served=True
        )
        is None
    )
    monkeypatch.setattr(
        adapter,
        "_fetch_models",
        lambda *_a, **_k: [{"context_window": True, "max_model_len": "bad"}],
    )
    assert context_budget.served_context_ceiling("http://localhost", "alias") is None


def test_offline_uses_shared_memory_fit(monkeypatch, tmp_path):
    from rapid_mlx import model_metadata
    from rapid_mlx.byom import preflight
    from rapid_mlx.kv_estimation import footprint_from_config, memory_fit_context
    from rapid_mlx.memory_budget import plan_metal_limit

    config = {
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_attention_heads": 4,
        "head_dim": 8,
        "max_position_embeddings": 262144,
    }
    (tmp_path / "model.safetensors").write_bytes(b"w" * 1000)
    metadata = model_metadata.ModelMetadata(config, None, tmp_path)
    monkeypatch.setattr(model_metadata, "read_model_metadata", lambda _: metadata)
    monkeypatch.setattr(
        preflight, "mlx_recommended_working_set_bytes", lambda: 10_000_000
    )
    monkeypatch.setattr(adapter, "_fetch_models", lambda *_a, **_k: [])
    plan = plan_metal_limit(weights_bytes=1000, device_budget_bytes=10_000_000)
    expected = memory_fit_context(
        footprint_from_config(config), plan.limit_bytes - 1000, 262144
    )
    assert context_budget.offline_context_ceiling("test") == expected
    assert context_budget.agent_context_budget(
        "pi", "http://localhost/v1", "test"
    ) == int(expected * 0.9)


def test_offline_native_and_unknown_fallbacks(monkeypatch):
    from rapid_mlx import model_metadata

    monkeypatch.setattr(adapter, "_fetch_models", lambda *_a, **_k: [])
    monkeypatch.setattr(
        model_metadata,
        "read_model_metadata",
        lambda _: SimpleNamespace(
            config={"text_config": {"max_position_embeddings": 40000}}
        ),
    )
    assert context_budget.offline_context_ceiling("test") == 40000
    assert (
        context_budget.agent_context_budget(
            "pi", "http://localhost", "test", offline=True
        )
        == 36000
    )
    monkeypatch.setattr(model_metadata, "read_model_metadata", lambda _: None)
    assert context_budget.agent_context_budget("pi", "http://localhost", "test") is None
    assert (
        context_budget.agent_context_budget(
            "pi", "http://localhost", "test", fallback=32768
        )
        == 32768
    )
    monkeypatch.setattr(
        model_metadata,
        "read_model_metadata",
        lambda _: (_ for _ in ()).throw(OSError()),
    )
    assert context_budget.offline_context_ceiling("test") is None


def test_fetch_models_passes_isolated_key(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return b'{"data":[{"id":"m","context_window":200,"max_model_len":100}]}'

    def fetch(request, timeout):
        assert request.full_url == "http://localhost:8000/v1/models"
        assert request.headers["Authorization"] == "Bearer isolated-key"
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    assert (
        context_budget.served_context_ceiling(
            "http://localhost:8000", "m", api_key="isolated-key"
        )
        == 100
    )


@pytest.mark.parametrize(
    "agent, expected",
    [
        ("opencode", 54000),
        ("codex", 60000),
        ("claude-code", 60000),
        ("pi", 54000),
        ("dsh", 54000),
        ("qwen-code", 54000),
        ("goose", 54000),
    ],
)
def test_agents_setup_cli_writer(
    served, monkeypatch, tmp_path, capsys, agent, expected
):
    from rapid_mlx import cli
    from rapid_mlx.launch import claude_code

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(
        claude_code, "current_config_path", lambda: tmp_path / "claude.json"
    )
    cli.agents_command(
        SimpleNamespace(
            agent_name=agent,
            base_url="http://localhost/v1",
            test=False,
            setup=True,
            model="local-model",
            agent_version="1.2.0" if agent == "opencode" else None,
            dry_run=False,
            yes=True,
            no_check=True,
        )
    )
    output = capsys.readouterr().out + "\n".join(
        path.read_text() for path in tmp_path.rglob("*") if path.is_file()
    )
    assert str(expected) in output
    assert "262144" not in output


@pytest.mark.parametrize(
    "agent, expected", [("opencode", 54000), ("codex", 60000), ("pi", 54000)]
)
def test_isolated_writer(served, monkeypatch, tmp_path, agent, expected):
    from rapid_mlx.agents import isolated

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(
        "rapid_mlx.agents.opencode_version.installed_version", lambda **_k: "1.2.0"
    )
    monkeypatch.setattr(
        "rapid_mlx.agents.opencode_version.resolve_executable", lambda: "opencode"
    )
    isolated.configure(agent, "local-model", port=8000)
    contents = "\n".join(
        path.read_text() for path in tmp_path.rglob("*") if path.is_file()
    )
    assert str(expected) in contents
    assert "262144" not in contents


def test_offline_catalog_weights_and_physical_budget(monkeypatch):
    from rapid_mlx import model_metadata, recommendations
    from rapid_mlx.byom import preflight

    monkeypatch.setattr(
        model_metadata,
        "read_model_metadata",
        lambda _: SimpleNamespace(
            config={
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "hidden_size": 32,
                "max_position_embeddings": 262144,
            }
        ),
    )
    monkeypatch.setattr(recommendations, "recommendation_footprint_gb", lambda _: 1)
    monkeypatch.setattr(preflight, "mlx_recommended_working_set_bytes", lambda: None)
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 2 * 1024**3)
    assert context_budget.offline_context_ceiling("test") == 262144


def test_run_writer_uses_served_budget(served, monkeypatch, tmp_path):
    from rapid_mlx.launch import claude_code
    from rapid_mlx.run import cli as run_cli

    path = tmp_path / "settings.json"
    monkeypatch.setattr(claude_code, "current_config_path", lambda: path)
    args = SimpleNamespace(no_setup=False, dry_run=False, yes=True)
    assert (
        run_cli._attach_and_configure(
            "http://localhost", "local-model", get_profile("claude-code"), args
        )
        == 0
    )
    assert (
        json.loads(path.read_text())["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "60000"
    )


def test_hermes_setup_and_isolated_reject_impossible_minimum(
    served, monkeypatch, tmp_path
):
    from rapid_mlx import cli
    from rapid_mlx.agents import isolated

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SystemExit):
        cli.agents_command(
            SimpleNamespace(
                agent_name="hermes",
                base_url="http://localhost/v1",
                test=False,
                setup=True,
                model="local-model",
                agent_version=None,
                dry_run=True,
                yes=True,
                no_check=True,
            )
        )
    with pytest.raises(ValueError, match="64,000"):
        isolated.configure("hermes", "local-model", port=8000)


def test_launch_claude_writer_uses_served_budget(served, monkeypatch, tmp_path):
    from rapid_mlx.launch import claude_code
    from rapid_mlx.launch import cli as launch_cli

    path = tmp_path / "settings.json"
    monkeypatch.setattr(claude_code, "current_config_path", lambda: path)
    monkeypatch.setattr(claude_code, "detect", lambda: True)
    launch_cli.launch_command(
        SimpleNamespace(
            client="claude-code",
            all=False,
            model="local-model",
            server_url="http://localhost:8000",
            port=8000,
            start_server=False,
            dry_run=False,
            json=False,
            legacy_config=True,
        )
    )
    assert (
        json.loads(path.read_text())["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "60000"
    )


def test_offline_known_weights_without_device_falls_back_to_native(monkeypatch):
    from rapid_mlx import model_metadata, recommendations

    monkeypatch.setattr(
        model_metadata,
        "read_model_metadata",
        lambda _: SimpleNamespace(
            config={
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "hidden_size": 32,
                "max_position_embeddings": 262144,
            }
        ),
    )
    monkeypatch.setattr(recommendations, "recommendation_footprint_gb", lambda _: 1)
    monkeypatch.setattr("rapid_mlx.byom.preflight.physical_ram_bytes", lambda: None)
    assert context_budget.offline_context_ceiling("test") == 262144
