"""Agent setup must observe a ready model before touching an agent config."""

import json
from io import BytesIO
from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx.agents import setup


def test_single_advertised_wrong_model_is_rejected(monkeypatch):
    class Response(BytesIO):
        status = 200

    def open_url(request, **_kwargs):
        if isinstance(request, str):
            return Response(b'{"ready": true}')
        return Response(b'{"data": [{"id": "other-model"}]}')

    monkeypatch.setattr(setup.urllib.request, "urlopen", open_url)
    with pytest.raises(RuntimeError, match="does not advertise model"):
        setup.verify_server("http://localhost:8000/v1", "wanted", agent="codex")


def _args(agent, model="wanted"):
    return SimpleNamespace(
        agent_name=agent,
        base_url="http://127.0.0.1:8000/v1",
        test=False,
        setup=True,
        model=model,
        agent_version=None,
        dry_run=False,
        yes=True,
        no_check=False,
    )


def test_wait_retries_until_model_is_advertised(monkeypatch):
    attempts = []

    def probe(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("server reported no models")
        return "wanted"

    monkeypatch.setattr(setup, "verify_server", probe)
    monkeypatch.setattr(setup.time, "sleep", lambda _seconds: None)
    assert (
        setup.wait_for_server("http://localhost:8000/v1", "wanted", agent="codex")
        == "wanted"
    )
    assert len(attempts) == 3


def test_wait_reports_only_final_model_failure(monkeypatch):
    failures = []
    monkeypatch.setattr(
        setup,
        "verify_server",
        lambda *_a, **_k: (_ for _ in ()).throw(
            setup.ModelNotAdvertisedError("server does not advertise model 'wanted'")
        ),
    )
    monkeypatch.setattr(
        setup, "track_agent_configure_failed", lambda *v: failures.append(v)
    )
    with pytest.raises(RuntimeError, match="does not advertise"):
        setup.wait_for_server(
            "http://localhost:8000/v1", "wanted", agent="codex", timeout=0
        )
    assert failures == [("model_not_advertised", "codex")]


def test_wait_reports_server_with_no_models_at_timeout(monkeypatch):
    class Response(BytesIO):
        status = 200

    def open_url(request, **_kwargs):
        if isinstance(request, str):
            return Response(b'{"ready": true}')
        return Response(b'{"data": []}')

    failures = []
    monkeypatch.setattr(setup.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(
        setup, "track_agent_configure_failed", lambda *values: failures.append(values)
    )
    with pytest.raises(RuntimeError, match="reported no models"):
        setup.wait_for_server(
            "http://localhost:8000/v1", "wanted", agent="codex", timeout=0
        )
    assert failures == [("server_no_models", "codex")]


def test_plan_failure_gives_versioned_retry_without_writing(
    monkeypatch, tmp_path, capsys
):
    config = tmp_path / "config.yaml"
    monkeypatch.setattr(setup.continue_dev, "current_config_path", lambda: config)
    monkeypatch.setattr(setup, "wait_for_server", lambda *_a, **_k: "wanted")

    def fail_plan(*_args, **_kwargs):
        raise OSError("read-only config directory")

    monkeypatch.setattr(setup, "build_setup_plan", fail_plan)
    args = _args("continue")
    args.agent_version = "1.0 rc"
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(args)
    assert exc.value.code == 1
    assert not config.exists()
    output = capsys.readouterr().out
    assert "setup failed: read-only config directory" in output
    assert (
        "Retry: rapid-mlx agents continue --setup --yes --model wanted "
        "--base-url http://127.0.0.1:8000/v1 --agent-version '1.0 rc'"
    ) in output


def test_first_class_does_not_write_when_server_unready(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    monkeypatch.setattr(setup.continue_dev, "current_config_path", lambda: config)
    monkeypatch.setattr(
        setup,
        "wait_for_server",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("model not advertised")),
    )
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(_args("continue"))
    assert exc.value.code == 1
    assert not config.exists()
    output = capsys.readouterr().out
    assert "model not advertised" in output
    assert "rapid-mlx agents continue --setup --yes --model wanted" in output


def test_generic_does_not_write_when_server_unready(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(
        setup,
        "wait_for_server",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("health returned HTTP 503")
        ),
    )
    with pytest.raises(SystemExit) as exc:
        cli.agents_command(_args("codex"))
    assert exc.value.code == 1
    assert not (tmp_path / "config.toml").exists()
    output = capsys.readouterr().out
    assert "health returned HTTP 503" in output
    assert "rapid-mlx agents codex --setup --yes --model wanted" in output


def test_model_that_appears_during_wait_is_written(monkeypatch, tmp_path):
    from rapid_mlx.agents import adapter

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(adapter, "_detect_running_model", lambda _url: (None, None))
    monkeypatch.setattr(setup, "wait_for_server", lambda *_a, **_k: "now-ready")
    cli.agents_command(_args("codex", model=None))
    assert 'model = "now-ready"' in (tmp_path / "config.toml").read_text()


def test_explicit_model_context_is_fetched_after_readiness(monkeypatch, tmp_path):
    from rapid_mlx.agents import adapter

    config = tmp_path / "settings.json"
    monkeypatch.setattr(setup.claude_code, "current_config_path", lambda: config)
    ready = False

    def wait(*_args, **_kwargs):
        nonlocal ready
        ready = True
        return "wanted"

    def context(*_args):
        return 16384 if ready else None

    monkeypatch.setattr(setup, "wait_for_server", wait)
    monkeypatch.setattr(adapter, "fetch_context_window", context)
    cli.agents_command(_args("claude-code"))
    assert (
        json.loads(config.read_text())["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"]
        == "16384"
    )
