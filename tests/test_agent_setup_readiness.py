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


@pytest.mark.parametrize("owned", [False, True])
def test_loading_wait_is_bounded_and_visible(monkeypatch, capsys, owned):
    from rapid_mlx.agents import server_discovery as discovery

    clock = [0.0]
    monkeypatch.setattr(setup.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        setup.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    server = discovery.Server(42, 8123, "wanted", listening=True) if owned else None
    monkeypatch.setattr(discovery, "server_for_url", lambda _url: server)
    monkeypatch.setattr(discovery, "pid_alive", lambda _pid: True)
    shown = []
    monkeypatch.setattr(
        discovery.Progress, "show", lambda self, url, now: shown.append(url)
    )
    monkeypatch.setattr(
        setup,
        "verify_server",
        lambda *_a, **_k: (_ for _ in ()).throw(
            setup.ServerLoadingError("loading", "model_not_loaded")
        ),
    )
    failures = []
    monkeypatch.setattr(
        setup, "track_agent_configure_failed", lambda *args: failures.append(args)
    )
    with pytest.raises(setup.ServerReadinessError, match="did not become ready"):
        setup.wait_for_server(
            "http://localhost:8123/v1", "wanted", agent="opencode", timeout=1
        )
    assert 1 <= clock[0] <= 1.25
    assert shown
    assert failures == [("server_not_ready", "opencode", "startup_timeout")]


def test_loading_unowned_server_becomes_ready(monkeypatch):
    from rapid_mlx.agents import server_discovery as discovery

    monkeypatch.setattr(discovery, "server_for_url", lambda _url: None)
    monkeypatch.setattr(discovery.Progress, "show", lambda *_a: None)
    monkeypatch.setattr(setup.time, "sleep", lambda _s: None)
    attempts = []

    def verify(url, model, **kwargs):
        attempts.append(url)
        if len(attempts) == 1:
            raise setup.ServerLoadingError("loading", "model_not_loaded")
        return "wanted"

    monkeypatch.setattr(setup, "verify_server", verify)
    assert (
        setup.wait_for_server("http://localhost:8123/v1", "wanted", agent="pi")
        == "wanted"
    )
    assert attempts == ["http://localhost:8123/v1"] * 2


def test_health_liveness_does_not_skip_model_loading(monkeypatch):
    class Response(BytesIO):
        status = 200

    requests = []

    def open_url(request, **_kwargs):
        requests.append(request)
        return Response(b'{"ready": false}')

    monkeypatch.setattr(setup.urllib.request, "urlopen", open_url)
    with pytest.raises(setup.ServerLoadingError):
        setup.verify_server(
            "http://localhost:8123/v1", "wanted", agent="opencode", track_failure=False
        )
    assert requests == ["http://localhost:8123/health"]


def test_setup_ready_timeout_parser():
    from rapid_mlx.cli_parser import build_parser

    parser = build_parser()
    assert parser.parse_args(["agents", "opencode", "--setup"]).ready_timeout == 600
    assert (
        parser.parse_args(["agents", "opencode", "--ready-timeout", "20"]).ready_timeout
        == 20
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["agents", "opencode", "--ready-timeout", "0"])


def test_health_loading_then_ready_waits_before_model_discovery(monkeypatch):
    from rapid_mlx.agents import server_discovery as discovery

    class Response(BytesIO):
        status = 200

    calls = []

    def open_url(request, **_kwargs):
        url = request if isinstance(request, str) else request.full_url
        calls.append(url)
        if url.endswith("/health"):
            return Response(
                b'{"ready": false}' if len(calls) == 1 else b'{"ready": true}'
            )
        return Response(b'{"data": [{"id": "wanted"}]}')

    monkeypatch.setattr(setup.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(discovery, "server_for_url", lambda _url: None)
    shown = []
    monkeypatch.setattr(discovery.Progress, "show", lambda *_a: shown.append(True))
    monkeypatch.setattr(setup.time, "sleep", lambda _s: None)
    assert (
        setup.wait_for_server("http://localhost:8123/v1", "wanted", agent="opencode")
        == "wanted"
    )
    assert calls == ["http://localhost:8123/health"] * 2 + [
        "http://localhost:8123/v1/models"
    ]
    assert shown == [True]
